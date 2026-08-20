"""
gemma_inference_test_explain_then_classify_v1.py — Two-stage same-model stance detection on the blind test set
using Gemma 4 31B via Cerebras and temperature 0.1.

Stage 1: Arabic-to-English explanation/mental rewrite.
Stage 2: Stance classification using original tweet + Stage 1 explanation.

Usage:
    python src/gemma_inference_test_explain_then_classify_v1.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from typing import Optional

import pandas as pd

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import LABEL2ID, write_pred_txt

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")

TEST_CSV_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_PATH = os.path.join(PRED_DIR, "results_gemma4_31b_cerebras_test_explain_then_classify_v1.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_gemma4_31b_cerebras_test_explain_then_classify_v1_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID = "gemma-4-31b"
TOP_P = 0.95

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}

TARGET_AR = {
    "Women Driving": "قيادة المرأة للسيارة"
}

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",      # original 0-based index in ground_truth.csv
    "target",
    "gold_label",      # empty or absent for test set
    "stage1_explanation",
    "predicted_label",
    "model_id",
    "parse_success",
    "raw_response",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
STAGE1_SYSTEM_PROMPT = """You are an expert Arabic language analyst.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

### Task
Given an Arabic tweet, rewrite its intended literal meaning in clear English, preserving the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis. Do not classify stance. Do not mention Favor, Against, or None. Only explain what the tweet is actually saying and who or what it is directed at.

Respond with plain explanation only."""

STAGE2_SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

### Task
You will be given the original Arabic tweet and an explanation of its intended meaning. Classify the writer's stance toward the target as exactly one of:

- Favor
- Against
- None

Use the explanation to inform your reading, but weigh the original tweet's actual wording, tone, and emojis as well — the explanation is a aid, not a replacement for the tweet itself.

Then determine the stance toward the target itself, not toward other people, quoted opinions, related entities, or hashtags.

Guidelines:

- Favor: supports, defends, promotes, or welcomes the target.
- Against: opposes, criticizes, rejects, or mocks the target.
- None: no clear stance toward the target.

Important:

- Determine where praise or criticism is directed. Negative language toward opponents of the target is usually Favor, not Against.
- Hashtags may be ironic or hijacked. Never infer stance from hashtags alone.
- Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording.
- Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor.
- If the stance toward the target cannot reasonably be inferred, output None.

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text."""

def build_stage2_user_message(tweet_text: str, explanation: str) -> str:
    return f"Tweet: {tweet_text}\nExplanation: {explanation}"

def parse_response(raw: str) -> tuple[str, bool]:
    """Parse a model response and return (stance_label, parse_success)."""
    raw = raw.strip()

    # Attempt 1: exact word match
    if raw in VALID_LABELS:
        return raw, True

    # Attempt 2: case-insensitive exact match
    for label in VALID_LABELS:
        if raw.lower() == label.lower():
            return label, True

    # Attempt 3: bare label substring anywhere
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw.lower():
            return label, False

    # Attempt 4: JSON fallback
    try:
        obj = json.loads(raw)
        stance = str(obj.get("stance", "")).strip()
        if stance in VALID_LABELS:
            return stance, False
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r'"stance"\s*:\s*"(Favor|Against|None)"', raw)
    if match:
        return match.group(1), False

    return "None", False

# ── Rate-limit pacing logic (Cerebras 5 RPM ceiling) ──────────────────────────
def compute_pace_delay(headers: dict) -> float:
    base_delay = 12.0
    try:
        remaining_req_min = int(headers.get("x-ratelimit-remaining-requests-minute", 5))
    except (ValueError, TypeError):
        remaining_req_min = 5

    try:
        remaining_tok_min = int(headers.get("x-ratelimit-remaining-tokens-minute", 30_000))
    except (ValueError, TypeError):
        remaining_tok_min = 30_000

    print(
        f"    Cerebras RL — remaining requests (min): {remaining_req_min}  |  "
        f"remaining tokens (min): {remaining_tok_min}"
    )

    if remaining_req_min <= 1:
        print(f"    ⚠️  Cerebras requests budget almost exhausted ({remaining_req_min} left) → sleeping 60s to replenish")
        return 60.0

    if remaining_tok_min < 1000:
        print(f"    ⚠️  Cerebras token budget low ({remaining_tok_min} left) → sleeping 30s to replenish")
        return 30.0

    return base_delay

def _parse_reset_duration(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip()
    if s.endswith("s") and not s.endswith("ms"):
        try:
            return float(s[:-1])
        except ValueError:
            pass
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        pass
    
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if m and (m.group(1) or m.group(2)):
        return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return 0.0

def load_request_timestamps(raw_log_path: str) -> list[float]:
    timestamps = []
    if not os.path.exists(raw_log_path):
        return timestamps
    now = time.time()
    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Load both timestamps if they exist
                ts1 = obj.get("timestamp_stage1")
                ts2 = obj.get("timestamp_stage2")
                if ts1 and (now - ts1 < 3600):
                    timestamps.append(ts1)
                if ts2 and (now - ts2 < 3600):
                    timestamps.append(ts2)
            except Exception:
                pass
    return sorted(timestamps)

def load_completed_indices(results_path: str) -> set[int]:
    if not os.path.exists(results_path):
        return set()
    completed: set[int] = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                completed.add(int(row["row_index"]))
            except (KeyError, ValueError):
                pass
    return completed

def append_result_row(results_path: str, row: dict) -> None:
    write_header = not os.path.exists(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in RESULTS_FIELDNAMES})

def call_cerebras_stage1(
    client,
    row_index: int,
    tweet_text: str,
    max_retries: int = 4,
) -> tuple[str, dict]:
    """Stage 1: Explanation Generation (T=0.1, max_tokens=384)"""
    for attempt in range(max_retries):
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                    {"role": "user",   "content": tweet_text},
                ],
                temperature=0.1,
                top_p=TOP_P,
                max_tokens=384,
                reasoning_effort="none",
            )
            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()
            explanation = completion.choices[0].message.content or ""
            return explanation.strip(), rl_headers

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 60.0
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                    if "retry-after" in headers:
                        try:
                            retry_after = float(headers["retry-after"])
                        except ValueError:
                            pass
                print(f"    ⚠️  [Stage 1] 429 Rate Limit on row {row_index} (attempt {attempt + 1}/{max_retries}) — sleeping {retry_after:.1f}s")
                time.sleep(retry_after)
                continue

            wait = 5.0 * (2 ** attempt)
            print(f"    ❌ [Stage 1] Row {row_index} attempt {attempt + 1}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    return "ERROR: max retries exceeded for explanation", {}

def call_cerebras_stage2(
    client,
    row_index: int,
    tweet_text: str,
    explanation: str,
    max_retries: int = 4,
) -> tuple[dict, dict]:
    """Stage 2: Stance Classification (T=0.1, max_tokens=10)"""
    user_msg = build_stage2_user_message(tweet_text, explanation)
    for attempt in range(max_retries):
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                top_p=TOP_P,
                max_tokens=10,
                reasoning_effort="none",
            )
            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()
            raw_text = completion.choices[0].message.content or ""
            predicted, parse_ok = parse_response(raw_text)

            result = {
                "predicted_label": predicted,
                "model_id":        completion.model,
                "parse_success":   parse_ok,
                "raw_response":    raw_text,
            }
            return result, rl_headers

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 60.0
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                    if "retry-after" in headers:
                        try:
                            retry_after = float(headers["retry-after"])
                        except ValueError:
                            pass
                print(f"    ⚠️  [Stage 2] 429 Rate Limit on row {row_index} (attempt {attempt + 1}/{max_retries}) — sleeping {retry_after:.1f}s")
                time.sleep(retry_after)
                continue

            wait = 5.0 * (2 ** attempt)
            print(f"    ❌ [Stage 2] Row {row_index} attempt {attempt + 1}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    result = {
        "predicted_label": "None",
        "model_id":        MODEL_ID,
        "parse_success":   False,
        "raw_response":    "ERROR: max retries exceeded for classification",
    }
    return result, {}

def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df = df.rename(columns={"id": "ID", "tweet_text": "text"})
    df.columns = df.columns.astype(str).str.strip()
    return df

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        print("❌ CEREBRAS_API_KEY not found in .env or environment. Cannot proceed.")
        sys.exit(1)
    print("✅ CEREBRAS_API_KEY loaded.")

    print(f"\nLoading {TEST_CSV_PATH} …")
    test_df = load_test_data(TEST_CSV_PATH)
    print(f"  Loaded {len(test_df)} rows.")

    run_df = test_df.copy()
    selected_indices = list(run_df.index)
    results_path = RESULTS_PATH
    raw_log_path = RAW_LOG_PATH

    # Initialize client
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    completed = load_completed_indices(results_path)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done (up to index {max(completed) if completed else 0}).")
    else:
        print(f"\n🆕 Starting fresh run.")

    request_timestamps = load_request_timestamps(raw_log_path)
    if request_timestamps:
        print(f"📊 Loaded {len(request_timestamps)} request timestamps from last 60 minutes.")

    print(f"── Inference config ──")
    print(f"   Model:            {MODEL_ID} (Two-Stage)")
    print(f"   Stage 1 Prompt:   STAGE1_SYSTEM_PROMPT")
    print(f"   Stage 2 Prompt:   STAGE2_SYSTEM_PROMPT")
    print(f"   Results CSV:      {results_path}")
    print()

    total = len(run_df)
    new_calls = 0
    skipped = 0
    rl_headers = {}

    for position, (_, row) in enumerate(run_df.iterrows(), start=1):
        row_index = int(row.name)

        if row_index in completed:
            skipped += 1
            continue

        target = str(row["target"])
        text   = str(row["text"])

        # Pacing checkpoint before Stage 1 Call
        now = time.time()
        request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        if len(request_timestamps) >= 150:
            oldest_ts = request_timestamps[0]
            sleep_time = oldest_ts + 3600 - now + 1.0
            
            if rl_headers:
                reset_req_hour_str = rl_headers.get("x-ratelimit-reset-requests-hour") or rl_headers.get("x-ratelimit-reset-requests-day")
                if reset_req_hour_str:
                    try:
                        header_sleep = _parse_reset_duration(str(reset_req_hour_str))
                        if header_sleep > sleep_time:
                            sleep_time = header_sleep + 1.0
                    except Exception:
                        pass

            resume_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + sleep_time))
            print(f"\n⏳ HOURLY RATE LIMIT REACHED (150 requests in last 60 minutes).")
            print(f"   Entering cooldown: sleeping for {sleep_time:.1f} seconds (~{sleep_time/60:.1f} minutes).")
            print(f"   Script will resume at: {resume_time}\n")
            time.sleep(sleep_time)

            now = time.time()
            request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r}")
        
        # --- Stage 1 Call ---
        explanation, rl_headers = call_cerebras_stage1(client, row_index, text)
        ts_s1 = time.time()
        request_timestamps.append(ts_s1)
        new_calls += 1

        delay = compute_pace_delay(rl_headers)
        if delay > 0:
            time.sleep(delay)

        # Pacing checkpoint before Stage 2 Call
        now = time.time()
        request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        if len(request_timestamps) >= 150:
            oldest_ts = request_timestamps[0]
            sleep_time = oldest_ts + 3600 - now + 1.0
            
            resume_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + sleep_time))
            print(f"\n⏳ HOURLY RATE LIMIT REACHED between stages (150 requests in last 60 minutes).")
            print(f"   Entering cooldown: sleeping for {sleep_time:.1f} seconds (~{sleep_time/60:.1f} minutes).")
            print(f"   Script will resume at: {resume_time}\n")
            time.sleep(sleep_time)

            now = time.time()
            request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        # --- Stage 2 Call ---
        result, rl_headers = call_cerebras_stage2(client, row_index, text, explanation)
        ts_s2 = time.time()
        request_timestamps.append(ts_s2)
        new_calls += 1

        result["row_index"] = row_index
        result["target"] = target
        result["gold_label"] = ""
        result["stage1_explanation"] = explanation

        append_result_row(results_path, result)
        completed.add(row_index)

        log_entry = {
            "row_index":         row_index,
            "target":            target,
            "gold_label":        "",
            "stage1_explanation": explanation,
            "raw_response":      result["raw_response"],
            "predicted_label":   result["predicted_label"],
            "model_id":          result["model_id"],
            "parse_success":     result["parse_success"],
            "timestamp_stage1":  ts_s1,
            "timestamp_stage2":  ts_s2,
        }
        with open(raw_log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → pred={result['predicted_label']!r}")

        delay = compute_pace_delay(rl_headers)
        if delay > 0:
            time.sleep(delay)

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # Write predictions to TXT and package as ZIP
    results_df = pd.read_csv(results_path, encoding="utf-8", keep_default_na=False)
    results_df = results_df[results_df["row_index"].isin(selected_indices)].copy()

    if results_df.empty:
        print("⚠️ No results found to evaluate.")
        return

    # Sort by row_index to guarantee correct line-by-line alignment
    results_df = results_df.sort_values("row_index").copy()
    results_df.to_csv(results_path, index=False)

    y_pred = results_df["predicted_label"].tolist()
    
    txt_path = results_path.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"Saved test predictions to {txt_path}")

    # Compress into ZIP
    zip_path = results_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(txt_path, arcname=os.path.basename(txt_path))
    print(f"Saved zipped submission package to {zip_path}")

    print("\nPredicted stance distribution:")
    counts = results_df["predicted_label"].value_counts()
    print(counts.to_string())

if __name__ == "__main__":
    main()

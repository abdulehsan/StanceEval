"""
gemma_inference_test_revised_zeroshot.py — Zero-shot stance detection on the blind test set (ground_truth.csv)
using Gemma 4 31B via Cerebras with the revised prompt and temperature 0.1.

Usage:
    python src/gemma_inference_test_revised_zeroshot.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
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
RESULTS_PATH = os.path.join(PRED_DIR, "results_gemma4_31b_cerebras_test_revised_zeroshot.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_gemma4_31b_cerebras_test_revised_zeroshot_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID = "gemma-4-31b"
TEMPERATURE = 0.1  # Set to 0.1 for optimal single-word deterministic mapping
TOP_P = 0.95
MAX_TOKENS = 10  # single-word label only

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}

TARGET_AR = {
    "Women Driving": "قيادة المرأة للسيارة"
}

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",      # original 0-based index in ground_truth.csv
    "target",
    "gold_label",      # empty or absent for test set
    "predicted_label",
    "model_id",
    "parse_success",
    "raw_response",
]

# ── System Prompt (Revised) ───────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert annotator for Arabic stance detection.\n\n"
    "Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:\n\n"
    "• Favor\n"
    "• Against\n"
    "• None\n\n"
    "Before assigning a stance, mentally rewrite the tweet into its intended literal meaning while preserving the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis.\n\n"
    "Then determine the stance toward the target itself, not toward other people, quoted opinions, related entities, or hashtags.\n\n"
    "Guidelines:\n\n"
    "• Favor: supports, defends, promotes, or welcomes the target.\n"
    "• Against: opposes, criticizes, rejects, or mocks the target.\n"
    "• None: no clear stance toward the target.\n\n"
    "Important:\n\n"
    "• Determine where praise or criticism is directed. Negative language toward opponents of the target is usually Favor, not Against.\n"
    "• Hashtags may be ironic or hijacked. Never infer stance from hashtags alone.\n"
    "• Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording.\n"
    "• Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance.\n"
    "• If the stance toward the target cannot reasonably be inferred, output None.\n\n"
    "Respond with ONLY one word:\n\n"
    "Favor\n"
    "Against\n"
    "None\n\n"
    "Do not provide any explanation, punctuation, or additional text."
)

def build_user_message(target: str, text: str) -> str:
    arabic_target = TARGET_AR.get(target, target)
    return f"Target: {arabic_target}\nTweet: {text}"

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
                ts = obj.get("timestamp")
                if ts and (now - ts < 3600):
                    timestamps.append(ts)
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

def call_cerebras_once(
    client,
    row_index: int,
    target: str,
    text: str,
    max_retries: int = 4,
) -> tuple[dict, dict]:
    user_msg = build_user_message(target, text)

    for attempt in range(max_retries):
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                reasoning_effort="none",
            )

            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()

            raw_text = completion.choices[0].message.content or ""
            model_returned = completion.model
            predicted, parse_ok = parse_response(raw_text)

            result = {
                "row_index":       row_index,
                "target":          target,
                "gold_label":      "",
                "predicted_label": predicted,
                "model_id":        model_returned,
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
                print(
                    f"    ⚠️  429 / Rate Limit on row {row_index} (attempt {attempt + 1}/{max_retries}) — "
                    f"sleeping {retry_after:.1f}s"
                )
                time.sleep(retry_after)
                continue

            wait = 5.0 * (2 ** attempt)
            print(f"    ❌ Row {row_index} attempt {attempt + 1}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print(f"    ❌ Row {row_index}: all {max_retries} attempts failed; recording as 'None'.")
    result = {
        "row_index":       row_index,
        "target":          target,
        "gold_label":      "",
        "predicted_label": "None",
        "model_id":        MODEL_ID,
        "parse_success":   False,
        "raw_response":    "ERROR: max retries exceeded",
    }
    return result, {}

def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    # Rename columns to match standard format
    df = df.rename(columns={"id": "ID", "tweet_text": "text"})
    df.columns = df.columns.astype(str).str.strip()
    return df

def main() -> None:
    # Load env variables
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
        window_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(request_timestamps[0]))
        print(f"📊 Loaded {len(request_timestamps)} request timestamps from last 60 minutes.")

    print(f"── Inference config ──")
    print(f"   Model:            {MODEL_ID}")
    print(f"   Prompt:           Revised Zero-Shot Prompt")
    print(f"   Targets:          Translated to Arabic ({TARGET_AR['Women Driving']})")
    print(f"   temperature={TEMPERATURE}  top_p={TOP_P}")
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

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r} ({TARGET_AR.get(target)})")

        result, rl_headers = call_cerebras_once(client, row_index, target, text)
        request_timestamps.append(time.time())
        result["gold_label"] = ""  # No gold label for test set

        append_result_row(results_path, result)
        completed.add(row_index)
        new_calls += 1

        log_entry = {
            "row_index":       row_index,
            "target":          target,
            "gold_label":      "",
            "raw_response":    result["raw_response"],
            "predicted_label": result["predicted_label"],
            "model_id":        result["model_id"],
            "parse_success":   result["parse_success"],
            "timestamp":       time.time(),
        }
        with open(raw_log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → pred={result['predicted_label']!r}")

        delay = compute_pace_delay(rl_headers)
        if delay > 0:
            time.sleep(delay)

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # Write predictions to TXT
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
    print("\nPredicted stance distribution:")
    print(results_df["predicted_label"].value_counts().to_string())

if __name__ == "__main__":
    main()

"""
gemma_inference.py — Zero-shot stance detection using Gemma 4 31B via Cerebras.

Compares gemma-4-31b (Cerebras, zero-shot) against our fine-tuned AraBERT
checkpoints and Qwen baselines on the Mawqif-v2 dev set for StanceEval-2026.

Runs on exactly 100 rows selected with balanced target coverage (reusing the
same stratified selection logic to guarantee a clean comparison).

Usage:
    python src/gemma_inference.py

Requirements:
    - CEREBRAS_API_KEY in .env (loaded via python-dotenv; never printed or committed)
    - pip install cerebras_cloud_sdk python-dotenv pandas scikit-learn
"""

from __future__ import annotations

import argparse
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
from data_utils import LABEL2ID, load_data
from metrics import favg2, per_topic_and_overall_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")  # Keep logs in same directory structure

DEV_CSV_PATH = os.path.join(DATA_DIR, "dev.csv")
# ── Output paths ──────────────────────────────────────────────────────────
# NEW — separate from baseline, so resumability doesn't skip these rows
RESULTS_PATH = os.path.join(PRED_DIR, "results_gemma4_31b_cerebras_v4_fewshot.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_gemma4_31b_cerebras_v4_fewshot_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID = "gemma-4-31b"

# Recommended parameters for gemma-4-31b on Cerebras
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 10  # single-word label only

# Sampling constants
N_ROWS = 100
ROWS_PER_TARGET = 33  # 33 + 33 + 34 = 100 across 3 targets

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}
TARGETS = ["Covid Vaccine", "Digital Transformation", "Women empowerment"]

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",      # original 0-based index in dev.csv
    "target",
    "gold_label",
    "predicted_label",
    "model_id",       # exact model string returned by the API
    "parse_success",  # True if response was a clean single-word label
    "raw_response",
]

# ── Prompt (Identical to Qwen runs) ───────────────────────────────────────────
SYSTEM_PROMPT_V1 = (
    "You are an expert annotator for Arabic stance detection. "
    "Given an Arabic tweet and a target topic, classify the writer's stance toward "
    "that target as exactly one of three labels:\n\n"
    "- Favor: the writer expresses support for or a positive position toward the target\n"
    "- Against: the writer expresses opposition to or a negative position toward the target\n"
    "- None: no clear stance — neutral, off-topic, or ambiguous. This includes cases where "
    "sarcasm makes the literal tone misleading about the writer's actual position — do not "
    "infer a stance from tone alone if the underlying position isn't clear.\n\n"
    "Respond with ONLY a single word: Favor, Against, or None. "
    "No explanation, no punctuation, no other text."
)

SYSTEM_PROMPT_V3 = (
    "You are an expert annotator for Arabic stance detection.\n\n"

    "Given an Arabic tweet and a target topic, determine the writer's stance "
    "toward the specified target based on the overall meaning of the tweet.\n\n"

    "Reason carefully before assigning a label:\n"
    "• Identify the target topic.\n"
    "• Determine who or what the writer is supporting, criticizing, agreeing with, or opposing.\n"
    "• Distinguish opinions directed at the target from opinions directed at other people, organizations, implementations, events, or related entities.\n"
    "• Base your decision on the complete meaning and context of the tweet, not isolated words or overall sentiment.\n"
    "• Do not assume that negative language means an Against stance or that positive language means a Favor stance. The stance depends on whether the opinion is directed toward the target itself.\n"
    "• A stance may be expressed explicitly or implicitly through praise, criticism, recommendations, concerns, consequences, comparisons, rhetorical questions, or sarcasm. Infer the most plausible stance toward the target from the overall context rather than requiring explicit statements of support or opposition.\n"
    "• Use the label None only when the tweet genuinely expresses no identifiable stance toward the target or when the stance cannot be reasonably inferred.\n\n"

    "Assign exactly one label:\n"
    "- Favor: the writer supports or expresses a positive stance toward the target.\n"
    "- Against: the writer opposes or expresses a negative stance toward the target.\n"
    "- None: the writer expresses no identifiable stance toward the target.\n\n"

    "Respond with exactly one word and nothing else:\n"
    "Favor\n"
    "Against\n"
    "None"
)

SYSTEM_PROMPT_V4 = (
    "You are an expert annotator for Arabic stance detection. "
    "Given an Arabic tweet and a target topic, classify the writer's stance toward "
    "that target as exactly one of three labels:\n\n"
    "- Favor: the writer expresses support for or a positive position toward the target\n"
    "- Against: the writer expresses opposition to or a negative position toward the target\n"
    "- None: no clear stance — neutral, off-topic, or ambiguous. This includes cases where "
    "sarcasm makes the literal tone misleading about the writer's actual position — do not "
    "infer a stance from tone alone if the underlying position isn't clear.\n\n"
    "Examples:\n\n"
    "Tweet: \"نعم الجائحة سرعت نحو التحول الإلكتروني وغيرت في سلوك المستهلكين لا شك. "
    "من الممكن الإعتماد على التجارة الإلكترونية لكن من غير الممكن استبعاد متاجر (الطوب) "
    "من المنظومة التجارية لأسباب حيوية للغاية مثل النشاط العمراني والبطالة.\"\n"
    "Target: Digital Transformation\n"
    "Stance: Favor\n\n"
    "Tweet: \"ههههههههههههههه ماعليك زمن تمكين المرأة ماراح يسوون لك شي 😂\"\n"
    "Target: Women empowerment\n"
    "Stance: None\n\n"
    "Respond with ONLY a single word: Favor, Against, or None. "
    "No explanation, no punctuation, no other text."
)


SYSTEM_PROMPT = SYSTEM_PROMPT_V4


def build_user_message(target: str, text: str) -> str:
    return f"Target: {target}\nTweet: {text}"


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


# ── Step 1: Stratified 100-row selection (same as Qwen) ────────────────────────
def select_100_rows(dev_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], str]:
    first_100 = dev_df.head(N_ROWS)
    counts = first_100["target"].value_counts()

    print(f"\n── Step 1: Target distribution — first {N_ROWS} rows (original order) ──")
    print(counts.to_string())

    min_count = min(counts.get(t, 0) for t in TARGETS)
    balanced_threshold = 30

    if min_count >= balanced_threshold:
        path = "as-is (first 100 rows, original order — no resampling needed)"
        run_df = first_100.copy()
    else:
        path = "stratified: first 33 rows from Covid Vaccine, 33 from Digital Transformation, 34 from Women empowerment"
        per_target_n = {
            "Covid Vaccine":          33,
            "Digital Transformation": 33,
            "Women empowerment":      34,
        }
        parts = []
        for target, n in per_target_n.items():
            slice_ = dev_df[dev_df["target"] == target].head(n)
            print(f"  → {target}: rows {list(slice_.index[:3])} … (first {n})")
            parts.append(slice_)
        run_df = pd.concat(parts).sort_index()

    selected_indices = list(run_df.index)

    print(f"\n── Path taken: {path} ──")
    print(f"Selected original row indices (first 5): {selected_indices[:5]}")
    print(f"Selected original row indices (last 5):  {selected_indices[-5:]}")
    print(f"Total rows selected: {len(selected_indices)}")
    print(f"\nResulting target distribution:")
    print(run_df["target"].value_counts().to_string())
    print(f"\nResulting gold stance distribution:")
    print(run_df["stance"].value_counts().to_string())

    return run_df, selected_indices, path


# ── Rate-limit pacing logic (Cerebras 5 RPM ceiling) ──────────────────────────
def compute_pace_delay(headers: dict) -> float:
    """Determine pacing delay based on Cerebras specific rate-limit headers.

    The free tier limit is 5 RPM (requests per minute).
    This implies a default hard floor of 12.0 seconds between calls.
    We also inspect remaining requests/tokens minute headers.
    """
    # Hard base delay of 12.0 seconds to keep request frequency strictly under 5 RPM
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

    # If remaining requests in the minute drops to 1, we wait a full 60 seconds to replenish
    if remaining_req_min <= 1:
        print(f"    ⚠️  Cerebras requests budget almost exhausted ({remaining_req_min} left) → sleeping 60s to replenish")
        return 60.0

    # If remaining tokens in the minute drops below 1,000, we wait 30 seconds
    if remaining_tok_min < 1000:
        print(f"    ⚠️  Cerebras token budget low ({remaining_tok_min} left) → sleeping 30s to replenish")
        return 30.0

    return base_delay


def _parse_reset_duration(s: str) -> float:
    """Parse a reset-time header like '1.234s', '60s', or '1m30.5s' → seconds."""
    if not s:
        return 0.0
    s = s.strip()
    # Fast path: plain float/int with optional 's' or 'ms'
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
    
    # General: minutes and seconds
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if m and (m.group(1) or m.group(2)):
        return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return 0.0


def load_request_timestamps(raw_log_path: str) -> list[float]:
    """Load timestamps of requests made in the last 60 minutes from JSONL log."""
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


# ── Single API call (Cerebras SDK) ────────────────────────────────────────────
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
                reasoning_effort="none",  # disable reasoning explicitly
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

            # Detect rate-limit errors (429)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 60.0  # default 60 seconds backoff for Cerebras
                
                # Check for standard HTTP retry-after headers in case exception object exposes them
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

            # General transient error
            wait = 5.0 * (2 ** attempt)
            print(f"    ❌ Row {row_index} attempt {attempt + 1}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    # Exceeded max retries
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot stance detection using Gemma 4 31B via Cerebras."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on the full 619-row dev set instead of the 100-row stratified subset."
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

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

    # Load dev data
    print(f"\nLoading {DEV_CSV_PATH} …")
    dev_df = load_data(DEV_CSV_PATH)
    print(f"  Loaded {len(dev_df)} rows.")

    # Resolve scope, paths and selection
    if args.full:
        run_df = dev_df.copy()
        selected_indices = list(run_df.index)
        path_taken = "full (all 619 rows, original order)"
        results_path = os.path.join(PRED_DIR, "results_gemma4_31b_cerebras_full.csv")
        raw_log_path = os.path.join(LOG_DIR, "results_gemma4_31b_cerebras_full_raw.jsonl")
        print(f"\n── Path taken: {path_taken} ──")
        print(f"Total rows to process: {len(run_df)}")
    else:
        run_df, selected_indices, path_taken = select_100_rows(dev_df)
        results_path = RESULTS_PATH
        raw_log_path = RAW_LOG_PATH

    # Initialize client
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)

    # Ensure output directories exist
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Resumability check
    completed = load_completed_indices(results_path)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done (up to index {max(completed) if completed else 0}).")
    else:
        print(f"\n🆕 Starting fresh run.")

    # Load rolling request timestamps from JSONL to enforce hourly cap (150 req/hour)
    request_timestamps = load_request_timestamps(raw_log_path)
    if request_timestamps:
        window_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(request_timestamps[0]))
        print(f"📊 Loaded {len(request_timestamps)} request timestamps from last 60 minutes from {raw_log_path}.")
        print(f"   Rolling window starts at: {window_start_time} (oldest request in window)")

    # Inference config summary
    print(f"── Inference config ──")
    print(f"   Model:            {MODEL_ID}")
    print(f"   reasoning_effort: none  (non-thinking mode)")
    print(f"   temperature={TEMPERATURE}  top_p={TOP_P}")
    print(f"   max_tokens:       {MAX_TOKENS}")
    print(f"   Sequential calls; rate pacing designed for 5 RPM and 150 RPH limits")
    print(f"   Results CSV:      {results_path}")
    print(f"   Raw log:          {raw_log_path}")
    print()

    total = len(run_df)
    new_calls = 0
    skipped = 0
    _header_keys_logged = False
    rl_headers = {}

    for position, (_, row) in enumerate(run_df.iterrows(), start=1):
        row_index = int(row.name)

        if row_index in completed:
            skipped += 1
            continue

        target = str(row["target"])
        text   = str(row["text"])
        gold   = str(row.get("stance", ""))

        # Hourly rate limiting tracking check
        now = time.time()
        request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        if len(request_timestamps) >= 150:
            oldest_ts = request_timestamps[0]
            # Sleep until the oldest request ages out
            sleep_time = oldest_ts + 3600 - now + 1.0
            
            # Check for hourly reset headers in case they exist
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

            # Refresh current time and request timestamps window after sleep
            now = time.time()
            request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r}")

        result, rl_headers = call_cerebras_once(client, row_index, target, text)
        request_timestamps.append(time.time())
        result["gold_label"] = gold

        # Incremental write
        append_result_row(results_path, result)
        completed.add(row_index)
        new_calls += 1

        # JSONL Log
        log_entry = {
            "row_index":       row_index,
            "target":          target,
            "gold_label":      gold,
            "raw_response":    result["raw_response"],
            "predicted_label": result["predicted_label"],
            "model_id":        result["model_id"],
            "parse_success":   result["parse_success"],
            "timestamp":       time.time(),
        }
        with open(raw_log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(
            f"    → pred={result['predicted_label']!r}  gold={gold!r}  "
            f"model={result['model_id']!r}  parse_ok={result['parse_success']}"
        )

        # Log raw headers once on the first response
        if rl_headers and not _header_keys_logged:
            rl_keys = [k for k in rl_headers if "ratelimit" in k.lower()]
            print(f"    ── Rate-limit header keys present in first response ──")
            for k in sorted(rl_keys):
                print(f"       {k}: {rl_headers[k]}")
            _header_keys_logged = True

        # Pacing sleep (per-minute RPM ceiling)
        delay = compute_pace_delay(rl_headers)
        if delay > 0:
            time.sleep(delay)

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # Step 5: Evaluation
    results_df = pd.read_csv(results_path, encoding="utf-8", keep_default_na=False)
    results_df = results_df[results_df["row_index"].isin(selected_indices)].copy()

    if results_df.empty:
        print("⚠️ No results found to evaluate.")
        return

    print(f"── Step 5: Evaluation on {len(results_df)} rows ──\n")
    print("Predicted label distribution:")
    print(results_df["predicted_label"].value_counts().to_string())
    print("\nGold label distribution:")
    print(results_df["gold_label"].value_counts().to_string())

    correct = (results_df["predicted_label"] == results_df["gold_label"]).sum()
    accuracy = correct / len(results_df)
    print(f"\nOverall accuracy: {correct}/{len(results_df)} = {accuracy:.4f}")

    y_true = results_df["gold_label"].tolist()
    y_pred = results_df["predicted_label"].tolist()
    overall_f2 = favg2(y_true, y_pred)
    print(f"Overall Favg2:    {overall_f2:.4f}")

    results_df["true_stance"]      = results_df["gold_label"]
    results_df["predicted_stance"] = results_df["predicted_label"]
    metrics = per_topic_and_overall_metrics(
        results_df,
        true_col="true_stance",
        pred_col="predicted_stance",
        target_col="target",
    )

    print(f"\nPer-target metrics:")
    print(f"{'Target':<28} {'Favg2':>8} {'Favg3':>8}")
    print("-" * 48)
    for key in sorted(metrics):
        if key == "Overall":
            continue
        m = metrics[key]
        print(f"{key:<28} {m['Favg2']:>8.4f} {m['Favg3']:>8.4f}")
    print("-" * 48)
    ov = metrics.get("Overall", {})
    print(f"{'Overall':<28} {ov.get('Favg2', float('nan')):>8.4f} {ov.get('Favg3', float('nan')):>8.4f}")

    print(f"\nResults CSV : {results_path}")
    print(f"Raw JSONL   : {raw_log_path}")
    if args.full:
        print(
            f"\n✅ Metrics computed on the full {len(results_df)} dev rows."
            f"\n    Compare these results directly against other full-dev runs in results_summary.csv."
        )
    else:
        print(
            f"\n⚠️ NOTE: Metrics computed on {len(results_df)}/619 rows (100-row subset)."
            f"\n    Do NOT add to results_summary.csv — not comparable to full-dev scores."
            f"\n    Row-selection path: {path_taken}"
        )


if __name__ == "__main__":
    main()

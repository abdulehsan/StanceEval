"""
gemma_inference_test_refined.py — Zero-shot stance detection on the blind test set (ground_truth.csv)
using Gemma 4 31B via Cerebras with the refined prompt.

Outputs raw predictions and supports resumability.

Usage:
    python src/gemma_inference_test_refined.py
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
RESULTS_PATH = os.path.join(PRED_DIR, "results_gemma4_31b_cerebras_test_refined.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_gemma4_31b_cerebras_test_refined_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID = "gemma-4-31b"
TEMPERATURE = 1.0
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

# ── System Prompt (Refined) ───────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert annotator for Arabic stance detection. Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels:\n\n"
    "- Favor: the writer expresses support for or a positive position toward the target.\n"
    "- Against: the writer expresses opposition to or a negative position toward the target.\n"
    "- None: no clear stance — neutral, off-topic, or ambiguous.\n\n"
    "### Stance Disambiguation Guidelines:\n\n"
    "1. **Identify the Opinion Target First**:\n"
    "   Before judging sentiment, determine WHO or WHAT the tweet's negative or positive language is actually directed at. Negative sentiment alone does not imply \"Against\" — first identify who or what is being criticized.\n\n"
    "2. **Defending the Target by Attacking Opponents (Negativity Diversion)**:\n"
    "   If the writer uses negative terms (e.g., \"backwardness\" / التخلف, \"reactionary\" / الرجعية, \"extremism\" / التطرف, \"fossilization\" / التحجر, \"pseudo-men\" / أشباه رجال) to attack the *opponents* of the target, the writer's stance is actually **Favor** (supporting the target by defending it against its detractors).\n\n"
    "3. **Target Resolution (Diversion)**:\n"
    "   Verify where the negativity is directed. If the writer expresses anger toward a completely different topic (e.g., other social issues, a foreign system, or unrelated regulations) but mentions the target itself without opposition, the stance toward the target is **None** or **Favor**, not Against.\n\n"
    "4. **Factual Reporting vs. Opinion**:\n"
    "   If a tweet merely reports a news event, announcement, or statistic about the target without any implicit or explicit opinion/emotion, classify it as **None**.\n\n"
    "5. **Questions and Speculation Without Opinion**:\n"
    "   Questions, speculation, or observations that do not express a clear personal opinion should be classified as **None**. Example: \"كم حادث صار؟\" → None.\n\n"
    "6. **Sarcasm and Hijacked Hashtags**:\n"
    "   Arabic social media often uses sarcasm. If a tweet uses a pro-target hashtag but sarcastically mocks it or describes it as destructive, the stance is **Against**.\n\n"
    "### Few-Shot Examples (from train.csv):\n\n"
    "Example 1 (Explicit Favor):\n"
    "Target: تمكين المرأة\n"
    "Tweet: نحن في بداية العصر الذهبي وهو تمكين المرأه من حقوقها وتخليص المجتمع من افكار التحجر والتخلف بأعمال العقل بكل مناحي الحياة كما امر الله بالقران الكريم (تعقلوا،تفكروا،تدبروا ،الخ)\n"
    "Stance: Favor\n\n"
    "Example 2 (Implicit Favor - Attacking Opponents):\n"
    "Target: تمكين المرأة\n"
    "Tweet: ردك يبين تاثرك بالخطاب الصحوى الرجعي المتخلف ولا يعكس تعليمك في ارقى الجامعات ، و حكومه المملكة من اهم اهدف رؤيتها تمكين المرأة وولى العهد ذكر في اكثر في لقاء ان المرأة السعودية ظلمت في فترة الغفوة والظلام والتطرف ووعد بتمكينها\n"
    "Stance: Favor\n\n"
    "Example 3 (Explicit Against):\n"
    "Target: تمكين المرأة\n"
    "Tweet: لايخدعونك بكذبة تمكين المرأة!.\n"
    "Stance: Against\n\n"
    "Example 4 (None - Target Resolution / Sentiment Diversion):\n"
    "Target: تمكين المرأة\n"
    "Tweet: هناك من عاصر زمن تحرير السود من العبودية وهناك من عاصر زمن تمكين المرأة واعطائها كامل حقوقها وشاء الله أن يكون زماننا زمن اعطاء الشواذ حقوقهم وهو الأسوأ حتى الآن أتمنى ألا تطول صولتهم\n"
    "Stance: None\n\n"
    "Example 5 (Against - Hijacked Hashtag / Sarcasm):\n"
    "Target: تمكين المرأة\n"
    "Tweet: #تمكين_المرأة بمفهوم الفارغون والفارغات والسطحيون والسطحيات والتافهون والتافهات هو اهلاك للمجتمع\n"
    "Stance: Against\n\n"
    "### Response Format:\n"
    "Return exactly one token:\n\n"
    "Favor\n"
    "Against\n"
    "None\n\n"
    "Do not explain.\n"
    "Do not repeat the tweet.\n"
    "Do not output anything else."
)

def build_user_message(target: str, text: str) -> str:
    arabic_target = TARGET_AR.get(target, target)
    return f"Target: {arabic_target}\nTweet: {text}"

def parse_response(raw: str) -> tuple[str, bool]:
    """Parse a model response and return (stance_label, parse_success).
    parse_success is True if and only if the output is a clean single token.
    """
    raw_clean = raw.strip()

    # Clean single token (allowing case-insensitive match of exact labels)
    for label in ["Favor", "Against", "None"]:
        if raw_clean.lower() == label.lower():
            return label, True

    # Fallback to salvage the label using substring lookup
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw_clean.lower():
            return label, False

    return "None", False

# ── Dynamic Rate-Limit Pacing Logic (Cerebras 5 RPM & 150 RPH) ────────────────
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

def pace_request(timestamps: list[float], headers: dict):
    now = time.time()
    
    # Filter timestamps to last 3600 seconds
    timestamps[:] = [ts for ts in timestamps if now - ts < 3600.0]
    
    # 1. Minute Limit (5 RPM)
    ts_min = [ts for ts in timestamps if now - ts < 60.0]
    
    delay = 0.0
    
    remaining_req_min = int(headers.get("x-ratelimit-remaining-requests-minute", 5)) if headers else 5
    if remaining_req_min <= 0 or len(ts_min) >= 5:
        if ts_min:
            delay = max(delay, 60.0 - (now - ts_min[0]))
            
    # 2. Hourly Limit (150 RPH)
    remaining_req_hour = int(headers.get("x-ratelimit-remaining-requests-hour", 150)) if headers else 150
    if remaining_req_hour <= 0 or len(timestamps) >= 150:
        if timestamps:
            delay = max(delay, 3600.0 - (now - timestamps[0]))
            
    # 3. Minimum spacing between requests to stay under 5 RPM (12 seconds)
    if ts_min:
        spacing_delay = 12.0 - (now - ts_min[-1])
        delay = max(delay, spacing_delay)
        
    if delay > 0:
        resume_time = time.strftime("%H:%M:%S", time.localtime(time.time() + delay))
        print(
            f"    ⏳ Pacing delay: sleeping {delay:.2f}s (Minute remaining: {remaining_req_min}, "
            f"Hour remaining: {remaining_req_hour}, Minute count: {len(ts_min)}, Hour count: {len(timestamps)}). "
            f"Resume at {resume_time}"
        )
        time.sleep(delay)

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
    last_rl_headers = {}
    attempt = 0

    while attempt < max_retries:
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
            last_rl_headers = rl_headers
            completion = raw_resp.parse()

            raw_text = completion.choices[0].message.content or ""
            predicted, parse_ok = parse_response(raw_text)

            result = {
                "row_index":       row_index,
                "target":          target,
                "gold_label":      "",
                "predicted_label": predicted,
                "model_id":        "gemma-4-31b-refined-v2",  # Hardcoded model_id for refined run
                "parse_success":   parse_ok,
                "raw_response":    raw_text,
            }
            return result, rl_headers

        except Exception as exc:
            exc_str = str(exc)
            if hasattr(exc, "response") and exc.response is not None:
                last_rl_headers = dict(exc.response.headers)
                
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 60.0
                if last_rl_headers and "retry-after" in last_rl_headers:
                    try:
                        retry_after = float(last_rl_headers["retry-after"])
                    except ValueError:
                        pass
                print(
                    f"    ⚠️  429 / Rate Limit on row {row_index} — "
                    f"sleeping {retry_after:.1f}s (does not count as failed retry attempt)"
                )
                time.sleep(retry_after)
                continue

            attempt += 1
            wait = 5.0 * (2 ** attempt)
            print(f"    ❌ Row {row_index} attempt {attempt}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries:
                time.sleep(wait)

    print(f"    ❌ Row {row_index}: all {max_retries} attempts failed; recording as 'None'.")
    result = {
        "row_index":       row_index,
        "target":          target,
        "gold_label":      "",
        "predicted_label": "None",
        "model_id":        "gemma-4-31b-refined-v2",
        "parse_success":   False,
        "raw_response":    "ERROR: max retries exceeded",
    }
    return result, last_rl_headers

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
    print(f"   Model ID (stored): gemma-4-31b-refined-v2")
    print(f"   Prompt:           Refined Stance Detection Prompt")
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

        # Dynamically pace request based on past timestamps and response headers
        pace_request(request_timestamps, rl_headers)

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r} ({TARGET_AR.get(target)})")

        result, new_rl_headers = call_cerebras_once(client, row_index, target, text)
        if new_rl_headers:
            rl_headers = new_rl_headers
        request_timestamps.append(time.time())

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

        print(f"    → pred={result['predicted_label']!r} (clean_token={result['parse_success']})")

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # Write predictions to TXT
    results_df = pd.read_csv(results_path, encoding="utf-8", keep_default_na=False)
    results_df = results_df[results_df["row_index"].isin(selected_indices)].copy()

    if results_df.empty:
        print("⚠️ No results found.")
        return

    # Sort results by row_index to guarantee correct line-by-line alignment
    results_df = results_df.sort_values("row_index").copy()
    
    # Save the sorted CSV file
    results_df.to_csv(results_path, index=False)

    y_pred = results_df["predicted_label"].tolist()
    txt_path = results_path.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"Saved sorted test predictions to {txt_path}")
    print("\nPredicted stance distribution:")
    print(results_df["predicted_label"].value_counts().to_string())

if __name__ == "__main__":
    main()

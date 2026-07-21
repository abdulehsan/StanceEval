"""
antigravity_inference_test.py — Stance detection and reasoning on the blind test set (ground_truth.csv)
as Antigravity (using Groq's Llama-3.3-70B model for fast and high-quality generation).

Generates stance prediction and detailed reasoning in Arabic, and supports resumability.

Usage:
    python src/antigravity_inference_test.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
import pandas as pd

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import LABEL2ID

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")

TEST_CSV_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_PATH = os.path.join(PRED_DIR, "results_antigravity_test.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_antigravity_test_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID = "llama-3-3-70b"  # Logical name for logging
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.5
MAX_TOKENS = 150

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}

TARGET_AR = "قيادة المرأة للسيارة"

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",
    "target",
    "antigravity_stance",
    "antigravity_reasoning",
    "parse_success",
    "raw_response",
]

# ── System Prompt (Act as Antigravity) ────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Antigravity, the expert stance detection AI. "
    "Given an Arabic tweet and a target topic, classify the writer's stance toward "
    "that target as exactly one of three labels: Favor, Against, or None.\n\n"
    "You must return your response as a raw JSON object containing exactly two keys:\n"
    '- "stance": exactly one of "Favor", "Against", or "None"\n'
    '- "reasoning": a brief explanation (maximum 15 words) in Arabic of why you chose this stance.\n\n'
    "Respond with ONLY the JSON object, do not include markdown blocks (```json) or any other text."
)

def build_user_message(text: str) -> str:
    return f"Target: {TARGET_AR}\nTweet: {text}"

def parse_json_response(raw: str) -> tuple[str, str, bool]:
    """Parse JSON response and extract stance and reasoning."""
    raw = raw.strip()
    
    # Try to strip markdown code blocks if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    
    raw = raw.strip()
    
    try:
        data = json.loads(raw)
        stance = str(data.get("stance", "None")).strip()
        reasoning = str(data.get("reasoning", "")).strip()
        
        # Normalize stance label capitalization
        for label in VALID_LABELS:
            if stance.lower() == label.lower():
                stance = label
                break
                
        if stance not in VALID_LABELS:
            stance = "None"
            
        return stance, reasoning, True
    except Exception:
        # Regex fallback
        stance_match = re.search(r'"stance"\s*:\s*"([^"]+)"', raw)
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
        
        stance = "None"
        reasoning = "Error parsing reasoning"
        
        if stance_match:
            cand = stance_match.group(1).strip()
            for label in VALID_LABELS:
                if cand.lower() == label.lower():
                    stance = label
                    break
        if reason_match:
            reasoning = reason_match.group(1).strip()
            
        return stance, reasoning, False

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

def call_groq_once(
    client,
    row_index: int,
    text: str,
    max_retries: int = 4,
) -> tuple[dict, dict]:
    user_msg = build_user_message(text)

    for attempt in range(max_retries):
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"}
            )

            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()

            raw_text = completion.choices[0].message.content or ""
            model_returned = completion.model
            stance, reasoning, parse_ok = parse_json_response(raw_text)

            result = {
                "row_index":             row_index,
                "target":                "Women Driving",
                "antigravity_stance":    stance,
                "antigravity_reasoning": reasoning,
                "model_id":              model_returned,
                "parse_success":         parse_ok,
                "raw_response":          raw_text,
            }
            return result, rl_headers

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 30.0
                print(f"    ⚠️  Groq Rate Limit (429) on row {row_index} — sleeping {retry_after}s")
                time.sleep(retry_after)
                continue

            wait = 2.0 * (2 ** attempt)
            print(f"    ❌ Row {row_index} attempt {attempt + 1}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print(f"    ❌ Row {row_index}: all {max_retries} attempts failed; recording default.")
    result = {
        "row_index":             row_index,
        "target":                "Women Driving",
        "antigravity_stance":    "None",
        "antigravity_reasoning": "API error",
        "model_id":              GROQ_MODEL,
        "parse_success":         False,
        "raw_response":          "ERROR: max retries exceeded",
    }
    return result, {}

def main() -> None:
    # Load env variables
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        print("❌ GROQ_API_KEY not found in .env or environment. Cannot proceed.")
        sys.exit(1)
    print("✅ GROQ_API_KEY loaded.")

    print(f"\nLoading {TEST_CSV_PATH} …")
    test_df = pd.read_csv(TEST_CSV_PATH, keep_default_na=False)
    # Rename columns to match standard format
    test_df = test_df.rename(columns={"id": "ID", "tweet_text": "text"})
    print(f"  Loaded {len(test_df)} rows.")

    from groq import Groq
    client = Groq(api_key=groq_key)

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    completed = load_completed_indices(RESULTS_PATH)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done (up to index {max(completed)}).")
    else:
        print(f"\n🆕 Starting fresh Antigravity run.")

    total = len(test_df)
    new_calls = 0
    skipped = 0

    for position, (_, row) in enumerate(test_df.iterrows(), start=1):
        row_index = int(row.name)

        if row_index in completed:
            skipped += 1
            continue

        text = str(row["text"])
        print(f"  [{position}/{total}] Antigravity row_index={row_index}")

        result, rl_headers = call_groq_once(client, row_index, text)
        append_result_row(RESULTS_PATH, result)
        completed.add(row_index)
        new_calls += 1

        log_entry = {
            "row_index":             row_index,
            "raw_response":          result["raw_response"],
            "antigravity_stance":    result["antigravity_stance"],
            "antigravity_reasoning": result["antigravity_reasoning"],
            "parse_success":         result["parse_success"],
            "timestamp":             time.time(),
        }
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → stance={result['antigravity_stance']!r}  reasoning={result['antigravity_reasoning']!r}")

        # Pace calls (Groq limit is very high, 0.5s is safe and fast)
        time.sleep(0.5)

    print(f"\n── Antigravity predictions complete: {new_calls} new calls, {skipped} skipped ──\n")

if __name__ == "__main__":
    main()

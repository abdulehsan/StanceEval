"""
qwen_inference_ci_v3.py — Zero-shot stance detection on the blind test set (ground_truth.csv)
using Qwen 3.6 27B via Groq with the CI v3 prompt and temperature 0.1.

Usage:
    python src/qwen_inference_ci_v3.py

IMPORTANT — Groq rate limits (free tier):
  - 60 RPM, 6,000 TPM, 1,000 RPD
  - We pace at ~10 RPM (6s between calls) to stay well below limits.
  - Conservative retry: max 2 retries on 429 with 120s backoff.
  - DO NOT increase pacing without reviewing your Groq tier.
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
LOG_DIR  = os.path.join(_ROOT, "qwen_raw_logs")

TEST_CSV_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_PATH  = os.path.join(PRED_DIR, "results_qwen3.6_27b_ci_v3.csv")
RAW_LOG_PATH  = os.path.join(LOG_DIR,  "results_qwen3.6_27b_ci_v3_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID    = "qwen/qwen3.6-27b"
TEMPERATURE = 0.1
TOP_P       = 0.95
MAX_TOKENS  = 10    # single-word label only

VALID_LABELS: set[str] = set(LABEL2ID.keys())   # {"Against", "Favor", "None"}

TARGET_AR = {
    "Women Driving": "قيادة المرأة للسيارة"
}

# ── Pacing ─────────────────────────────────────────────────────────────────────
PACE_DELAY = 6.0         # seconds between calls (~10 RPM, well under 60 RPM)
RETRY_BACKOFF = 120.0    # seconds to wait on 429 (conservative — avoid ban)
MAX_RETRIES = 2          # only 2 retries on 429 — better to skip than get banned

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",
    "target",
    "gold_label",
    "predicted_label",
    "model_id",
    "parse_success",
    "raw_response",
]

# ── System Prompt (CI v3 — identical to gemma ci_v3) ──────────────────────────
SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

### Task
Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:

• Favor
• Against
• None

Before assigning a stance, mentally rewrite the tweet into its intended literal meaning while preserving the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis.

Then determine the stance toward the target itself, not toward other people, quoted opinions, related entities, or hashtags.

Guidelines:

• Favor: supports, defends, promotes, or welcomes the target.
• Against: opposes, criticizes, rejects, or mocks the target.
• None: no clear stance toward the target.

Important:

• Determine where praise or criticism is directed. Negative language toward opponents of the target is usually Favor, not Against.
• Hashtags may be ironic or hijacked. Never infer stance from hashtags alone.
• Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording.
• Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor.
• If the stance toward the target cannot reasonably be inferred, output None.

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text."""


def build_user_message(target: str, text: str) -> str:
    arabic_target = TARGET_AR.get(target, target)
    return f"Target: {arabic_target}\nTweet: {text}"


def parse_response(raw: str) -> tuple[str, bool]:
    """Parse a model response and return (stance_label, parse_success)."""
    raw = raw.strip()

    if raw in VALID_LABELS:
        return raw, True

    for label in VALID_LABELS:
        if raw.lower() == label.lower():
            return label, True

    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw.lower():
            return label, False

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


# ── Resume support ─────────────────────────────────────────────────────────────
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
    target: str,
    text: str,
) -> dict:
    user_msg = build_user_message(target, text)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )

            raw_text = resp.choices[0].message.content or ""
            model_returned = resp.model
            predicted, parse_ok = parse_response(raw_text)

            return {
                "row_index":       row_index,
                "target":          target,
                "gold_label":      "",
                "predicted_label": predicted,
                "model_id":        model_returned,
                "parse_success":   parse_ok,
                "raw_response":    raw_text,
            }

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                print(
                    f"    ⚠️  429 on row {row_index} (attempt {attempt + 1}/{MAX_RETRIES}) — "
                    f"sleeping {RETRY_BACKOFF:.0f}s (conservative to avoid ban)"
                )
                time.sleep(RETRY_BACKOFF)
                continue

            wait = 10.0 * (2 ** attempt)
            print(f"    ❌ Row {row_index} attempt {attempt + 1}/{MAX_RETRIES}: {exc!r} — retrying in {wait}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    print(f"    ❌ Row {row_index}: all {MAX_RETRIES} attempts failed; recording as 'None'.")
    return {
        "row_index":       row_index,
        "target":          target,
        "gold_label":      "",
        "predicted_label": "None",
        "model_id":        MODEL_ID,
        "parse_success":   False,
        "raw_response":    "ERROR: max retries exceeded",
    }


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

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env or environment. Cannot proceed.")
        sys.exit(1)
    print("✅ GROQ_API_KEY loaded.")

    print(f"\nLoading {TEST_CSV_PATH} …")
    test_df = load_test_data(TEST_CSV_PATH)
    print(f"  Loaded {len(test_df)} rows.")

    run_df = test_df.copy()
    selected_indices = list(run_df.index)

    from groq import Groq
    client = Groq(api_key=api_key)

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    completed = load_completed_indices(RESULTS_PATH)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done (up to index {max(completed)}).")
    else:
        print(f"\n🆕 Starting fresh run.")

    print(f"── Inference config ──")
    print(f"   Model:       {MODEL_ID}")
    print(f"   Prompt:      CI v3 (identical to Gemma ci_v3)")
    print(f"   temperature={TEMPERATURE}  top_p={TOP_P}")
    print(f"   Pacing:      {PACE_DELAY}s between calls (~{60/PACE_DELAY:.0f} RPM)")
    print(f"   Results CSV: {RESULTS_PATH}")
    print()

    total = len(run_df)
    new_calls = 0
    skipped   = 0

    for position, (_, row) in enumerate(run_df.iterrows(), start=1):
        row_index = int(row.name)

        if row_index in completed:
            skipped += 1
            continue

        target = str(row["target"])
        text   = str(row["text"])

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r} ({TARGET_AR.get(target)})")

        result = call_groq_once(client, row_index, target, text)

        append_result_row(RESULTS_PATH, result)
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
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → pred={result['predicted_label']!r}  parse_ok={result['parse_success']}")

        # Pace: sleep between calls
        time.sleep(PACE_DELAY)

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # ── Write predictions TXT ─────────────────────────────────────────────────
    results_df = pd.read_csv(RESULTS_PATH, encoding="utf-8", keep_default_na=False)
    results_df = results_df[results_df["row_index"].isin(selected_indices)].copy()

    if results_df.empty:
        print("⚠️ No results found to evaluate.")
        return

    results_df = results_df.sort_values("row_index").copy()
    results_df.to_csv(RESULTS_PATH, index=False)

    y_pred = results_df["predicted_label"].tolist()

    txt_path = RESULTS_PATH.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"Saved test predictions to {txt_path}")
    print("\nPredicted stance distribution:")
    print(results_df["predicted_label"].value_counts().to_string())


if __name__ == "__main__":
    main()

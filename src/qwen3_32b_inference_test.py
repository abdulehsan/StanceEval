"""
qwen3_32b_inference_test.py — Stance detection on the blind test set (ground_truth.csv)
using OpenRouter's Qwen 3 32B model, returning only the stance label, with strict exit-on-error behavior.

Usage:
    python src/qwen3_32b_inference_test.py
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
RESULTS_PATH = os.path.join(PRED_DIR, "results_qwen_3_32b_test.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_qwen_3_32b_test_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
OPENROUTER_MODEL = "qwen/qwen3-32b"
TEMPERATURE = 0.5
MAX_TOKENS = 1500  # Gives budget for thinking, though final content is just a JSON stance

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}
TARGET_AR = "قيادة المرأة للسيارة"

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",
    "target",
    "qwen_stance",
    "parse_success",
    "raw_response",
]

# ── System Prompt (No reasoning output, only stance) ─────────────────────────
SYSTEM_PROMPT = (
    "You are an expert annotator and stance detection system for Arabic social media.\n\n"
    "Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels: Favor, Against, or None.\n\n"
    "### Reasoning Guidelines:\n"
    "1. Identify the target topic (provided in Arabic).\n"
    "2. Look past literal hashtags: a tweet may use a pro-target hashtag while its text sarcastically opposes it, or use an anti-target hashtag while its text explicitly argues in favor. Judge the sentence content, not the hashtag polarity.\n"
    "3. Distinguish the target of the opinion from other entities: pay attention to whether sentiment is directed at the target itself or a different topic mentioned in the same tweet.\n"
    "4. Reporting vs. stance: if the tweet only reports factual news (official statements, announcements) without explicit or implicit personal opinion, classify as \"None\".\n"
    "5. Sarcasm & implicit expression: sarcasm in dialectal Arabic can express a strong stance. Use context and tone, not literal word polarity, to determine the writer's actual position.\n"
    "6. Use \"None\" when the tweet is neutral, off-topic, or when the stance cannot be reasonably inferred.\n\n"
    "### Few-Shot Examples (from Training Set):\n\n"
    "Example 1 (Implicit Stance / Humor):\n"
    "Tweet: \"رسميًا صرت ملكة حجوزات تطعيم كورونا اتوقع باقي القطوه الي بالشارع م حجزت لها ههه\"\n"
    "Target: لقاح كورونا\n"
    "Stance: Favor\n\n"
    "Example 2 (Target Resolution / Sentiment Diversion):\n"
    "Tweet: \"هناك من عاصر زمن تحرير السود من العبودية وهناك من عاصر زمن تمكين المرأة واعطائها كامل حقوقها وشاء الله أن يكون زماننا زمن اعطاء الشواذ حقوقهم وهو الأسوأ حتى الآن أتمنى ألا تطول صولتهم\"\n"
    "Target: تمكين المرأة\n"
    "Stance: None\n\n"
    "Example 3 (Factual Reporting / News):\n"
    "Tweet: \"السديس يؤكد تفعيل التحول الإلكتروني في جميع تعاملات الرئاسة\"\n"
    "Target: التحول الرقمي\n"
    "Stance: None\n\n"
    "Example 4 (Genuine Against):\n"
    "Tweet: \"لايخدعونك بكذبة تمكين المرأة!.\"\n"
    "Target: تمكين المرأة\n"
    "Stance: Against\n\n"
    "Example 5 (Hijacked Hashtag Pattern):\n"
    "Tweet: \"#تمكين_المرأة بمفهوم الفارغون والفارغات والسطحيون والسطحيات والتافهون والتافهات هو اهلاك للمجتمع\"\n"
    "Target: تمكين المرأة\n"
    "Stance: Against\n\n"
    "### Output Format:\n"
    "You must return your output as a raw JSON object containing exactly one key:\n"
    "- \"stance\": exactly one of \"Favor\", \"Against\", or \"None\"\n\n"
    "Do not include any markdown formatting (like ```json), markdown code blocks, or introductory text. Respond only with the JSON object."
)

def build_user_message(text: str) -> str:
    return f"Target: {TARGET_AR}\nTweet: {text}"

def parse_json_response(raw: str) -> tuple[str, bool]:
    """Parse JSON response and extract stance."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        data = json.loads(raw)
        stance = str(data.get("stance", "None")).strip()
        for label in VALID_LABELS:
            if stance.lower() == label.lower():
                stance = label
                break
        if stance not in VALID_LABELS:
            return "None", False
        return stance, True
    except Exception:
        # Regex fallback
        stance_match = re.search(r'"stance"\s*:\s*"([^"]+)"', raw)
        if stance_match:
            cand = stance_match.group(1).strip()
            for label in VALID_LABELS:
                if cand.lower() == label.lower():
                    return label, True
        return "None", False

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

def call_openrouter_once(
    client,
    row_index: int,
    text: str,
) -> dict:
    user_msg = build_user_message(text)

    try:
        print("    Calling OpenRouter API...")
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=20.0,  # Prevent indefinite hangs
            extra_headers={
                "HTTP-Referer": "https://github.com/stanceeval",
                "X-Title": "StanceEval"
            }
        )

        raw_text = completion.choices[0].message.content or ""
        stance, parse_ok = parse_json_response(raw_text)

        if not parse_ok:
            print(f"\n❌ Parse Error on row_index={row_index}: Raw response: {repr(raw_text)}")
            print("Stopping script immediately as requested.")
            sys.exit(1)

        return {
            "row_index":     row_index,
            "target":        "Women Driving",
            "qwen_stance":   stance,
            "parse_success": True,
            "raw_response":  raw_text,
        }

    except Exception as exc:
        print(f"\n❌ API/Rate limit error on row_index={row_index}: {exc!r}")
        print("Stopping script immediately as requested.")
        sys.exit(1)

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY not found in .env or environment. Cannot proceed.")
        sys.exit(1)
    print("✅ OPENROUTER_API_KEY loaded.")

    print(f"\nLoading {TEST_CSV_PATH} …")
    test_df = pd.read_csv(TEST_CSV_PATH, keep_default_na=False)
    test_df = test_df.rename(columns={"id": "ID", "tweet_text": "text"})
    print(f"  Loaded {len(test_df)} rows.")

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    completed = load_completed_indices(RESULTS_PATH)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done (up to index {max(completed)}).")
    else:
        print(f"\n🆕 Starting fresh Qwen 3 32B run.")

    total = len(test_df)
    new_calls = 0
    skipped = 0

    for position, (_, row) in enumerate(test_df.iterrows(), start=1):
        row_index = int(row.name)

        if row_index in completed:
            skipped += 1
            continue

        text = str(row["text"])
        print(f"  [{position}/{total}] Qwen3-32B row_index={row_index}")

        result = call_openrouter_once(client, row_index, text)
        append_result_row(RESULTS_PATH, result)
        completed.add(row_index)
        new_calls += 1

        log_entry = {
            "row_index":     row_index,
            "raw_response":  result["raw_response"],
            "qwen_stance":   result["qwen_stance"],
            "parse_success": result["parse_success"],
            "timestamp":     time.time(),
        }
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → stance={result['qwen_stance']!r}")

        # Strict 5-second pacing delay
        time.sleep(5.0)

    print(f"\n── Qwen 3 32B predictions complete: {new_calls} new calls, {skipped} skipped ──\n")

if __name__ == "__main__":
    main()

"""
qwen_inference_test.py — Stance detection and reasoning on the blind test set (ground_truth.csv)
using Groq's Qwen 3.6 27B model, utilizing a local token budget tracker to prevent 429 daily rate limits.

Usage:
    python src/qwen_inference_test.py
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
RESULTS_PATH = os.path.join(PRED_DIR, "results_qwen_3.6_27b_test.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_qwen_3.6_27b_test_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
GROQ_MODEL = "qwen/qwen3.6-27b"
TEMPERATURE = 0.5
MAX_TOKENS = 150

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}
TARGET_AR = "قيادة المرأة للسيارة"

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",
    "target",
    "qwen_stance",
    "qwen_reasoning",
    "parse_success",
    "raw_response",
]

# ── System Prompt (V3 Expert annotator with few-shots) ────────────────────────
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
    "Reasoning: The writer jokingly highlights booking many vaccine appointments for others, implying active positive engagement with the vaccine.\n"
    "Stance: Favor\n\n"
    "Example 2 (Target Resolution / Sentiment Diversion):\n"
    "Tweet: \"هناك من عاصر زمن تحرير السود من العبودية وهناك من عاصر زمن تمكين المرأة واعطائها كامل حقوقها وشاء الله أن يكون زماننا زمن اعطاء الشواذ حقوقهم وهو الأسوأ حتى الآن أتمنى ألا تطول صولتهم\"\n"
    "Target: تمكين المرأة\n"
    "Reasoning: Negative language targets a different topic (LGBTQ+ rights), not women's empowerment itself, which the writer never opposes.\n"
    "Stance: None\n\n"
    "Example 3 (Factual Reporting / News):\n"
    "Tweet: \"السديس يؤكد تفعيل التحول الإلكتروني في جميع تعاملات الرئاسة\"\n"
    "Target: التحول الرقمي\n"
    "Reasoning: The tweet only reports an official's statement on activating electronic services, with no personal opinion expressed.\n"
    "Stance: None\n\n"
    "Example 4 (Genuine Against):\n"
    "Tweet: \"لايخدعونك بكذبة تمكين المرأة!.\"\n"
    "Target: تمكين المرأة\n"
    "Reasoning: The writer directly labels women's empowerment a lie/deception, explicitly rejecting the concept itself.\n"
    "Stance: Against\n\n"
    "Example 5 (Hijacked Hashtag Pattern):\n"
    "Tweet: \"#تمكين_المرأة بمفهوم الفارغون والفارغات والسطحيون والسطحيات والتافهون والتافهات هو اهلاك للمجتمع\"\n"
    "Target: تمكين المرأة\n"
    "Reasoning: Despite using the pro-empowerment hashtag, the writer calls the concept destructive to society, showing clear opposition.\n"
    "Stance: Against\n\n"
    "### Output Format:\n"
    "You must return your output as a raw JSON object containing exactly two keys:\n"
    "- \"stance\": exactly one of \"Favor\", \"Against\", or \"None\"\n"
    "- \"reasoning\": a brief explanation (maximum 15 words) in Arabic of why you chose this stance.\n\n"
    "Do not include any markdown formatting (like ```json), markdown code blocks, or introductory text. Respond only with the JSON object."
)

def build_user_message(text: str) -> str:
    return f"Target: {TARGET_AR}\nTweet: {text}"

def parse_json_response(raw: str) -> tuple[str, str, bool]:
    """Parse JSON response and extract stance and reasoning."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        data = json.loads(raw)
        stance = str(data.get("stance", "None")).strip()
        reasoning = str(data.get("reasoning", "")).strip()
        for label in VALID_LABELS:
            if stance.lower() == label.lower():
                stance = label
                break
        if stance not in VALID_LABELS:
            stance = "None"
        return stance, reasoning, True
    except Exception:
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

# ── Local Token Budget Tracker (Option A) ──────────────────────────────────────
def get_local_tpd_budget(log_path: str, limit_tpd: int = 200000) -> tuple[float, float]:
    """
    Scans the JSONL log file to calculate tokens consumed in the last 24 hours.
    Returns:
        (remaining_tokens, wait_seconds)
        If remaining_tokens < 1500, wait_seconds will be the time until the oldest
        request rolls out of the sliding 24-hour window.
    """
    if not os.path.exists(log_path):
        return limit_tpd, 0.0

    current_time = time.time()
    used_tokens = 0
    active_requests = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                log_time = data.get("timestamp", current_time)
                age = current_time - log_time
                if age < 86400:  # 24 hours
                    raw_resp = data.get("raw_response", "")
                    prompt_tokens = 1000  # Conservative estimate
                    resp_tokens = len(raw_resp.split()) * 1.5 if raw_resp else 40
                    total = prompt_tokens + resp_tokens
                    used_tokens += total
                    active_requests.append((log_time, total))
            except Exception:
                pass

    remaining_tokens = limit_tpd - used_tokens

    if remaining_tokens < 1500 and active_requests:
        active_requests.sort()
        oldest_time, _ = active_requests[0]
        # Rolls out exactly 24h after it was created
        wait_seconds = max(10.0, (oldest_time + 86400) - current_time)
        return remaining_tokens, wait_seconds

    return remaining_tokens, 0.0

def print_daily_budget_status(log_path: str, limit_tpd: int = 200000) -> None:
    if not os.path.exists(log_path):
        return
    current_time = time.time()
    used_uncached_tokens = 0
    used_requests = 0
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                log_time = data.get("timestamp", current_time)
                if current_time - log_time < 86400: # 24h
                    raw_resp = data.get("raw_response", "")
                    resp_tokens = len(raw_resp.split()) * 1.5 if raw_resp else 40
                    used_uncached_tokens += (40 + resp_tokens) # prompt cached (0 input count)
                    used_requests += 1
            except Exception:
                pass
                
    pct_used = (used_uncached_tokens / limit_tpd) * 100
    print(
        f"    📊 Groq Daily Budget: {int(used_uncached_tokens):,} / {limit_tpd:,} tokens used ({pct_used:.2f}%) "
        f"| Requests: {used_requests} / 1,000 used"
    )

# ── Groq API Pacing ───────────────────────────────────────────────────────────
def _parse_reset_duration(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?s", s):
        return float(s[:-1])
    if re.fullmatch(r"\d+(?:\.\d+)?ms", s):
        return float(s[:-2]) / 1000.0
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if m and (m.group(1) or m.group(2)):
        return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return 0.0

def compute_pace_delay(headers: dict) -> float:
    try:
        remaining_tok = int(headers.get("x-ratelimit-remaining-tokens", 10_000))
    except (ValueError, TypeError):
        remaining_tok = 10_000
    try:
        remaining_req = int(headers.get("x-ratelimit-remaining-requests", 100))
    except (ValueError, TypeError):
        remaining_req = 100

    reset_tok_str = headers.get("x-ratelimit-reset-tokens", "0s")
    reset_req_str = headers.get("x-ratelimit-reset-requests", "0s")

    if remaining_tok < 1200:
        wait = _parse_reset_duration(reset_tok_str)
        print(f"    ⚠️  Minute Token budget low ({remaining_tok} left) → sleeping {wait:.2f}s for reset")
        return wait

    if remaining_req < 2:
        wait = _parse_reset_duration(reset_req_str)
        print(f"    ⚠️  Daily Request budget low ({remaining_req} left) → sleeping {wait:.2f}s for reset")
        return wait

    return 0.0

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
    attempt = 0

    while True:
        try:
            print("    Calling Groq API...")
            raw_resp = client.chat.completions.with_raw_response.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                top_p=0.8,
                max_tokens=MAX_TOKENS,
                reasoning_effort="none",
                timeout=20.0,  # Prevent indefinite hangs
            )

            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()

            raw_text = completion.choices[0].message.content or ""
            model_returned = completion.model
            stance, reasoning, parse_ok = parse_json_response(raw_text)

            result = {
                "row_index":      row_index,
                "target":         "Women Driving",
                "qwen_stance":    stance,
                "qwen_reasoning": reasoning,
                "model_id":       model_returned,
                "parse_success":  parse_ok,
                "raw_response":   raw_text,
            }
            return result, rl_headers

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                wait_sec = 30.0
                match = re.search(r"try again in (\d+m)?(\d+(?:\.\d+)?)s", exc_str)
                if match:
                    mins = float(match.group(1)[:-1]) if match.group(1) else 0.0
                    secs = float(match.group(2))
                    wait_sec = mins * 60 + secs + 2.0
                    print(f"    ⚠️  Groq Rate Limit (429) on row {row_index} — parsed wait time: {int(wait_sec)}s")
                else:
                    print(f"    ⚠️  Groq Rate Limit (429) on row {row_index} — sleeping default 30.0s")
                time.sleep(wait_sec)
                continue

            attempt += 1
            wait = 2.0 * (2 ** (attempt - 1))
            print(f"    ❌ Row {row_index} attempt {attempt}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries:
                time.sleep(wait)
            else:
                break

    print(f"    ❌ Row {row_index}: all {max_retries} attempts failed; recording default.")
    result = {
        "row_index":      row_index,
        "target":         "Women Driving",
        "qwen_stance":    "None",
        "qwen_reasoning": "API error",
        "model_id":       GROQ_MODEL,
        "parse_success":  False,
        "raw_response":   "ERROR: max retries exceeded",
    }
    return result, {}

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        print("❌ GROQ_API_KEY not found in .env or environment. Cannot proceed.")
        sys.exit(1)

    print(f"\nLoading {TEST_CSV_PATH} …")
    test_df = pd.read_csv(TEST_CSV_PATH, keep_default_na=False)
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
        print(f"\n🆕 Starting fresh Qwen run.")

    total = len(test_df)
    new_calls = 0
    skipped = 0

    for position, (_, row) in enumerate(test_df.iterrows(), start=1):
        row_index = int(row.name)

        if row_index in completed:
            skipped += 1
            continue



        text = str(row["text"])
        print(f"  [{position}/{total}] Qwen row_index={row_index}")

        result, rl_headers = call_groq_once(client, row_index, text)
        append_result_row(RESULTS_PATH, result)
        completed.add(row_index)
        new_calls += 1

        log_entry = {
            "row_index":      row_index,
            "raw_response":   result["raw_response"],
            "qwen_stance":    result["qwen_stance"],
            "qwen_reasoning": result["qwen_reasoning"],
            "parse_success":  result["parse_success"],
            "timestamp":      time.time(),
        }
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → stance={result['qwen_stance']!r}  reasoning={result['qwen_reasoning']!r}")
        print_daily_budget_status(RAW_LOG_PATH)

        # Pace requests
        delay = compute_pace_delay(rl_headers)
        time.sleep(max(delay, 2.0))

    print(f"\n── Qwen predictions complete: {new_calls} new calls, {skipped} skipped ──\n")

if __name__ == "__main__":
    main()

"""
run_kb_test.py — best_86_61 + hashtag/entity KB (isolated addition, nothing else changed)
against all 352 blind-test tweets via Cerebras Gemma 4 31B.

USAGE:
    python run_kb_test.py

Output: writes predictions to predictions/kb_test_results.csv and
        raw logs to qwen_raw_logs/kb_test_raw.jsonl.
        Prints Favor/Against/None split to compare against the known
        distribution: Favor 44.89% / Against 45.45% / None 9.66%.

IMPORTANT — before trusting this run:
  - T=0.1, same as best_86_61's confirmed setting. Do not change without
    a separate isolated test.
  - This script does NOT submit to CodaBench. It only produces local
    predictions + the distribution sanity check. Submission is a separate,
    deliberate step after you've reviewed this output.
  - This is the ONLY isolated test of the KB addition. Do not bundle with
    the implementation-complaint clause or any other pending change.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time

# ── Portable paths ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE  # run_kb_test.py lives at repo root

# prompt_builder.py lives next to this file (repo root)
sys.path.insert(0, _HERE)

# Windows: force UTF-8 so Arabic text doesn't crash the terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from prompt_builder import build_prompt_for_tweet

DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR  = os.path.join(_ROOT, "qwen_raw_logs")

INPUT_CSV   = os.path.join(DATA_DIR, "ground_truth.csv")
OUTPUT_CSV  = os.path.join(PRED_DIR, "kb_test_results.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "kb_test_raw.jsonl")

# ── Model / API constants ───────────────────────────────────────────────────────
MODEL_ID    = "gemma-4-31b"
TEMPERATURE = 0.1
TOP_P       = 0.95
MAX_TOKENS  = 10  # single-word label only

VALID_LABELS = {"Favor", "Against", "None"}

RESULTS_FIELDNAMES = [
    "row_index",
    "id",
    "tweet_id",
    "target",
    "predicted_label",
    "model_id",
    "parse_success",
    "raw_response",
]

TARGET_AR = {
    "Women Driving": "قيادة المرأة للسيارة"
}


# ── Parsing ─────────────────────────────────────────────────────────────────────
def parse_response(raw: str) -> tuple[str, bool]:
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


# ── Rate-limit pacing (mirrors ci_v3) ──────────────────────────────────────────
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
        print(f"    ⚠️  Requests budget almost exhausted ({remaining_req_min} left) → sleeping 60s")
        return 60.0

    if remaining_tok_min < 1000:
        print(f"    ⚠️  Token budget low ({remaining_tok_min} left) → sleeping 30s")
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


# ── Resume support ──────────────────────────────────────────────────────────────
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


def append_result_row(results_path: str, row: dict) -> None:
    write_header = not os.path.exists(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in RESULTS_FIELDNAMES})


# ── API call with retry ─────────────────────────────────────────────────────────
def call_cerebras_once(
    client,
    row_index: int,
    tweet_text: str,
    target: str,
    max_retries: int = 4,
) -> tuple[dict, dict]:
    system_prompt = build_prompt_for_tweet(tweet_text)
    arabic_target = TARGET_AR.get(target, target)
    user_msg = f"Tweet: {tweet_text}\nTarget: {arabic_target}"

    for attempt in range(max_retries):
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
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
                "id":              "",   # filled in by caller
                "tweet_id":        "",   # filled in by caller
                "target":          target,
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
                    f"    ⚠️  429 on row {row_index} (attempt {attempt + 1}/{max_retries}) — "
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
        "id":              "",
        "tweet_id":        "",
        "target":          target,
        "predicted_label": "None",
        "model_id":        MODEL_ID,
        "parse_success":   False,
        "raw_response":    "ERROR: max retries exceeded",
    }
    return result, {}


# ── Main ────────────────────────────────────────────────────────────────────────
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

    print(f"\nLoading {INPUT_CSV} …")
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  Loaded {len(rows)} rows.")

    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    completed = load_completed_indices(OUTPUT_CSV)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done.")
    else:
        print(f"\n🆕 Starting fresh run.")

    request_timestamps = load_request_timestamps(RAW_LOG_PATH)

    print(f"── Inference config ──")
    print(f"   Model:       {MODEL_ID}")
    print(f"   Prompt:      best_86_61 + KB (per-tweet dynamic injection)")
    print(f"   temperature={TEMPERATURE}  top_p={TOP_P}")
    print(f"   Results CSV: {OUTPUT_CSV}")
    print()

    total = len(rows)
    new_calls = 0
    skipped = 0
    rl_headers: dict = {}

    for position, row in enumerate(rows, start=1):
        row_index = position - 1  # 0-based

        if row_index in completed:
            skipped += 1
            continue

        tweet_text = str(row["tweet_text"])
        target     = str(row["target"])

        # ── Hourly rate-limit guard (150 req / 60 min) ──
        now = time.time()
        request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        if len(request_timestamps) >= 150:
            oldest_ts  = request_timestamps[0]
            sleep_time = oldest_ts + 3600 - now + 1.0

            if rl_headers:
                reset_str = rl_headers.get("x-ratelimit-reset-requests-hour") or \
                            rl_headers.get("x-ratelimit-reset-requests-day")
                if reset_str:
                    try:
                        header_sleep = _parse_reset_duration(str(reset_str))
                        if header_sleep > sleep_time:
                            sleep_time = header_sleep + 1.0
                    except Exception:
                        pass

            resume_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + sleep_time))
            print(f"\n⏳ HOURLY RATE LIMIT REACHED (150 requests in last 60 minutes).")
            print(f"   Sleeping {sleep_time:.1f}s — resuming at {resume_at}\n")
            time.sleep(sleep_time)
            now = time.time()
            request_timestamps = [ts for ts in request_timestamps if now - ts < 3600]

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r}")

        result, rl_headers = call_cerebras_once(client, row_index, tweet_text, target)
        request_timestamps.append(time.time())

        # patch in id/tweet_id from source row
        result["id"]       = row.get("id", "")
        result["tweet_id"] = row.get("tweet_id", "")

        append_result_row(OUTPUT_CSV, result)
        completed.add(row_index)
        new_calls += 1

        log_entry = {
            "row_index":       row_index,
            "target":          target,
            "raw_response":    result["raw_response"],
            "predicted_label": result["predicted_label"],
            "model_id":        result["model_id"],
            "parse_success":   result["parse_success"],
            "timestamp":       time.time(),
        }
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    → pred={result['predicted_label']!r}  parse_ok={result['parse_success']}")

        delay = compute_pace_delay(rl_headers)
        if delay > 0:
            time.sleep(delay)

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # ── Distribution sanity check ───────────────────────────────────────────────
    counts: dict[str, int] = {"Favor": 0, "Against": 0, "None": 0, "UNPARSED": 0}
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = r.get("predicted_label", "UNPARSED")
            counts[label] = counts.get(label, 0) + 1

    total_preds = sum(counts.values())
    print("=== Predicted distribution vs. known true distribution ===")
    for label, true_pct in [("Favor", 44.89), ("Against", 45.45), ("None", 9.66)]:
        pred_pct = counts.get(label, 0) / total_preds * 100
        print(f"{label:8s}  predicted {pred_pct:5.2f}%   true {true_pct:5.2f}%   delta {pred_pct - true_pct:+.2f}")
    if counts.get("UNPARSED", 0):
        print(f"UNPARSED  {counts['UNPARSED']} rows — inspect these before drawing conclusions")

    print(f"\nWrote predictions → {OUTPUT_CSV}")
    print(f"Raw log          → {RAW_LOG_PATH}")


if __name__ == "__main__":
    main()

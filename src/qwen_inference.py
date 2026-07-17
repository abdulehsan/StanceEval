"""
qwen_inference.py — Zero-shot stance detection using Qwen3-32B via Groq.

Compares qwen/qwen3-32b (Groq, zero-shot) against our fine-tuned AraBERT
checkpoints on the Mawqif-v2 dev set for StanceEval-2026.

Runs on exactly 100 rows selected with balanced target coverage (Step 1 checks
whether the first 100 rows are balanced; if not, takes ~33 per target instead).

Usage:
    # Step 1 only — distribution check, no inference:
    python src/qwen_inference.py --check_only

    # Full run (100 rows, stratified if needed):
    python src/qwen_inference.py

    # Resume from last completed row (reads existing results CSV automatically):
    python src/qwen_inference.py

Requirements:
    - GROQ_API_KEY in .env (loaded via python-dotenv; never printed or committed)
    - pip install groq python-dotenv pandas scikit-learn

Model: qwen/qwen3-32b (Groq, Preview)
Non-thinking mode params: temperature=0.7, top_p=0.8, top_k=20, min_p=0
Toggle: reasoning_effort="none"  (confirmed from Groq docs; "none"/"default" only)

Rate limiting: fully header-driven.  x-ratelimit-remaining-{tokens,requests} and
x-ratelimit-reset-{tokens,requests} are read from every response; the next call
is delayed accordingly.  On a 429 the retry-after header is respected.
No hardcoded sleep values.

Output:
    predictions/qwen3_32b_groq_100row_results.csv  (incremental, one row/call)
    qwen_raw_logs/qwen3_32b_groq_raw.jsonl         (full prompt+response log)

Resumability: completed row_index values are read from the results CSV at startup;
already-done rows are skipped without re-calling the API.
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
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")

DEV_CSV_PATH = os.path.join(DATA_DIR, "dev.csv")
RESULTS_PATH = os.path.join(PRED_DIR, "results_qwen3_32b.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_qwen3_32b_raw.jsonl")

# ── Model / API constants ─────────────────────────────────────────────────────
MODEL_ID = "qwen/qwen3-32b"  # updated for 32b run with new system prompt

# Non-thinking mode params (Groq model card recommendation for qwen/qwen3-32b):
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0
MAX_TOKENS = 10  # single-word label only

# Sampling constants
N_ROWS = 100
ROWS_PER_TARGET = 33  # 33 + 33 + 34 = 100 across 3 targets

VALID_LABELS: set[str] = set(LABEL2ID.keys())  # {"Against", "Favor", "None"}
TARGETS = ["Covid Vaccine", "Digital Transformation", "Women empowerment"]

# ── Results CSV schema ────────────────────────────────────────────────────────
RESULTS_FIELDNAMES = [
    "row_index",      # original 0-based index in dev.csv (preserved for traceability)
    "target",
    "gold_label",
    "predicted_label",
    "model_id",       # exact model string returned by the API in each response
    "parse_success",  # True if response was a clean single-word label
    "raw_response",
]


# ── Prompt ────────────────────────────────────────────────────────────────────
PREVIOUS_SYSTEM_PROMPT = (
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

SYSTEM_PROMPT = (
    "You are an expert annotator for Arabic stance detection.\n\n"

    "Your task is to determine the writer's stance toward a specified target topic "
    "based on the overall meaning of the Arabic tweet.\n\n"

    "Before assigning a label, carefully reason about the tweet:\n"
    "1. Identify the target topic.\n"
    "2. Determine whether the writer expresses an opinion.\n"
    "3. Identify who or what the opinion is directed toward.\n"
    "4. Distinguish the stance toward the target from opinions about other people, "
    "organizations, events, policies, implementations, or related entities.\n"
    "5. If multiple entities or viewpoints are mentioned, determine the writer's "
    "stance only toward the specified target.\n"
    "6. Base your decision on the complete meaning and context of the tweet rather "
    "than isolated words or overall sentiment.\n"
    "7. Do not assume that negative sentiment implies an Against stance or that "
    "positive sentiment implies a Favor stance. The stance depends on whether the "
    "opinion is directed toward the target itself.\n"
    "8. If the tweet discusses consequences, criticisms, sarcasm, rhetorical "
    "questions, or indirect references, infer the writer's underlying position "
    "toward the target whenever it is reasonably supported by the tweet.\n"
    "9. Choose None only when the stance toward the target is genuinely absent, "
    "unclear, neutral, or cannot be reasonably inferred.\n\n"

    "Assign exactly one of the following labels:\n"
    "- Favor: The writer expresses support for or a positive stance toward the target.\n"
    "- Against: The writer expresses opposition to or a negative stance toward the target.\n"
    "- None: The writer expresses no identifiable stance toward the target.\n\n"

    "Respond with exactly one word and nothing else:\n"
    "Favor\n"
    "Against\n"
    "None"
)


def build_user_message(target: str, text: str) -> str:
    return f"Target: {target}\nTweet: {text}"


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str, bool]:
    """Parse a model response and return (stance_label, parse_success).

    parse_success=True  → clean single-word hit (reliable parse).
    parse_success=False → fallback match; prediction is still made but flagged.
    """
    raw = raw.strip()

    # Attempt 1: exact word match (expected happy path for single-word prompt)
    if raw in VALID_LABELS:
        return raw, True

    # Attempt 2: case-insensitive exact match
    for label in VALID_LABELS:
        if raw.lower() == label.lower():
            return label, True

    # Attempt 3: bare label substring anywhere (degraded — flag as parse_success=False)
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw.lower():
            return label, False

    # Attempt 4: JSON fallback (old format, unlikely with this prompt but kept for safety)
    try:
        obj = json.loads(raw)
        stance = str(obj.get("stance", "")).strip()
        if stance in VALID_LABELS:
            return stance, False  # not a clean single-word response
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r'"stance"\s*:\s*"(Favor|Against|None)"', raw)
    if match:
        return match.group(1), False

    # Fallback — keep as None and flag
    return "None", False


# ── Step 1: Stratified 100-row selection ─────────────────────────────────────

def select_100_rows(dev_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], str]:
    """Select the 100 rows for inference with balanced target coverage.

    Checks the first 100 rows of the dev set as-is. If every target has at
    least 20 rows out of 100 (roughly balanced), uses them in original order.
    Otherwise, takes the first 33/33/34 rows from each target group and
    concatenates in original index order.

    Returns:
        (run_df, selected_original_indices, path_taken_description)
    """
    first_100 = dev_df.head(N_ROWS)
    counts = first_100["target"].value_counts()

    print(f"\n── Step 1: Target distribution — first {N_ROWS} rows (original order) ──")
    print(counts.to_string())

    min_count = min(counts.get(t, 0) for t in TARGETS)
    balanced_threshold = 30  # ≥ 30 rows per target (≥ 30% of 100) → "reasonably balanced"
                               # 23 Covid rows in the first 100 is a 2:1 skew → triggers stratify

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
        run_df = pd.concat(parts).sort_index()  # restore original index order

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


# ── Rate-limit header utilities ───────────────────────────────────────────────

def _parse_reset_duration(s: str) -> float:
    """Parse a Groq reset-time header like '1.234s', '60s', or '1m30.5s' → seconds."""
    if not s:
        return 0.0
    s = s.strip()
    # Fast path: plain seconds e.g. "1.5s"
    if re.fullmatch(r"\d+(?:\.\d+)?s", s):
        return float(s[:-1])
    # Milliseconds e.g. "500ms"
    if re.fullmatch(r"\d+(?:\.\d+)?ms", s):
        return float(s[:-2]) / 1000.0
    # General: optional minutes + optional seconds e.g. "1m30.5s" or "2m"
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if m and (m.group(1) or m.group(2)):
        return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return 0.0


def compute_pace_delay(headers: dict) -> float:
    """Read rate-limit headers and return how long to sleep before the next call.

    Strategy:
      - If remaining tokens < 500 (≈ 1 short response), wait for token reset.
      - If remaining requests < 2, wait for request reset.
      - Otherwise, return 0.0 (proceed immediately).
    """
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

    print(
        f"    RL — remaining tokens: {remaining_tok} (reset in {reset_tok_str})"
        f"  |  remaining requests: {remaining_req} (reset in {reset_req_str})"
    )

    if remaining_tok < 500:
        wait = _parse_reset_duration(reset_tok_str)
        print(f"    ⚠️  Token budget low ({remaining_tok} left) → sleeping {wait:.2f}s for token reset")
        return wait

    if remaining_req < 2:
        wait = _parse_reset_duration(reset_req_str)
        print(f"    ⚠️  Request budget low ({remaining_req} left) → sleeping {wait:.2f}s for request reset")
        return wait

    return 0.0


# ── Resumability ──────────────────────────────────────────────────────────────

def load_completed_indices(results_path: str) -> set[int]:
    """Return the set of row_index values already written to the results CSV."""
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
    """Incrementally append one result row to the CSV (creates header if new file)."""
    write_header = not os.path.exists(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in RESULTS_FIELDNAMES})


# ── Single API call ───────────────────────────────────────────────────────────

def call_groq_once(
    client,
    row_index: int,
    target: str,
    text: str,
    max_retries: int = 3,
) -> tuple[dict, dict]:
    """Make one synchronous Groq API call with retry logic.

    Uses with_raw_response to capture rate-limit headers from every response.

    Returns:
        (result_dict, rate_limit_headers)
        result_dict keys match RESULTS_FIELDNAMES (gold_label filled by caller).
    """
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
                reasoning_effort="none",  # non-thinking mode — confirmed param name
                # extra_body omitted: Groq rejects top_k and min_p with 400.
                # Only standard OpenAI-compat params (temperature, top_p) accepted.
            )

            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()

            raw_text = completion.choices[0].message.content or ""
            model_returned = completion.model  # exact model string from API (may differ from request)
            predicted, parse_ok = parse_response(raw_text)

            result = {
                "row_index":       row_index,
                "target":          target,
                "gold_label":      "",  # filled by caller
                "predicted_label": predicted,
                "model_id":        model_returned,
                "parse_success":   parse_ok,
                "raw_response":    raw_text,
            }
            return result, rl_headers

        except Exception as exc:
            exc_str = str(exc)

            # 429 rate-limit error — read retry-after from the exception message
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                # Groq embeds retry-after in the error body; try to extract it
                retry_after = 60.0  # safe default
                m = re.search(r"(?:retry.after|retry_after)[^\d]*(\d+(?:\.\d+)?)", exc_str, re.IGNORECASE)
                if m:
                    retry_after = float(m.group(1))
                print(
                    f"    ⚠️  429 on row {row_index} (attempt {attempt + 1}/{max_retries}) — "
                    f"sleeping {retry_after:.1f}s (retry-after from error)"
                )
                time.sleep(retry_after)
                continue  # retry same attempt count

            # Other transient errors — exponential backoff
            wait = 2 ** attempt
            print(f"    ❌ Row {row_index} attempt {attempt + 1}/{max_retries}: {exc!r} — retrying in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    # All retries exhausted — return a failure sentinel
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load GROQ_API_KEY from .env ─────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass  # fall through — key may already be in environment

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        print("❌ GROQ_API_KEY not found in .env or environment. Cannot proceed.")
        sys.exit(1)
    print("✅ GROQ_API_KEY loaded.")  # never printed, only confirmed present

    # ── Load dev set ────────────────────────────────────────────────────────
    print(f"\nLoading {DEV_CSV_PATH} …")
    dev_df = load_data(DEV_CSV_PATH)
    print(f"  Loaded {len(dev_df)} rows.")

    # ── Step 1: distribution check + row selection ──────────────────────────
    run_df, selected_indices, path_taken = select_100_rows(dev_df)

    if args.check_only:
        print("\n── --check_only flag set: stopping after Step 1. ──")
        return

    # ── Init Groq client ────────────────────────────────────────────────────
    try:
        from groq import Groq
    except ImportError:
        print("❌ groq package not installed. Run: pip install groq")
        sys.exit(1)

    client = Groq(api_key=groq_key)

    # ── Ensure output dirs exist ─────────────────────────────────────────────
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Resumability ────────────────────────────────────────────────────────
    completed = load_completed_indices(RESULTS_PATH)
    if completed:
        print(f"\n📂 Resuming: {len(completed)} rows already done (up to index {max(completed)}).")
    else:
        print(f"\n🆕 Starting fresh run.")

    # ── Inference config summary ────────────────────────────────────────────
    print(f"── Inference config ──")
    print(f"   Model:            qwen3.6-27b")
    print(f"   reasoning_effort: none  (non-thinking mode)")
    print(f"   temperature={TEMPERATURE}  top_p={TOP_P}")
    print(f"   max_tokens:       {MAX_TOKENS}")
    print(f"   Note: top_k and min_p omitted — rejected by Groq API (400)")
    print(f"   Sequential calls; rate pacing from response headers")
    print(f"   Results CSV:      results_qwen3.6-27b.csv")
    print(f"   Raw log:          raw_qwen3.6-27b.jsonl")
    print()

    total = len(run_df)
    new_calls = 0
    skipped = 0
    _header_keys_logged = False  # log raw RL header keys once on first response

    # ── Sequential inference loop ────────────────────────────────────────────
    for position, (_, row) in enumerate(run_df.iterrows(), start=1):
        row_index = int(row.name)  # original 0-based index in dev.csv

        if row_index in completed:
            skipped += 1
            continue

        target    = str(row["target"])
        text      = str(row["text"])
        gold      = str(row.get("stance", ""))

        print(f"  [{position}/{total}] row_index={row_index} | target={target!r}")

        result, rl_headers = call_groq_once(client, row_index, target, text)
        result["gold_label"] = gold

        # ── Incremental write (one row per call, never all-at-end) ───────────
        append_result_row(RESULTS_PATH, result)
        completed.add(row_index)
        new_calls += 1

        # ── JSONL raw log ────────────────────────────────────────────────────
        log_entry = {
            "row_index":       row_index,
            "target":          target,
            "gold_label":      gold,
            "raw_response":    result["raw_response"],
            "predicted_label": result["predicted_label"],
            "model_id":        result["model_id"],
            "parse_success":   result["parse_success"],
        }
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(
            f"    → pred={result['predicted_label']!r}  gold={gold!r}  "
            f"model={result['model_id']!r}  parse_ok={result['parse_success']}"
        )

        # ── Log raw header keys once (first response only) ───────────────────
        if rl_headers and not _header_keys_logged:
            rl_keys = [k for k in rl_headers if "ratelimit" in k.lower()]
            print(f"    ── Rate-limit header keys present in first response ──")
            for k in sorted(rl_keys):
                print(f"       {k}: {rl_headers[k]}")
            _header_keys_logged = True

        # ── Header-driven pacing ─────────────────────────────────────────────
        if rl_headers:
            delay = compute_pace_delay(rl_headers)
            if delay > 0:
                time.sleep(delay)

    print(f"\n── Inference complete: {new_calls} new calls, {skipped} skipped ──\n")

    # ── Step 5: Evaluation ──────────────────────────────────────────────────
    results_df = pd.read_csv(RESULTS_PATH, encoding="utf-8", keep_default_na=False)
    # Restrict to the exact rows selected for this run (handles partial-resume edge cases)
    results_df = results_df[results_df["row_index"].isin(selected_indices)].copy()

    if results_df.empty:
        print("⚠️  No results found to evaluate. Check the results CSV.")
        return

    print(f"── Step 5: Evaluation on {len(results_df)} rows ──\n")

    print("Predicted label distribution:")
    print(results_df["predicted_label"].value_counts().to_string())
    print("\nGold label distribution:")
    print(results_df["gold_label"].value_counts().to_string())

    # Accuracy
    correct = (results_df["predicted_label"] == results_df["gold_label"]).sum()
    accuracy = correct / len(results_df)
    print(f"\nOverall accuracy: {correct}/{len(results_df)} = {accuracy:.4f}")

    # Favg2 — reuses the existing favg2() from metrics.py (do not reimplement)
    y_true = results_df["gold_label"].tolist()
    y_pred = results_df["predicted_label"].tolist()
    overall_f2 = favg2(y_true, y_pred)
    print(f"Overall Favg2:    {overall_f2:.4f}")

    # Per-target Favg2 / Favg3
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

    print(f"\nResults CSV : {RESULTS_PATH}")
    print(f"Raw JSONL   : {RAW_LOG_PATH}")
    print(
        f"\n⚠️  NOTE: Metrics computed on {len(results_df)}/619 rows (100-row subset)."
        f"\n    Do NOT add to results_summary.csv — not comparable to full-dev scores."
        f"\n    Row-selection path: {path_taken}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot stance detection with Qwen3-32B (Groq) on 100 stratified dev rows."
        )
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Run Step 1 (distribution check + row selection) only, then stop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

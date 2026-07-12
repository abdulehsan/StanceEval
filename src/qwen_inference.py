"""
qwen_inference.py — Zero-shot stance detection using Qwen 2.5 72B via HF Router.

Runs on the 619 dev.csv rows (or a subset for A/B testing prompt framings).
No training — purely inference. Logs every prompt+response to JSONL.

Usage:
    # A/B test: run 50 rows with English framing first
    python src/qwen_inference.py --framing en --subset 50

    # Then run Arabic framing on the same 50 rows
    python src/qwen_inference.py --framing ar --subset 50

    # Full run (619 rows) after committing to a framing
    python src/qwen_inference.py --framing en

Requirements:
    - HF_TOKEN in .env (loaded via python-dotenv)
    - pip install openai python-dotenv

Costs: ~250K input / 10K output tokens ≈ $0.05-0.15 at DeepInfra rates.
Check your HF billing page after the first 50-row subset run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import random
import re
import sys
import time
from typing import Optional

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (
    LABEL2ID,
    load_data,
    write_pred_csv,
    write_pred_txt,
)
from metrics import per_topic_and_overall_metrics, run_official_eval

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")
RESULTS_CSV = os.path.join(_ROOT, "results_summary.csv")

RAW_LOG_PATH = os.path.join(LOG_DIR, "qwen25_72b_zeroshot_raw.jsonl")
PRED_CSV_PATH = os.path.join(PRED_DIR, "qwen25_72b_zeroshot_dev_preds.csv")
PRED_TXT_PATH = os.path.join(PRED_DIR, "qwen25_72b_zeroshot_dev_preds.txt")

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
BASE_URL = "https://router.huggingface.co/v1"

VALID_LABELS = set(LABEL2ID.keys())


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_EN = """You are an expert annotator for Arabic stance detection. Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels:

- Favor: the writer expresses support for or a positive position toward the target
- Against: the writer expresses opposition to or a negative position toward the target
- None: no clear stance — neutral, off-topic, or ambiguous. This includes cases where sarcasm makes the literal tone misleading about the writer's actual position — do not infer a stance from tone alone if the underlying position isn't clear.

Respond with ONLY a JSON object in this exact format, no other text:
{"stance": "Favor"} or {"stance": "Against"} or {"stance": "None"}"""

# Arabic system prompt — JSON output schema stays in English for reliable parsing
SYSTEM_AR = """أنت خبير في التعليق التلقائي على المواقف في النصوص العربية. بناءً على تغريدة عربية وموضوع مستهدف، صنّف موقف كاتب التغريدة من الموضوع المستهدف كواحد فقط من ثلاثة تسميات:

- Favor: يُعبّر الكاتب عن دعمه أو موقف إيجابي تجاه الموضوع
- Against: يُعبّر الكاتب عن معارضته أو موقف سلبي تجاه الموضوع
- None: لا يوجد موقف واضح — محايد أو خارج الموضوع أو غامض. يشمل ذلك الحالات التي يكون فيها السخرية مضللة حول الموقف الفعلي للكاتب — لا تستنتج الموقف من النبرة وحدها إذا كان الموقف الفعلي غير واضح.

أجب فقط بكائن JSON بهذا الشكل بالضبط، بدون أي نص إضافي:
{"stance": "Favor"} أو {"stance": "Against"} أو {"stance": "None"}"""


def build_user_message(target: str, text: str) -> str:
    return f"Target: {target}\nTweet: {text}"


def get_system_prompt(framing: str) -> str:
    if framing == "en":
        return SYSTEM_EN
    elif framing == "ar":
        return SYSTEM_AR
    raise ValueError(f"Unknown framing: {framing!r}. Use 'en' or 'ar'.")


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str, bool]:
    """Parse a model response and return (stance_label, parse_success).

    Tries JSON first, then regex fallback, then defaults to 'None'.
    """
    raw = raw.strip()

    # Attempt 1: parse as JSON
    try:
        obj = json.loads(raw)
        stance = str(obj.get("stance", "")).strip()
        if stance in VALID_LABELS:
            return stance, True
    except (json.JSONDecodeError, AttributeError):
        pass

    # Attempt 2: regex — find "stance": "Favor|Against|None"
    match = re.search(r'"stance"\s*:\s*"(Favor|Against|None)"', raw)
    if match:
        return match.group(1), True

    # Attempt 3: bare label anywhere in the response
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw.lower():
            return label, False  # found but not via JSON — flag as parse_success=False

    # Fallback
    return "None", False


# ── Async inference ───────────────────────────────────────────────────────────

async def call_model(
    client,
    row_id: int,
    target: str,
    text: str,
    framing: str,
    semaphore: asyncio.Semaphore,
    model_id: str,
    max_retries: int = 3,
) -> dict:
    """Make one API call with retry-backoff. Returns a log dict."""
    system_prompt = get_system_prompt(framing)
    user_msg = build_user_message(target, text)
    prompt_str = f"System: {system_prompt}\n\nUser: {user_msg}"

    last_error = None
    for attempt in range(max_retries):
        async with semaphore:
            try:
                loop = asyncio.get_event_loop()
                # openai client is sync; run in thread pool to avoid blocking
                response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0,
                        max_tokens=20,
                    ),
                )
                raw_response = response.choices[0].message.content or ""
                parsed_stance, parse_success = parse_response(raw_response)
                return {
                    "ID": row_id,
                    "prompt_framing": framing,
                    "prompt": prompt_str,
                    "raw_response": raw_response,
                    "parsed_stance": parsed_stance,
                    "parse_success": parse_success,
                }
            except Exception as e:
                last_error = e
                # Exponential backoff: 2^attempt seconds
                wait = 2 ** attempt
                print(f"  ⚠️  Row {row_id} attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait}s ...")
                await asyncio.sleep(wait)

    # All retries exhausted
    print(f"  ❌ Row {row_id} failed after {max_retries} attempts: {last_error}")
    return {
        "ID": row_id,
        "prompt_framing": framing,
        "prompt": prompt_str,
        "raw_response": "",
        "parsed_stance": "None",
        "parse_success": False,
        "error": str(last_error),
    }


async def run_inference(
    df,
    framing: str,
    concurrency: int,
    api_key: str,
    base_url: str,
    model_id: str,
) -> list[dict]:
    """Run async inference on all rows in df. Returns list of log dicts."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        call_model(
            client,
            row["ID"],
            row["target"],
            row["text"],
            framing,
            semaphore,
            model_id,
        )
        for _, row in df.iterrows()
    ]

    results = []
    total = len(tasks)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            successes = sum(1 for r in results if r.get("parse_success"))
            print(f"  Progress: {i+1}/{total} | Parse success: {successes}/{i+1}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load credentials from .env ──────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass  # dotenv not installed; check environment variables directly

    sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()

    if sambanova_key:
        print("Using SambaNova Cloud serverless endpoint...")
        base_url = "https://api.sambanova.ai/v1"
        api_key = sambanova_key
        model_id = "Qwen2.5-72B-Instruct"  # SambaNova Qwen model ID
    else:
        # Fall back to Hugging Face Router
        base_url = BASE_URL
        api_key = os.environ.get("HF_TOKEN", "").strip()
        model_id = MODEL_ID
        if not api_key:
            print("❌ Error: No credentials found. Please set either SAMBANOVA_API_KEY or HF_TOKEN in your .env file.")
            sys.exit(1)

    # ── Load dev data ─────────────────────────────────────────────────────
    print(f"\nLoading data/dev.csv ...")
    dev_df = load_data(os.path.join(DATA_DIR, "dev.csv"))
    print(f"  Loaded {len(dev_df)} rows.")

    # ── Subset selection ──────────────────────────────────────────────────
    if args.subset and args.subset < len(dev_df):
        print(f"\nSubset mode: running on {args.subset} rows (seed=42).")
        print("  Use the same subset for both framings when A/B testing.")
        rng = random.Random(42)
        indices = rng.sample(range(len(dev_df)), args.subset)
        indices_sorted = sorted(indices)
        run_df = dev_df.iloc[indices_sorted].reset_index(drop=True)
        is_subset = True
    else:
        run_df = dev_df.reset_index(drop=True)
        is_subset = False

    print(f"\nRunning Qwen 2.5 72B zero-shot inference ...")
    print(f"  Framing: {args.framing}")
    print(f"  Rows: {len(run_df)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Endpoint: {base_url}")
    print(f"  Model ID: {model_id}")

    # ── Run inference ─────────────────────────────────────────────────────
    start = time.time()
    results = asyncio.run(run_inference(run_df, args.framing, args.concurrency, api_key, base_url, model_id))
    elapsed = time.time() - start
    print(f"\n  Finished in {elapsed:.1f}s ({elapsed/len(results):.2f}s/row)")

    # Sort results by original row order (asyncio.as_completed returns in completion order)
    id_to_result = {r["ID"]: r for r in results}

    # ── Log to JSONL ──────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    mode = "a" if os.path.exists(RAW_LOG_PATH) else "w"
    with open(RAW_LOG_PATH, mode, encoding="utf-8") as f:
        for _, row in run_df.iterrows():
            result = id_to_result.get(row["ID"], {})
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n  Raw log written to: {RAW_LOG_PATH}")

    # ── Parse stats ───────────────────────────────────────────────────────
    total = len(results)
    successes = sum(1 for r in results if r.get("parse_success"))
    print(f"  Parse success rate: {successes}/{total} ({100*successes/total:.1f}%)")

    label_dist = {}
    for r in results:
        label_dist[r["parsed_stance"]] = label_dist.get(r["parsed_stance"], 0) + 1
    print(f"  Label distribution: {label_dist}")

    # ── Write predictions (full run only) ─────────────────────────────────
    if not is_subset:
        ordered_preds = [id_to_result[row["ID"]]["parsed_stance"] for _, row in run_df.iterrows()]
        write_pred_csv(run_df, ordered_preds, f"qwen25_72b_{args.framing}", PRED_CSV_PATH)
        write_pred_txt(ordered_preds, PRED_TXT_PATH)
        print(f"\n  Predictions written:")
        print(f"    {PRED_CSV_PATH}")
        print(f"    {PRED_TXT_PATH}")

        # ── Evaluate ──────────────────────────────────────────────────────
        print("\n  Running official evaluation ...")
        try:
            official = run_official_eval(
                os.path.join(DATA_DIR, "dev.csv"),
                PRED_TXT_PATH,
            )
            print("  Official scores:")
            for k, v in sorted(official.items()):
                print(f"    {k}: {v:.6f}")
        except Exception as e:
            print(f"  ⚠️  Official eval failed: {e}")

        # ── results_summary.csv ───────────────────────────────────────────
        run_df["predicted_stance"] = ordered_preds
        local_metrics = per_topic_and_overall_metrics(
            run_df, true_col="stance", pred_col="predicted_stance"
        )
        _append_results_row(f"qwen25_72b_{args.framing}", local_metrics)
    else:
        print(f"\n  (Subset run — predictions not written. Run without --subset for full output.)")
        print(f"  Review the parse success rate and label distribution above.")
        print(f"  If framing looks good, commit to it and run without --subset.")


def _append_results_row(model_name: str, metrics: dict) -> None:
    """Append row to results_summary.csv."""
    fieldnames = [
        "model_name",
        "favg2_covid", "favg2_digital", "favg2_women", "favg2_overall",
        "favg3_covid", "favg3_digital", "favg3_women", "favg3_overall",
    ]
    row = {"model_name": model_name}
    target_key_map = {
        "Covid Vaccine":          ("favg2_covid",   "favg3_covid"),
        "Digital Transformation": ("favg2_digital", "favg3_digital"),
        "Women empowerment":      ("favg2_women",   "favg3_women"),
        "Overall":                ("favg2_overall", "favg3_overall"),
    }
    for target, (f2_key, f3_key) in target_key_map.items():
        if target in metrics:
            row[f2_key] = f"{metrics[target]['Favg2']:.6f}"
            row[f3_key] = f"{metrics[target]['Favg3']:.6f}"
        else:
            row[f2_key] = row[f3_key] = ""

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended results to {RESULTS_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot stance detection with Qwen 2.5 72B via HF router."
    )
    parser.add_argument(
        "--framing",
        choices=["en", "ar"],
        default="en",
        help="Prompt language framing (default: en). A/B test on --subset 50 first.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run on N rows only (for A/B testing prompt framings). Omit for full run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent API requests (default 8, spec recommends 5-10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

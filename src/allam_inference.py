"""
Runs Gemma's CI v3 prompt against ALLaM-7B-Instruct-preview,
served locally via llama.cpp.

PREREQUISITE - start the local server first, in a separate terminal:
    llama-server --hf-repo Omartificial-Intelligence-Space/ALLaM-7B-Instruct-preview-Q4_K_M-GGUF \
      --hf-file allam-7b-instruct-preview-q4_k_m.gguf \
      -c 2048 -ngl 20 --port 8080

Then run this script:
    python src/allam_inference.py
"""

import csv
import io
import json
import os
import re
import sys
import time
from openai import OpenAI

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import write_pred_txt

INPUT_CSV = os.path.join(_ROOT, "data", "ground_truth.csv")
OUTPUT_CSV = os.path.join(_ROOT, "predictions", "allam_test_results.csv")
RAW_LOG_PATH = os.path.join(_ROOT, "predictions", "allam_test_raw.jsonl")

# llama.cpp server exposes an OpenAI-compatible endpoint locally
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
TEMPERATURE = 0.1

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

VALID_LABELS = {"Favor", "Against", "None"}

RESULTS_FIELDNAMES = [
    "row_index",
    "target",
    "gold_label",
    "predicted_label",
    "model_id",
    "parse_success",
    "raw_response"
]


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


def classify_row(row_index: int, tweet_text: str, target: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="allam-7b-instruct-preview",
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Tweet: {tweet_text}\nTarget: {target}"},
            ],
            max_tokens=10
        )
        raw = resp.choices[0].message.content or ""
        predicted, parse_ok = parse_response(raw)
        model_returned = resp.model
    except Exception as e:
        print(f"  ❌ Error on row {row_index}: {e}")
        predicted, parse_ok, raw, model_returned = "None", False, f"ERROR: {e}", "allam-7b-instruct-preview"

    return {
        "row_index": row_index,
        "target": target,
        "gold_label": "",
        "predicted_label": predicted,
        "model_id": model_returned,
        "parse_success": parse_ok,
        "raw_response": raw
    }


def load_completed_indices(results_path: str) -> set[int]:
    if not os.path.exists(results_path):
        return set()
    completed = set()
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


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input CSV not found: {INPUT_CSV}")
        sys.exit(1)

    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} tweets. Running CI v3 prompt against ALLaM-7B...")

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(RAW_LOG_PATH) or ".", exist_ok=True)

    completed = load_completed_indices(OUTPUT_CSV)
    if completed:
        print(f"📂 Resuming: {len(completed)} rows already completed.")
    else:
        print("🆕 Starting fresh run.")

    t0 = time.time()
    for i, row in enumerate(rows):
        row_index = i
        if row_index in completed:
            continue

        text_val = row.get("tweet_text") or row.get("text", "")
        target_val = row.get("target", "Women Driving")
        pred_dict = classify_row(row_index, text_val, target_val)
        append_result_row(OUTPUT_CSV, pred_dict)
        completed.add(row_index)

        elapsed = time.time() - t0
        rate = len(completed) / elapsed if elapsed > 0 else 0.1
        eta = (len(rows) - len(completed)) / rate
        print(f"  [{len(completed)}/{len(rows)}] row_index={row_index} | pred={pred_dict['predicted_label']!r} | rate={rate:.2f} tweets/sec | ETA={eta:.0f}s")

    print(f"\nWrote predictions to {OUTPUT_CSV}")
    final_rows = []
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            final_rows.append(r)

    y_pred = [r["predicted_label"] for r in sorted(final_rows, key=lambda x: int(x["row_index"]))]
    txt_path = OUTPUT_CSV.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"Saved test predictions to {txt_path}")


if __name__ == "__main__":
    main()

"""
Runs best_86_61 (Gemma's winning prompt) against ALLaM-7B-Instruct-preview,
served locally via llama.cpp on your machine.

PREREQUISITE - start the local server first, in a separate terminal:
    llama-server --hf-repo Omartificial-Intelligence-Space/ALLaM-7B-Instruct-preview-Q4_K_M-GGUF \
      --hf-file allam-7b-instruct-preview-q4_k_m.gguf \
      -c 2048 -ngl 20 --port 8080

Then run this script:
    python run_allam_test.py
"""

import csv
import io
import json
import os
import re
import sys
import time
from openai import OpenAI

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
from data_utils import write_pred_txt

INPUT_CSV = os.path.join(_HERE, "data", "ground_truth.csv")
OUTPUT_CSV = os.path.join(_HERE, "allam_test_results.csv")
RAW_LOG_PATH = os.path.join(_HERE, "qwen_raw_logs", "allam_test_raw.jsonl")


# llama.cpp server exposes an OpenAI-compatible endpoint locally
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

TEMPERATURE = 0.1  # match best_86_61's setting for now - single variable = model only

# best_86_61, verbatim, unchanged
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


def classify_row(row_index: int, tweet_text: str, target: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="allam-7b-instruct-preview",  # name is arbitrary for local llama.cpp server
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
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} tweets. Running best_86_61 prompt (unchanged) against ALLaM-7B...")

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(RAW_LOG_PATH) or ".", exist_ok=True)

    completed = load_completed_indices(OUTPUT_CSV)
    if completed:
        print(f"📂 Resuming: {len(completed)} rows already completed.")
    else:
        print("🆕 Starting fresh run.")

    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        row_index = i
        if row_index in completed:
            continue

        pred_dict = classify_row(row_index, row["tweet_text"], row["target"])
        append_result_row(OUTPUT_CSV, pred_dict)
        completed.add(row_index)
        
        # Log raw outputs for backup
        log_entry = {
            "row_index": row_index,
            "target": row["target"],
            "gold_label": "",
            "raw_response": pred_dict["raw_response"],
            "predicted_label": pred_dict["predicted_label"],
            "model_id": pred_dict["model_id"],
            "parse_success": pred_dict["parse_success"],
            "timestamp": time.time(),
        }
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Live prints matching our Gemma script style
        elapsed = time.time() - t0
        rate = len(completed) / elapsed if elapsed > 0 else 0.1
        eta = (len(rows) - len(completed)) / rate
        print(f"  [{len(completed)}/{len(rows)}] row_index={row_index} | pred={pred_dict['predicted_label']!r} | rate={rate:.2f} tweets/sec | ETA={eta:.0f}s")

    print(f"\nWrote predictions to {OUTPUT_CSV} in {time.time()-t0:.0f}s total")

    # Load all results to output distribution summary
    final_rows = []
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            final_rows.append(r)

    counts = {"Favor": 0, "Against": 0, "None": 0, "UNPARSED": 0}
    for r in final_rows:
        pred_val = r["predicted_label"]
        parse_ok = r["parse_success"] == "True"
        if parse_ok and pred_val in counts:
            counts[pred_val] += 1
        else:
            counts["UNPARSED"] += 1
            
    total = len(final_rows)
    print("\n=== Predicted distribution vs. known true distribution ===")
    for label, true_pct in [("Favor", 44.89), ("Against", 45.45), ("None", 9.66)]:
        pred_pct = counts.get(label, 0) / total * 100 if total > 0 else 0
        print(f"{label:8s}  predicted {pred_pct:5.2f}%   true {true_pct:5.2f}%   delta {pred_pct - true_pct:+.2f}")
    if counts.get("UNPARSED", 0):
        print(f"UNPARSED  {counts['UNPARSED']} rows - inspect before drawing conclusions")

    # Write predictions to TXT matching CodaBench requirement
    y_pred = [r["predicted_label"] for r in sorted(final_rows, key=lambda x: int(x["row_index"]))]
    txt_path = OUTPUT_CSV.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"Saved test predictions to {txt_path}")


if __name__ == "__main__":
    main()

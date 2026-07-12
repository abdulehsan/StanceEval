"""
evaluate.py — Run the official eval script on prediction files, populate results_summary.csv.

Usage:
    # Evaluate a single prediction file
    python src/evaluate.py --pred_txt predictions/arabertv02_twitter_base_dev_preds.txt

    # Evaluate all prediction .txt files in predictions/
    python src/evaluate.py --all

    # Print current results_summary.csv as a table
    python src/evaluate.py --show_results

The official script (Evaluation Script/evaluate.py) is called via subprocess.
It expects:
    input_dir/ref/gold.csv   (dev.csv)
    input_dir/res/preds.txt  (one label per line)
And writes:
    output_dir/scores.txt    (key=value format)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import run_official_eval

_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV_CSV = os.path.join(_ROOT, "data", "dev.csv")
PRED_DIR = os.path.join(_ROOT, "predictions")
RESULTS_CSV = os.path.join(_ROOT, "results_summary.csv")

# Maps safe_name keys from evaluate.py back to human-readable targets
_SAFE_TO_HUMAN = {
    "Covid_Vaccine":          "Covid Vaccine",
    "Digital_Transformation": "Digital Transformation",
    "Women_empowerment":      "Women empowerment",
    "Overall":                "Overall",
}


def scores_to_row(model_name: str, scores: dict) -> dict:
    """Convert official score dict to results_summary.csv row."""
    row = {"model_name": model_name}
    mapping = {
        "Covid_Vaccine_Favg2":          "favg2_covid",
        "Digital_Transformation_Favg2": "favg2_digital",
        "Women_empowerment_Favg2":      "favg2_women",
        "Overall_Favg2":                "favg2_overall",
        "Covid_Vaccine_Favg3":          "favg3_covid",
        "Digital_Transformation_Favg3": "favg3_digital",
        "Women_empowerment_Favg3":      "favg3_women",
        "Overall_Favg3":                "favg3_overall",
    }
    for score_key, csv_col in mapping.items():
        row[csv_col] = f"{scores.get(score_key, float('nan')):.6f}"
    return row


def write_results_row(row: dict) -> None:
    fieldnames = [
        "model_name",
        "favg2_covid", "favg2_digital", "favg2_women", "favg2_overall",
        "favg3_covid", "favg3_digital", "favg3_women", "favg3_overall",
    ]
    # Check if model_name already exists and update, or append
    existing_rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    updated = False
    for i, r in enumerate(existing_rows):
        if r.get("model_name") == row["model_name"]:
            existing_rows[i] = row
            updated = True
            break

    if not updated:
        existing_rows.append(row)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def evaluate_one(pred_txt: str, model_name: Optional[str] = None) -> dict:
    """Run official eval on a single .txt file. Returns the scores dict."""
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(pred_txt))[0]

    print(f"\n── Evaluating: {model_name} ──")
    print(f"   Pred file: {pred_txt}")

    if not os.path.exists(pred_txt):
        print(f"   ❌ File not found: {pred_txt}")
        return {}

    scores = run_official_eval(DEV_CSV, pred_txt)
    print("   Official scores:")
    for k in sorted(scores):
        print(f"     {k}: {scores[k]:.6f}")

    row = scores_to_row(model_name, scores)
    write_results_row(row)
    print(f"   Written to results_summary.csv")
    return scores


def show_results() -> None:
    """Print results_summary.csv as a formatted table."""
    if not os.path.exists(RESULTS_CSV):
        print("No results_summary.csv found. Run --all first.")
        return

    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("results_summary.csv is empty.")
        return

    col_width = 10
    headers = ["model_name", "favg2_covid", "favg2_digital", "favg2_women",
               "favg2_overall", "favg3_overall"]
    name_width = max(len(r["model_name"]) for r in rows) + 2

    header_line = f"{'Model':<{name_width}}" + "".join(f"{h:>{col_width}}" for h in headers[1:])
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("=" * len(header_line))
    for row in rows:
        line = f"{row['model_name']:<{name_width}}"
        for h in headers[1:]:
            val = row.get(h, "")
            try:
                line += f"{float(val):>{col_width}.4f}"
            except (ValueError, TypeError):
                line += f"{'N/A':>{col_width}}"
        print(line)
    print("=" * len(header_line))


def main():
    args = parse_args()

    if args.show_results:
        show_results()
        return

    if args.all:
        txt_files = sorted(
            f for f in os.listdir(PRED_DIR) if f.endswith(".txt")
        )
        if not txt_files:
            print(f"No .txt files found in {PRED_DIR}")
            sys.exit(1)
        print(f"Found {len(txt_files)} prediction files in {PRED_DIR}:")
        for f in txt_files:
            print(f"  {f}")
        for f in txt_files:
            pred_txt = os.path.join(PRED_DIR, f)
            model_name = os.path.splitext(f)[0]
            # Strip the trailing _dev_preds suffix for cleaner model names
            model_name = model_name.replace("_dev_preds", "")
            evaluate_one(pred_txt, model_name)
        print("\n── Summary ──")
        show_results()

    elif args.pred_txt:
        model_name = args.model_name
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(args.pred_txt))[0]
            model_name = model_name.replace("_dev_preds", "")
        evaluate_one(args.pred_txt, model_name)

    else:
        print("Specify --pred_txt, --all, or --show_results.")
        sys.exit(1)


# needed at module level for Optional import
from typing import Optional  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction files using the official StanceEval eval script."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pred_txt",
        help="Path to a single prediction .txt file (one label per line).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all .txt files in the predictions/ directory.",
    )
    group.add_argument(
        "--show_results",
        action="store_true",
        help="Print the current results_summary.csv as a formatted table.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override model name for results_summary.csv (default: derived from filename).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

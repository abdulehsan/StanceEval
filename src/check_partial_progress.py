"""
check_partial_progress.py — Standalone progress checker for in-flight Gemma full dev run.
"""

from __future__ import annotations

import io
import os
import sys
import pandas as pd

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add src/ to path so we can import siblings
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from metrics import favg2, favg3, per_topic_and_overall_metrics

def main():
    print("==========================================================================")
    print("⚠️  PARTIAL / DIRECTIONAL ONLY — not the official score, small-sample")
    print("   and row-order effects apply. Do not report this number anywhere.")
    print("==========================================================================\n")

    _ROOT = os.path.join(os.path.dirname(__file__), "..")
    results_path = os.path.join(_ROOT, "predictions", "results_gemma4_31b_cerebras_zeroshot_arabic_target_temp01.csv")
    dev_path = os.path.join(_ROOT, "data", "dev.csv")

    if not os.path.exists(results_path):
        print(f"❌ No partial results file found at: {results_path}")
        print("   Wait for the gemma_inference.py --full script to start producing outputs.")
        sys.exit(0)

    try:
        # Read the partial results file (read-only, no locks)
        results_df = pd.read_csv(results_path, encoding="utf-8", keep_default_na=False)
    except Exception as e:
        print(f"❌ Error reading partial results file: {e}")
        sys.exit(1)

    if results_df.empty:
        print("📂 Results file is currently empty. Wait for first rows to write.")
        sys.exit(0)

    # Load gold dev set
    try:
        dev_df = load_data(dev_path)
    except Exception as e:
        print(f"❌ Error loading dev.csv: {e}")
        sys.exit(1)

    total_rows = len(dev_df)
    completed_rows = len(results_df)

    print(f"📈 Progress Check:")
    print(f"   Completed: {completed_rows} / {total_rows} rows ({completed_rows / total_rows * 100:.1f}%)")
    print()

    # Convert row_index to int for safe mapping
    results_df["row_index"] = results_df["row_index"].astype(int)
    dev_df.index = dev_df.index.astype(int)

    # Map target and stance directly from dev_df to handle indexing issues safely
    results_df["true_stance"] = results_df["row_index"].map(dev_df["stance"])
    results_df["predicted_stance"] = results_df["predicted_label"]

    # Overall metrics
    y_true = results_df["true_stance"].tolist()
    y_pred = results_df["predicted_stance"].tolist()

    overall_f2 = favg2(y_true, y_pred)
    overall_f3 = favg3(y_true, y_pred)

    print(f"📊 Overall Metrics So Far:")
    print(f"   Favg2 (Favor/Against): {overall_f2:.4f}")
    print(f"   Favg3 (3-class macro): {overall_f3:.4f}")
    print()

    # Topic-wise metrics
    try:
        metrics = per_topic_and_overall_metrics(
            results_df,
            true_col="true_stance",
            pred_col="predicted_stance",
            target_col="target",
        )
    except Exception as e:
        print(f"⚠️  Error computing per-target metrics: {e}")
        metrics = {}

    print(f"🎯 Target-wise Favg2 Breakdown:")
    print(f"   {'Target Topic':<28} {'Rows':<6} {'Favg2':<8} {'Status'}")
    print(f"   {'-'*70}")

    for target in ["Covid Vaccine", "Digital Transformation", "Women empowerment"]:
        if target in metrics:
            m = metrics[target]
            count = len(results_df[results_df["target"] == target])
            status = ""
            if count < 20:
                status = "⚠️  [Noisy: <20 rows]"
            else:
                status = "✅ [Meaningful]"
            print(f"   {target:<28} {count:<6} {m['Favg2']:<8.4f} {status}")
        else:
            print(f"   {target:<28} {'0':<6} {'—':<8} ⚠️  [No data]")

    print()

    # Parse failures / blank predictions detection
    # Identify row_index values where parse_success is false
    results_df["parse_success_bool"] = results_df["parse_success"].astype(str).str.lower().str.strip() == "true"
    parse_failures_df = results_df[
        (~results_df["parse_success_bool"]) | 
        (results_df["predicted_label"] == "") |
        (results_df["predicted_label"].isna())
    ]

    fail_count = len(parse_failures_df)
    print(f"🔍 Parse Failures / Malformed Responses:")
    if fail_count > 0:
        fail_indices = parse_failures_df["row_index"].tolist()
        print(f"   Count: {fail_count} failure(s)")
        print(f"   Row Indices with issues: {fail_indices}")
    else:
        print("   ✅ No parse failures or blank predictions detected so far.")

    print("\n==========================================================================")

if __name__ == "__main__":
    main()

"""
verify_metrics.py — Gate check: local Favg2/Favg3 must match the official eval script.

Run this BEFORE any AraBERT training. If it fails, fix metrics.py first.
The entire checkpoint selection for all 4 variants depends on the metric being correct.

Usage:
    python src/verify_metrics.py

What it does:
    1. Generates 3 synthetic prediction .txt files (all-Favor, all-Against, random mix)
    2. Runs local favg2/favg3 on each
    3. Runs the official evaluate.py on the same inputs
    4. Asserts all values match to 6 decimal places
    5. Prints a summary and exits 0 (pass) or 1 (fail)
"""

from __future__ import annotations

import io
import os
import random
import sys
import tempfile

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add src/ to path so we can import siblings
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data, write_pred_txt, LABEL2ID
from metrics import favg2, favg3, per_topic_and_overall_metrics, run_official_eval

# Paths
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV_CSV = os.path.join(_ROOT, "data", "dev.csv")
TOLERANCE = 1e-6  # must match to 6 decimal places


def _assert_close(local_val: float, official_val: float, name: str) -> bool:
    diff = abs(local_val - official_val)
    if diff > TOLERANCE:
        print(f"  ❌ MISMATCH {name}: local={local_val:.8f} official={official_val:.8f} diff={diff:.2e}")
        return False
    print(f"  ✅ {name}: {local_val:.6f} (diff={diff:.2e})")
    return True


def verify_one(dev_df, pred_labels: list[str], label: str, tmpdir: str) -> bool:
    """Verify local metrics match official script for one set of predictions."""
    print(f"\n-- {label} --")

    # Write pred txt
    pred_txt = os.path.join(tmpdir, f"{label}.txt")
    write_pred_txt(pred_labels, pred_txt)

    # Local metrics
    true_labels = dev_df["stance"].tolist()
    local_favg2 = favg2(true_labels, pred_labels)
    local_favg3 = favg3(true_labels, pred_labels)

    # Official metrics
    try:
        official = run_official_eval(DEV_CSV, pred_txt)
    except Exception as e:
        print(f"  ❌ Official script failed: {e}")
        return False

    ok = True
    ok &= _assert_close(local_favg2, official["Overall_Favg2"], "Overall_Favg2")
    ok &= _assert_close(local_favg3, official["Overall_Favg3"], "Overall_Favg3")

    # Per-topic spot check
    dev_df = dev_df.copy()
    dev_df["predicted_stance"] = pred_labels
    local_per_topic = per_topic_and_overall_metrics(
        dev_df, true_col="stance", pred_col="predicted_stance"
    )

    for target, safe_key_map in [
        ("Covid Vaccine",          "Covid_Vaccine"),
        ("Digital Transformation", "Digital_Transformation"),
        ("Women empowerment",      "Women_empowerment"),
    ]:
        if target in local_per_topic:
            for metric in ["Favg2", "Favg3"]:
                official_key = f"{safe_key_map}_{metric}"
                if official_key in official:
                    ok &= _assert_close(
                        local_per_topic[target][metric],
                        official[official_key],
                        f"{target} {metric}"
                    )
    return ok


def main() -> int:
    if not os.path.exists(DEV_CSV):
        print(f"ERROR: dev.csv not found at {DEV_CSV}")
        print("Run from the repo root or ensure data/dev.csv exists.")
        return 1

    print("Loading dev.csv ...")
    dev_df = load_data(DEV_CSV)
    n = len(dev_df)
    print(f"  Loaded {n} rows.")

    labels = list(LABEL2ID.keys())  # ["Against", "Favor", "None"]
    random.seed(42)

    all_passed = True
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: All predictions = "Favor"
        all_passed &= verify_one(dev_df, ["Favor"] * n, "all_favor", tmpdir)

        # Test 2: All predictions = "Against"
        all_passed &= verify_one(dev_df, ["Against"] * n, "all_against", tmpdir)

        # Test 3: All predictions = "None"
        all_passed &= verify_one(dev_df, ["None"] * n, "all_none", tmpdir)

        # Test 4: Random predictions (reproducible via seed=42)
        random_preds = [random.choice(labels) for _ in range(n)]
        all_passed &= verify_one(dev_df, random_preds, "random_preds", tmpdir)

        # Test 5: Perfect predictions (local should be 1.0, official should be 1.0)
        perfect_preds = dev_df["stance"].tolist()
        all_passed &= verify_one(dev_df, perfect_preds, "perfect_preds", tmpdir)

    print()
    if all_passed:
        print("=" * 50)
        print("✅ All metric verification checks PASSED.")
        print("   Local metrics match official evaluate.py on all test cases.")
        print("   Safe to proceed with AraBERT training.")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("❌ METRIC VERIFICATION FAILED.")
        print("   Fix metrics.py before proceeding with any training.")
        print("   Checkpoint selection depends on this being correct.")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())

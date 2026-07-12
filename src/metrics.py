"""
metrics.py — Local implementation of Favg2/Favg3 metrics for StanceEval-2026.

These exactly replicate the official evaluate.py formula so they can be used
for early stopping during AraBERT training without shelling out to the script
on every epoch.

IMPORTANT: Before using for any training decision, run verify_metrics.py to
confirm local and official numbers agree to 6 decimal places on a test file.

Official formula (from evaluate.py):
    f_against = f1_score(y_true, y_pred, labels=[0], average="macro")
    f_favor   = f1_score(y_true, y_pred, labels=[1], average="macro")
    f_none    = f1_score(y_true, y_pred, labels=[2], average="macro")
    Favg2 = (f_favor + f_against) / 2.0
    Favg3 = (f_favor + f_against + f_none) / 3.0

Note: sklearn's f1_score with a single label in `labels` and average="macro"
returns the F1 for that class only (same as average="binary" for multi-class
when a single label is specified).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

# Must match official evaluate.py
LABEL2ID: dict[str, int] = {"Against": 0, "Favor": 1, "None": 2}
VALID_LABELS: list[str] = ["Against", "Favor", "None"]

# Path to the official eval script (relative to repo root)
_OFFICIAL_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "Evaluation Script", "evaluate.py"
)


# ── Core metric functions ─────────────────────────────────────────────────────

def _labels_to_ids(labels: list[str]) -> list[int]:
    return [LABEL2ID[x] for x in labels]


def favg2(y_true: list[str], y_pred: list[str]) -> float:
    """Compute Favg2 = (F1_Favor + F1_Against) / 2 over all samples.

    Matches the global computation in the official evaluate.py.
    """
    t = _labels_to_ids(y_true)
    p = _labels_to_ids(y_pred)
    f_against = f1_score(t, p, labels=[0], average="macro")
    f_favor   = f1_score(t, p, labels=[1], average="macro")
    return (f_favor + f_against) / 2.0


def favg3(y_true: list[str], y_pred: list[str]) -> float:
    """Compute Favg3 = (F1_Favor + F1_Against + F1_None) / 3 over all samples."""
    t = _labels_to_ids(y_true)
    p = _labels_to_ids(y_pred)
    f_against = f1_score(t, p, labels=[0], average="macro")
    f_favor   = f1_score(t, p, labels=[1], average="macro")
    f_none    = f1_score(t, p, labels=[2], average="macro")
    return (f_favor + f_against + f_none) / 3.0


def per_topic_and_overall_metrics(
    df: pd.DataFrame,
    true_col: str = "true_stance",
    pred_col: str = "predicted_stance",
    target_col: str = "target",
) -> dict:
    """Compute per-topic and overall Favg2/Favg3.

    Replicates the full output of the official evaluate.py.

    Returns a dict like:
    {
        "Covid Vaccine":          {"Favg2": 0.xx, "Favg3": 0.xx},
        "Digital Transformation": {"Favg2": 0.xx, "Favg3": 0.xx},
        "Women empowerment":      {"Favg2": 0.xx, "Favg3": 0.xx},
        "Overall":                {"Favg2": 0.xx, "Favg3": 0.xx},  # global, not averaged
    }
    """
    results = {}
    for target in sorted(df[target_col].unique()):
        sub = df[df[target_col] == target]
        t = sub[true_col].tolist()
        p = sub[pred_col].tolist()
        results[target] = {
            "Favg2": favg2(t, p),
            "Favg3": favg3(t, p),
        }

    # Global (all rows together, NOT per-topic average) — matches official script
    results["Overall"] = {
        "Favg2": favg2(df[true_col].tolist(), df[pred_col].tolist()),
        "Favg3": favg3(df[true_col].tolist(), df[pred_col].tolist()),
    }
    return results


def format_metrics_table(results: dict) -> str:
    """Pretty-print a metrics results dict as a table."""
    lines = [f"{'Target':<28} {'Favg2':>8} {'Favg3':>8}"]
    lines.append("-" * 48)
    for key in sorted(results.keys()):
        if key == "Overall":
            continue
        m = results[key]
        lines.append(f"{key:<28} {m['Favg2']:>8.4f} {m['Favg3']:>8.4f}")
    lines.append("-" * 48)
    m = results.get("Overall", {})
    lines.append(f"{'Overall':<28} {m.get('Favg2', float('nan')):>8.4f} {m.get('Favg3', float('nan')):>8.4f}")
    return "\n".join(lines)


# ── Official eval script integration ─────────────────────────────────────────

def run_official_eval(
    gold_csv_path: str,
    pred_txt_path: str,
    python_exe: Optional[str] = None,
) -> dict:
    """Shell out to the official evaluate.py and return parsed scores.

    Creates a temporary directory structure that matches the official script's
    expected layout:
        input_dir/
            ref/  ← gold CSV
            res/  ← prediction TXT

    Args:
        gold_csv_path: Path to dev.csv (or any gold CSV with target+stance cols).
        pred_txt_path: Path to the prediction .txt file (one label per line).
        python_exe: Python interpreter to use. Defaults to sys.executable.

    Returns:
        Dict with keys like 'Covid_Vaccine_Favg2', 'Overall_Favg2', etc.
        (uses safe_name format from the official script)

    Raises:
        FileNotFoundError: if the official script is not found.
        RuntimeError: if the script exits with a non-zero code.
    """
    if python_exe is None:
        python_exe = sys.executable

    script_path = os.path.abspath(_OFFICIAL_SCRIPT)
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Official eval script not found at: {script_path}\n"
            "Download it from: https://github.com/StanceEval/stanceeval.github.io/tree/main/Evaluation%20Script"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build the directory layout the official script expects
        ref_dir = os.path.join(tmpdir, "input", "ref")
        res_dir = os.path.join(tmpdir, "input", "res")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(ref_dir)
        os.makedirs(res_dir)
        os.makedirs(out_dir)

        # Copy gold CSV to ref/
        shutil.copy(gold_csv_path, os.path.join(ref_dir, "gold.csv"))

        # Copy prediction TXT to res/
        shutil.copy(pred_txt_path, os.path.join(res_dir, "predictions.txt"))

        input_dir = os.path.join(tmpdir, "input")
        result = subprocess.run(
            [python_exe, script_path, input_dir, out_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Official eval script failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        scores_path = os.path.join(out_dir, "scores.txt")
        if not os.path.exists(scores_path):
            raise RuntimeError(f"Official script ran but scores.txt not found in {out_dir}")

        with open(scores_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    scores = {}
    for line in lines:
        line = line.strip()
        if "=" in line:
            key, val = line.split("=", 1)
            scores[key.strip()] = float(val.strip())

    return scores


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("metrics.py loaded OK. Run verify_metrics.py to validate against official script.")

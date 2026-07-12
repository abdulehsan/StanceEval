"""
data_utils.py — Shared data loading and preprocessing utilities for StanceEval-2026.

Handles:
  - Target canonicalization (raw CSV strings → canonical strings matching dev.csv exactly)
  - Data loading with validation
  - Stratified internal train/val split (90/10) for early stopping
  - Class weight computation for weighted cross-entropy
  - Label <-> ID mappings (matching the official eval script's LABEL2ID)

Target strings confirmed from actual train.csv and dev.csv:
  'Women empowerment', 'Covid Vaccine', 'Digital Transformation'

These are the canonical strings — the TARGET_MAP maps any messy variants to these.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Label mappings ──────────────────────────────────────────────────────────
# Must match the official evaluate.py exactly: Against=0, Favor=1, None=2
LABEL2ID: dict[str, int] = {"Against": 0, "Favor": 1, "None": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS: int = 3

# ── Canonical target strings ─────────────────────────────────────────────────
# Confirmed by inspecting train.csv and dev.csv with pandas.
# The official eval script groups by gold['target'] verbatim (post .strip()),
# so these strings must appear in prediction files' corresponding gold CSVs.
CANONICAL_TARGETS: list[str] = [
    "Women empowerment",
    "Covid Vaccine",
    "Digital Transformation",
]

# Lookup: lowercase-stripped input → canonical string
# Covers all casing variants seen in the wild.
_TARGET_LOOKUP: dict[str, str] = {t.lower().strip(): t for t in CANONICAL_TARGETS}

# Additional explicit aliases for safety (add more if new variants are found)
_TARGET_ALIASES: dict[str, str] = {
    "covid-19 vaccine": "Covid Vaccine",
    "covid19 vaccine":  "Covid Vaccine",
    "covid_vaccine":    "Covid Vaccine",
    "women_empowerment": "Women empowerment",
    "digital_transformation": "Digital Transformation",
}


def canonicalize_target(raw: str) -> str:
    """Map a raw target string to its canonical form.

    Raises ValueError if the string cannot be mapped.
    """
    key = raw.lower().strip()
    if key in _TARGET_LOOKUP:
        return _TARGET_LOOKUP[key]
    if key in _TARGET_ALIASES:
        return _TARGET_ALIASES[key]
    raise ValueError(
        f"Unknown target string: {raw!r}. "
        f"Expected one of {CANONICAL_TARGETS}. "
        f"Update TARGET_MAP in data_utils.py if this is a new valid variant."
    )


def load_data(path: str, canonicalize: bool = True) -> pd.DataFrame:
    """Load a StanceEval CSV and return a clean DataFrame.

    Args:
        path: Path to train.csv or dev.csv.
        canonicalize: If True, map target column to canonical strings.

    Returns:
        DataFrame with at minimum columns: ID, text, target, stance.
        If the file has no 'stance' column (e.g. blind test set), that column
        will be absent but the rest will be loaded normally.

    Raises:
        FileNotFoundError: if path doesn't exist.
        ValueError: if required columns are missing or stance labels are invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, keep_default_na=False)
    df.columns = df.columns.astype(str).str.strip()

    required = ["ID", "text", "target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # Canonicalize target strings
    if canonicalize:
        df["target"] = df["target"].apply(canonicalize_target)

    # Validate stance labels if the column exists
    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str).str.strip()
        valid = set(LABEL2ID.keys())
        invalid = sorted(set(df["stance"]) - valid)
        if invalid:
            raise ValueError(
                f"Invalid stance labels in {path}: {invalid}. "
                f"Expected one of {sorted(valid)}"
            )

    return df


def make_internal_split(
    df: pd.DataFrame,
    test_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve an internal val split out of training data for early stopping.

    Stratified by (stance, target) combo so the imbalanced None class and
    the skewed Digital Transformation target aren't further skewed by the split.

    Args:
        df: Full training DataFrame (from load_data on train.csv).
        test_size: Fraction to use as internal val (default 0.10 → ~350 rows).
        random_state: Seed for reproducibility.

    Returns:
        (train_df, val_df) — val_df is for early stopping only, never reported.

    Note:
        dev.csv is NEVER passed here. It remains untouched until final evaluation.
    """
    if "stance" not in df.columns or "target" not in df.columns:
        raise ValueError("DataFrame must have 'stance' and 'target' columns for stratification.")

    # Combine stance+target into a single stratify key
    stratify_col = df["stance"] + "|||" + df["target"]

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def compute_class_weights(df: pd.DataFrame) -> "list[float]":
    """Compute inverse-frequency class weights for weighted cross-entropy.

    Returns a list of floats indexed by LABEL2ID (Against=0, Favor=1, None=2).
    Weights are normalized so the minimum weight is 1.0.

    Args:
        df: Training DataFrame with 'stance' column.

    Returns:
        List of 3 floats: [weight_Against, weight_Favor, weight_None]
    """
    counts = df["stance"].value_counts()
    total = len(df)

    weights = []
    for label in ["Against", "Favor", "None"]:
        n = counts.get(label, 1)
        weights.append(total / (NUM_LABELS * n))

    # Normalize so min weight = 1.0
    min_w = min(weights)
    weights = [w / min_w for w in weights]
    return weights


# ── Prediction file helpers ───────────────────────────────────────────────────

def write_pred_csv(
    df: pd.DataFrame,
    predicted_stances: list[str],
    model_name: str,
    output_path: str,
) -> None:
    """Write a prediction CSV for error analysis (not for the official eval script).

    Schema: ID, text, target, true_stance, predicted_stance, model_name

    Args:
        df: The dev DataFrame (from load_data on dev.csv).
        predicted_stances: List of predicted labels, same order as df.
        model_name: Identifier string for the model (e.g. 'arabertv02_twitter_base').
        output_path: Where to write the CSV.
    """
    assert len(predicted_stances) == len(df), (
        f"Length mismatch: {len(predicted_stances)} predictions vs {len(df)} rows."
    )
    out = df[["ID", "text", "target"]].copy()
    out["true_stance"] = df["stance"].values if "stance" in df.columns else ""
    out["predicted_stance"] = predicted_stances
    out["model_name"] = model_name
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")


def write_pred_txt(predicted_stances: list[str], output_path: str) -> None:
    """Write a prediction .txt file for the official eval script.

    Format: one label per line (Favor/Against/None), no header.
    The official script optionally strips a 'stance' header line, but we
    omit the header to keep it clean.

    Args:
        predicted_stances: List of predicted labels in the same row order
            as the gold dev CSV (i.e., same order as df from load_data).
        output_path: Where to write the .txt file.
    """
    valid = set(LABEL2ID.keys())
    invalid = sorted(set(predicted_stances) - valid)
    if invalid:
        raise ValueError(f"Invalid prediction labels before writing txt: {invalid}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for label in predicted_stances:
            f.write(label + "\n")


# ── __main__ sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(data_dir, "train.csv")
    dev_path = os.path.join(data_dir, "dev.csv")

    print("Loading train.csv ...")
    train = load_data(train_path)
    print(f"  Train shape: {train.shape}")
    print(f"  Target unique: {sorted(train['target'].unique())}")
    print(f"  Stance counts:\n{train['stance'].value_counts().to_string()}")

    print("\nMaking internal split ...")
    tr, val = make_internal_split(train)
    print(f"  Train split: {len(tr)} rows, Val split: {len(val)} rows")
    print(f"  Val stance distribution:\n{val['stance'].value_counts().to_string()}")

    print("\nClass weights (global):")
    weights = compute_class_weights(train)
    for i, label in enumerate(["Against", "Favor", "None"]):
        print(f"  {label} (id={i}): {weights[i]:.4f}")

    print("\nLoading dev.csv ...")
    dev = load_data(dev_path)
    print(f"  Dev shape: {dev.shape}")

    print("\nAll checks passed.")
    sys.exit(0)

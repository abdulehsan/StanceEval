"""
arabert_finetune.py — Fine-tune AraBERT variants for Arabic stance detection.

Supports all 4 variants from the spec:
  - arabertv02_base           (aubmindlab/bert-base-arabertv02)           No Farasa
  - arabertv2_base            (aubmindlab/bert-base-arabertv2)            Needs Farasa + Java
  - arabertv02_twitter_base   (aubmindlab/bert-base-arabertv02-twitter)   No Farasa
  - arabertv02_twitter_large  (aubmindlab/bert-large-arabertv02-twitter)  No Farasa

Run order per spec: arabertv02_twitter_base first → arabertv02_base → arabertv2_base → arabertv02_twitter_large

Usage:
    python src/arabert_finetune.py \\
        --model_key arabertv02_twitter_base \\
        --epochs 5 \\
        --batch_size 16 \\
        --lr 2e-5

Key design decisions (matching the spec):
  - Internal 90/10 stratified split from train.csv for early stopping
  - dev.csv is NEVER touched during training — only after selecting final checkpoint
  - Class-weighted cross-entropy (inverse-frequency weights, global across all targets)
  - Early stopping on internal-val Favg2 (not dev Favg2)
  - After training: runs inference on dev.csv once, writes:
      predictions/{model_key}_dev_preds.csv   (for error analysis)
      predictions/{model_key}_dev_preds.txt   (for official eval script)
  - Appends row to results_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import warnings
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add src/ to path
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    compute_class_weights,
    load_data,
    make_internal_split,
    write_pred_csv,
    write_pred_txt,
)
from metrics import favg2, favg3, per_topic_and_overall_metrics, run_official_eval

warnings.filterwarnings("ignore", category=UserWarning)

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, str] = {
    "arabertv02_base":          "aubmindlab/bert-base-arabertv02",
    "arabertv2_base":           "aubmindlab/bert-base-arabertv2",
    "arabertv02_twitter_base":  "aubmindlab/bert-base-arabertv02-twitter",
    "arabertv02_twitter_large": "aubmindlab/bert-large-arabertv02-twitter",
}

# Models that require Farasa segmentation (needs Java runtime)
FARASA_REQUIRED: set[str] = {"arabertv2_base"}

# Paths (relative to repo root)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
CKPT_DIR = os.path.join(_ROOT, "checkpoints")
RESULTS_CSV = os.path.join(_ROOT, "results_summary.csv")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def check_java_available() -> bool:
    """Return True if `java` is on PATH and executable."""
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def preflight_checks(model_key: str) -> None:
    """Abort with a clear message if prerequisites for model_key are missing."""
    if model_key in FARASA_REQUIRED:
        if not check_java_available():
            print(
                f"\n❌ MODEL REQUIRES JAVA: '{model_key}' uses arabertv2 which needs "
                f"Farasa segmentation via farasapy, which requires a Java runtime.\n"
                f"   Java was not found on PATH.\n"
                f"   Install Java (JDK 11+ recommended) and ensure 'java' is on PATH.\n"
                f"   Then re-run: pip install farasapy\n"
                f"   Only after both are available should you run this model.\n"
                f"\n   Consider running arabertv02_twitter_base first (no Java needed)."
            )
            sys.exit(1)
        try:
            import farasapy  # noqa: F401
        except ImportError:
            print(
                f"\n❌ MISSING DEPENDENCY: '{model_key}' requires farasapy.\n"
                f"   pip install farasapy\n"
                f"   (Also ensure Java is installed — farasapy requires a JVM.)"
            )
            sys.exit(1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StanceDataset(torch.utils.data.Dataset):
    """PyTorch dataset for sequence-pair stance classification."""

    def __init__(self, encodings: dict, labels: Optional[list[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataset(df, tokenizer, preprocessor, max_length: int = 128) -> StanceDataset:
    """Preprocess text and tokenize as sequence pair (target, tweet).

    Args:
        df: DataFrame with 'target', 'text', and optionally 'stance'.
        tokenizer: HuggingFace tokenizer.
        preprocessor: ArabertPreprocessor instance.
        max_length: Max token sequence length.

    Returns:
        StanceDataset
    """
    targets = df["target"].tolist()
    texts = df["text"].tolist()

    # Apply ArabertPreprocessor to tweets (targets are short English strings,
    # preprocessing them is fine — the preprocessor handles mixed text)
    processed_texts = [preprocessor.preprocess(str(t)) for t in texts]
    processed_targets = [preprocessor.preprocess(str(t)) for t in targets]

    encodings = tokenizer(
        processed_targets,  # text_a (goes after [CLS])
        processed_texts,    # text_b (goes after [SEP])
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,  # return lists, converted in __getitem__
    )

    labels = None
    if "stance" in df.columns:
        labels = [LABEL2ID[s] for s in df["stance"].tolist()]

    return StanceDataset(encodings, labels)


# ── Weighted loss trainer ─────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: Optional[list[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float).to(device)
            )
        else:
            self.loss_fct = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_fct is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Compute metrics for Trainer ───────────────────────────────────────────────

def make_compute_metrics(val_df):
    """Return a compute_metrics function that the Trainer calls each eval step.

    Uses internal-val split rows to compute Favg2 for early stopping.
    The val_df is used to retrieve the true stances (in the same row order
    as the val dataset).
    """
    true_labels = val_df["stance"].tolist()

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        pred_labels = [ID2LABEL[i] for i in pred_ids]
        score = favg2(true_labels, pred_labels)
        score3 = favg3(true_labels, pred_labels)
        return {"favg2": score, "favg3": score3}

    return compute_metrics


# ── Results summary ───────────────────────────────────────────────────────────

def append_results_row(model_name: str, metrics: dict) -> None:
    """Append a row to results_summary.csv.

    Creates the file with a header if it doesn't exist.
    """
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


# ── Main training function ────────────────────────────────────────────────────

def train(
    model_key: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    early_stopping_patience: int = 2,
    seed: int = 42,
) -> None:
    print(f"\n{'='*60}")
    print(f"  StanceEval-2026 — AraBERT Fine-tune")
    print(f"  Model: {model_key} ({MODEL_REGISTRY[model_key]})")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"{'='*60}\n")

    # ── Pre-flight ──────────────────────────────────────────────────────────
    preflight_checks(model_key)

    hf_id = MODEL_REGISTRY[model_key]

    # ── Paths ───────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, model_key)
    pred_csv = os.path.join(PRED_DIR, f"{model_key}_dev_preds.csv")
    pred_txt = os.path.join(PRED_DIR, f"{model_key}_dev_preds.txt")
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    print("Loading train.csv and making internal split ...")
    train_df = load_data(os.path.join(DATA_DIR, "train.csv"))
    tr_df, val_df = make_internal_split(train_df, test_size=0.10, random_state=seed)
    print(f"  Train: {len(tr_df)} rows | Internal-val: {len(val_df)} rows")
    print(f"  Internal-val stance distribution:")
    print(val_df["stance"].value_counts().to_string())

    # ── Class weights (global, from full train before split) ─────────────
    class_weights = compute_class_weights(train_df)
    print(f"\n  Class weights [Against, Favor, None]: {[f'{w:.4f}' for w in class_weights]}")

    # ── Tokenizer & preprocessor ─────────────────────────────────────────
    print(f"\nLoading tokenizer: {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    print("Loading ArabertPreprocessor ...")
    from arabert.preprocess import ArabertPreprocessor
    preprocessor = ArabertPreprocessor(model_name=hf_id)

    # ── Build datasets ────────────────────────────────────────────────────
    print("Building train dataset ...")
    train_dataset = build_dataset(tr_df, tokenizer, preprocessor, max_length)
    print("Building val dataset ...")
    val_dataset = build_dataset(val_df, tokenizer, preprocessor, max_length)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model: {hf_id} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Training args ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=ckpt_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="favg2",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),  # use fp16 only on GPU
        dataloader_num_workers=0,        # Windows-safe
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(val_df),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        class_weights=class_weights,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    print("  (Early stopping on internal-val Favg2 — dev.csv not touched)")
    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\n  Best checkpoint: {best_ckpt}")
    print(f"  Best internal-val Favg2: {trainer.state.best_metric:.4f}")

    # ── Dev inference (one-shot, final) ──────────────────────────────────
    print("\nRunning inference on data/dev.csv (one-shot, final) ...")
    dev_df = load_data(os.path.join(DATA_DIR, "dev.csv"))

    # Capture row IDs in the exact order load_data returns them (original CSV order).
    # build_dataset iterates df.iterrows() in this same order — no shuffle.
    # trainer.predict() uses SequentialSampler (no shuffle) per HF Trainer contract.
    # We assert length and zip positionally, which is safe given SequentialSampler,
    # and then write in dev_df's original row order so the .txt aligns with the gold CSV.
    dev_ids_in_order = dev_df["ID"].tolist()

    dev_dataset = build_dataset(dev_df, tokenizer, preprocessor, max_length)

    predictions = trainer.predict(dev_dataset)
    pred_ids_raw = np.argmax(predictions.predictions, axis=-1)

    # Positional assertion: DataLoader must not have dropped or reordered rows.
    if len(pred_ids_raw) != len(dev_df):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(pred_ids_raw)} predictions "
            f"for {len(dev_df)} dev rows. DataLoader may have dropped rows — "
            f"check dataloader_drop_last or dataset length."
        )

    # Build ID → label map from the positional pairing (SequentialSampler guarantee).
    id_to_pred = {
        dev_ids_in_order[i]: ID2LABEL[int(pred_ids_raw[i])]
        for i in range(len(dev_ids_in_order))
    }

    # Write in dev_df's original CSV row order (the order the gold CSV uses).
    pred_labels = [id_to_pred[row_id] for row_id in dev_ids_in_order]

    # ── Write outputs ─────────────────────────────────────────────────────
    write_pred_csv(dev_df, pred_labels, model_key, pred_csv)
    write_pred_txt(pred_labels, pred_txt)          # ← labels in original dev.csv row order
    print(f"  Written: {pred_csv}")
    print(f"  Written: {pred_txt}")

    # ── Evaluate with official script ─────────────────────────────────────
    print("\nRunning official evaluation ...")
    try:
        official_scores = run_official_eval(
            gold_csv_path=os.path.join(DATA_DIR, "dev.csv"),
            pred_txt_path=pred_txt,
        )
        print("  Official scores:")
        for k, v in sorted(official_scores.items()):
            print(f"    {k}: {v:.6f}")
    except Exception as e:
        print(f"  ⚠️  Official eval failed: {e}")
        print("  Falling back to local metrics ...")
        official_scores = None

    # ── Local metrics for results_summary ────────────────────────────────
    dev_df["predicted_stance"] = pred_labels
    local_metrics = per_topic_and_overall_metrics(
        dev_df, true_col="stance", pred_col="predicted_stance"
    )
    print("\n  Local metrics:")
    for target, m in sorted(local_metrics.items()):
        print(f"    {target}: Favg2={m['Favg2']:.4f}  Favg3={m['Favg3']:.4f}")

    append_results_row(model_key, local_metrics)

    print(f"\n✅ Done: {model_key}")
    print(f"   Checkpoint saved to: {ckpt_path}")
    print(f"   Predictions: {pred_csv}, {pred_txt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an AraBERT variant for stance detection."
    )
    parser.add_argument(
        "--model_key",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which AraBERT variant to fine-tune.",
    )
    parser.add_argument("--epochs",     type=int,   default=5,   help="Max epochs (default 5)")
    parser.add_argument("--batch_size", type=int,   default=16,  help="Batch size (default 16)")
    parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate (default 2e-5)")
    parser.add_argument("--max_length", type=int,   default=128, help="Max token length (default 128)")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Early stopping patience in epochs (default 2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_key=args.model_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )

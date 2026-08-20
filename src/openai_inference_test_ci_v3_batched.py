"""
openai_inference_test_ci_v3_batched.py — Zero-shot stance detection on the blind test set (ground_truth.csv)
using gpt-5.6-luna via OpenAI API with prompt-level batching (BATCH_SIZE = 20) and the CI v3 guidelines.

Usage:
    python src/openai_inference_test_ci_v3_batched.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
import pandas as pd
from openai import OpenAI

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")

TEST_CSV_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_PATH = os.path.join(PRED_DIR, "results_luna_openai_test_ci_v3_batched.csv")
RAW_LOG_PATH = os.path.join(LOG_DIR, "results_luna_openai_test_ci_v3_batched_raw.jsonl")

MODEL_ID = "gpt-5.6-luna"
TEMPERATURE = 0.0
MAX_TOKENS = 4000
BATCH_SIZE = 20

VALID_LABELS = {"Favor", "Against", "None"}

TARGET_AR = {
    "Women Driving": "قيادة المرأة للسيارة"
}

# ── System Prompt (CI v3 adapted for JSON batching) ───────────────────────────
SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

### Task
Given a batch of Arabic tweets and a target topic, classify the writer's stance toward the target as exactly one of:
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

Output Format:
Return ONLY a valid JSON object mapping each tweet ID to its predicted stance.
Do NOT use markdown code blocks (e.g., do not wrap the output in ```json ... ```).
Do NOT include any introduction, explanation, or additional text.
Example structure:
{
  "0": "Favor",
  "1": "Against"
}"""

def build_batch_prompt(batch_df: pd.DataFrame) -> str:
    prompt = "Tweets to classify:\n\n"
    for idx, row in batch_df.iterrows():
        target = str(row["target"])
        arabic_target = TARGET_AR.get(target, target)
        prompt += f"ID: {idx}\nTarget: {arabic_target}\nTweet: {row['text']}\n\n"
    prompt += "Provide stance predictions for all the above IDs. Return only JSON object, no wrappers or explanation."
    return prompt

def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df = df.rename(columns={"id": "ID", "tweet_text": "text"})
    df.columns = df.columns.astype(str).str.strip()
    return df

def clean_response_text(text: str) -> str:
    text = text.strip()
    # Remove markdown code blocks if the model ignored instructions
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text)
    return text.strip()

def predict_batch_with_retry(
    client: OpenAI,
    batch_df: pd.DataFrame,
    max_retries: int = 5,
) -> dict[int, str]:
    prompt = build_batch_prompt(batch_df)
    expected_ids = {int(idx) for idx in batch_df.index.tolist()}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                max_completion_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_content = response.choices[0].message.content
            if raw_content is None:
                raise ValueError("Model returned empty message body.")
            
            cleaned_text = clean_response_text(raw_content)
            parsed = json.loads(cleaned_text)

            # Convert keys to int and values to cleaned stance strings
            predictions = {}
            for k, v in parsed.items():
                pred_label = str(v).strip().capitalize()
                # Simple normalization (e.g. favor -> Favor, against -> Against, none -> None)
                if pred_label in VALID_LABELS:
                    predictions[int(k)] = pred_label
                else:
                    raise ValueError(f"Invalid label returned: {pred_label}")

            returned_ids = set(predictions.keys())
            if returned_ids != expected_ids:
                raise ValueError(
                    f"ID mismatch. Expected: {expected_ids}, Returned: {returned_ids}"
                )

            # Log the raw transaction to the raw log
            log_entry = {
                "batch_ids": list(expected_ids),
                "raw_response": raw_content,
                "parsed_predictions": {str(k): v for k, v in predictions.items()},
                "timestamp": time.time(),
            }
            with open(RAW_LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            return predictions

        except Exception as e:
            wait_time = 3.0 * (attempt + 1)
            print(f"    ⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # If all retries fail, return a fallback of "None" for all IDs in the batch
    print(f"    ❌ All {max_retries} attempts failed for batch. Falling back to 'None' for these rows.")
    return {int(idx): "None" for idx in batch_df.index.tolist()}

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Load from .env
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(_ROOT, ".env"))
            api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            pass

    # Read from .env manually if needed
    if not api_key and os.path.exists(os.path.join(_ROOT, ".env")):
        try:
            with open(os.path.join(_ROOT, ".env"), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip().strip('"').strip("'")
                        if k == "OPENAI_API_KEY":
                            api_key = v
                            break
        except Exception:
            pass

    if not api_key:
        print("❌ OPENAI_API_KEY not configured. Cannot proceed.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"\nLoading test data from {TEST_CSV_PATH}...")
    df = load_test_data(TEST_CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Split into batches
    batches = []
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        batches.append(df.iloc[start:end])

    print(f"Total batches to run: {len(batches)}")
    print(f"Model ID: {MODEL_ID}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Starting inference loop...")

    all_predictions: dict[int, str] = {}

    for idx, batch_df in enumerate(batches, start=1):
        print(f"Running batch {idx}/{len(batches)} (IDs: {batch_df.index.tolist()[0]} to {batch_df.index.tolist()[-1]})")
        batch_preds = predict_batch_with_retry(client, batch_df)
        all_predictions.update(batch_preds)

    # Save results to CSV
    results_rows = []
    for idx, row in df.iterrows():
        row_id = int(idx)
        pred_label = all_predictions.get(row_id, "None")
        results_rows.append({
            "row_index": row_id,
            "target": row["target"],
            "gold_label": "",
            "predicted_label": pred_label,
            "model_id": MODEL_ID,
            "parse_success": True,
            "raw_response": ""
        })

    results_df = pd.DataFrame(results_rows)
    # Ensure ordered by row_index
    results_df = results_df.sort_values("row_index")
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"✅ Saved CSV predictions to {RESULTS_PATH}")

    # Import write_pred_txt from data_utils
    sys.path.insert(0, os.path.dirname(__file__))
    from data_utils import write_pred_txt

    y_pred = results_df["predicted_label"].tolist()
    txt_path = RESULTS_PATH.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"✅ Saved TXT predictions to {txt_path}")

    # Print distribution
    print("\nStance Distribution:")
    print(results_df["predicted_label"].value_counts().to_string())

if __name__ == "__main__":
    main()

"""
gemma_inference_dev_runs.py — Run all 4 prompt configurations on the 619-row dev set
using 4-key API rotation to avoid rate limits and errors.

Prompt configurations:
  - Revised (No context, v1 rules)
  - CI v1 (Context + v1 rules)
  - CI v3 (Context + v3 rules)
  - CI v5 (Context + v5 rules)

Usage:
    python src/gemma_inference_dev_runs.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)

import builtins
def print(*args, **kwargs):
    builtins.print(*args, **kwargs, flush=True)

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import LABEL2ID, write_pred_txt
from metrics import favg2, run_official_eval
from evaluate import scores_to_row, write_results_row

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")
LOG_DIR = os.path.join(_ROOT, "qwen_raw_logs")

DEV_CSV_PATH = os.path.join(DATA_DIR, "dev.csv")

# ── Model Config ──────────────────────────────────────────────────────────────
MODEL_ID = "gemma-4-31b"
TEMPERATURE = 0.1  # Greedy decoding for consistent classification
TOP_P = 0.95
MAX_TOKENS = 10

VALID_LABELS: set[str] = set(LABEL2ID.keys())

# Target translations
TARGET_AR = {
    "Covid Vaccine": "لقاح كورونا",
    "Digital Transformation": "التحول الرقمي",
    "Women empowerment": "تمكين المرأة"
}

RESULTS_FIELDNAMES = [
    "row_index",
    "target",
    "gold_label",
    "predicted_label",
    "model_id",
    "parse_success",
    "raw_response",
]

# ── API Key Setup ─────────────────────────────────────────────────────────────
load_dotenv(os.path.join(_ROOT, ".env"))

api_keys = [
    os.environ.get("CEREBRAS_API_KEY", "").strip(),
    os.environ.get("API_KEY", "").strip(),
    os.environ.get("api2", "").strip(),
    os.environ.get("api3", "").strip(),
    os.environ.get("api_key_24", "").strip(),
    os.environ.get("api_key_200", "").strip()
]
api_keys = [k for k in api_keys if k]

if not api_keys:
    print("❌ No Cerebras API keys found in .env. Cannot proceed.")
    sys.exit(1)

print(f"✅ Loaded {len(api_keys)} API keys for rotation:")
for idx, key in enumerate(api_keys):
    print(f"   Key {idx + 1}: {key[:6]}...{key[-6:]}")

from cerebras.cloud.sdk import Cerebras
clients = [Cerebras(api_key=key, timeout=10.0) for key in api_keys]

# ── Prompt Builder ────────────────────────────────────────────────────────────
def get_system_prompt(target: str, version: str) -> str:
    background_contexts = {
        "Covid Vaccine": (
            "The target concerns the COVID-19 vaccine rollout and vaccination policies in Saudi Arabia. "
            "Tweets discuss vaccine safety, side effects, mandatory vaccination, registration via the Sehhaty app, "
            "vaccine centers, and public health campaigns. Emojis, sarcasm, and Saudi dialect are common."
        ),
        "Digital Transformation": (
            "The target concerns Saudi Arabia's national digital transformation and electronic government services "
            "(such as Absher, Tawakkalna, and paperless systems). Tweets discuss administrative efficiency, ease of registration, "
            "system outages, service complaints, and the transition away from physical office visits."
        ),
        "Women empowerment": (
            "The target concerns Saudi social reforms and policies promoting women's empowerment, gender equality, "
            "and female labor participation. Tweets discuss job opportunities, leadership roles, travel rights, "
            "societal reception, and changes in traditional gender norms."
        )
    }
    
    context = background_contexts.get(target, "")
    
    bg_block = ""
    if version != "Revised" and context:
        bg_block = f"\n### Background Context\n{context}\n"
        
    task_block = (
        "\n### Task\n"
        "Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:\n\n"
        "• Favor\n"
        "• Against\n"
        "• None\n\n"
        "Before assigning a stance, mentally rewrite the tweet into its intended literal meaning while preserving the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis.\n\n"
        "Then determine the stance toward the target itself, not toward other people, quoted opinions, related entities, or hashtags.\n"
    )
    
    guidelines_block = (
        "\nGuidelines:\n\n"
        "• Favor: supports, defends, promotes, or welcomes the target.\n"
        "• Against: opposes, criticizes, rejects, or mocks the target.\n"
        "• None: no clear stance toward the target.\n"
    )
    
    rules = [
        "Determine where praise or criticism is directed. Negative language toward opponents of the target is usually Favor, not Against.",
        "Hashtags may be ironic or hijacked. Never infer stance from hashtags alone.",
        "Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording."
    ]
    
    if version in ["CI v1", "Revised"]:
        rules.append("Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance.")
    else: # CI v3 and CI v5
        rules.append("Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor.")
        
    if version == "CI v5":
        rules.append("Rhetorical questions or seemingly factual statements that imply doubt, hidden motive, or negative consequence specifically about the target indicate Against, even without explicit negative words.")
        
    rules.append("If the stance toward the target cannot reasonably be inferred, output None.")
    
    important_block = "\nImportant:\n\n" + "\n".join([f"• {r}" for r in rules]) + "\n"
    
    format_block = (
        "\nRespond with ONLY one word:\n\n"
        "Favor\n"
        "Against\n"
        "None\n\n"
        "Do not provide any explanation, punctuation, or additional text."
    )
    
    prompt = (
        "You are an expert annotator for Arabic stance detection."
        f"{bg_block}"
        f"{task_block}"
        f"{guidelines_block}"
        f"{important_block}"
        f"{format_block}"
    )
    return prompt

# ── API Call Function with Rotation ───────────────────────────────────────────
blacklisted_indices = set()

def call_cerebras_with_rotation(
    system_prompt: str,
    target: str,
    text: str,
    row_index: int,
    max_retries: int = 40
) -> tuple[dict, str]:
    user_msg = f"Target: {TARGET_AR.get(target, target)}\nTweet: {text}"
    
    for attempt in range(max_retries):
        # Filter active clients
        active_idxs = [i for i in range(len(clients)) if i not in blacklisted_indices]
        if not active_idxs:
            active_idxs = list(range(len(clients)))  # fallback if all fail
            
        idx_in_active = (call_cerebras_with_rotation.counter + attempt) % len(active_idxs)
        client_idx = active_idxs[idx_in_active]
        client = clients[client_idx]
        key_label = f"Key {client_idx + 1}"
        
        try:
            print(f"      → Trying {key_label}...")
            raw_resp = client.chat.completions.with_raw_response.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                reasoning_effort="none",
            )
            
            rl_headers = dict(raw_resp.headers)
            completion = raw_resp.parse()
            
            # Increment call counter
            call_cerebras_with_rotation.counter += 1
            
            raw_text = (completion.choices[0].message.content or "").strip()
            model_returned = completion.model
            
            # Parse stance
            predicted = "None"
            parse_ok = False
            for label in VALID_LABELS:
                if raw_text.lower() == label.lower() or raw_text.lower().startswith(label.lower()):
                    predicted = label
                    parse_ok = True
                    break
                    
            if not parse_ok:
                for label in ["Favor", "Against", "None"]:
                    if label.lower() in raw_text.lower():
                        predicted = label
                        break
                        
            result = {
                "row_index":       row_index,
                "target":          target,
                "gold_label":      "",
                "predicted_label": predicted,
                "model_id":        model_returned,
                "parse_success":   parse_ok,
                "raw_response":    raw_text,
            }
            return result, key_label
            
        except Exception as exc:
            exc_str = str(exc)
            print(f"    ⚠️ Warning on {key_label} row {row_index} (attempt {attempt+1}/{max_retries}): {exc_str}")
            if "402" in exc_str or "payment_required" in exc_str.lower():
                print(f"    🚫 Blacklisting {key_label} due to payment/quota limits (402).")
                blacklisted_indices.add(client_idx)
                continue
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                print(f"    ⏳ Rate limit (429) hit on {key_label}. Sleeping 30 seconds...")
                time.sleep(30.0)
                continue
            time.sleep(1.0)
            
    print(f"    ❌ Row {row_index}: All retries failed. Defaulting to None.")
    result = {
        "row_index":       row_index,
        "target":          target,
        "gold_label":      "",
        "predicted_label": "None",
        "model_id":        MODEL_ID,
        "parse_success":   False,
        "raw_response":    "ERROR: max retries exceeded",
    }
    return result, "Failed"

call_cerebras_with_rotation.counter = 0

# ── Helper Utils ──────────────────────────────────────────────────────────────
def load_completed_indices(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, keep_default_na=False)
        return set(df["row_index"].astype(int).tolist())
    except Exception:
        return set()

def append_result_row(path: str, row: dict) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in RESULTS_FIELDNAMES})

# ── Main Run Controller ───────────────────────────────────────────────────────
def run_prompt_version(version: str) -> None:
    slug = version.lower().replace(" ", "_")
    results_path = os.path.join(PRED_DIR, f"results_gemma4_31b_cerebras_dev_{slug}.csv")
    raw_log_path = os.path.join(LOG_DIR, f"results_gemma4_31b_cerebras_dev_{slug}_raw.jsonl")
    
    print(f"\n========================================================")
    print(f"🚀 RUNNING VERSION: {version}")
    print(f"   Output CSV: {results_path}")
    print(f"========================================================")
    
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    dev_df = pd.read_csv(DEV_CSV_PATH, keep_default_na=False)
    dev_df = dev_df.rename(columns={"id": "ID", "tweet_text": "text"})
    
    completed = load_completed_indices(results_path)
    if completed:
        print(f"📂 Resuming: {len(completed)}/619 rows already complete.")
    else:
        print(f"🆕 Starting fresh run on 619 rows.")
        
    total = len(dev_df)
    new_calls = 0
    
    for idx, row in dev_df.iterrows():
        row_index = int(row.name)
        
        if row_index in completed:
            continue
            
        target = str(row["target"])
        text = str(row["text"])
        gold = str(row["stance"])
        
        # Build prompt dynamically per target
        sys_prompt = get_system_prompt(target, version)
        
        print(f"  [{idx+1}/{total}] row_index={row_index} | target={target}")
        
        result, key_label = call_cerebras_with_rotation(sys_prompt, target, text, row_index)
        result["gold_label"] = gold
        
        append_result_row(results_path, result)
        completed.add(row_index)
        new_calls += 1
        
        log_entry = {
            "row_index":       row_index,
            "target":          target,
            "gold_label":      gold,
            "raw_response":    result["raw_response"],
            "predicted_label": result["predicted_label"],
            "model_id":        result["model_id"],
            "parse_success":   result["parse_success"],
            "timestamp":       time.time(),
            "key_used":        key_label
        }
        with open(raw_log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        print(f"    → pred={result['predicted_label']!r}  gold={gold!r}  ({key_label})")
        
        # Safe pacing delay (2.0s between requests total, rotated)
        time.sleep(2.0)
        
    print(f"\n✅ Finished predictions for {version} ({new_calls} new calls).")
    
    # Sort and evaluate
    results_df = pd.read_csv(results_path, encoding="utf-8", keep_default_na=False)
    results_df = results_df.drop_duplicates(subset=["row_index"]).sort_values("row_index").copy()
    results_df.to_csv(results_path, index=False)
    
    y_true = results_df["gold_label"].tolist()
    y_pred = results_df["predicted_label"].tolist()
    
    overall_f2 = favg2(y_true, y_pred)
    print(f"🏆 Final Favg2 for {version}: {overall_f2:.4f}")
    
    # Write pred TXT file
    txt_path = results_path.replace(".csv", ".txt")
    write_pred_txt(y_pred, txt_path)
    print(f"Saved predictions to {txt_path}")
    
    # Run official eval script
    scores = run_official_eval(DEV_CSV_PATH, txt_path)
    print(f"\nOfficial evaluation scores for {version}:")
    for k in sorted(scores):
        print(f"  {k}: {scores[k]:.6f}")
        
    # Log to results_summary.csv
    summary_model_name = f"gemma4_31b_cerebras_dev_{slug}"
    summary_row = scores_to_row(summary_model_name, scores)
    write_results_row(summary_row)
    print(f"Successfully logged scores to results_summary.csv under '{summary_model_name}'")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, choices=["Revised", "CI v1", "CI v3", "CI v5"], help="Run specific prompt version")
    args = parser.parse_args()
    
    if args.version:
        run_prompt_version(args.version)
    else:
        # Run all 4 prompt configurations in sequence
        for v in ["Revised", "CI v1", "CI v3", "CI v5"]:
            run_prompt_version(v)

if __name__ == "__main__":
    main()

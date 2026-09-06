"""
Gemma 4 31B Test Set Inference Runner (352-row Mawqif-v2 Women Driving)

Supports evaluating key prompt calibration variants reported in the paper:
  - v1:        Initial Zero-Shot baseline
  - revised:   Revised v1 (Mental translation & target normalization, Favg2 = 0.8444)
  - ci_v1:     Context Injection v1 (Background context + mental rewrite, Favg2 = 0.8485)
  - ci_v3:     Final Best Calibrated Prompt (Positive framing reporting, Favg2 = 0.8661)
  - ci_v5:     Skepticism Calibration (Doubt / traffic negative consequence rules, Favg2 = 0.8635)

Usage:
    python src/gemma_inference_test.py --variant ci_v3
"""

import os
import sys
import io
import time
import json
import re
import argparse
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(_ROOT, "data")
PRED_DIR = os.path.join(_ROOT, "predictions")

DEFAULT_TEST_CSV = os.path.join(DATA_DIR, "ground_truth.csv")
MODEL_ID = "gemma-4-31b"
TEMPERATURE = 0.1
TOP_P = 0.95
MAX_TOKENS = 10

VALID_LABELS = {"Against", "Favor", "None"}
TARGET_AR = {"Women Driving": "قيادة المرأة للسيارة"}

BACKGROUND_CONTEXT = (
    "The target concerns the 2017–2018 Saudi policy change allowing women to drive. "
    "Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this "
    "period often discuss the royal decree, implementation, licensing, religion, tradition, "
    "safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, "
    "sarcasm, humor, rhetorical questions, and indirect expressions."
)


def get_system_prompt(variant: str) -> str:
    if variant == "v1":
        return (
            "You are an expert annotator for Arabic stance detection. Given a target and an Arabic tweet, "
            "determine whether the author is Favor, Against, or None toward the target.\n\n"
            "Guidelines:\n"
            "- Favor: the writer expresses support for or a positive position toward the target.\n"
            "- Against: the writer expresses opposition to or a negative position toward the target.\n"
            "- None: no clear stance toward the target.\n\n"
            "Respond with ONLY a single word: Favor, Against, or None. No explanation, no punctuation, no other text."
        )

    bg_block = f"\n### Background Context\n{BACKGROUND_CONTEXT}\n" if variant != "revised" else ""

    task_block = (
        "\n### Task\n"
        "Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:\n\n"
        "• Favor\n"
        "• Against\n"
        "• None\n\n"
        "Before assigning a stance, mentally rewrite the tweet into its intended literal meaning while preserving "
        "the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis.\n\n"
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
        "Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording.",
    ]

    if variant in ["revised", "ci_v1"]:
        rules.append("Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance.")
    else:  # ci_v3 and ci_v5
        rules.append(
            "Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, "
            "unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor."
        )

    if variant == "ci_v5":
        rules.append(
            "Rhetorical questions or seemingly factual statements that imply doubt, hidden motive, or negative consequence "
            "specifically about the target indicate Against, even without explicit negative words."
        )

    rules.append("If the stance toward the target cannot reasonably be inferred, output None.")

    important_block = "\nImportant:\n\n" + "\n".join([f"• {r}" for r in rules]) + "\n"
    format_block = (
        "\nRespond with ONLY one word:\n\n"
        "Favor\n"
        "Against\n"
        "None\n\n"
        "Do not provide any explanation, punctuation, or additional text."
    )

    return f"You are an expert annotator for Arabic stance detection.{bg_block}{task_block}{guidelines_block}{important_block}{format_block}"


def parse_response(raw: str) -> tuple[str, bool]:
    raw_str = raw.strip()
    if raw_str in VALID_LABELS:
        return raw_str, True
    for label in VALID_LABELS:
        if raw_str.lower() == label.lower():
            return label, True
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw_str.lower():
            return label, False
    return "None", False


def main():
    parser = argparse.ArgumentParser(description="Run Gemma 4 31B on 352-row test set")
    parser.add_argument("--variant", choices=["v1", "revised", "ci_v1", "ci_v3", "ci_v5"], default="ci_v3",
                        help="Prompt variant to evaluate (default: ci_v3)")
    parser.add_argument("--data_path", default=DEFAULT_TEST_CSV, help="Path to evaluation CSV")
    parser.add_argument("--output_csv", default=None, help="Custom output CSV path")
    args = parser.parse_args()

    variant_tag = args.variant.replace(" ", "_").lower()
    out_csv = args.output_csv or os.path.join(PRED_DIR, f"results_gemma4_31b_cerebras_test_{variant_tag}.csv")
    out_txt = out_csv.replace(".csv", ".txt")

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        print("❌ Error: CEREBRAS_API_KEY environment variable not set.")
        sys.exit(1)

    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)

    if not os.path.exists(args.data_path):
        print(f"❌ Input data not found at: {args.data_path}")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    text_col = "text" if "text" in df.columns else "tweet_text"
    target_col = "target"

    system_prompt = get_system_prompt(args.variant)
    print(f"=== Gemma 4 31B Inference (Variant: {args.variant}) ===")
    print(f"Loaded {len(df)} rows from {args.data_path}")

    results = []
    for idx, row in df.iterrows():
        target_name = row[target_col]
        arabic_target = TARGET_AR.get(target_name, target_name)
        tweet_text = row[text_col]
        user_content = f"Target: {arabic_target}\nTweet: {tweet_text}"

        retries = 3
        pred_label = "None"
        parse_ok = False
        raw_text = ""

        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    reasoning_effort="none"
                )
                raw_text = resp.choices[0].message.content or ""
                pred_label, parse_ok = parse_response(raw_text)
                break
            except Exception as e:
                time.sleep(2.0 * (attempt + 1))

        results.append({
            "row_index": idx,
            "target": target_name,
            "predicted_label": pred_label,
            "parse_success": parse_ok,
            "raw_response": raw_text
        })
        time.sleep(0.5)

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in res_df["predicted_label"]:
            f.write(f"{p}\n")

    print(f"\nCompleted! Predictions saved to:\n - CSV: {out_csv}\n - TXT: {out_txt}")
    print("\nPredicted distribution:")
    print(res_df["predicted_label"].value_counts())


if __name__ == "__main__":
    main()

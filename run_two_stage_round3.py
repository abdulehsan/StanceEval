import os
import csv
import io
import re
import sys
import time
import json
from openai import OpenAI

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

# Load environment variables manually from .env
dotenv_path = os.path.join(_HERE, ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().strip('"').strip("'")
                os.environ[key] = val

from prompt_builder import build_kb_section

INPUT_CSV = os.path.join(_HERE, "data", "ground_truth.csv")
BASELINE_CSV = os.path.join(_HERE, "predictions", "results_gemma4_31b_cerebras_test_ci_v3.csv")
ROUND2_CSV = os.path.join(_HERE, "predictions", "results_two_stage_pipeline_25.csv")

OUTPUT_CSV = os.path.join(_HERE, "results_two_stage_pipeline_round3_25.csv")
OUTPUT_CSV_PRED = os.path.join(_HERE, "predictions", "results_two_stage_pipeline_round3_25.csv")
SUMMARY_TXT = os.path.join(_HERE, "summary_two_stage_round3_25.txt")

# Initialize OpenAI client for local ALLaM server
allam_client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Initialize Cerebras client
cerebras_api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
if not cerebras_api_key:
    print("❌ ERROR: CEREBRAS_API_KEY not found in environment or .env file.")
    sys.exit(1)

from cerebras.cloud.sdk import Cerebras
gemma_client = Cerebras(api_key=cerebras_api_key)

# Models and settings
ALLAM_MODEL = "allam-7b-instruct-preview"
GEMMA_MODEL = "gemma-4-31b"

ALLAM_SYSTEM_PROMPT_TEMPLATE = """You are an expert in Saudi Arabic dialect, sarcasm, and cultural context.

Given a tweet, explain what the writer literally means beneath the surface — resolve any
sarcasm, dialect-specific phrasing, or rhetorical devices. Preserve the writer's emotional
tone and attitude (e.g. celebratory, mocking, frustrated, joking, worried) — do not flatten
it into a neutral factual summary. Focus on tone and intent, not on classifying any stance.
Do not mention "Favor," "Against," or "None," and do not use evaluative words that imply
approval or disapproval (e.g. "positive," "negative," "disapproving," "supportive") —
describe the tone itself, not a judgment of it.

{kb_reference}
Use any reference information above only to correctly understand what a hashtag or named
entity refers to — it does not tell you how the writer feels about it. The writer may be
using that hashtag or entity ironically, defiantly, or dismissively; your job is still to
read that from the tweet's own wording and tone."""

GEMMA_SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

### Native-Speaker Interpretation
A native Arabic speaker has provided the following read on tone and meaning for this
specific tweet. This is a supplementary aid for resolving dialect or sarcasm you might
otherwise miss — it is NOT a substitute for the tweet itself. The tweet's own wording,
emojis, and hashtags remain the primary evidence for your stance judgment. If the
interpretation's tone seems flatter or more neutral than what the tweet's actual wording
and emojis convey, trust the tweet. Additional context should sharpen your confidence in
a stance, not soften it — do not default to None simply because more information is
present; only use None when the stance is genuinely unclear even with this added context.

{allam_interpretation}

### Task
Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:

- Favor
- Against
- None

Before assigning a stance, mentally rewrite the tweet into its intended literal meaning while preserving the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis.

Then determine the stance toward the target itself, not toward other people, quoted opinions, related entities, or hashtags.

Guidelines:

- Favor: supports, defends, promotes, or welcomes the target.
- Against: opposes, criticizes, rejects, or mocks the target.
- None: no clear stance toward the target.

Important:

- Determine where praise or criticism is directed. Negative language toward opponents of the target is usually Favor, not Against.
- Hashtags may be ironic or hijacked. Never infer stance from hashtags alone.
- Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording.
- Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor.
- If the stance toward the target cannot reasonably be inferred, output None.

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text."""

VALID_LABELS = {"Favor", "Against", "None"}

def parse_response(raw: str) -> tuple[str, bool]:
    raw = raw.strip()
    if raw in VALID_LABELS:
        return raw, True
    for label in VALID_LABELS:
        if raw.lower() == label.lower():
            return label, True
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw.lower():
            return label, False
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

def load_baseline_and_round2():
    baselines = {}
    with open(BASELINE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            baselines[int(row["row_index"])] = row["predicted_label"]

    round2 = {}
    with open(ROUND2_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            round2[int(row["row_index"])] = row["two_stage_label"]
            
    return baselines, round2

def get_allam_interpretation(tweet_text: str) -> str:
    kb_block = build_kb_section(tweet_text)
    sys_prompt = ALLAM_SYSTEM_PROMPT_TEMPLATE.format(kb_reference=kb_block)
    
    # Simple retry loop to wait for server if it takes a moment
    for attempt in range(5):
        try:
            resp = allam_client.chat.completions.create(
                model=ALLAM_MODEL,
                temperature=0.7,
                max_tokens=384,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Tweet: {tweet_text}"}
                ]
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"    ⚠️ ALLaM attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise Exception("ALLaM server is not responding.")

def get_gemma_stance(tweet_text: str, interpretation: str) -> tuple[str, str]:
    sys_prompt = GEMMA_SYSTEM_PROMPT.format(allam_interpretation=interpretation)
    user_msg = f"Target: قيادة المرأة للسيارة\nTweet: {tweet_text}"
    
    # 12s floor delay for Cerebras rate limit
    time.sleep(12.0)
    
    for attempt in range(5):
        try:
            raw_resp = gemma_client.chat.completions.with_raw_response.create(
                model=GEMMA_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=10,
                reasoning_effort="none"
            )
            completion = raw_resp.parse()
            raw_text = completion.choices[0].message.content or ""
            predicted, parse_ok = parse_response(raw_text)
            return predicted, raw_text
        except Exception as e:
            exc_str = str(e)
            retry_after = 6.0 * (2 ** attempt)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 60.0
                if hasattr(e, "response") and e.response is not None:
                    headers = dict(e.response.headers)
                    if "retry-after" in headers:
                        try:
                            retry_after = float(headers["retry-after"])
                        except ValueError:
                            pass
            print(f"    ⚠️ Gemma attempt {attempt+1} failed (sleeping {retry_after}s): {e}")
            time.sleep(retry_after)
            
    return "None", "ERROR: all attempts failed"

def main():
    baselines, round2 = load_baseline_and_round2()
    
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    print(f"Loaded {len(rows)} tweets. Evaluating first 25 rows on Round 3 pipeline...")
    
    results = []
    for i in range(25):
        row = rows[i]
        tweet_text = row["tweet_text"]
        target = row["target"]
        
        print(f"Processing Row {i}...")
        
        # Stage 1: ALLaM
        interpretation = get_allam_interpretation(tweet_text)
        kb_block = build_kb_section(tweet_text)
        has_kb = "Yes" if kb_block.strip() else "No"
        print(f"  Stage 1 (ALLaM) done. KB match: {has_kb}")
        
        # Stage 2: Gemma
        predicted, raw_response = get_gemma_stance(tweet_text, interpretation)
        print(f"  Stage 2 (Gemma) done. Label: {predicted}")
        
        plain_gemma = baselines.get(i, "None")
        changed = "True" if predicted != plain_gemma else "False"
        
        results.append({
            "row_index": i,
            "tweet_text": tweet_text,
            "allam_interpretation": interpretation,
            "two_stage_label": predicted,
            "plain_gemma_label": plain_gemma,
            "changed": changed
        })
        
    # Write output CSVs
    fieldnames = ["row_index", "tweet_text", "allam_interpretation", "two_stage_label", "plain_gemma_label", "changed"]
    for path in [OUTPUT_CSV, OUTPUT_CSV_PRED]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                
    print(f"predictions saved to {OUTPUT_CSV} and {OUTPUT_CSV_PRED}")

    # Generate summary report
    print("Generating summary report...")
    summary_lines = []
    summary_lines.append("FINAL PIPELINE RUN SUMMARY RESULTS (ROUND 3):")
    summary_lines.append("=" * 120)
    summary_lines.append(f"{'Row':<4} | {'Plain Gemma':<12} | {'Two-Stage':<12} | {'Changed?':<8} | ALLaM Interpretation")
    summary_lines.append("=" * 120)
    
    total_changed = 0
    flips_breakdown = {
        "Favor -> Against": 0,
        "Favor -> None": 0,
        "Against -> Favor": 0,
        "Against -> None": 0,
        "None -> Favor": 0,
        "None -> Against": 0
    }
    
    for r in results:
        idx = r["row_index"]
        plain = r["plain_gemma_label"]
        two_stage = r["two_stage_label"]
        changed = r["changed"] == "True"
        interp = r["allam_interpretation"].replace("\n", " ")
        
        summary_lines.append(f"{idx:<4} | {plain:<12} | {two_stage:<12} | {str(changed):<8} | {interp}")
        
        if changed:
            total_changed += 1
            key = f"{plain} -> {two_stage}"
            if key in flips_breakdown:
                flips_breakdown[key] += 1
                
    summary_lines.append("=" * 120)
    summary_lines.append(f"Total Changed Rows: {total_changed} / 25")
    summary_lines.append("Change Details:")
    for r in results:
        if r["changed"] == "True":
            summary_lines.append(f"  - Row {r['row_index']}: {r['plain_gemma_label']} → {r['two_stage_label']}")
            
    # Add flip breakdown to summary
    summary_lines.append("\nFlip breakdown vs. plain best_86_61:")
    for k, v in flips_breakdown.items():
        summary_lines.append(f"  - {k}: {v}")
        
    # Check row 8 details
    row8_data = results[8]
    summary_lines.append(f"\nRow 8 status: Plain Gemma={row8_data['plain_gemma_label']}, Two-stage Round 3={row8_data['two_stage_label']}")
    summary_lines.append(f"Row 8 ALLaM Interpretation:\n{row8_data['allam_interpretation']}")
    
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        
    print(f"Summary saved to {SUMMARY_TXT}")

if __name__ == "__main__":
    main()

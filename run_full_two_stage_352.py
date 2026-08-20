import os
import csv
import io
import re
import sys
import time
import json
import threading
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
from data_utils import write_pred_txt

INPUT_CSV = os.path.join(_HERE, "data", "ground_truth.csv")
BASELINE_CSV = os.path.join(_HERE, "predictions", "results_gemma4_31b_cerebras_test_ci_v3.csv")

ALLAM_CSV = os.path.join(_HERE, "predictions", "results_allam_interpretations_352.csv")
GEMMA_CSV = os.path.join(_HERE, "predictions", "results_two_stage_gemma_352.csv")
GEMMA_TXT = os.path.join(_HERE, "predictions", "results_two_stage_gemma_352.txt")

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

def load_baseline():
    baselines = {}
    if os.path.exists(BASELINE_CSV):
        with open(BASELINE_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                baselines[int(row["row_index"])] = row["predicted_label"]
    else:
        print(f"⚠️ Warning: baseline {BASELINE_CSV} not found. Comparison will fall back.")
    return baselines

# Thread-safe lists and locks
allam_results = [None] * 352
allam_results_lock = threading.Lock()

allam_csv_lock = threading.Lock()
gemma_csv_lock = threading.Lock()

# Statistics tracker
allam_wall_clock = 0.0
gemma_wall_clock = 0.0

def get_allam_interpretation(tweet_text: str, row_index: int) -> str:
    kb_block = build_kb_section(tweet_text)
    sys_prompt = ALLAM_SYSTEM_PROMPT_TEMPLATE.format(kb_reference=kb_block)
    
    # 5 attempts maximum
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
            print(f"    ⚠️ ALLaM attempt {attempt+1} failed on row {row_index}: {e}")
            time.sleep(2)
    raise Exception(f"ALLaM server failed on row {row_index}")

def get_gemma_stance(tweet_text: str, interpretation: str, row_index: int) -> tuple[str, str]:
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
            print(f"    ⚠️ Gemma attempt {attempt+1} failed on row {row_index} (sleeping {retry_after}s): {e}")
            time.sleep(retry_after)
            
    return "None", "ERROR: all attempts failed"

# Load existing state to resume
def load_allam_progress() -> int:
    if not os.path.exists(ALLAM_CSV):
        return 0
    completed_count = 0
    with open(ALLAM_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["row_index"])
            allam_results[idx] = row["allam_interpretation"]
            completed_count += 1
    return completed_count

def load_gemma_progress() -> int:
    if not os.path.exists(GEMMA_CSV):
        return 0
    completed = set()
    with open(GEMMA_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(int(row["row_index"]))
    return len(completed)

# Producer Thread: Stage 1 (ALLaM)
def allam_producer(rows: list[dict]):
    global allam_wall_clock
    print("🚀 Starting Stage 1 (ALLaM Producer Thread)...")
    
    # Setup CSV header if not exists
    os.makedirs(os.path.dirname(ALLAM_CSV) or ".", exist_ok=True)
    if not os.path.exists(ALLAM_CSV):
        with open(ALLAM_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_index", "id", "tweet_id", "tweet_text", "allam_interpretation", "kb_matched"])
            
    completed_count = load_allam_progress()
    if completed_count > 0:
        print(f"📂 ALLaM resumed: {completed_count} interpretations loaded.")
        
    start_time = time.time()
    last_printed_idx = -1
    
    for i, row in enumerate(rows):
        # Skip if already completed (from resume file)
        with allam_results_lock:
            if allam_results[i] is not None:
                continue
                
        tweet_text = row["tweet_text"]
        kb_block = build_kb_section(tweet_text)
        has_kb = "Yes" if kb_block.strip() else "No"
        
        # Call local ALLaM
        interp = get_allam_interpretation(tweet_text, i)
        
        # Store in list
        with allam_results_lock:
            allam_results[i] = interp
            
        # Append row to CSV
        with allam_csv_lock:
            with open(ALLAM_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["row_index", "id", "tweet_id", "tweet_text", "allam_interpretation", "kb_matched"])
                writer.writerow({
                    "row_index": i,
                    "id": row["id"],
                    "tweet_id": row["tweet_id"],
                    "tweet_text": tweet_text,
                    "allam_interpretation": interp,
                    "kb_matched": has_kb
                })
                
        # Progress reporting
        completed_now = sum(1 for x in allam_results if x is not None)
        if completed_now % 25 == 0 and completed_now != last_printed_idx:
            last_printed_idx = completed_now
            elapsed = time.time() - start_time
            rate = elapsed / (completed_now - completed_count) if (completed_now - completed_count) > 0 else 0
            eta = rate * (352 - completed_now)
            print(f"📈 [ALLaM] Processed {completed_now}/352 rows. Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
    allam_wall_clock = time.time() - start_time
    print(f"✅ Stage 1 (ALLaM) complete in {allam_wall_clock:.1f} seconds.")

# Consumer Thread: Stage 2 (Gemma)
def gemma_consumer(rows: list[dict], baselines: dict):
    global gemma_wall_clock
    print("🚀 Gemma Consumer Thread is waiting for the startup threshold (22 ALLaM rows)...")
    
    # Wait until producer has processed at least 22 rows
    while True:
        with allam_results_lock:
            ready = sum(1 for x in allam_results if x is not None)
        if ready >= 22:
            print(f"🔔 Threshold reached ({ready} rows ready). Starting Stage 2 (Gemma Consumer)...")
            break
        time.sleep(2)
        
    # Setup CSV header if not exists
    os.makedirs(os.path.dirname(GEMMA_CSV) or ".", exist_ok=True)
    if not os.path.exists(GEMMA_CSV):
        with open(GEMMA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_index", "id", "tweet_id", "target", "two_stage_label", "plain_gemma_label", "changed", "allam_interpretation", "raw_response"])
            
    # Load gemma resume state
    gemma_completed = set()
    if os.path.exists(GEMMA_CSV):
        with open(GEMMA_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gemma_completed.add(int(row["row_index"]))
    gemma_completed_count = len(gemma_completed)
    if gemma_completed_count > 0:
        print(f"📂 Gemma resumed: {gemma_completed_count} rows already processed.")
        
    start_time = time.time()
    last_printed_idx = -1
    
    for j in range(352):
        if j in gemma_completed:
            continue
            
        row = rows[j]
        tweet_text = row["tweet_text"]
        target = row["target"]
        plain_gemma = baselines.get(j, "None")
        
        # Wait until ALLaM interpretation is ready
        while True:
            with allam_results_lock:
                interp = allam_results[j]
            if interp is not None:
                break
            time.sleep(1)
            
        # Query Gemma (takes 12s sleep inside function)
        predicted, raw_response = get_gemma_stance(tweet_text, interp, j)
        
        changed = "True" if predicted != plain_gemma else "False"
        
        # Append row to CSV
        with gemma_csv_lock:
            with open(GEMMA_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["row_index", "id", "tweet_id", "target", "two_stage_label", "plain_gemma_label", "changed", "allam_interpretation", "raw_response"])
                writer.writerow({
                    "row_index": j,
                    "id": row["id"],
                    "tweet_id": row["tweet_id"],
                    "target": target,
                    "two_stage_label": predicted,
                    "plain_gemma_label": plain_gemma,
                    "changed": changed,
                    "allam_interpretation": interp,
                    "raw_response": raw_response
                })
                
        gemma_completed.add(j)
        completed_now = len(gemma_completed)
        
        if completed_now % 25 == 0 and completed_now != last_printed_idx:
            last_printed_idx = completed_now
            elapsed = time.time() - start_time
            rate = elapsed / (completed_now - gemma_completed_count) if (completed_now - gemma_completed_count) > 0 else 0
            eta = rate * (352 - completed_now)
            print(f"📈 [Gemma] Processed {completed_now}/352 rows. Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
    gemma_wall_clock = time.time() - start_time
    print(f"✅ Stage 2 (Gemma) complete in {gemma_wall_clock:.1f} seconds.")

def check_distribution():
    # True target distribution for reference
    # Known split: Favor 44.89% / Against 45.45% / None 9.66%
    true_dist = {"Favor": 44.89, "Against": 45.45, "None": 9.66}
    
    # Load prediction labels
    predictions = []
    plain_gemma_labels = []
    
    with open(GEMMA_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row["two_stage_label"])
            plain_gemma_labels.append(row["plain_gemma_label"])
            
    total = len(predictions)
    if total == 0:
        return
        
    counts = {"Favor": 0, "Against": 0, "None": 0}
    plain_counts = {"Favor": 0, "Against": 0, "None": 0}
    
    for p in predictions:
        if p in counts:
            counts[p] += 1
    for pg in plain_gemma_labels:
        if pg in plain_counts:
            plain_counts[pg] += 1
            
    print("\n" + "="*80)
    print("SANITY DISTRIBUTION CHECK:")
    print("="*80)
    print(f"{'Stance':<8} | {'True Split %':<12} | {'Plain Gemma %':<15} | {'Two-Stage Round 3 %':<20} | {'Count':<6}")
    print("-"*80)
    for k in ["Favor", "Against", "None"]:
        p_pct = (counts[k] / total) * 100
        pg_pct = (plain_counts[k] / total) * 100
        print(f"{k:<8} | {true_dist[k]:<12.2f}% | {pg_pct:<15.2f}% | {p_pct:<20.2f}% | {counts[k]:<6}")
    print("="*80)
    
    # Calculate flip statistics
    flips = 0
    flip_breakdown = {
        "Favor -> Against": 0,
        "Favor -> None": 0,
        "Against -> Favor": 0,
        "Against -> None": 0,
        "None -> Favor": 0,
        "None -> Against": 0
    }
    
    for pg, ts in zip(plain_gemma_labels, predictions):
        if pg != ts:
            flips += 1
            key = f"{pg} -> {ts}"
            if key in flip_breakdown:
                flip_breakdown[key] += 1
                
    print(f"Total Flip Count: {flips} / {total} ({ (flips / total)*100 :.2f}%)")
    print("Flip Direction Breakdown:")
    for k, v in flip_breakdown.items():
        print(f"  - {k}: {v}")
    print("="*80)
    
    # Format final predictions to txt
    print(f"Writing CodaBench submission file to {GEMMA_TXT}...")
    write_pred_txt(predictions, GEMMA_TXT)
    print("✅ CodaBench text file formatted and written successfully.")

def main():
    # Load dataset
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    if len(rows) != 352:
        print(f"⚠️ Warning: Dataset length is {len(rows)} (expected 352)")
        
    baselines = load_baseline()
    
    # Run threads
    prod_thread = threading.Thread(target=allam_producer, args=(rows,))
    cons_thread = threading.Thread(target=gemma_consumer, args=(rows, baselines))
    
    t0 = time.time()
    
    prod_thread.start()
    cons_thread.start()
    
    prod_thread.join()
    cons_thread.join()
    
    total_time = time.time() - t0
    print(f"\n🎉 ALL RUNS COMPLETED in {total_time:.1f} seconds (~{total_time/60:.1f} minutes).")
    print(f"  Stage 1 (ALLaM) wall clock: {allam_wall_clock:.1f}s")
    print(f"  Stage 2 (Gemma) wall clock: {gemma_wall_clock:.1f}s")
    
    check_distribution()

if __name__ == "__main__":
    main()

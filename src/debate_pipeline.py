import os
import csv
import io
import re
import sys
import time
import json
import argparse
import threading
from groq import Groq
from cerebras.cloud.sdk import Cerebras

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_HERE = _ROOT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import write_pred_txt

# ── Load environment variables ────────────────────────────────────────────────
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

# ── API clients ────────────────────────────────────────────────────────────────
cerebras_api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
if not cerebras_api_key:
    print("❌ ERROR: CEREBRAS_API_KEY not found in env.")
    sys.exit(1)
gemma_client = Cerebras(api_key=cerebras_api_key)

groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY not found in env.")
    sys.exit(1)
qwen_client = Groq(api_key=groq_api_key)

# ── File Paths ────────────────────────────────────────────────────────────────
INPUT_CSV = os.path.join(_HERE, "data", "ground_truth.csv")
BASELINE_CSV = os.path.join(_HERE, "predictions", "results_gemma4_31b_cerebras_test_ci_v3.csv")

# Outputs
STAGE1_CSV = os.path.join(_HERE, "predictions", "debate_stage1.csv")
STAGE2_CSV = os.path.join(_HERE, "predictions", "debate_stage2.csv")
STAGE3_CSV = os.path.join(_HERE, "predictions", "debate_stage3.csv")
FINAL_TXT = os.path.join(_HERE, "predictions", "debate_final.txt")

# Rate limit tracking log
CEREBRAS_LOG = os.path.join(_HERE, "predictions", "debate_cerebras_log.jsonl")

# Target translation
TARGET_AR = {"Women Driving": "قيادة المرأة للسيارة"}

# ── Prompts ───────────────────────────────────────────────────────────────────
STAGE1_SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

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

Respond in exactly this format:
Stance: <Favor/Against/None>
Argument: <justification for this stance>"""

STAGE2_SYSTEM_PROMPT = """You are a critical reviewer of Arabic stance detection judgments.

You will be given a tweet, a target topic, and another annotator's classification with
their justification. Your job is to challenge it:

- If you believe the classification is wrong, argue for what you believe the correct
  stance is (Favor, Against, or None) and explain why, citing specific words, tone,
  sarcasm, or context in the tweet.
- If you believe the classification is correct, still identify any weaknesses or
  alternative readings in the original justification — do not simply agree without
  scrutiny.

Be concise"""

STAGE3_SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection, making
a final decision after reviewing two perspectives.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

You will be given a tweet, a target, an initial classification with its argument, and a
challenge/critique of that classification. Weigh both perspectives on their merits — do
not automatically favor the initial classification just because it came first.

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

# ── Shared State ──────────────────────────────────────────────────────────────
s1_completed = {}
s2_completed = {}
s3_completed = {}

s1_lock = threading.Lock()
s2_lock = threading.Lock()

s1_csv_lock = threading.Lock()
s2_csv_lock = threading.Lock()

timestamps = []
rows = []

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_stage1_response(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    stance = "None"
    argument = ""
    
    stance_match = re.search(r'(?:Stance|stance)\s*:\s*(Favor|Against|None)', raw, re.IGNORECASE)
    if stance_match:
        val = stance_match.group(1).lower()
        if val == "favor":
            stance = "Favor"
        elif val == "against":
            stance = "Against"
        elif val == "none":
            stance = "None"
            
    argument_match = re.search(r'(?:Argument|argument)\s*:\s*(.*)', raw, re.DOTALL | re.IGNORECASE)
    if argument_match:
        argument = argument_match.group(1).strip()
        
    if stance == "None" and not stance_match:
        for label in ["Favor", "Against", "None"]:
            if label.lower() in raw.lower():
                stance = label
                break
                
    if not argument:
        argument = re.sub(r'(?:Stance|stance)\s*:\s*(Favor|Against|None)', '', raw, flags=re.IGNORECASE).strip()
        
    return stance, argument

def parse_stage3_response(raw: str) -> str:
    raw = raw.strip()
    if raw in {"Favor", "Against", "None"}:
        return raw
    for label in {"Favor", "Against", "None"}:
        if raw.lower() == label.lower():
            return label
    for label in ["Favor", "Against", "None"]:
        if label.lower() in raw.lower():
            return label
    return "None"

def extract_qwen_stance(challenge_text: str) -> str:
    match = re.search(r'(?:correct stance is|should be|stance is|correct classification is)\s*\**"?\+?(Favor|Against|None)', challenge_text, re.IGNORECASE)
    if match:
        return match.group(1).title()
    return None

# ── Rate Limiting (Cerebras) ──────────────────────────────────────────────────
def load_request_timestamps(log_path: str) -> list[float]:
    t_list = []
    if not os.path.exists(log_path):
        return t_list
    now = time.time()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = obj.get("timestamp")
                if ts and (now - ts < 3600):
                    t_list.append(ts)
            except Exception:
                pass
    return sorted(t_list)

def log_request_timestamp(log_path: str, ts: float) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": ts}) + "\n")

def check_cerebras_hourly_limit(log_path: str) -> None:
    global timestamps
    now = time.time()
    timestamps = [ts for ts in timestamps if now - ts < 3600]
    
    if len(timestamps) >= 148:
        oldest_ts = timestamps[0]
        sleep_time = oldest_ts + 3600 - now + 5.0
        resume_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + sleep_time))
        print(f"\n⏳ HOURLY RATE LIMIT WARNING: {len(timestamps)} requests in last 60m.")
        print(f"   Sleeping for {sleep_time:.1f}s (~{sleep_time/60:.1f} minutes).")
        print(f"   Script will resume at: {resume_time}\n")
        time.sleep(sleep_time)
        
        now = time.time()
        timestamps = [ts for ts in timestamps if now - ts < 3600]

# ── Inference wrappers ────────────────────────────────────────────────────────
def call_cerebras(prompt: str, user_msg: str, row_idx: int, max_tokens: int = 256) -> str:
    global timestamps
    
    # RPM limit pacing sleep
    time.sleep(12.5)
    
    # Hourly limit pacing check
    check_cerebras_hourly_limit(CEREBRAS_LOG)
    
    for attempt in range(4):
        try:
            resp = gemma_client.chat.completions.create(
                model="gemma-4-31b",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                reasoning_effort="none"
            )
            now_ts = time.time()
            log_request_timestamp(CEREBRAS_LOG, now_ts)
            timestamps.append(now_ts)
            
            return resp.choices[0].message.content or ""
        except Exception as e:
            exc_str = str(e)
            retry_after = 15.0 * (2 ** attempt)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 60.0
            print(f"    ⚠️ Cerebras Row {row_idx} attempt {attempt+1} failed: {e}. Sleeping {retry_after}s...")
            time.sleep(retry_after)
            
    return "ERROR: all attempts failed"

def call_groq(prompt: str, user_msg: str, row_idx: int) -> str:
    # Groq RPM pacing sleep
    time.sleep(6.0)
    
    for attempt in range(3):
        try:
            resp = qwen_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                max_tokens=256,
                reasoning_effort="none"
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            exc_str = str(e)
            retry_after = 10.0 * (2 ** attempt)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "too_many_requests" in exc_str.lower():
                retry_after = 120.0
            print(f"    ⚠️ Groq Row {row_idx} attempt {attempt+1} failed: {e}. Sleeping {retry_after}s...")
            time.sleep(retry_after)
            
    return "ERROR: all attempts failed"

# ── Workers (Stage 1 and Stage 2 run in parallel) ─────────────────────────────
def stage1_worker(limit: int, s1_path: str):
    print("🚀 Stage 1 Worker (Cerebras) started.")
    s1_new = 0
    s1_start = time.time()
    
    for idx in range(limit):
        with s1_lock:
            if idx in s1_completed:
                continue
                
        row = rows[idx]
        tweet = row["tweet_text"]
        target = TARGET_AR.get(row["target"], row["target"])
        user_msg = f"Tweet: {tweet}\nTarget: {target}"
        
        raw_resp = call_cerebras(STAGE1_SYSTEM_PROMPT, user_msg, idx, max_tokens=256)
        stance, argument = parse_stage1_response(raw_resp)
        
        with s1_lock:
            s1_completed[idx] = {"stage1_stance": stance, "stage1_argument": argument}
            
        with s1_csv_lock:
            with open(s1_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["row_id", "tweet_text", "target", "stage1_stance", "stage1_argument"])
                writer.writerow({
                    "row_id": idx,
                    "tweet_text": tweet,
                    "target": row["target"],
                    "stage1_stance": stance,
                    "stage1_argument": argument
                })
        s1_new += 1
        if s1_new % 25 == 0 or idx == limit - 1:
            elapsed = time.time() - s1_start
            rate = elapsed / s1_new if s1_new > 0 else 0
            eta = rate * (limit - idx - 1)
            print(f"📈 [Stage 1] Processed {idx+1}/{limit} rows. Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
    print("✅ Stage 1 Worker completed.")

def stage2_worker(limit: int, s2_path: str):
    print("🚀 Stage 2 Worker (Groq) started.")
    s2_new = 0
    s2_start = time.time()
    
    for idx in range(limit):
        with s2_lock:
            if idx in s2_completed:
                continue
                
        # Wait for Stage 1 to process this row
        while True:
            with s1_lock:
                ready = idx in s1_completed
            if ready:
                break
            time.sleep(1.0)
            
        row = rows[idx]
        tweet = row["tweet_text"]
        target = TARGET_AR.get(row["target"], row["target"])
        
        with s1_lock:
            s1_data = s1_completed[idx]
            s1_stance = s1_data["stage1_stance"]
            s1_arg = s1_data["stage1_argument"]
            
        user_msg = f"Tweet: {tweet}\nTarget: {target}\nOriginal classification: {s1_stance}\nOriginal argument: {s1_arg}"
        
        challenge = call_groq(STAGE2_SYSTEM_PROMPT, user_msg, idx)
        
        with s2_lock:
            s2_completed[idx] = challenge
            
        with s2_csv_lock:
            with open(s2_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["row_id", "stage2_challenge"])
                writer.writerow({
                    "row_id": idx,
                    "stage2_challenge": challenge
                })
        s2_new += 1
        if s2_new % 25 == 0 or idx == limit - 1:
            elapsed = time.time() - s2_start
            rate = elapsed / s2_new if s2_new > 0 else 0
            eta = rate * (limit - idx - 1)
            print(f"📈 [Stage 2] Processed {idx+1}/{limit} rows. Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
    print("✅ Stage 2 Worker completed.")

# ── Main Runner ────────────────────────────────────────────────────────────────
def main():
    global timestamps, rows
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", type=int, default=None, help="If set, run only on first N rows using test output names")
    args = parser.parse_args()

    s1_path = STAGE1_CSV if not args.test_run else STAGE1_CSV.replace(".csv", "_test.csv")
    s2_path = STAGE2_CSV if not args.test_run else STAGE2_CSV.replace(".csv", "_test.csv")
    s3_path = STAGE3_CSV if not args.test_run else STAGE3_CSV.replace(".csv", "_test.csv")
    final_path = FINAL_TXT if not args.test_run else FINAL_TXT.replace(".txt", "_test.txt")

    print(f"Loading inputs from {INPUT_CSV}...")
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    limit = args.test_run if args.test_run else len(rows)
    print(f"Total rows to process: {limit}")

    # Resume Stage 1 Progress
    if os.path.exists(s1_path):
        with open(s1_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s1_completed[int(r["row_id"])] = {
                    "stage1_stance": r["stage1_stance"],
                    "stage1_argument": r["stage1_argument"]
                }
        print(f"📂 Loaded {len(s1_completed)} completed rows for Stage 1.")
    else:
        with open(s1_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "tweet_text", "target", "stage1_stance", "stage1_argument"])

    # Resume Stage 2 Progress
    if os.path.exists(s2_path):
        with open(s2_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s2_completed[int(r["row_id"])] = r["stage2_challenge"]
        print(f"📂 Loaded {len(s2_completed)} completed rows for Stage 2.")
    else:
        with open(s2_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "stage2_challenge"])

    # Resume Stage 3 Progress
    if os.path.exists(s3_path):
        with open(s3_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s3_completed[int(r["row_id"])] = r["final_stance"]
        print(f"📂 Loaded {len(s3_completed)} completed rows for Stage 3.")
    else:
        with open(s3_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "tweet_id", "target", "final_stance"])

    # Load Cerebras timestamps
    timestamps = load_request_timestamps(CEREBRAS_LOG)
    print(f"Loaded {len(timestamps)} Cerebras request timestamps from the last 60 minutes.")

    t_start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: Run Stage 1 & Stage 2 in Parallel
    # ──────────────────────────────────────────────────────────────────────────
    if len(s1_completed) < limit or len(s2_completed) < limit:
        print("\n==================================================")
        print("PART 1: Stage 1 (Gemma) & Stage 2 (Qwen) running parallel")
        print("==================================================")
        
        t1 = threading.Thread(target=stage1_worker, args=(limit, s1_path))
        t2 = threading.Thread(target=stage2_worker, args=(limit, s2_path))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: Run Stage 3 Sequentially
    # ──────────────────────────────────────────────────────────────────────────
    if len(s3_completed) < limit:
        print("\n==================================================")
        print("PART 2: Stage 3 (Gemma Final Judge) running sequentially")
        print("==================================================")
        s3_new = 0
        s3_start = time.time()
        
        for idx in range(limit):
            if idx in s3_completed:
                continue
                
            row = rows[idx]
            tweet = row["tweet_text"]
            target = TARGET_AR.get(row["target"], row["target"])
            
            s1_stance = s1_completed[idx]["stage1_stance"]
            s1_arg = s1_completed[idx]["stage1_argument"]
            s2_challenge = s2_completed[idx]
            
            user_msg = f"Tweet: {tweet}\nTarget: {target}\nInitial classification: {s1_stance}\nInitial argument: {s1_arg}\nChallenge: {s2_challenge}"
            
            # Query Cerebras (max_tokens=10 for single word)
            raw_resp = call_cerebras(STAGE3_SYSTEM_PROMPT, user_msg, idx, max_tokens=10)
            
            final_stance = parse_stage3_response(raw_resp)
            s3_completed[idx] = final_stance
            s3_new += 1
            
            with open(s3_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["row_id", "tweet_id", "target", "final_stance"])
                writer.writerow({
                    "row_id": idx,
                    "tweet_id": row["tweet_id"],
                    "target": row["target"],
                    "final_stance": final_stance
                })
                
            if s3_new % 25 == 0 or idx == limit - 1:
                elapsed = time.time() - s3_start
                rate = elapsed / s3_new if s3_new > 0 else 0
                eta = rate * (limit - idx - 1)
                print(f"📈 [Stage 3] Processed {idx+1}/{limit} rows. Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    total_time = time.time() - t_start
    print(f"\n🎉 ALL RUNS COMPLETED IN {total_time:.1f} seconds (~{total_time/60:.1f} minutes).")
    
    # ──────────────────────────────────────────────────────────────────────────
    # POST-RUN STATISTICS & COMPARISONS
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Distribution
    predicted_labels = [s3_completed[i] for i in range(limit)]
    total = len(predicted_labels)
    
    counts = {"Favor": 0, "Against": 0, "None": 0}
    for p in predicted_labels:
        if p in counts:
            counts[p] += 1
            
    # True split: Favor 44.89% / Against 45.45% / None 9.66%
    true_split = {"Favor": 44.89, "Against": 45.45, "None": 9.66}
    
    print("\n" + "="*80)
    print("predicted Favor/Against/None distribution vs. true split:")
    print("="*80)
    print(f"{'Stance':<8} | {'True Split %':<15} | {'Debate Final %':<18} | {'Count':<6}")
    print("-"*80)
    for k in ["Favor", "Against", "None"]:
        p_pct = (counts[k] / total) * 100
        print(f"{k:<8} | {true_split[k]:<15.2f}% | {p_pct:<18.2f}% | {counts[k]:<6}")
    print("="*80)

    # 2. Compare against plain best_86_61 baseline
    baseline_labels = {}
    if os.path.exists(BASELINE_CSV):
        with open(BASELINE_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    baseline_labels[int(r["row_index"])] = r["predicted_label"]
                except (ValueError, KeyError):
                    pass
    
    flips = 0
    flip_breakdown = {
        "Favor -> Against": 0, "Favor -> None": 0,
        "Against -> Favor": 0, "Against -> None": 0,
        "None -> Favor": 0, "None -> Against": 0
    }
    
    for idx in range(limit):
        if idx in baseline_labels:
            base = baseline_labels[idx]
            final = s3_completed[idx]
            if base != final:
                flips += 1
                key = f"{base} -> {final}"
                if key in flip_breakdown:
                    flip_breakdown[key] += 1
                else:
                    flip_breakdown[key] = 1

    print("\n" + "="*80)
    print("FLIP STATISTICS VS. PLAIN best_86_61 BASELINE:")
    print("="*80)
    print(f"Total Flips: {flips} / {limit} ({ (flips / limit)*100 :.2f}%)")
    print("Direction Breakdown:")
    for k, v in sorted(flip_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k}: {v}")
    print("="*80)

    # 3. Self-Bias Check: Judge Siding
    sided_stage1 = 0
    sided_challenge = 0
    other_decision = 0
    disagreed_count = 0
    
    for idx in range(limit):
        s1_stance = s1_completed[idx]["stage1_stance"]
        final_stance = s3_completed[idx]
        challenge_txt = s2_completed[idx]
        qwen_stance = extract_qwen_stance(challenge_txt)
        
        if qwen_stance and qwen_stance != s1_stance:
            disagreed_count += 1
            if final_stance == s1_stance:
                sided_stage1 += 1
            elif final_stance == qwen_stance:
                sided_challenge += 1
            else:
                other_decision += 1
        else:
            if final_stance == s1_stance:
                sided_stage1 += 1
            else:
                sided_challenge += 1
                
    print("\n" + "="*80)
    print("JUDGE SELF-BIAS CHECK:")
    print("="*80)
    print(f"Judge remained consistent with its Stage 1 claim: {sided_stage1} / {limit} ({ (sided_stage1/limit)*100 :.2f}%)")
    print(f"Judge changed stance from Stage 1: {sided_challenge + other_decision} / {limit} ({ ((sided_challenge+other_decision)/limit)*100 :.2f}%)")
    if disagreed_count > 0:
        sided_stage1_disagree = 0
        sided_qwen_disagree = 0
        went_to_third = 0
        for idx in range(limit):
            s1_st = s1_completed[idx]["stage1_stance"]
            fin_st = s3_completed[idx]
            qw_st = extract_qwen_stance(s2_completed[idx])
            if qw_st and qw_st != s1_st:
                if fin_st == s1_st:
                    sided_stage1_disagree += 1
                elif fin_st == qw_st:
                    sided_qwen_disagree += 1
                else:
                    went_to_third += 1
                    
        print(f"\nWhen Qwen explicitly disagreed & proposed alternative ({disagreed_count} rows):")
        print(f"  - Judge sided with Stage 1 (self-claim): {sided_stage1_disagree} / {disagreed_count} ({ (sided_stage1_disagree/disagreed_count)*100 :.2f}%)")
        print(f"  - Judge sided with Qwen Challenge: {sided_qwen_disagree} / {disagreed_count} ({ (sided_qwen_disagree/disagreed_count)*100 :.2f}%)")
        print(f"  - Judge went to a third stance: {went_to_third} / {disagreed_count}")
    print("="*80)

    # 4. Format CodaBench predictions
    print(f"\nWriting CodaBench submission file to {final_path}...")
    write_pred_txt(predicted_labels, final_path)
    print("✅ CodaBench text file formatted and written successfully.")

if __name__ == "__main__":
    main()

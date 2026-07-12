# StanceEval-2026 — Track 1 Baseline Spec
**Scope:** Track 1 (seen targets) only. Track 2 (unseen) is a stretch goal, not committed.
**Deadline:** self-imposed July 18, 2026 (registration + blind test release July 20).
**Budget:** ~15-20 hrs total, background track.

---

## 0. Environment
Use a venv for this project, isolated from global Python — don't let AraBERT/transformers/torch versions collide with other projects (FeedMind, etc.):
```bash
python3 -m venv stanceeval-venv
source stanceeval-venv/bin/activate   # Windows: stanceeval-venv\Scripts\activate
pip install transformers torch datasets accelerate arabert openai
```
`arabert` package (`pip install arabert`) provides the official `ArabertPreprocessor` used in Approach A. `openai` client is used for the HF router calls in Approach B (OpenAI-compatible endpoint, not an actual OpenAI dependency).

## 1. Task
Given an Arabic tweet + a target (COVID-19 Vaccine / Digital Transformation / Women Empowerment), classify stance as **Favor / Against / None**.

Primary metric: **Favg2** = macro-F1 over Favor+Against, per topic, then averaged.
Secondary: **Favg3** = macro-F1 over all 3 classes.
Use the official eval script: https://github.com/StanceEval/stanceeval.github.io/tree/main/Evaluation%20Script — don't hand-roll the metric.

## 2. Data
Source: https://github.com/StanceEval/stanceeval.github.io/tree/main/MawqifV2
Local files already downloaded (both .csv):

| Split | Targets | Size | Status |
|---|---|---|---|
| Training | 3 seen targets | 3,502 | have it |
| Development | same 3 targets, seen during training | 619 | have it |
| Test (Track 1, seen target) | held-out | TBD | **not released yet** — drops July 20 with registration |

⚠️ Do not label the 3,502-row file "test" anywhere in code/filenames — it's train. Rename on disk now if it isn't already `train.csv` / `dev.csv`, to avoid a mixup once the real test file lands.

**Actual schema (confirmed from files):**
`ID, text, target, stance, stance:confidence, against_reason, favor_reason, none_reason, sarcasm, sarcasm:confidence, sentiment, sentiment:confidence, datetime, Date`

- `text`: raw Arabic tweet, includes emojis, hashtags, diacritics — normalize per-model (AraBERT preprocessor for Approach A, lighter cleanup for Approach B)
- `target`: **not canonically formatted** — seen as `"Women empowerment"`, `"Covid Vaccine"`, `"Digital Transformation"` with inconsistent casing. Map to one fixed string per target at load time (e.g. a small lookup dict) before tokenizing — don't pass raw strings through, or the model may see near-duplicate targets as distinct.
- `stance`: label — `Favor` / `Against` / `None`
- `stance:confidence`: float, annotator agreement strength. Not used in baseline v1, but a lever if dev results look noisy — filter low-confidence rows or use as sample weight.
- `against_reason` / `favor_reason` / `none_reason`: free text, only one populated per row matching the stance. Not fed to the classifier — save for error analysis in the paper.
- `sarcasm` (Yes/No) + `sarcasm:confidence`: auxiliary. Flag in the Qwen prompt that sarcasm can invert apparent sentiment vs. actual stance (e.g. clearly negative-toned text can still be labeled `None` if it's sarcastic and ambiguous about actual position) — zero-shot models tend to over-read literal sentiment as stance.
- `sentiment` + `sentiment:confidence`: auxiliary, not core to the task.
- `datetime` / `Date`: not needed for classification.

**Class imbalance is real** — e.g. Digital Transformation train split is ~77% Favor, ~11% None. Don't ignore this: use class-weighted loss (fine-tune) and note it explicitly when prompting the LLM (zero-shot models tend to under-predict rare classes without guidance).

## 3. Approach A — AraBERT fine-tune (all 4 variants)

**What CLS/SEP mean:** `[CLS]` is a special token prepended to every input; its final hidden state after the model runs is what feeds the classification head — it's the model's learned "summary" of the whole sequence. `[SEP]` separates two distinct text spans fed in together (here: target and tweet) so the model knows where one ends and the other begins. Both are inserted automatically by `tokenizer(target, tweet)` in pair mode — nothing to hand-write.

**Variants to test** (same fine-tune recipe, swap checkpoint):

| Model | HF ID | Size | Twitter-pretrained? |
|---|---|---|---|
| AraBERTv0.2-base | `aubmindlab/bert-base-arabertv02` | 543MB/136M | No |
| AraBERTv2-base | `aubmindlab/bert-base-arabertv2` | 543MB/136M | Yes* |
| AraBERTv0.2-Twitter-base | `aubmindlab/bert-base-arabertv02-twitter` | 543MB/136M | Yes (60M dialect tweets) |
| AraBERTv0.2-Twitter-large | `aubmindlab/bert-large-arabertv02-twitter` | 1.38GB/371M | Yes (60M dialect tweets) |

*Run order recommendation (not a hard requirement): **Twitter-base first** — cheapest run, domain-matched, gives you a fast signal before spending time on the other 3. Large-Twitter is the most expensive run (2.5x params) — do it last so it doesn't eat hours if the smaller variants already tell you what you need.

- **Input format:** sequence-pair — `[CLS] target [SEP] tweet [SEP]`, standard 3-way classification head
- **Preprocessing:** use AraBERT's own `arabert.preprocess.ArabertPreprocessor` (handles Farasa segmentation, normalization) — don't build a custom normalizer, it'll fight the tokenizer's expectations
- **Training:** joint model across all 3 targets (more data, shared representation) rather than per-target models
  - **Do not use the 619-row dev set for early stopping** — that causes leakage (the model gets tuned to the exact set you later report scores on). Instead, carve an internal validation split out of the 3,502 train rows: ~90/10 split → ~3,150 train / ~350 internal-val, stratified by `stance` and `target` if possible (`sklearn.model_selection.train_test_split(..., stratify=df[['stance','target']])`) so the imbalanced None class isn't skewed further by the split.
  - Use the internal-val split for early stopping / checkpoint selection during training.
  - Only run the final chosen checkpoint on the full 619-row `dev.csv` once, for the actual comparison metric that goes in `results_summary.csv` — keep dev untouched until then, across all 4 variants and Qwen, so the comparison is apples-to-apples.
  - lr 2e-5, batch size 16, 3-5 epochs, early stop on internal-val Favg2
  - class-weighted cross-entropy (weights inverse to class frequency, computed globally first — per-topic weighting is a fallback if global doesn't fix the low-None topics)
- **Output:** dev predictions → run through official eval script → Favg2/Favg3 per topic + average, per variant

## 4. Approach B — Qwen 2.5 72B zero-shot
- **Access:** Hugging Face Inference Providers router, routed to DeepInfra — only needs `HF_TOKEN`, no separate DeepInfra account/key:
  ```python
  import os
  from openai import OpenAI
  client = OpenAI(
      base_url="https://router.huggingface.co/v1",
      api_key=os.environ["HF_TOKEN"],
  )
  completion = client.chat.completions.create(
      model="Qwen/Qwen2.5-72B-Instruct:deepinfra",
      messages=[...]
  )
  ```
  Billing: HF applies monthly free Inference Providers credits first, then pay-as-you-go on your HF account at DeepInfra's rate. Given the volume here (619 short prompts), this may stay entirely within free credits — check your HF billing page after a first small batch to confirm before running the full set.
- **Run only on the 619 dev rows** — this is zero-shot inference, no training, so the 3,502 train rows are irrelevant to this approach (they're only used for Approach A fine-tuning)
- No fine-tuning, inference only
- **Generation settings:** `temperature=0` (or 0.1 max — want deterministic, repeatable labels, not sampling variety), `max_tokens=20` (output is a tiny JSON object, no reason to allow more)
- **Concurrency/reliability:** 619 sequential calls is well within DeepInfra's capacity, no expected timeout issues — but run with modest concurrency (5-10 parallel requests, not all 619 at once) for speed, and add basic retry-with-backoff on failures (timeout/429/500) as standard practice
- **Cost estimate:** ~300-450 input tokens/call (Arabic tokenizes less efficiently than English) + ~15 output tokens, × 619 calls ≈ 250K input / 10K output tokens → roughly **$0.05-0.15** at DeepInfra's published rate, likely absorbed entirely by HF's free monthly Inference Providers credits at this volume
- **Prompt — exact text, two framings to A/B test on a ~50-row subset first:**

  **Framing 1 (English instructions):**
  ```
  System:
  You are an expert annotator for Arabic stance detection. Given an Arabic tweet
  and a target topic, classify the writer's stance toward that target as exactly
  one of three labels:

  - Favor: the writer expresses support for or a positive position toward the target
  - Against: the writer expresses opposition to or a negative position toward the target
  - None: no clear stance — neutral, off-topic, or ambiguous. This includes cases
    where sarcasm makes the literal tone misleading about the writer's actual
    position — do not infer a stance from tone alone if the underlying position
    isn't clear.

  Respond with ONLY a JSON object in this exact format, no other text:
  {"stance": "Favor"} or {"stance": "Against"} or {"stance": "None"}

  User:
  Target: {target}
  Tweet: {text}
  ```

  **Framing 2 (Arabic-translated system instructions):** same content translated to Arabic, but keep the JSON output schema in English (keys and values) — don't translate the output format itself, or parsing gets messy.

  Run both on the same ~50-row subset, compare JSON-parse success rate and label quality, commit to the winner for the full 619-row run. Log which framing was used per row in the output.
- **No training** — run directly on dev set, evaluate with the same official script
- **Log full prompt + response pairs** — useful for the custom-addition writeup later (e.g. error analysis: does the model over-predict Favor? confuse sarcasm for stance?)

## 5. Directory structure & output schema
Antigravity sets this up first — move existing CSVs into `data/`, create the rest empty:
```bash
mkdir -p data predictions checkpoints qwen_raw_logs
mv train.csv dev.csv data/
```
```
stanceeval2026/
├── data/
│   ├── train.csv
│   └── dev.csv
├── predictions/
│   ├── arabertv02_base_dev_preds.csv
│   ├── arabertv2_base_dev_preds.csv
│   ├── arabertv02_twitter_base_dev_preds.csv
│   ├── arabertv02_twitter_large_dev_preds.csv
│   └── qwen25_72b_zeroshot_dev_preds.csv
├── checkpoints/               # AraBERT weights — see retention note below
├── qwen_raw_logs/
│   └── qwen25_72b_zeroshot_raw.jsonl
├── results_summary.csv
└── error_analysis.md
```

**Prediction file schema — same columns across all 5 files, kept lean:**
```
ID, text, target, true_stance, predicted_stance, model_name
```
Don't duplicate `stance:confidence`, `*_reason`, `sentiment*` columns into these — they already live in `data/dev.csv` keyed by `ID`. Join on `ID` when doing error analysis instead of copying static data 5 times.

**Qwen raw log — JSONL, separate from predictions:**
```json
{"ID": 3, "prompt_framing": "en", "prompt": "...", "raw_response": "...", "parsed_stance": "None", "parse_success": true}
```
Keep every prompt+response pair even when parsing succeeds — this is the source material for error analysis (sarcasm confusion, over-prediction patterns, etc.), not just a debug log.

**results_summary.csv — one row per model, feeds the paper's results table directly:**
```
model_name, favg2_covid, favg2_digital, favg2_women, favg2_avg, favg3_covid, favg3_digital, favg3_women, favg3_avg
```
Generate by running the official eval script against each prediction file and appending a row.

**Checkpoint retention:** keep all 4 AraBERT checkpoints only until dev comparison is done and a winner (or top 1-2) is picked — then delete the rest. At least one checkpoint needs to survive past July 20 to run the actual blind test set.

## 6. Comparison & next step
Once both are scored on dev (Favg2/Favg3 per topic):
- If AraBERT-twitter clearly beats Qwen zero-shot → baseline is the fine-tune, LLM result becomes a reported comparison point in the paper
- If Qwen zero-shot is competitive or better → worth noting as the more surprising result, and reconsider whether the "custom addition" (budget: ~1 clear idea) should build on top of the LLM path instead (e.g. few-shot prompting, or fine-tuning Qwen via LoRA if compute allows) rather than the BERT path
- **Only then** decide whether MARBERT / JAIS runs are worth the remaining hours — don't run four models on principle, run what the dev results justify

## 7. Deliverables checklist
- [ ] Directory structure set up, CSVs moved to `data/`
- [ ] Data loader + preprocessing script (shared by both approaches where possible)
- [ ] Stratified internal train/val split (90/10) carved from train.csv for early stopping
- [ ] AraBERT fine-tune script, run across all 4 variants → checkpoints + prediction CSVs
- [ ] Qwen 2.5 72B zero-shot script via HF router → raw JSONL log + prediction CSV
- [ ] Eval script wired to official Favg2/Favg3 metric → results_summary.csv
- [ ] Delete losing AraBERT checkpoints once comparison is done, keep at least 1 for blind test
- [ ] Error analysis notes (for the paper's discussion section)
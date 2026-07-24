# StanceEval 2026 — Project State & End-to-End Pipeline Documentation

This file tracks the current state of the StanceEval 2026 Track 1 baseline implementation, detailed end-to-end data pipelines, verification audits, and active experimental results.

---

## 1. Directory Tree & File Inventory

Below is the directory tree of the workspace, including file descriptions, pipeline roles, and active statuses:

```
StanceEval/
├── .env                                # Active. Private tokens storage (GROQ_API_KEY, CEREBRAS_API_KEY).
├── .gitignore                          # Active. Ignores venvs, credentials, and model checkpoints.
├── error_analysis.md                  # Active. Fine-tuning error analysis report for AraBERT runs.
├── README.md                          # Active. Project readme documenting files, setup, running, and results.
├── requirements.txt                   # Active. Python package requirements.
├── results_summary.csv                # Active. Summary database of final scores (auto-updated by fine-tune).
├── stanceeval2026_baseline_spec.md    # Active. Baseline specifications and prompt instructions.
├── state.md                           # Active. End-to-end pipeline and status tracker (this file).
├── checkpoints/                        # Active. Holds fine-tuned model checkpoints (eval chooses best model).
│   ├── arabertv02_base/
│   ├── arabertv02_twitter_base/
│   ├── arabertv02_twitter_large/
│   └── arabertv02_twitter_large_fulldata/
├── data/                               # Active. Ground truth datasets.
│   ├── train.csv                      # (1.40 MB) 2,473 training tweets.
│   └── dev.csv                        # (254 KB) 619 validation tweets used as eval source.
├── Evaluation Script/                 # Active. Competition-provided evaluation framework.
│   └── evaluate.py                    # Official evaluation script used to grade submissions.
├── predictions/                        # Active. Evaluation outputs and prediction logs.
│   ├── arabertv02_base_dev_preds.csv
│   ├── arabertv02_base_dev_preds.txt
│   ├── arabertv02_twitter_base_dev_preds.csv
│   ├── arabertv02_twitter_base_dev_preds.txt
│   ├── arabertv02_twitter_large_dev_preds.csv
│   ├── arabertv02_twitter_large_dev_preds.txt
│   ├── arabertv02_twitter_large_fulldata_dev_preds.csv
│   ├── arabertv02_twitter_large_fulldata_dev_preds.txt
│   ├── results_qwen3_32b.csv          # Archived Qwen 3 32B (Groq) 100-row stratified subset predictions.
│   ├── results_qwen3.6_27b.csv        # Qwen 3.6 27B (Groq) 100-row stratified subset predictions.
│   ├── results_gemma4_31b_cerebras.csv# Gemma 4 31B (Cerebras) 100-row stratified subset predictions.
│   └── results_gemma4_31b_cerebras_full.csv # Gemma 4 31B (Cerebras) full 619-row dev set predictions.
├── qwen_raw_logs/                      # Active. Raw API request/response completion logs.
│   ├── results_qwen3_32b_raw.jsonl
│   ├── results_qwen3.6_27b_raw.jsonl
│   ├── results_gemma4_31b_cerebras_raw.jsonl
│   └── results_gemma4_31b_cerebras_full_raw.jsonl # Raw logs for full Gemma run.
└── src/                                # Active. Source code modules.
    ├── arabert_finetune.py            # Core AraBERT fine-tuning workflow script.
    ├── check_partial_progress.py      # Standalone progress checker for in-flight Gemma dev runs.
    ├── data_utils.py                  # Data loading, splitting, and preprocessing utilities.
    ├── evaluate.py                    # Script to evaluate saved AraBERT checkpoints against dev set.
    ├── gemma_inference.py             # Pipeline for running Gemma 4 zero-shot inference on Cerebras.
    ├── metrics.py                     # Single source of truth for metric scoring (Favg2/Favg3).
    ├── qwen_inference.py              # Pipeline for running Qwen zero-shot inference on Groq.
    └── verify_metrics.py              # Verification suite validating math equations in metrics.py.
```

---

## 2. End-to-End Pipeline Data Flow

Our project uses three distinct methodologies. Each progresses from raw input to final scored output through a highly standardized data pipeline.

### Method 1: AraBERT Fine-Tuning
1. **Data Loading:** `src/arabert_finetune.py` loads `data/train.csv` (2,473 rows) via `data_utils.load_data()`.
2. **Splitting:** A 90/10 stratified split splits the data into internal train (2,225 rows) and internal val (248 rows) sets via `data_utils.make_internal_split()`.
3. **Training & Validation:** Training occurs on the internal train set. Token validation happens at the end of each epoch using the internal val set to score `favg2`. Early stopping selects the best checkpoint based on validation loss.
4. **Dev Evaluation:** The best checkpoint loads into memory. `data/dev.csv` (619 rows) loads. Inference is run sequentially over all 619 dev rows.
5. **Output Writing:** Position-aligned predictions are written to `predictions/{model}_dev_preds.csv` (for analysis) and `predictions/{model}_dev_preds.txt` (one label per line).
6. **Scoring:** The script calls `metrics.run_official_eval()`, which triggers `Evaluation Script/evaluate.py` to evaluate the predictions against `dev.csv` and appends final metrics to `results_summary.csv`.

### Method 2: Qwen Zero-Shot via Groq (32B / 27B)
1. **Data Loading:** `src/qwen_inference.py` loads `data/dev.csv` via `data_utils.load_data()`.
2. **Resampling / Stratification:** Target counts in `dev.csv` are checked. Because targets are skewed in the first 100 rows, stratified sampling triggers, picking the first 33 COVID-19 Vaccine, 33 Digital Transformation, and 34 Women Empowerment rows (100 total rows) sorting to maintain original index order.
3. **Inference Loop:** The script initiates a sequential, single-threaded query loop. Groq is called with `reasoning_effort="none"`, `temperature=0.7`, and `top_p=0.8` using `x-ratelimit-remaining` response headers to compute pacing delays.
4. **Incremental Storage:** Each row completion is instantly appended to `predictions/results_qwen{version}.csv`. Raw response logs are appended to `qwen_raw_logs/results_qwen{version}_raw.jsonl`.
5. **Evaluation:** `predictions/results_qwen{version}.csv` is read using `keep_default_na=False` to prevent NaN coercion. `metrics.favg2()` and `metrics.per_topic_and_overall_metrics()` compute final scores on the 100-row subset.

### Method 3: Gemma Zero-Shot via Cerebras (4 31B)
1. **Data Loading:** `src/gemma_inference.py` loads `data/dev.csv` via `data_utils.load_data()`.
2. **Resampling / Stratification:** If running subset (default), reuses the identical stratified 100-row selection logic (COVID-19 Vaccine: 33, Digital Transformation: 33, Women Empowerment: 34) ensuring perfect alignment. If running full dev set (`--full`), processes all 619 rows in their original order.
3. **Inference Loop:** The script connects to Cerebras API using `reasoning_effort="none"`, `temperature=1.0`, and `top_p=0.95`. Calls are sequentially throttled at 5 RPM (using a hard floor delay of 12.0 seconds) and capped at 150 requests per hour (using rolling hourly tracking to sleep until oldest requests age out). In case of minute budget depletion, it sleeps 30-60 seconds.
4. **Incremental Storage:** Completed items write instantly to `predictions/results_gemma4_31b_cerebras_full.csv` (or `results_gemma4_31b_cerebras.csv` for the subset) and corresponding log `results_gemma4_31b_cerebras_full_raw.jsonl` (or `results_gemma4_31b_cerebras_raw.jsonl` for the subset) with timestamp annotations to support state-aware resumability across restarts.
5. **Evaluation:** Predictions are read back using `keep_default_na=False`. `metrics.favg2()` and `metrics.per_topic_and_overall_metrics()` calculate full set or subset score statistics.

---

## 3. Results Summary Database

Primary evaluation metric: **Overall Favg2** (Macro-F1 score over Favor and Against, averaged across COVID-19 Vaccine, Digital Transformation, and Women Empowerment).

### Full-Dev Set runs (619 rows)
| Model / Fine-Tuning Setup | COVID-19 Favg2 | Digital Favg2 | Women Favg2 | Overall Favg2 | Overall Favg3 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Reference: Qwen 2.5 (72B, zero-shot)* | - | - | - | **84.25** | - | *Reference Target* |
| *Reference: AraBERT-twitter* | - | - | - | **83.90** | - | *Reference Target* |
| **AraBERTv0.2-Twitter-Large** (90/10 split) | 0.8313 | 0.7282 | 0.8581 | **83.46** | 0.7157 | **Completed** — best checkpoint |
| **AraBERTv0.2-Twitter-Large** (100% data) | 0.8231 | 0.7167 | 0.8602 | **83.26** | 0.7233 | **Completed** |
| **AraBERTv0.2-Twitter-Base** (Run 2) | 0.7808 | 0.7694 | 0.8627 | **82.43** | 0.7129 | **Completed** |
| **AraBERTv0.2-Twitter-Base** (Run 1) | 0.7825 | 0.7666 | 0.8480 | **82.08** | 0.7139 | **Completed** |
| **AraBERTv0.2-Base** (Non-Twitter) | 0.7231 | 0.7160 | 0.8375 | **78.55** | 0.6677 | **Completed** |
| **Gemma 4 31B** (Cerebras, zero-shot) | 0.9066 | 0.7685 | 0.8996 | **0.8848** | 0.7820 | **Completed** — best overall score |

### Zero-Shot LLM Subset runs (100-row stratified subset)
| Model / API Provider | COVID-19 Favg2 | Digital Favg2 | Women Favg2 | Overall Favg2 | Overall Favg3 | Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 4 31B** (Cerebras, revised prompt, T=0.1) | **0.8857** | **0.8583** | **1.0000** | **0.9365** | **0.8542** | **90%** | **Completed (best Gemma config)** |
| **Gemma 4 31B** (Cerebras, prompt v1) | 0.8857 | 0.8583 | 0.9654 | **0.9249** | 0.8235 | 88% | **Completed** |
| **Gemma 4 31B** (Cerebras, prompt v3) | 0.8730 | 0.6600 | 0.9583 | **0.9094** | 0.7729 | 86% | **Completed** |
| **Qwen 3 32B** (Groq, prompt v3) | 0.7376 | 0.7917 | 0.9137 | **0.8550** | 0.7429 | **81%** | **Completed (best Qwen config)** |
| **Qwen 3 32B** (Groq, prompt v2) | 0.7529 | 0.7681 | 0.8904 | **0.8424** | 0.7225 | 79% | **Completed** |
| **Qwen 3 32B** (Groq, prompt v1) | 0.7141 | **0.8583** | 0.9178 | **0.8485** | 0.7273 | 79% | **Completed** |
| **Qwen 3.6 27B** (Groq) | 0.7048 | 0.6875 | 0.9227 | **0.8270** | 0.7365 | 78% | **Completed** |

---

## 4. Leakage & Correctness Audit Findings

To ensure experimental integrity before the July 18 deadline, we conducted a programmatic pipeline audit:

*   **Audit Check 1: AraBERT Validation Isolation (No Leakage)**
    *   *Finding:* **PASS**
    *   *Verification:* Traced `src/arabert_finetune.py` (lines 319-328, 467). Training utilizes ONLY `train.csv`. The validation set is drawn entirely from a 90/10 stratified split of `train.csv`. `dev.csv` is loaded into memory only *after* training has completely finished.
*   **Audit Check 2: Dataset Order Consistency**
    *   *Finding:* **PASS**
    *   *Verification:* Checked loader functions in all 3 pipelines. They load the same `data/dev.csv` file using `keep_default_na=False` and preserve original index ordering.
*   **Audit Check 3: Zero-Shot Subset Row Index Alignment**
    *   *Finding:* **PASS**
    *   *Verification:* Programmatically diffed `results_qwen3_32b.csv`, `results_qwen3.6_27b.csv`, and `results_gemma4_31b_cerebras.csv`. All three subsets are exactly 100 rows long and contain the identical list of row indices in identical order (first: `0, 1, 2, 3, 4`; last: `115, 119, 122, 127, 128`).
*   **Audit Check 4: Scoring Formula Consistency**
    *   *Finding:* **PASS**
    *   *Verification:* Verified imports in `arabert_finetune.py` (line 67), `qwen_inference.py` (line 61), and `gemma_inference.py` (line 36). All three scripts call `src/metrics.py` directly for Favg2/Favg3 evaluation.
*   **Audit Check 5: NaN Coercion ("None" stance bug) Prevention**
    *   *Finding:* **PASS**
    *   *Verification:* Ran a global workspace grep search for `read_csv`. Every active script (`evaluate.py` line 34, `data_utils.py` line 91, `qwen_inference.py` line 515, `gemma_inference.py` line 425) includes the `keep_default_na=False` guard.
*   **Audit Check 6: Summary Database Integrity (Duplicate check)**
    *   *Finding:* **WARNING**
    *   *Verification:* Inspected `results_summary.csv` and found a duplicate row for `arabertv02_twitter_base` (Run 1) at lines 2 and 3. Checked that it does not affect runtime behaviour but is recommended to clean up.


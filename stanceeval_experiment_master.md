# StanceEval 2026 Experiment Master Record

This document serves as the single source of truth for the experimental history, parameters, prompt versions, and results across both the Development and Evaluation Phases of the StanceEval 2026 Track 1 baseline implementation.

---

## 1. Task and Dataset

The task requires classifying the writer's stance toward three topics in Saudi Arabic tweets: **COVID-19 Vaccine (لقاح كورونا)**, **Digital Transformation (التحول الرقمي)**, and **Women Empowerment (تمكين المرأة)**. 

### Dataset Statistics
*   **Training Set (`train.csv`):** 3,502 rows containing gold labels. Used exclusively for AraBERT fine-tuning.
    *   *Source:* [data/train.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/train.csv) | Verified via `len(pd.read_csv())`
*   **Development Set (`dev.csv`):** 619 rows containing gold labels. Used for validation, error analysis, and zero-shot system comparison.
    *   *Source:* [data/dev.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/dev.csv) | Verified via `len(pd.read_csv())`
*   **Blind Test Set (`ground_truth.csv`):** 352 rows without public gold labels. Used for final CodaBench evaluation.
    *   *Source:* [data/ground_truth.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/ground_truth.csv) | Verified via `len(pd.read_csv())`

### Known True Test Set Distribution
*   **Favor:** 44.89%
*   **Against:** 45.45%
*   **None:** 9.66%
    *   *Source:* [run_allam_test.py#L241-246](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/run_allam_test.py#L241-246)

---

## 2. Development Phase

The Development Phase utilized the 619-row validation set (`dev.csv`) to benchmark and iterate on models before testing on the blind test set.

### 2.1 Submission History
The best known result on the Development Set was achieved by **Gemma 4 31B (Cerebras, zero-shot)**.

| Metric | Score | Verification Status | Source |
| :--- | :---: | :--- | :--- |
| **Overall FAVG2** | **0.8848** (0.884824) | `VERIFIED` | [results_summary.csv#L6](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L6) |
| **Overall FAVG3** | **0.7820** (0.781973) | `VERIFIED` | [results_summary.csv#L6](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L6) |
| **Overall Accuracy** | **85.95%** (0.8595) | `VERIFIED` | Local prediction evaluation via `scratch/analyze_dev_errors.py` |
| **COVID FAVG2** | **0.9066** (0.906609) | `VERIFIED` | [results_summary.csv#L6](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L6) |
| **Digital FAVG2** | **0.7685** (0.768467) | `VERIFIED` | [results_summary.csv#L6](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L6) |
| **Women FAVG2** | **0.8996** (0.899564) | `VERIFIED` | [results_summary.csv#L6](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L6) |
| **Known Ranking** | **#2** | `REPORTED IN HISTORY` | Supervisor's CodaBench account records (reported in project notes) |

---

### 2.2 AraBERT Experiments
All AraBERT variants were fine-tuned on `train.csv` (using a 90/10 stratified split for validation and early stopping) and evaluated on the full `dev.csv` (619 rows).

*   **Training Parameters:**
    *   *Epochs:* `5` (with early stopping patience of `2`)
    *   *Batch Size:* `16` (gradient accumulation = 1)
    *   *Learning Rate:* `2e-5` (with weight decay of 0.01 and warmup ratio of 0.1)
    *   *Max Sequence Length:* `128`
    *   *Loss:* Class-weighted cross-entropy (Weights: `[Against: 1.1433, Favor: 0.5435, None: 3.5065]` derived dynamically from `train.csv`)
    *   *Seed:* `42`
    *   *Source:* [src/arabert_finetune.py#L278-284](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/arabert_finetune.py#L278-284)

| Model Registry Key | Pretrained Model Path | Favg2 Covid | Favg2 Digital | Favg2 Women | Overall Favg2 | Overall Favg3 | Status | Evidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **arabertv02_twitter_large** | `aubmindlab/bert-large-arabertv02-twitter` | 0.8313 | 0.7282 | 0.8581 | **0.8346** | 0.7157 | `VERIFIED` | [results_summary.csv#L4](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L4) |
| **arabertv02_twitter_large_fulldata** | `aubmindlab/bert-large-arabertv02-twitter` (100% train) | 0.8231 | 0.7167 | 0.8602 | **0.8326** | 0.7233 | `VERIFIED` | [results_summary.csv#L5](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L5) |
| **arabertv02_twitter_base** | `aubmindlab/bert-base-arabertv02-twitter` | 0.7808 | 0.7694 | 0.8627 | **0.8243** | 0.7129 | `VERIFIED` | [results_summary.csv#L2](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L2) |
| **arabertv02_base** | `aubmindlab/bert-base-arabertv02` | 0.7231 | 0.7160 | 0.8375 | **0.7855** | 0.6677 | `VERIFIED` | [results_summary.csv#L3](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L3) |
| **arabertv2_base** | `aubmindlab/bert-base-arabertv2` | — | — | — | — | — | `UNRECOVERABLE` | Skipped due to Java dependency for Farasa |

---

### 2.3 Gemma Experiments
Zero-shot inference runs on Gemma 4 31B (Cerebras API) evaluated on the full `dev.csv` (619 rows) or the 100-row stratified subset.

*   **Decoding Settings:** `temperature=0.1` (greedy classifier logit base), `top_p=0.95`, `max_tokens=10`, `reasoning_effort="none"`.
*   **Hardware/Pacing:** Cerebras sequential loop with 12.0s floor delay (5 RPM limit) and rolling 150 requests/hour limit cooldown check.

| Model / Prompt Variant | Set Evaluated | Favg2 Covid | Favg2 Digital | Favg2 Women | Overall Favg2 | Overall Favg3 | Evidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 4 31B** (Prompt v1 + Arabic Targets) | 100-row Subset | 0.8857 | 0.8583 | 1.0000 | **0.9365** | 0.8542 | [results_summary.csv#L15](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L15) |
| **Gemma 4 31B** (Prompt v1) | 100-row Subset | 0.8857 | 0.8583 | 0.9654 | **0.9249** | 0.8235 | [results_summary.csv#L9](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L9) |
| **Gemma 4 31B** (Prompt v3) | 100-row Subset | 0.8730 | 0.6600 | 0.9583 | **0.9094** | 0.7729 | [results_summary.csv#L7](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L7) |
| **Gemma 4 31B** (Original prompt, T=1.0) | 619-row Dev | 0.9066 | 0.7685 | 0.8996 | **0.8848** | 0.7820 | [results_summary.csv#L6](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L6) |
| **Gemma 4 31B** (Original prompt, Arabic Targets, T=1.0) | 619-row Dev | 0.9066 | 0.7785 | 0.8970 | **0.8844** | 0.7816 | [results_summary.csv#L16](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L16) |

---

### 2.4 Qwen Experiments
Zero-shot inference runs on Qwen models (Groq API) evaluated on the 100-row stratified subset.

*   **Decoding Settings:** `temperature=0.7`, `top_p=0.8`, `max_tokens=256` or `10`.

| Model / Prompt Variant | Favg2 Covid | Favg2 Digital | Favg2 Women | Overall Favg2 | Overall Favg3 | Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen 3 32B** (Groq, prompt v3) | 0.7376 | 0.7917 | 0.9137 | **0.8550** | 0.7429 | [results_summary.csv#L11](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L11) |
| **Qwen 3 32B** (Groq, prompt v1) | 0.7141 | 0.8583 | 0.9178 | **0.8485** | 0.7273 | [results_summary.csv#L12](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L12) |
| **Qwen 3 32B** (Groq, prompt v2) | 0.7529 | 0.7681 | 0.8904 | **0.8424** | 0.7225 | [results_summary.csv#L13](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L13) |
| **Qwen 3.6 27B** (Groq) | 0.7048 | 0.6875 | 0.9227 | **0.8270** | 0.7365 | [results_summary.csv#L10](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L10) |

---

### 2.5 Few-shot Experiments
Gemma 4 31B evaluated on the 100-row stratified subset.

| Model / Configuration | Favg2 Covid | Favg2 Digital | Favg2 Women | Overall Favg2 | Overall Favg3 | Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 4 31B** (Prompt v4 few-shot) | 0.8444 | 0.8694 | 0.9872 | **0.9240** | 0.8229 | [results_summary.csv#L17](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L17) |
| **Gemma 4 31B** (Prompt v5 + Arabic targets) | 0.8571 | 0.8468 | 0.9872 | **0.9178** | 0.8202 | [results_summary.csv#L14](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv#L14) |

---

### 2.6 Development Error Analysis (Gemma-4-31B Full Dev)
Quantitative metrics for the best development system (Gemma-4-31B on the 619-row dev set) generated via `scratch/analyze_dev_errors.py`:

*   **Overall Accuracy:** **85.95%** (532 / 619)
*   **Confusion Matrix:**
    ```
                  Pred_Favor  Pred_Against  Pred_None
    True_Favor           330            28         22
    True_Against           9           168          3
    True_None             15            10         34
    ```
*   **Per-Class Metrics:**
    *   *Favor:* Precision = 93.22%, Recall = 86.84%, F1 = 89.92% (Support: 380)
    *   *Against:* Precision = 81.55%, Recall = 93.33%, F1 = 87.05% (Support: 180)
    *   *None:* Precision = 57.63%, Recall = 57.63%, F1 = 57.63% (Support: 59)
*   **Top Error Transition Types:**
    *   `Favor -> Against` (28 occurrences): Confusing positive stance with opposition (often due to sarcasm or hijack hashtags).
    *   `Favor -> None` (22 occurrences): Misinterpreting positive logistics or implementation complaints.
    *   `None -> Favor` (15 occurrences): Conflating target-referencing descriptions with positive stance.
    *   `None -> Against` (10 occurrences): Misinterpreting negative vocabulary in objective tweets.

---

### 2.7 Best Development System
The strongest system on the Development Set was **Gemma 4 31B (Cerebras, zero-shot, prompt v1)** which scored **0.8848** overall FAVG2 on the full 619 rows, and **0.9365** on the 100-row subset when targets were translated to Arabic.

---

## 3. Evaluation Phase

The Evaluation Phase utilized the 352-row blind test set (`ground_truth.csv`) with submissions made directly to CodaBench.

### 3.1 Submission History
Exhaustive compilation of all 39 submissions to CodaBench:

| ID | Date | Filename | Overall Favg2 | Platform Status | Evidence |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **879203** | 2026-08-05 15:49 | `confidence_thresholded_submission.zip` | **0.8548** | Finished | [predictions/confidence_thresholded_submission.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/confidence_thresholded_submission.txt) |
| **879155** | 2026-08-05 15:23 | `majority_fallback_sol.zip` | **0.8620** | Finished (Best Ensemble) | [predictions/majority_fallback_sol.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/majority_fallback_sol.txt) |
| **879154** | 2026-08-05 15:23 | `majority_fallback_luna.zip` | **0.8550** | Finished | [predictions/majority_fallback_luna.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/majority_fallback_luna.txt) |
| **879153** | 2026-08-05 15:23 | `majority_fallback_gemma.zip` | **0.8600** | Finished | [predictions/majority_fallback_gemma.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/majority_fallback_gemma.txt) |
| **879152** | 2026-08-05 15:23 | `disagree_pick_sol.zip` | **0.8308** | Finished | [predictions/disagree_pick_sol.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/disagree_pick_sol.txt) |
| **879151** | 2026-08-05 15:23 | `disagree_pick_luna.zip` | **0.8341** | Finished | [predictions/disagree_pick_luna.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/disagree_pick_luna.txt) |
| **879149** | 2026-08-05 15:22 | `disagree_pick_gemma.zip` | **0.8661** | Finished | [predictions/disagree_pick_gemma.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/disagree_pick_gemma.txt) |
| **879138** | 2026-08-05 15:13 | `results_sol_openai_test_ci_v3_batch.zip` | **0.8308** | Finished | [predictions/results_sol_openai_test_ci_v3_batched.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_sol_openai_test_ci_v3_batched.txt) |
| **879123** | 2026-08-05 14:58 | `results_luna_openai_test_ci_v3_batc.zip` | **0.8341** | Finished | [predictions/results_luna_openai_test_ci_v3_batched.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_luna_openai_test_ci_v3_batched.txt) |
| **872683** | 2026-08-01 05:14 | `agree_or_baseline_test.zip` | **0.8661** | Finished | [predictions/agree_or_baseline_test.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/agree_or_baseline_test.txt) |
| **872675** | 2026-08-01 05:01 | `majority_vote_test.zip` | **0.8478** | Finished | [predictions/majority_vote_test.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/majority_vote_test.txt) |
| **872660** | 2026-08-01 04:47 | `results_gemma4_31b_cerebras_test_ex.zip` | **0.8296** | Finished | [predictions/results_gemma4_31b_cerebras_test_explain_then_classify_v1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_explain_then_classify_v1.txt) |
| **872420** | 2026-08-01 00:36 | `results_gemma4_31b_cerebras_test_ev.zip` | **0.8613** | Finished | [predictions/results_gemma4_31b_cerebras_test_event_favor_clause_v2.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_event_favor_clause_v2.txt) |
| **872418** | 2026-08-01 00:34 | `debate_stage1.zip` | **0.8352** | Finished | [predictions/debate_stage1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/debate_stage1.txt) |
| **872386** | 2026-07-31 23:53 | `results_gemma4_31b_cerebras_test_pe.zip` | **0.8528** | Finished | [predictions/results_gemma4_31b_cerebras_test_personal_action_clause_v1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_personal_action_clause_v1.txt) |
| **872336** | 2026-07-31 22:41 | `debate_stage1.zip` | **0.8352** | Finished (Duplicate) | [predictions/debate_stage1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/debate_stage1.txt) |
| **872279** | 2026-07-31 21:31 | `results_gemma4_31b_cerebras_test_ev.zip` | **0.8613** | Finished (Duplicate) | [predictions/results_gemma4_31b_cerebras_test_event_favor_clause_v2.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_event_favor_clause_v2.txt) |
| **871980** | 2026-07-31 17:15 | `results_gemma4_31b_cerebras_test_sa.zip` | **0.8573** | Finished | [predictions/results_gemma4_31b_cerebras_test_sarcasm_direction_v1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_sarcasm_direction_v1.txt) |
| **870992** | 2026-07-31 03:36 | `debate_final.zip` | **0.8123** | Finished | [predictions/debate_final.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/debate_final.txt) |
| **869250** | 2026-07-30 01:38 | `results_two_stage_gemma_352.zip` | **0.7414** | Finished | [predictions/results_two_stage_gemma_352.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_two_stage_gemma_352.txt) |
| **867568** | 2026-07-29 01:46 | `results_gemma4_31b_cerebras_test_ci.zip` | **0.8538** | Finished (CI v3.1) | [predictions/results_gemma4_31b_cerebras_test_ci_v3_1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v3_1.txt) |
| **867227** | 2026-07-28 20:08 | `results_gemma4_31b_cerebras_test_ef.zip` | **0.8585** | Finished (EF-I) | [predictions/results_gemma4_31b_cerebras_test_ef_i.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ef_i.txt) |
| **866950** | 2026-07-28 16:54 | `kb_test_results.zip` | **0.8382** | Finished (KB tests) | [predictions/kb_test_results.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/kb_test_results.txt) |
| **865403** | 2026-07-27 16:45 | `results_gemma4_31b_sc_v1.zip` | **0.8506** | Finished (Self-Consistency) | [predictions/results_gemma4_31b_final_sc.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_final_sc.txt) |
| **864982** | 2026-07-27 12:25 | `results_gemma4_31b_ci_h1_v2.zip` | **0.8489** | Finished | [predictions/results_gemma4_31b_ci_h1_v2.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_ci_h1_v2.csv) |
| **864490** | 2026-07-27 02:43 | `results_gemma4_31b_ci_ha_v1.zip` | **0.8420** | Finished | [predictions/results_gemma4_31b_ci_ha_v1.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_ci_ha_v1.csv) |
| **864371** | 2026-07-26 23:19 | `results_gemma4_31b_cerebras_test_ci.zip` | **0.8635** | Finished (CI v5) | [predictions/results_gemma4_31b_cerebras_test_ci_v5.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v5.csv) |
| **864239** | 2026-07-26 20:48 | `results_gemma4_31b_cerebras_test_ci.zip` | **0.8473** | Finished (CI v2) | [predictions/results_gemma4_31b_cerebras_test_ci_v2.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v2.csv) |
| **863999** | 2026-07-26 17:50 | `results_gemma4_31b_cerebras_test_ci.zip` | **0.8661** | Finished (CI v3, Champion) | [predictions/results_gemma4_31b_cerebras_test_ci_v3.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v3.txt) |
| **863330** | 2026-07-26 05:05 | `results_gemma4_31b_cerebras_test_ci.zip` | **0.8228** | Finished (CI v4) | [predictions/results_gemma4_31b_cerebras_test_ci_v4.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v4.csv) |
| **863078** | 2026-07-26 00:26 | `results_gemma4_31b_cerebras_test_ci.zip` | **0.8485** | Finished (CI v1) | [predictions/results_gemma4_31b_cerebras_test_ci_v1.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v1.txt) |
| **862754** | 2026-07-25 19:02 | `results_qwen_3_32b_test.zip` | **0.7074** | Finished | [predictions/results_qwen_3_32b_test.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_qwen_3_32b_test.csv) |
| **862695** | 2026-07-25 17:52 | `results_qwen_3.6_27b_test.zip` | **0.7098** | Finished | [predictions/results_qwen_3.6_27b_test.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_qwen_3.6_27b_test.csv) |
| **862582** | 2026-07-25 15:36 | `results_gemma4_31b_cerebras_test_re.zip` | **0.8376** | Finished (Revised v3) | [predictions/results_gemma4_31b_cerebras_test_revised_zeroshot.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_revised_zeroshot.csv) |
| **861982** | 2026-07-25 05:12 | `results_gemma4_31b_cerebras_test_re.zip` | **0.8409** | Finished (Revised v2) | [predictions/results_gemma4_31b_cerebras_test_revised_zeroshot_backup.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_revised_zeroshot_backup.csv) |
| **861684** | 2026-07-24 22:17 | `results_gemma4_31b_cerebras_test_re.zip` | **0.8444** | Finished (Revised v1) | [predictions/results_gemma4_31b_cerebras_test_revised_zeroshot_v2_backup.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_revised_zeroshot_v2_backup.csv) |
| **858731** | 2026-07-22 23:07 | `results_gemma4_31b_cerebras_test_re.zip` | **0.7940** | Finished (Refined) | [predictions/results_gemma4_31b_cerebras_test_refined.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_refined.csv) |
| **858470** | 2026-07-22 19:28 | `qwen3.6.zip` | **0.7759** | Finished | [predictions/agree_or_baseline_test.zip](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/agree_or_baseline_test.zip) |
| **857271** | 2026-07-22 00:37 | `1.zip` | **0.7108** | Finished | [predictions/agree_or_baseline_test.zip](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/agree_or_baseline_test.zip) |

---

### 3.2 Prompt Iterations
1.  **Refined Prompt:** Basic zero-shot framework containing only single-sentence definition guidelines. (Test Set FAVG2: **0.7940**)
2.  **Revised v1 Prompt:** Added the "Mental Translation" guideline to convert dialect/slang into modern Arabic prior to classification. (Test Set FAVG2: **0.8444**)
3.  **Revised v2 Prompt:** Added the "Face-Value Guard" to reduce false-positive Favor stances from neutral reporting. (Test Set FAVG2: **0.8409**)
4.  **Revised v3 Prompt:** Added specific target-resolution rules to restrict stance to the exact target itself. (Test Set FAVG2: **0.8376**)

---

### 3.3 CI v1
*   **Prompt changes:** Injected context about the historical Saudi driving decree (Background Context).
*   **FAVG2:** **0.8485**
*   **FAVG3:** **0.6702**
*   **Accuracy:** **80.97%**
*   *Evidence:* [predictions/gemma4_31b_test_ci_v1_flips_report.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/gemma4_31b_test_ci_v1_flips_report.md)

---

### 3.4 CI v2
*   **Prompt changes:** Added rules detailing how Against opposes the target and directed the model to filter general complaints about traffic or sibling banter into None.
*   **FAVG2:** **0.8473**
*   *Evidence:* [predictions/gemma4_31b_test_ci_v2_flips_report.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/gemma4_31b_test_ci_v2_flips_report.md)

---

### 3.5 CI v3 (Champion Zero-Shot System)
*   **Prompt changes:** Calibrated positive framing guidelines, clarifying that positive news reporting (police distributing roses, license celebration) implies Favor.
*   **FAVG2:** **0.8661** (0.8661)
*   **FAVG3:** **0.6746** (0.6746)
*   **Accuracy:** **82.67%** (0.8267)
*   *Evidence:* [README.md#L40-43](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/README.md)

---

### 3.6 CI v5
*   **Prompt changes:** Attempted to add detailed instructions for Against (skepticism and doubt) and target normalization.
*   **FAVG2:** **0.8635**
*   *Evidence:* [predictions/results_gemma4_31b_cerebras_test_ci_v5.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v5.csv)

---

### 3.7 Negative Prompt Experiments
*   **CI v4:** Modified the prompt to simplify instructions and remove target context, which led to a catastrophic performance drop to **0.8228** FAVG2.
*   *Evidence:* [predictions/results_gemma4_31b_cerebras_test_ci_v4.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/results_gemma4_31b_cerebras_test_ci_v4.csv)

---

### 3.8 Knowledge/Context Injection
*   **CI v3.1:** Clarified that passing references to dates or unrelated personal commercial posts do not imply Favor, scoring **0.8538** FAVG2.
*   *Evidence:* [src/gemma_inference_test_ci_v3_1.py#L90](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/gemma_inference_test_ci_v3_1.py#L90)

---

### 3.9 Entity/KB Pipeline
*   **KB Run:** Zero-shot Gemma with hashtag-entity context injection scoring **0.8382** FAVG2.
*   *Evidence:* [predictions/kb_test_results.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/kb_test_results.txt)

---

### 3.10 ALLaM
*   **ALLaM-7B-Instruct:** Local pipeline running dialect-to-literal interpretations prior to classification, scoring **0.7414** FAVG2.
*   *Evidence:* [run_full_two_stage_352.py#L37](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/run_full_two_stage_352.py#L37)

---

### 3.11 Debate / Multi-stage Pipeline
*   **debate_final:** Gemma-Qwen-Gemma multi-agent debate pipeline, scoring **0.8123** FAVG2.
*   *Evidence:* [run_debate_pipeline.py#L208](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/run_debate_pipeline.py#L208)

---

### 3.12 GPT Experiments
*   **gpt-5.6-luna:** OpenAI batched JSON model, scoring **0.8341** FAVG2.
*   **gpt-5.6-sol:** OpenAI batched JSON model, scoring **0.8308** FAVG2.
*   *Evidence:* [src/openai_inference_test_ci_v3_batched.py#L35](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/openai_inference_test_ci_v3_batched.py#L35)

---

### 3.13 Ensemble Experiments
*   **majority_fallback_sol:** Best ensemble strategy (Gemma, Luna, Sol majority vote with Sol fallback on split), scoring **0.8620** FAVG2.
*   *Evidence:* [predictions/majority_fallback_sol.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions/majority_fallback_sol.txt)

---

### 3.14 CI v3 Error Analysis (Evaluation Phase)
Class prediction distribution of the CI v3 zero-shot Gemma model on the 352-row test set:
*   **Favor:** 156 (44.32%) — Expected: 44.89% (Delta: -0.57%)
*   **Against:** 182 (51.70%) — Expected: 45.45% (Delta: +6.25%)
*   **None:** 14 (3.98%) — Expected: 9.66% (Delta: -5.68%)
    *   *Source:* `scratch/analyze_eval_ci_v3.py`
    *   *Conclusion:* The None class was severely under-predicted, and Against was over-predicted on the evaluation test set.

---

## 4. Cross-Experiment Comparison

| Phase | Experiment | Model | Main Change | FAVG2 | FAVG3 | Accuracy | Status | Evidence |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Dev** | AraBERT-Base | `bert-base-arabertv02` | Joint multi-target training baseline | 0.7855 | 0.6677 | — | `VERIFIED` | `results_summary.csv#L3` |
| **Dev** | AraBERT-Twitter-Base | `bert-base-arabertv02-twitter` | Twitter-pretrained model base | 0.8243 | 0.7129 | — | `VERIFIED` | `results_summary.csv#L2` |
| **Dev** | AraBERT-Twitter-Large | `bert-large-arabertv02-twitter` | Stratified 90/10 training split check | 0.8346 | 0.7157 | — | `VERIFIED` | `results_summary.csv#L4` |
| **Dev** | AraBERT-Twitter-Large-Full | `bert-large-arabertv02-twitter` | Retraining on 100% training set | 0.8326 | 0.7233 | — | `VERIFIED` | `results_summary.csv#L5` |
| **Dev** | Gemma-ZeroShot-Base | `gemma-4-31b` | Zero-shot greedy inference baseline | 0.8848 | 0.7820 | 85.95% | `VERIFIED` | `results_summary.csv#L6` |
| **Eval** | Early Baseline | `gemma-4-31b` | Early zero-shot test run | 0.8063 | 0.6183 | 75.28% | `VERIFIED` | `README.md#L45` |
| **Eval** | Revised v1 | `gemma-4-31b` | Injected "Mental Translation" rules | 0.8444 | 0.6429 | 80.11% | `VERIFIED` | `README.md#L44` |
| **Eval** | Revised v2 | `gemma-4-31b` | Added "Face-Value Guard" | 0.8409 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_revised_zeroshot_backup.csv` |
| **Eval** | Revised v3 | `gemma-4-31b` | Target-specific resolution logic | 0.8376 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_revised_zeroshot.csv` |
| **Eval** | CI v1 | `gemma-4-31b` | Added Background Context ( decree context) | 0.8485 | 0.6702 | 80.97% | `VERIFIED` | `results_gemma4_31b_cerebras_test_ci_v1.txt` |
| **Eval** | CI v2 | `gemma-4-31b` | Added general complaint filters to Against | 0.8473 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_ci_v2.csv` |
| **Eval** | **CI v3 (Champion)** | `gemma-4-31b` | Calibrated positive framing guidelines | **0.8661** | **0.6746** | **82.67%** | `VERIFIED` | `README.md#L40-43` |
| **Eval** | CI v4 | `gemma-4-31b` | Simplified prompt, removed context | 0.8228 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_ci_v4.csv` |
| **Eval** | CI v5 | `gemma-4-31b` | Target normalization and doubt rules | 0.8635 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_ci_v5.csv` |
| **Eval** | CI v3.1 | `gemma-4-31b` | Incident reporting limits on Favor | 0.8538 | — | 79.43% | `VERIFIED` | `results_gemma4_31b_cerebras_test_ci_v3_1.csv` |
| **Eval** | EF-I | `gemma-4-31b` | Implementation complaints mapped to Favor | 0.8585 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_ef_i.csv` |
| **Eval** | Sarcasm Direction | `gemma-4-31b` | Directed sarcasm rules targeting opponents | 0.8573 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_sarcasm_direction_v1.csv` |
| **Eval** | Personal Action | `gemma-4-31b` | Logistics/access questions mapped to Favor | 0.8528 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_personal_action_clause_v1.csv` |
| **Eval** | Event Favor Clause | `gemma-4-31b` | Limits campaign events reporting | 0.8613 | — | — | `VERIFIED` | `results_gemma4_31b_cerebras_test_event_favor_clause_v2.csv` |
| **Eval** | KB Run | `gemma-4-31b` | Dynamic Entity/Hashtag KB prompt | 0.8382 | — | — | `VERIFIED` | `kb_test_results.csv` |
| **Eval** | Self-Consistency | `gemma-4-31b` | SC early stopped (T=0.1, 0.5, 1.0) | 0.8506 | — | — | `VERIFIED` | `results_gemma4_31b_sc_v1.csv` |
| **Eval** | ALLaM Stage 2 | `allam-7b` | Local rephraser + Gemma classifier | 0.7414 | — | — | `VERIFIED` | `results_two_stage_gemma_352.csv` |
| **Eval** | Debate Final | `gemma/qwen` | Parallel claim-challenge-judge debate | 0.8123 | — | — | `VERIFIED` | `debate_final.txt` |
| **Eval** | **Ensemble Best** | Gemma/Luna/Sol | Majority vote ensemble (Sol fallback) | **0.8620** | — | — | `VERIFIED` | `majority_fallback_sol.txt` |

---

## 5. Prompt Archive

### Gemma Champion Prompt (CI v3)
> *Source:* [src/gemma_inference_test_final_sc.py#L65-102](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/gemma_inference_test_final_sc.py#L65-102)
```
You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

### Task
Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:

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

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text.
```

---

## 6. Hyperparameters

*   **AraBERT Training:** Epochs = 5, Batch Size = 16, LR = 2e-5, Weight Decay = 0.01, Max Length = 128, stratified splits = 90/10. (Source: `src/arabert_finetune.py#L278-284`)
*   **Gemma zero-shot:** Temperature = 0.1, Top_p = 0.95, Max tokens = 10, reasoning_effort = "none". Hard floor throttle delay = 12s. (Source: `src/gemma_inference_test_ci_v3_1.py#L40-43`)
*   **OpenAI batching:** Batch size = 20, Temperature = 0.0, Max tokens = 4000. (Source: `src/openai_inference_test_ci_v3_batched.py#L35-38`)
*   **Self-Consistency Voting:** Pass 1 Temp = 0.1, Pass 2 Temp = 0.5, Pass 3 Temp = 1.0. Early stop on Pass 1/2 agreement, fallback to Pass 1 on 3-way split. (Source: `src/gemma_inference_test_final_sc.py#L394-428`)

---

## 7. Error Analysis (Qualitative Cases)

### Mode 1: Target-Resolution Failure (None classified as Favor/Against)
*   **ID: 157** | **Text:** "تمكين المراة" | **Gold:** `None` | **Predicted:** `Favor`
    *   *Analysis:* The tweet simply states the target topic name with no opinion or stance. The model failed target resolution and assigned `Favor` because of the raw target string.
*   **ID: 91** | **Text:** "ثانيا انا ما قلت ان النظام الالكتروني او التحول الرقمي ما ينجح انا قلت يحتاج اشياء لدعمة اكثر... الانتقال الالكتروني مجاله كبير و واسع..." | **Gold:** `None` | **Predicted:** `Favor`
    *   *Analysis:* Objective explanation of necessary infrastructure for digital transformation. The model is swayed by positive tokens ("مجاله كبير و واسع") and misclassifies it as support (`Favor`).

### Mode 2: Implementation Criticism Confused with Opposition (Favor classified as Against)
*   **ID: 530** | **Text:** "للاسف خدماتكم سيئة للغاية نريد نسجل لازم نروح مكتب سند ... أين التحول الإلكتروني" | **Gold:** `Favor` | **Predicted:** `Against`
    *   *Analysis:* Highly negative language complaining about administrative loops and demanding electronic transformation. The model misinterprets the negative tone as opposition to digital transformation rather than frustration at its slow execution.

### Mode 3: Factual Event Reporting Confused with Stance (None classified as Favor)
*   **ID: 127** | **Text:** "... وبلادنا هي اللي قررت تطعيم الشعب بلقاح كورونا ..." | **Gold:** `None` | **Predicted:** `Favor`
    *   *Analysis:* Explaining that the country decided to vaccinate citizens. The model treats mentioning the decision as an endorsement of the vaccine target.

---

## 8. Key Findings

1.  **Context-Injection Value:** Injecting decree-specific timeline context resolved a massive sarcasm class confusion, driving test set Favg2 up from **0.8063** to **0.8485** (CI v1).
2.  **Calibration over Simplification:** Adding positive news-reporting boundaries (CI v3) unlocked the top standalone system (**0.8661**), whereas simplifying prompts and removing context (CI v4) collapsed performance to **0.8228**.
3.  **Ensemble Fallbacks:** Combining local open-source API pipelines with closed-source batch annotators via majority vote with a Sol fallback achieved the best overall ensemble performance (**0.8620**).

---

## 9. Unverified / Missing Evidence
*   **arabertv2_base checkpoint & predictions:** No local files or prediction logs exist for this run, as training requires Java for Farasa segmentation which was unavailable on the execution machine. Mark as `UNRECOVERABLE`.
*   **supervisor account credentials/logs:** Leaderboard rankings from the development phase are logged as `REPORTED IN HISTORY` based on project records; direct credentials or session history for that account were not recovered.

# StanceEval 2026 — Project State

This file tracks the current state of the StanceEval 2026 Track 1 baseline implementation, progress against reference scores, and active workarounds.

---

## 1. Baseline Performance Summary (Development Set)

Primary evaluation metric: **Overall Favg2** (Macro-F1 score over Favor and Against, averaged across COVID-19 Vaccine, Digital Transformation, and Women Empowerment).

| Model / Source | COVID-19 Favg2 | Digital Favg2 | Women Favg2 | Overall Favg2 | Overall Favg3 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference: Qwen 2.5 (72B, zero-shot)** | - | - | - | **84.25** | - | *Reference Target* |
| **Reference: AraBERT-twitter** | - | - | - | **83.90** | - | *Reference Target* |
| **Reference: JAIS (70B, zero-shot)** | - | - | - | **83.37** | - | *Reference Target* |
| **Reference: MARBERT** | - | - | - | **81.91** | - | *Reference Target* |
| ─── *Our Fine-Tuned Runs* ─── | | | | | | |
| **AraBERTv0.2-Twitter-Large** | 0.8313 | 0.7282 | 0.8581 | **83.46** | 0.7157 | **Completed** (0.44% behind reference) |
| **AraBERTv0.2-Twitter-Base** (Run 2) | 0.7808 | 0.7694 | 0.8627 | **82.43** | 0.7129 | **Completed** |
| **AraBERTv0.2-Twitter-Base** (Run 1) | 0.7825 | 0.7666 | 0.8480 | **82.08** | 0.7139 | **Completed** |
| **AraBERTv0.2-Base** (Non-Twitter) | 0.7231 | 0.7160 | 0.8375 | **78.55** | 0.6677 | **Completed** |

---

## 2. Approach B (Qwen 2.5 72B Zero-Shot) API Status

To run the Qwen 2.5 72B Instruct zero-shot baseline, we have configured `src/qwen_inference.py`. However, we hit the following API quota restrictions:

1. **Hugging Face Router (`https://router.huggingface.co/v1`)**
   * *Issue:* The user's account has depleted monthly included credits, returning **Error 402** on paid providers (DeepInfra/Novita).
   * *Status:* Bypassed for smaller models (which run on HF's free serverless tier), but Qwen 2.5 72B Instruct always triggers provider routing and fails.

2. **SambaNova Cloud (`https://api.sambanova.ai/v1`)**
   * *Issue:* We attempted to use SambaNova's free API, but `Qwen2.5-72B-Instruct` has been **deprecated and removed** from their active model catalog (replaced by Llama 3.3 70B and DeepSeek V3).

3. **Active Workarounds Available:**
   * **Fireworks AI:** New accounts receive $1.00 in free credits (no credit card required), which is enough to run the full 619-row evaluation multiple times. Model ID: `accounts/fireworks/models/qwen2p5-72b-instruct`.
   * **Model Studio (Alibaba Cloud):** New accounts receive 70M free tokens, which easily covers the 0.05M tokens needed for the run.

---

## 3. Project File Directory

*   [data/](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/): Contains training and validation splits (`train.csv`, `dev.csv`).
*   [src/arabert_finetune.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/arabert_finetune.py): Fine-tuning script for AraBERT models.
*   [src/qwen_inference.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/qwen_inference.py): Pipeline script for zero-shot LLM inference.
*   [src/verify_metrics.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/verify_metrics.py): Hardened metric verification suite.
*   [results_summary.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv): Raw baseline scores CSV.
*   [stanceeval2026_baseline_spec.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/stanceeval2026_baseline_spec.md): Baseline specification and prompt details.

---

## 4. Next Steps

1.  **Configure Qwen Zero-Shot Provider:**
    *   Sign up for **Fireworks AI** (or a similar free-credit provider) to obtain a key.
    *   Run the 50-row subset tests to select the best framing (English vs. Arabic).
    *   Execute the full 619-row run to complete the Qwen 2.5 72B baseline.
2.  **Optimize AraBERT-Twitter-Large:**
    *   Fine-tune on the full **100% training set** (using the optimal epoch count determined during validation) to push our score past the **83.90** reference.

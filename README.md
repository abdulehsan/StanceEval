# StanceEval-2026: Arabic Stance Detection

Official implementation and experimental resources for our system submitted to **StanceEval-2026 Track 1** (Arabic stance detection in social media on the Mawqif-v2 dataset).

The system investigates fine-tuned AraBERT encoder baselines alongside zero-shot generative large language models (primarily Gemma 4 31B), systematically calibrated through Context Injection (CI) and targeted decision guidelines.

---

## 🏆 Official Results

### Official Blind Test Set (352 rows — Target: Women Driving)
Evaluated on the official CodaBench platform for Mawqif-v2 Track 1:

| Model / Submission | Setup & Prompting | $F_{avg2}$ (Primary) | Accuracy | $F_{avg3}$ | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gemma 4 31B (CI v3)** | Context Injection + Mental Rewrite + Positive Framing *(Final)* | **0.8661** | **82.67%** | **0.6746** | **7th** |
| **Gemma 4 31B (CI v5)** | Context Injection + Skepticism / Traffic Doubt Rules | **0.8635** | 82.39% | 0.6626 | — |
| **Gemma 4 31B (CI v1)** | Context Injection (Historical Context + Mental Rewrite) | **0.8485** | 80.97% | 0.6702 | — |
| **Gemma 4 31B (Revised v1)** | Mental Translation & Target Normalization *(Pre-final)* | **0.8444** | 80.11% | 0.6429 | 3rd* |
| **Gemma 4 31B (Prompt v1)** | Initial Zero-Shot Baseline | **0.8063** | 75.28% | 0.6183 | — |

*\*Rank at the time of submission (July 24).*

---

### Development Set Evaluation (619 rows — 3 Targets)

| Model / Fine-Tuning Setup | Covid-19 Vaccine | Digital Transformation | Women Empowerment | Overall $F_{avg2}$ | Overall $F_{avg3}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 31B (Zero-Shot)** | 0.9066 | 0.7685 | 0.8996 | **0.8848** | **0.7820** |
| **AraBERTv0.2-Twitter-Large** | 0.8313 | 0.7282 | 0.8581 | **0.8346** | **0.7157** |
| **AraBERTv0.2-Twitter-Base** | 0.7808 | 0.7694 | 0.8627 | **0.8243** | **0.7129** |
| **AraBERTv0.2-Base** | 0.7231 | 0.7160 | 0.8375 | **0.7855** | **0.6677** |

---

## 📁 Repository Structure

```
StanceEval/
├── data/                                 # Official Mawqif-v2 data splits
│   ├── train.csv                         # 2,473 training tweets
│   ├── dev.csv                           # 619 development tweets (3 targets)
│   └── ground_truth.csv                  # 352 blind evaluation tweets (Women Driving)
├── Evaluation Script/                    # Official task evaluation script
│   └── evaluate.py                       # Computes Favg2 and Favg3 macro-F1 metrics
├── predictions/                          # Official prediction files
│   ├── results_gemma4_31b_cerebras_test_ci_v3.txt
│   └── results_gemma4_31b_cerebras_test_revised_zeroshot.txt
├── src/                                  # Source code
│   ├── arabert_finetune.py               # Supervised fine-tuning for AraBERT variants
│   ├── data_utils.py                     # Data loading, preprocessing, and export helpers
│   ├── evaluate.py                       # Evaluation interface
│   ├── metrics.py                        # Metric definitions (Favg2, Favg3, accuracy)
│   ├── verify_metrics.py                 # Verification test suite matching official eval
│   ├── gemma_inference_dev_runs.py       # Multi-key Dev set runner across CI prompt iterations
│   ├── gemma_inference_test.py           # Unified test set runner for all calibration variants
│   ├── gemma_inference_test_ci_v3.py     # Standalone runner for the winning CI v3 configuration
│   ├── gemma_inference_test_explain_then_classify_v1.py # Explain-then-classify ablation
│   ├── gemma_inference_test_final_sc.py  # Self-consistency voting pipeline (T=0.1, 0.5, 1.0)
│   ├── debate_pipeline.py                # Multi-model Gemma-Qwen-Gemma collaborative debate
│   ├── allam_inference.py                # ALLaM-7B local dialect interpreter pipeline
│   ├── openai_inference_test_ci_v3_batched.py     # Batched OpenAI / GPT inference
│   └── openai_inference_test_ci_v3_batched_sol.py # Alternative GPT pipeline
├── references.bib                        # BibTeX references
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/abdulehsan/StanceEval.git
cd StanceEval
pip install -r requirements.txt
```

### 2. Metric Verification

Verify that local metric computations strictly match the official evaluation script across all edge cases:

```bash
python src/verify_metrics.py
```

### 3. AraBERT Fine-Tuning

To train and evaluate the supervised AraBERT baselines:

```bash
python src/arabert_finetune.py --model_variant arabertv02_twitter_large
```

Available variants: `arabertv02_base`, `arabertv02_twitter_base`, `arabertv02_twitter_large`.

### 4. Gemma 4 31B Inference (Development Set)

To evaluate prompt versions (`Revised`, `CI v1`, `CI v3`, `CI v5`) on the full 619-row development set:

```bash
python src/gemma_inference_dev_runs.py --version "CI v3"
```

### 5. Gemma 4 31B Inference (Test Set)

To generate predictions for the 352-row blind test set using the final calibrated configuration:

```bash
python src/gemma_inference_test.py --variant ci_v3
```

You can also run other variants:
```bash
python src/gemma_inference_test.py --variant v1         # Initial zero-shot baseline
python src/gemma_inference_test.py --variant revised    # Mental translation (0.8444 Favg2)
python src/gemma_inference_test.py --variant ci_v1      # Context Injection v1
python src/gemma_inference_test.py --variant ci_v5      # Skepticism rules (0.8635 Favg2)
```

### 6. Advanced Pipelines

- **Self-Consistency Voting ($T=0.1, 0.5, 1.0$):**
  ```bash
  python src/gemma_inference_test_final_sc.py
  ```
- **Gemma-Qwen-Gemma Debate:**
  ```bash
  python src/debate_pipeline.py
  ```
- **ALLaM-7B Dialect Interpretation:**
  ```bash
  python src/allam_inference.py
  ```

---

## ⚙️ Inference Configuration

For all reported zero-shot generative results:
- **Model:** `gemma-4-31b` via Cerebras Cloud API
- **Decoding Strategy:** Deterministic greedy decoding
- **Temperature ($T$):** `0.1`
- **Top-$p$:** `0.95`
- **Max Tokens:** `10` (constrained single-token output: `Favor`, `Against`, `None`)

---

## 📖 Citation

If you use this repository or our findings in your work, please cite:

```bibtex
@inproceedings{ehsan-2026-stanceeval,
  title={Contextual Calibration and Prompt Engineering for Arabic Stance Detection in Social Media},
  author={Ehsan, Abdullah},
  booktitle={Proceedings of the ArabicNLP 2026 Shared Tasks},
  year={2026}
}
```

# StanceEval-2026

Arabic stance detection baseline system for Mawqif-v2 dataset targeting COVID-19 Vaccine, Digital Transformation, and Women Empowerment topics evaluated using the macro-F1 Favg2 metric.

## Results

### Full-Dev Set Evaluation (619 rows)
Evaluation results on the Mawqif-v2 development set. Favg2 is the primary evaluation metric (macro-average over Favor and Against stance classes, averaged across topics). Favg3 is the secondary evaluation metric (macro-average over Favor, Against, and None stance classes, averaged across topics).

| Model / Fine-Tuning Setup | Covid Favg2 | Digital Favg2 | Women Favg2 | Overall Favg2 | Overall Favg3 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| *Organizer Reference: Qwen 2.5 (72B, zero-shot)* | — | — | — | **0.8425** | — | Baseline |
| *Organizer Reference: AraBERT-twitter* | — | — | — | **0.8390** | — | Baseline |
| **Gemma 4 31B** (Cerebras, zero-shot) | 0.9066 | 0.7685 | 0.8996 | **0.8848** | 0.7820 | Completed (best overall score) |
| **Gemma 4 31B** (Cerebras, zero-shot, Arabic targets) | 0.9066 | 0.7785 | 0.8970 | **0.8844** | 0.7816 | Completed |
| **AraBERTv0.2-Twitter-Large** (90/10 split) | 0.8313 | 0.7282 | 0.8581 | **0.8346** | 0.7157 | Completed (best checkpoint) |
| **AraBERTv0.2-Twitter-Large** (100% data) | 0.8231 | 0.7167 | 0.8602 | **0.8326** | 0.7233 | Completed |
| **AraBERTv0.2-Twitter-Base** (Run 2) | 0.7808 | 0.7694 | 0.8627 | **0.8243** | 0.7129 | Completed |
| **AraBERTv0.2-Twitter-Base** (Run 1) | 0.7825 | 0.7666 | 0.8480 | **0.8208** | 0.7139 | Completed |
| **AraBERTv0.2-Base** (Non-Twitter) | 0.7231 | 0.7160 | 0.8375 | **0.7855** | 0.6677 | Completed |

*Note: A duplicate row for `arabertv02_twitter_base` (Run 1) in `results_summary.csv` has been deduplicated in this table.*

### Zero-Shot LLM Subset Evaluation (100-row stratified subset)
Zero-shot model performance evaluated on a stratified 100-row subset of the dev set (composed of 33 Covid Vaccine, 33 Digital Transformation, and 34 Women Empowerment rows).

| Model / API Provider | Covid Favg2 | Digital Favg2 | Women Favg2 | Overall Favg2 | Overall Favg3 | Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 4 31B** (Cerebras, prompt v1 + Arabic targets) | 0.8857 | 0.8583 | 1.0000 | **0.9365** | 0.8542 | 90% | Completed (best subset score) |
| **Gemma 4 31B** (Cerebras, prompt v1) | 0.8857 | 0.8583 | 0.9654 | **0.9249** | 0.8235 | 88% | Completed |
| **Gemma 4 31B** (Cerebras, prompt v4 few-shot) | 0.8444 | 0.8694 | 0.9872 | **0.9240** | 0.8229 | 89% | Completed |
| **Gemma 4 31B** (Cerebras, prompt v5 + Arabic targets) | 0.8571 | 0.8468 | 0.9872 | **0.9178** | 0.8202 | 88% | Completed |
| **Gemma 4 31B** (Cerebras, prompt v3) | 0.8730 | 0.6600 | 0.9583 | **0.9094** | 0.7729 | 86% | Completed |
| **Qwen 3 32B** (Groq, prompt v3) | 0.7376 | 0.7917 | 0.9137 | **0.8550** | 0.7429 | 81% | Completed (best Qwen config) |
| **Qwen 3 32B** (Groq, prompt v2) | 0.7529 | 0.7681 | 0.8904 | **0.8424** | 0.7225 | 79% | Completed |
| **Qwen 3 32B** (Groq, prompt v1) | 0.7141 | 0.8583 | 0.9178 | **0.8485** | 0.7273 | 79% | Completed |
| **Qwen 3.6 27B** (Groq) | 0.7048 | 0.6875 | 0.9227 | **0.8270** | 0.7365 | 78% | Completed |

## Folder Structure

```
StanceEval/
├── Evaluation Script/
│   └── evaluate.py
├── data/
│   ├── dev.csv
│   ├── ground_truth.csv
│   └── train.csv
├── predictions/
│   ├── arabertv02_base_dev_preds.csv
│   ├── arabertv02_base_dev_preds.txt
│   ├── arabertv02_twitter_base_dev_preds.csv
│   ├── arabertv02_twitter_base_dev_preds.txt
│   ├── arabertv02_twitter_large_dev_preds.csv
│   ├── arabertv02_twitter_large_dev_preds.txt
│   ├── results_gemma4_31b_cerebras_test.csv
│   ├── results_gemma4_31b_cerebras_test.txt
│   ├── results_qwen_3.6_27b_test.csv
│   └── results_qwen_3_32b_test.csv
├── src/
│   ├── arabert_finetune.py
│   ├── check_partial_progress.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── gemma_inference.py
│   ├── gemma_inference_test.py
│   ├── metrics.py
│   ├── qwen3_32b_inference_test.py
│   ├── qwen_inference.py
│   ├── qwen_inference_test.py
│   └── verify_metrics.py
├── .gitignore
├── requirements.txt
├── results_summary.csv
├── stanceeval2026_baseline_spec.md
└── state.md
```

## File Descriptions

### Root Files
*   **[.gitignore](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/.gitignore)**: Specifies files and directories ignored by Git version control, such as environment variables, model checkpoints, venvs, and raw API logs.
*   **[requirements.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/requirements.txt)**: Python package dependency list for setting up the python virtual environment.
*   **[results_summary.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv)**: Summary CSV database of final dev set evaluation scores for each AraBERT model variant run.
*   **[stanceeval2026_baseline_spec.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/stanceeval2026_baseline_spec.md)**: Baseline specifications detailing model parameters, tokenization settings, data formats, prompting structures, and pipeline goals.
*   **[state.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/state.md)**: Master project status tracker and end-to-end pipeline guide mapping folder hierarchies, execution metrics, and correctness audit reviews.

### Data Directory
*   **[data/train.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/train.csv)**: Training dataset consisting of 2,473 annotated Arabic tweets with their respective targets and stance labels.
*   **[data/dev.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/dev.csv)**: Validation dataset consisting of 619 annotated Arabic tweets used as the validation source.
*   **[data/ground_truth.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/ground_truth.csv)**: Test dataset consisting of 352 unannotated Arabic tweets used for final blind testing.

### Predictions Directory
*   **[predictions/](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions)**: Folder storing output predictions in CSV format (for qualitative error analysis) and TXT format (for official metric evaluation) generated by fine-tuned checkpoints and zero-shot LLM inference runs.

### Source Directory
*   **[src/arabert_finetune.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/arabert_finetune.py)**: Fine-tunes AraBERT variants for Arabic stance detection, supporting aubmindlab base/large models. Orchestrates joint target training on an internal 90/10 stratified split, class-weighted cross-entropy loss, and post-training dev set evaluation.
*   **[src/check_partial_progress.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/check_partial_progress.py)**: Standalone, read-only progress checker that reads whatever predictions currently exist and reports target-wise and overall metrics so far without locking files or calling official evaluators.
*   **[src/data_utils.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/data_utils.py)**: Canonicalizes raw target strings to match dev.csv, loads and validates train/dev CSVs, produces a stratified 90/10 internal split for early stopping, computes inverse-frequency class weights, and writes prediction outputs in both CSV (error analysis) and txt (official eval script) formats.
*   **[src/evaluate.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/evaluate.py)**: Script to run the official competition evaluation script on saved prediction files and update results_summary.csv.
*   **[src/metrics.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/metrics.py)**: Shared implementation of Favg2/Favg3 metrics replicating the competition math formulas to support training validation and LLM evaluation.
*   **[src/verify_metrics.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/verify_metrics.py)**: Test suite that asserts local Favg2/Favg3 score implementations match the official competition evaluation outputs to 6 decimal places.

#### Blind Test Inference Scripts:
*   **[src/gemma_inference_test.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/gemma_inference_test.py)**: Runs Gemma 4 31B inference via Cerebras on the 352-row test dataset (`ground_truth.csv`).
*   **[src/qwen_inference_test.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/qwen_inference_test.py)**: Runs Qwen 3.6 27B inference via Groq on the 352-row test dataset (`ground_truth.csv`), utilizing prompt-caching and reactive 429 safety delays.
*   **[src/qwen3_32b_inference_test.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/qwen3_32b_inference_test.py)**: Runs Qwen 3 32B inference via OpenRouter on the 352-row test dataset (`ground_truth.csv`), utilizing stance-only predictions and connection timeouts.

## Setup Instructions

### 1. Environment Creation
Create and activate a Python virtual environment:
```bash
python -m venv stanceeval-venv
# Windows (PowerShell):
.\stanceeval-venv\Scripts\Activate.ps1
# Linux/macOS:
source stanceeval-venv/bin/activate
```

### 2. Dependency Installation
Install required packages from the dependency list:
```bash
pip install -r requirements.txt
```
*Note: To run the zero-shot LLM inference scripts, install the corresponding API clients:*
```bash
pip install groq cerebras-cloud-sdk openai
```

### 3. Environment Variables
Create a `.env` file in the repository root. Configure the API credentials for zero-shot LLM runs:
```env
# Required for Gemma (Cerebras API)
CEREBRAS_API_KEY=your_cerebras_api_key_here

# Required for Qwen 3.6 27B (Groq API)
GROQ_API_KEY=your_groq_api_key_here

# Required for Qwen 3 32B (OpenRouter API)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## How to Run

### 1. Verify Metrics Implementation
```bash
python src/verify_metrics.py
```

### 2. Fine-tune AraBERT Models
```bash
python src/arabert_finetune.py --model_key <model_key> [options]
```

### 3. Run Test Set LLM Inference

#### Gemma 4 31B (Cerebras)
```bash
python src/gemma_inference_test.py
```

#### Qwen 3.6 27B (Groq)
```bash
python src/qwen_inference_test.py
```

#### Qwen 3 32B (OpenRouter)
```bash
python src/qwen3_32b_inference_test.py
```

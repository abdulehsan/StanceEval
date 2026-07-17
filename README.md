# StanceEval-2026

Arabic stance detection baseline system for Mawqif-v2 dataset targeting COVID-19 Vaccine, Digital Transformation, and Women Empowerment topics evaluated using the macro-F1 Favg2 metric.

## Results

### Full-Dev Set Evaluation (619 rows)
Evaluation results on the Mawqif-v2 development set. Favg2 is the primary evaluation metric (macro-average over Favor and Against stance classes, averaged across topics). Favg3 is the secondary evaluation metric (macro-average over Favor, Against, and None stance classes, averaged across topics).

| Model / Fine-Tuning Setup | Covid Favg2 | Digital Favg2 | Women Favg2 | Overall Favg2 | Overall Favg3 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| *Organizer Reference: Qwen 2.5 (72B, zero-shot)* | — | — | — | **0.8425** | — | Baseline |
| *Organizer Reference: AraBERT-twitter* | — | — | — | **0.8390** | — | Baseline |
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
| **Gemma 4 31B** (Cerebras) | 0.8857 | 0.8583 | 0.9654 | **0.9249** | 0.8235 | 88% | Completed |
| **Qwen 3 32B** (Groq) | 0.7141 | 0.8583 | 0.9178 | **0.8485** | 0.7273 | 79% | Completed |
| **Qwen 3.6 27B** (Groq) | 0.7048 | 0.6875 | 0.9227 | **0.8270** | 0.7365 | 78% | Completed |

## Folder Structure

```
StanceEval/
├── Evaluation Script/
│   └── evaluate.py
├── data/
│   ├── dev.csv
│   └── train.csv
├── predictions/
│   ├── arabertv02_base_dev_preds.csv
│   ├── arabertv02_base_dev_preds.txt
│   ├── arabertv02_twitter_base_dev_preds.csv
│   ├── arabertv02_twitter_base_dev_preds.txt
│   ├── arabertv02_twitter_large_dev_preds.csv
│   ├── arabertv02_twitter_large_dev_preds.txt
│   ├── arabertv02_twitter_large_fulldata_dev_preds.csv
│   ├── arabertv02_twitter_large_fulldata_dev_preds.txt
│   ├── qwen3_32b_groq_100row_results.csv
│   ├── results_gemma4_31b_cerebras.csv
│   ├── results_qwen3.6_27b.csv
│   └── results_qwen3_32b.csv
├── src/
│   ├── arabert_finetune.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── gemma_inference.py
│   ├── metrics.py
│   ├── qwen_inference.py
│   └── verify_metrics.py
├── .gitignore
├── error_analysis.md
├── requirements.txt
├── results_summary.csv
├── stanceeval2026_baseline_spec.md
└── state.md
```

## File Descriptions

### Root Files
*   **[.gitignore](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/.gitignore)**: Specifies files and directories ignored by Git version control, such as environment variables, model checkpoints, venvs, and raw API logs.
*   **[error_analysis.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/error_analysis.md)**: Markdown document template for documenting error patterns, class distributions, and confusion matrices for the AraBERT and Qwen models.
*   **[requirements.txt](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/requirements.txt)**: Python package dependency list for setting up the python virtual environment.
*   **[results_summary.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/results_summary.csv)**: Summary CSV database of final dev set evaluation scores for each AraBERT model variant run.
*   **[stanceeval2026_baseline_spec.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/stanceeval2026_baseline_spec.md)**: Baseline specifications detailing model parameters, tokenization settings, data formats, prompting structures, and pipeline goals.
*   **[state.md](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/state.md)**: Master project status tracker and end-to-end pipeline guide mapping folder hierarchies, execution metrics, and correctness audit reviews.

### Data Directory
*   **[data/train.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/train.csv)**: Training dataset consisting of 2,473 annotated Arabic tweets with their respective targets and stance labels.
*   **[data/dev.csv](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/data/dev.csv)**: Validation dataset consisting of 619 annotated Arabic tweets used as the validation source.

### Evaluation Script Directory
*   **[Evaluation Script/evaluate.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/Evaluation%20Script/evaluate.py)**: *No docstring (purpose inferred).* Loads gold annotations from a reference directory and predictions from a submission directory (extracting zip files if present), validates matching row counts, computes target-wise and overall Favg2/Favg3 metrics, and writes the output to scores.txt.

### Predictions Directory
*   **[predictions/](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/predictions)**: Folder storing output predictions in CSV format (for qualitative error analysis) and TXT format (for official metric evaluation) generated by fine-tuned checkpoints and zero-shot LLM inference runs.

### Source Directory
*   **[src/arabert_finetune.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/arabert_finetune.py)**: Fine-tunes AraBERT variants for Arabic stance detection, supporting aubmindlab base/large models. Orchestrates joint target training on an internal 90/10 stratified split, class-weighted cross-entropy loss, and post-training dev set evaluation.
*   **[src/data_utils.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/data_utils.py)**: Canonicalizes raw target strings to match dev.csv, loads and validates train/dev CSVs, produces a stratified 90/10 internal split for early stopping, computes inverse-frequency class weights, and writes prediction outputs in both CSV (error analysis) and txt (official eval script) formats.
*   **[src/evaluate.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/evaluate.py)**: Script to run the official competition evaluation script on saved prediction files and update results_summary.csv.
*   **[src/gemma_inference.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/gemma_inference.py)**: Executes zero-shot stance detection using Gemma 4 31B via Cerebras API on a stratified 100-row subset of the dev set.
*   **[src/metrics.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/metrics.py)**: Shared implementation of Favg2/Favg3 metrics replicating the competition math formulas to support training validation and LLM evaluation.
*   **[src/qwen_inference.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/qwen_inference.py)**: Executes zero-shot stance detection using Qwen3-32B via Groq API on a stratified 100-row subset of the dev set, using rate-limit headers for pacing.
*   **[src/verify_metrics.py](file:///d:/Abdullah%20Files/Programmming/python/StanceEval/src/verify_metrics.py)**: Test suite that asserts local Favg2/Favg3 score implementations match the official competition evaluation outputs to 6 decimal places.

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
*Note: To run the zero-shot LLM inference scripts (`src/qwen_inference.py` and `src/gemma_inference.py`), install the corresponding API clients which are not listed in the default `requirements.txt`:*
```bash
pip install groq cerebras-cloud-sdk
```

### 3. Environment Variables
Create a `.env` file in the repository root (no `.env.example` file is provided). Configure the API credentials for zero-shot LLM runs:
```env
# Required for Qwen zero-shot runs (src/qwen_inference.py)
GROQ_API_KEY=your_groq_api_key_here

# Required for Gemma zero-shot runs (src/gemma_inference.py)
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

## How to Run

### 1. Verify Metrics Implementation
Before running any training, verify that local metric calculations match the official competition evaluator:
```bash
python src/verify_metrics.py
```

### 2. Fine-tune AraBERT Models
Train a specific AraBERT variant on the training dataset.

```bash
python src/arabert_finetune.py --model_key <model_key> [options]
```

**Required Arguments:**
*   `--model_key`: Choices are `arabertv02_base`, `arabertv2_base`, `arabertv02_twitter_base`, `arabertv02_twitter_large`.

**Optional Arguments:**
*   `--epochs`: Max epochs (default: `5`).
*   `--batch_size`: Per-device batch size (default: `16`).
*   `--gradient_accumulation_steps`: Gradient accumulation steps (default: `1`).
*   `--lr`: Learning rate (default: `2e-5`).
*   `--max_length`: Max token length (default: `128`).
*   `--early_stopping_patience`: Early stopping patience in epochs (default: `2`).
*   `--seed`: Random seed (default: `42`).
*   `--checkpoint_path`: Path to a specific model checkpoint to load (e.g. `checkpoints/arabertv02_twitter_base/checkpoint-591`).
*   `--predict_only`: Skip model training and run evaluation/prediction only using specified checkpoint.
*   `--full_train`: Train on 100% of train.csv (no internal split or early stopping).
*   `--run_name`: Override checkpoint directory name and results_summary.csv row label.

*Example command (base Twitter AraBERT training):*
```bash
python src/arabert_finetune.py --model_key arabertv02_twitter_base --epochs 5 --batch_size 16 --lr 2e-5
```

### 3. Evaluate Predictions
Evaluate output prediction files and view the current results summary database.

```bash
python src/evaluate.py <options>
```

**Options (Choose one of the mutually exclusive flags):**
*   `--pred_txt <path>`: Evaluate a single prediction TXT file.
*   `--all`: Evaluate all prediction TXT files in `predictions/`.
*   `--show_results`: Print the current `results_summary.csv` database as a formatted table.

**Additional Options:**
*   `--model_name <name>`: Override model name in the results CSV (default: derived from filename).

*Example command (evaluate single model output):*
```bash
python src/evaluate.py --pred_txt predictions/arabertv02_twitter_base_dev_preds.txt
```

### 4. Run Zero-shot LLM Inference

#### Qwen (Groq)
Runs Qwen3-32B inference on the 100-row stratified dev subset:
```bash
python src/qwen_inference.py [--check_only]
```
*   `--check_only`: Runs row selection and target distribution check only, then exits.

#### Gemma (Cerebras)
Runs Gemma 4 31B inference on the 100-row stratified dev subset:
```bash
python src/gemma_inference.py
```

### 5. Run Official Evaluator Verbatim
If you want to run the official evaluation script directly:
```bash
python "Evaluation Script/evaluate.py" <input_dir> <output_dir>
```
*   `<input_dir>` must contain `ref/gold.csv` and `res/predictions.txt`.
*   `<output_dir>` is where the script writes `scores.txt`.

## Known Issues & In-Progress Work
*   **Pending Ensemble Script**: An ensemble script is planned/in-progress to perform logit averaging or majority voting across predictions from the four checkpoints.
*   **Validation Set Isolation**: The development dataset `dev.csv` is completely held out and never used during model training or hyperparameter tuning. It is only touched after selecting the final best checkpoint to compute final scores.

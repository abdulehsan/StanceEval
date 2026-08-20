# Error Analysis — StanceEval-2026 Track 1 Baseline

**Purpose:** Document error patterns across models for the paper's Discussion section.
Fill this in after dev evaluation is complete and `results_summary.csv` is populated.

---

## 1. Class Distribution Recap (train.csv)

| Target                | Against | Favor | None | Total |
|-----------------------|---------|-------|------|-------|
| Covid Vaccine         | 508     | 509   | 150  | 1167  |
| Digital Transformation| 142     | 879   | 124  | 1145  |
| Women empowerment     | 371     | 760   | 59   | 1190  |
| **Total**             | **1021**|**2148**|**333**|**3502**|

Key imbalances:
- Digital Transformation: 77% Favor — risk of the model predicting Favor for everything
- Women empowerment: ~5% None — None class may be near-invisible for this topic

---

## 2. Results Summary

*(Fill in after running `python src/evaluate.py --all --show_results`)*

| Model | Favg2 Covid | Favg2 Digital | Favg2 Women | Favg2 Overall | Favg3 Overall |
|-------|-------------|---------------|-------------|---------------|---------------|
| arabertv02_twitter_base   | | | | | |
| arabertv02_base           | | | | | |
| arabertv2_base            | | | | | |
| arabertv02_twitter_large  | | | | | |
| qwen25_72b_en             | | | | | |
| qwen25_72b_ar (if run)    | | | | | |

---

## 3. AraBERT Error Patterns

*(Run after best model is selected)*

### 3.1 Confusion Matrix per Target

```
# To generate:
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# df = pd.read_csv('predictions/arabertv02_twitter_base_dev_preds.csv')
# for target in df['target'].unique():
#     sub = df[df['target'] == target]
#     print(target)
#     print(confusion_matrix(sub['true_stance'], sub['predicted_stance'],
#                            labels=['Favor', 'Against', 'None']))
```

- [ ] Covid Vaccine confusion matrix
- [ ] Digital Transformation confusion matrix
- [ ] Women empowerment confusion matrix

### 3.2 None class recall by target

The None class is the hardest to predict (smallest, most ambiguous). Note:
- Is recall higher on Covid (150 None train examples) vs Women empowerment (59)?
- Does class weighting help or does the model still miss most None rows?

**Findings:** (fill in)

### 3.3 Qualitative samples

List 5-10 representative errors per pattern:

**Pattern: Predicted Favor, True Against**
| ID | text | target | true | pred |
|----|------|--------|------|------|
| | | | | |

**Pattern: Predicted Favor, True None**
| ID | text | target | true | pred |
|----|------|--------|------|------|
| | | | | |

---

## 4. Qwen Zero-Shot Error Patterns

*(Run after `python src/qwen_inference.py` completes)*

### 4.1 Sarcasm confusion

The spec notes Qwen may over-read literal tone as stance. Check:
- Cross-reference predicted rows with `sarcasm=Yes` in `data/dev.csv` (join on ID)
- Does Qwen predict Favor/Against on rows where the human label is None because of sarcasm?

```python
# Quick check:
import pandas as pd, json
dev = pd.read_csv('data/dev.csv')
log = pd.read_json('qwen_raw_logs/qwen25_72b_zeroshot_raw.jsonl', lines=True)
merged = dev.merge(log[['ID','parsed_stance']], on='ID')
sarcastic = merged[merged['sarcasm'] == 'Yes']
print(sarcastic.groupby(['stance', 'parsed_stance']).size())
```

**Findings:** (fill in)

### 4.2 Parse failures

- Total parse_success=False rows:
- Were they assigned 'None' by fallback?
- Pattern in raw_response for failures?

### 4.3 Class over-prediction

- Does Qwen under-predict None vs AraBERT?
- Does Arabic framing help or hurt on any particular target?

### 4.4 Prompt framing comparison (from 50-row A/B test)

| Metric | English framing | Arabic framing |
|--------|----------------|----------------|
| Parse success rate | | |
| Favg2 on subset | | |
| Committed to: | | |

---

## 5. Model Comparison Summary

*(For paper Discussion section)*

**Best model on dev:** (fill in)
**Biggest gap between best and worst:** (fill in)

**Observations:**
- [ ] Does Twitter-pretraining help? (v02-twitter vs v02-base)
- [ ] Does model size help? (v02-twitter-base vs v02-twitter-large)
- [ ] Does Farasa segmentation help? (v02 vs v2, if Java available)
- [ ] Zero-shot vs fine-tuned gap: how large?
- [ ] Which target is hardest? (expect Digital Transformation due to imbalance)

---

## 6. Custom Addition Ideas (post-baseline)

Based on the comparison results, note which direction the custom addition should take:

- [ ] **Few-shot prompting for Qwen** — if Qwen is competitive, add 3-5 examples per label to the prompt
- [ ] **LoRA fine-tuning of Qwen** — if compute budget allows and Qwen shows promise
- [ ] **Per-topic class weights** — if global weights don't fix Digital Transformation's Favor bias
- [ ] **Ensemble** — if multiple AraBERT variants score well on different topics

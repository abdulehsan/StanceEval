# Exploratory Data Analysis: Ground Truth Test Set (`ground_truth.csv`)

This document presents a detailed exploratory data analysis (EDA) of the newly released blind test set `data/ground_truth.csv` and compares it with the training (`data/train.csv`) and validation (`data/dev.csv`) datasets.

---

## 1. Executive Summary

- **Shape & Columns**: The dataset contains **352 rows** and **4 columns** (`id`, `tweet_id`, `tweet_text`, `target`).
- **Missing Labels**: There is **no `stance` column** in `ground_truth.csv`. Since the official `evaluate.py` script requires a `stance` column to evaluate, this file is a blind test set where we must predict the stance labels.
- **Single Target**: 100% of the tweets in this dataset are annotated with the target **`Women Driving`**. 
- **Topic & Temporal Shift**: 
  - The training and dev sets contain the target `Women empowerment` (along with `Covid Vaccine` and `Digital Transformation`) spanning **2021 to 2022**. These tweets focus on general social/professional empowerment, International Women's Day, and women in technology.
  - The test set focuses exclusively on the specific debate surrounding **women driving in Saudi Arabia**, spanning **2016 to 2021** (concentrated around the April 2016 campaign and the June 2018 official lifting of the driving ban).
- **ID Overlap Warning**: While the values in the `id` column overlap with those in the training and dev sets, the actual tweets and targets are completely different.

---

## 2. General Dataset Statistics

| Metric | Value |
| :--- | :--- |
| **Total Rows** | 352 |
| **Columns** | `id`, `tweet_id`, `tweet_text`, `target` |
| **Missing Values** | None (0 missing in all columns) |
| **Duplicate Tweets** | None (352 unique texts) |

---

## 3. Target Distribution & Topic Shift

In the training and dev sets, the targets are split evenly between three topics. In the test set, only one target is present:

| Dataset | Covid Vaccine | Digital Transformation | Women empowerment | Women Driving | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **train.csv** | 1,167 | 1,145 | 1,190 | 0 | **3,502** |
| **dev.csv** | 206 | 203 | 210 | 0 | **619** |
| **ground_truth.csv** | 0 | 0 | 0 | 352 | **352** |

### Key Takeaway:
We are dealing with a **domain/topic shift**. The model needs to perform zero-shot stance detection on the target `Women Driving` (or map it to `Women empowerment` if using models trained on `Women empowerment`, though they represent slightly different nuances).

---

## 4. Tweet Length Distribution

We analyzed the character and word lengths of the tweets in the test set:

| Statistic | Character Length | Word Count |
| :--- | :---: | :---: |
| **Mean** | 100.3 | 15.8 |
| **Std Dev** | 55.6 | 9.5 |
| **Min** | 15.0 | 2.0 |
| **25%** | 56.8 | 8.0 |
| **50% (Median)** | 89.5 | 14.0 |
| **75%** | 134.0 | 21.0 |
| **Max** | 278.0 | 48.0 |

---

## 5. Hashtag Analysis

Hashtags are highly predictive of stance in this dataset, indicating a strong polarity in the debate. Out of **425 total hashtags** (75 unique), the top 10 are:

| Hashtag | Count | Stance Implication | Translation / Context |
| :--- | :---: | :---: | :--- |
| `#لن_تقودي` | 177 | **Against** | "You will not drive" (Conservative campaign) |
| `#المراه_السعوديه_تسوق` | 83 | **Neutral / Mixed** | "Saudi women drive" (General campaign hashtag) |
| `#قياده_المرأه_السعودية` | 26 | **Favor / Mixed** | "Saudi women driving" |
| `#المرأة_السعودية_تسوق` | 14 | **Favor / Mixed** | "Saudi women drive" |
| `#المراه_السعوديه_تقود_السياره` | 12 | **Favor / Mixed** | "Saudi women drive the car" |
| `#الملك_ينتصر_لقياده_المراه` | 9 | **Favor** | "The King supports women driving" |
| `#قيادة_المرأة_السعودية` | 7 | **Favor / Mixed** | "Saudi women driving" |
| `#المرأة_السعودية_تقود_السيارة` | 5 | **Favor / Mixed** | "Saudi women drive the car" |
| `#قيادة_المرأة_للسيارة` | 5 | **Favor / Mixed** | "Women driving cars" |
| `#قيده_المراه_السعوديه` | 5 | **Favor / Mixed** | "Saudi women driving" |

> [!NOTE]
> The hashtag `#لن_تقودي` alone appears in **50.3%** of all test set tweets. This indicates a massive presence of conservative or anti-driving discourse in the test set.

---

## 6. Temporal Analysis (Snowflake ID Extraction)

Twitter Snowflake IDs encode the creation timestamp. By shifting the `tweet_id` by 22 bits and adding the Twitter Epoch, we recovered the exact posting times of the tweets:

- **Date Range**: January 2, 2016, to February 12, 2021.
- **Top Posting Months**:
  - **June 2018**: 166 tweets (47.2%). This matches the historical event when the Saudi ban on women driving was officially lifted (June 24, 2018).
  - **April 2016**: 93 tweets (26.4%). This aligns with intense online debates and hashtag campaigns around women driving in Saudi Arabia.
  - **Other months**: Sparse distribution spanning 2016 to 2021.

### Contrast with Train/Dev Sets:
The train and dev sets' `Women empowerment` tweets span **January 2021 to March 2022**. This highlights a five-year temporal gap between the historical driving debate (test set) and the general empowerment discussions (train/dev set).

---

## 7. Word Frequency Highlights (Unigrams)

Excluding prepositions and pronouns, the most frequent words in the tweets are:

1. **`الله`** (God - 53 times): Often used in prayers or oaths (e.g., "ان شاء الله", "الله يستر").
2. **`لن`** (Not/Will not - 42 times): Strongly associated with opposition (e.g., `#لن_تقودي`, "لن تقود").
3. **`المرأة`** (The woman/women - 28 times).
4. **`تقودي`** (You drive - feminine - 28 times).
5. **`قيادة`** (Driving - 23 times).
6. **`السيارة`** (The car - 18 times).
7. **`تقود`** (She/You drive - 17 times).

---

## 8. Implications for Stance Prediction

1. **Target Mapping**: Since all targets are `"Women Driving"`, zero-shot prompting must explicitly instruct the LLM about this target. For models fine-tuned on `"Women empowerment"`, we must determine whether to feed `"Women Driving"` as the target directly or map it to `"Women empowerment"` so it aligns with the training vocabulary.
2. **Class Imbalance / Polarized Stance**: Given that `#لن_تقودي` (Against) makes up more than half of the hashtags, the test set likely has a substantial number of `Against` stances, contrasted with the training set's `Women empowerment` class which was heavily skewed towards `Favor`.
3. **Dialectal Nuances**: The tweets are written in Gulf/Saudi dialects from 2016-2018. Models must handle dialectal Arabic spelling variations (e.g., `المراه` vs `المرأة`, `قياده` vs `قيادة`).

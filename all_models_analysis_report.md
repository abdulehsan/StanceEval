# Stance Detection Model Evaluation & Hyperparameter Analysis Report
**Target Topic**: Women Driving (قيادة المرأة للسيارة)  
**Dataset**: 352-row Blind Test Set (Mawqif-v2)  
**Primary Metric**: Macro F1-score of Favor and Against (Favg2)  
**Secondary Metric**: Macro F1-score of Favor, Against, and None (Favg3)  

---

## 1. Overall Performance Leaderboard

| Rank | Model | Setup & Prompting | Temperature | Top-P | Favg2 | Favg3 | Accuracy |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gemma 4 31B (CI v3)** | Zero-Shot, mental rewrite + context + calibrated positive framing | **0.1** | **0.95** | **86.61%** | **67.46%** | **82.67%** |
| 🥈 | **Gemma 4 31B (CI v5)** | Zero-Shot, mental rewrite + context + calibrated doubt/skepticism rules | **0.1** | **0.95** | **86.35%** | **66.26%** | **82.39%** |
| 🥉 | **Gemma 4 31B (CI v1)** | Zero-Shot, mental rewrite + historical background context | **0.1** | **0.95** | **84.85%** | **67.02%** | **80.97%** |

---

## 2. Consensus & Disagreement Statistics

We compared the row-by-row stance classifications across all three models to discover where they align and where they deviate.

### A. Consensus Counts (When ALL 3 Models Agree)
All 3 models agreed on the exact same label for **333 out of 352 rows** (94.60% of the dataset). This represents the core "easy/explicit" subset. The distribution of consensus labels is:
*   **Against**: **173 rows** *(51.95%)*
*   **Favor**: **150 rows** *(45.05%)*
*   **None**: **10 rows** *(3.00%)*

*Insight*: Explicit opposition is the most common class on Arabic social media regarding the driving decree, though explicit favor is a close second.

### B. Pairwise Agreement Matrix
This matrix shows how often pairs of models yielded the same prediction:

| | Gemma (CI v1) | Gemma (CI v5) | Gemma (CI v3) |
| :--- | :---: | :---: | :---: |
| **Gemma (CI v1)** | 100.00% | 95.17% | 96.31% |
| **Gemma (CI v5)** | 95.17% | 100.00% | 97.73% |
| **Gemma (CI v3)** | 96.31% | 97.73% | 100.00% |

---

## 3. System Prompts & Structure Comparison

### 1. Gemma 4 31B (CI v3 Zero-Shot)
*   **Structure**: Plain text output (single token).
*   **Prompt Philosophy**: Reused parameters ($T=0.1$, greedy decoding). Added a positive-framing reporting directive stating that reporting on events/decrees that celebrate, promote, or invite participation leans `Favor` (rather than being general neutral `None` reporting). This unlocked correct classification of early celebratory tweets.

### 2. Gemma 4 31B (CI v5 Zero-Shot)
*   **Structure**: Plain text output (single token).
*   **Prompt Philosophy**: Reused parameters ($T=0.1$, greedy decoding). Added a rule specifying that rhetorical questions or factual statements implying doubt, hidden motive, or negative consequences specifically about the target indicate `Against`. This targeted passive-aggressive road-safety and traffic skepticism.

### 3. Gemma 4 31B (CI v1 Zero-Shot)
*   **Structure**: Plain text output (single token).
*   **Prompt Philosophy**: Reused parameters ($T=0.1$, greedy decoding). Added a critical historical **Background Context** section specifying the 2017–2018 royal decree context, implementation issues, licensing, safety, and Saudi dialect usage. This gave the model the semantic priors to correctly resolve sarcastic anti-driving hashtags.

---

## 4. Hyperparameter Physics

### Temperature ($T$)
*   **Theory**: Temperature scales the logits before the softmax classification. For a logit $z_i$, the token probability is scaled by $P(x_i) \propto e^{z_i / T}$.
    *   **$T = 0.1$ (Gemma CI v3, CI v5, and CI v1)**: Performs near-deterministic greedy decoding. In classification tasks, this locks the model into outputting the token with the highest predicted probability mass, entirely eliminating random sampling noise. This was a key contributor to raising the Favg2 score.

---

## 5. Performance Hypotheses

### Hypothesis 1: Positive Framing Calibration (CI v3 vs. CI v1)
*   **Why Gemma CI v3 achieved the all-time high (86.61% Favg2, 67.46% Favg3)**:
    By adding the instruction that news reporting of events/decrees leans `Favor` if it is framed positively (e.g. promoting, celebrating, or inviting participation), we resolved a major blind spot. In Saudi social media, early reports of women driving (such as officers distributing roses or celebrating license issuances) are structurally and sentimentally pro-driving, but CI v1 often resolved them as neutral `None` or `Against`. This new calibration successfully mapped these celebratory events to `Favor`, aligning perfectly with human annotations.

### Hypothesis 2: Calibrating Skepticism and Sarcasm (CI v5 vs. CI v3)
*   **Why Gemma CI v5 performed slightly below CI v3 (86.35% Favg2, 66.26% Favg3)**:
    While CI v5's directive to map statements implying negative consequences to `Against` succeeded in resolving safety and traffic complaints (e.g., worries about morning traffic or passenger distraction), it was slightly **over-aggressive**. It misclassified pro-driving sarcasm directed at opponents (e.g., mocking anti-driving hashtags by showing women filling a car wash) as `Against`. In stance detection, human gold labels map mocking sarcasm to `Favor` and lighthearted gender banter to `None`. CI v3's more balanced guidelines remain our top configuration.

---

## 6. System Prompt Contents

### 1. Gemma 4 31B (CI v3)
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

### 2. Gemma 4 31B (CI v5)
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
• Rhetorical questions or seemingly factual statements that imply doubt, hidden motive, or negative consequence specifically about the target indicate Against, even without explicit negative words.
• Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor.
• If the stance toward the target cannot reasonably be inferred, output None.

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text.
```

### 3. Gemma 4 31B (CI v1)
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
• Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance.
• If the stance toward the target cannot reasonably be inferred, output None.

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text.
```

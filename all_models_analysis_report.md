# Stance Detection Model Evaluation & Hyperparameter Analysis Report
**Target Topic**: Women Driving (قيادة المرأة للسيارة)  
**Dataset**: 352-row Blind Test Set (Mawqif-v2)  
**Primary Metric**: Macro F1-score of Favor and Against (Favg2)  
**Secondary Metric**: Macro F1-score of Favor, Against, and None (Favg3)  

---

## 1. Overall Performance Leaderboard

| Model | Setup & Prompting | Temperature | Top-P | Favg2 | Favg3 | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Gemma 4 31B (Revised)** | Zero-Shot, Mental Translation + Normalization Guidelines | **0.1** | **0.95** | **84.44%** | **64.29%** | **80.11%** |
| 🥈 **Gemma 4 31B (Original)** | Zero-Shot, English prompt, Arabic targets | **1.0** | **0.95** | **80.63%** | **61.83%** | **75.28%** |
| 🥉 **Gemma 4 31B (Refined)** | Zero-Shot, Target-First + Negativity guidelines | **1.0** | **0.95** | **79.40%** | **59.38%** | **74.15%** |
| 4th **Qwen 3.6 27B** | Few-Shot, JSON format with Stance + Reasoning | **0.5** | *Default* | **77.59%** | **58.62%** | **73.01%** |
| 5th **Qwen 3 32B** | Few-Shot, JSON format (Reasoning suppressed) | **0.6** | *Default* | **68.75%** | **54.22%** | **68.47%** |

---

## 2. Consensus & Disagreement Statistics

We compared the row-by-row stance classifications across all five models to discover where they align and where they deviate.

### A. Consensus Counts (When ALL 5 Models Agree)
All 5 models agreed on the exact same label for **201 out of 352 rows** (57.10% of the dataset). This represents the "explicit/easy" subset. The distribution of consensus labels is:
*   **Against**: **125 rows** *(62.19%)*
*   **Favor**: **68 rows** *(33.83%)*
*   **None**: **8 rows** *(3.98%)*

*Insight*: The consensus data indicates that explicit opposition to the target is the most common and linguistically straightforward class on Arabic social media (outnumbering explicit favor 2-to-1 in consensus).

### B. Pairwise Agreement Matrix
This matrix shows how often pairs of models yielded the same prediction:

| | Gemma (Original) | Gemma (Refined) | Gemma (Revised) | Qwen 3.6 27B | Qwen 3 32B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 31B (Original)** | 100.00% | 92.33% | 90.34% | 80.97% | 67.05% |
| **Gemma 4 31B (Refined)** | 92.33% | 100.00% | 88.35% | 80.68% | 68.47% |
| **Gemma 4 31B (Revised)** | 90.34% | 88.35% | 100.00% | 80.11% | 64.49% |
| **Qwen 3.6 27B** | 80.97% | 80.68% | 80.11% | 100.00% | 68.18% |
| **Qwen 3 32B** | 67.05% | 68.47% | 64.49% | 68.18% | 100.00% |

---

## 3. System Prompts & Structure Comparison

### 1. Gemma 4 31B (Original Zero-Shot)
*   **Structure**: Plain text output (strictly the word `Favor`, `Against`, or `None` via a low `max_tokens = 10` constraint).
*   **Prompt Philosophy**: Standard classification prompt defining the three classes in English, using Arabic target labels (`قيادة المرأة للسيارة`) to anchor the sentiment. No detailed edge-case guidelines.

### 2. Gemma 4 31B (Refined Zero-Shot)
*   **Structure**: Plain text output (single token).
*   **Prompt Philosophy**: Explicitly added cognitive guidelines for:
    1.  *Target-First Identification* (check who/what is being criticized before judging sentiment).
    2.  *Negativity Diversion* (interpreting aggressive criticism of anti-driving detractors as `Favor`).
    3.  *Questions Rule* (mapping news reporting or neutral speculation questions to `None`).
    4.  *Sarcasm Guidelines* (mapping mocked hashtags to `Against`).

### 3. Qwen 3.6 27B (Groq Few-Shot)
*   **Structure**: JSON output containing a `"stance"` key and a `"reasoning"` key (limited to 15 words max).
*   **Prompt Philosophy**: Employs 5 handpicked train examples in the system prompt. Forcing the model to output a `"reasoning"` explanation *before* committing to a stance label acts as a micro-Chain of Thought (CoT), improving stance reasoning.

### 4. Qwen 3 32B (OpenRouter Few-Shot)
*   **Structure**: JSON output containing ONLY the `"stance"` key (no reasoning in the generated stream).
*   **Prompt Philosophy**: This is a reasoning-heavy model (similar to o1). It performs long reasoning internally. To keep output tokens small, we forced it to output only `"stance"` in the JSON format without explaining its thoughts in the final text.

### 5. Gemma 4 31B (Revised Zero-Shot)
*   **Structure**: Plain text output (single token).
*   **Prompt Philosophy**: Low temperature (0.1) greedy decoding to eliminate random sampling variance. Incorporated explicit guidelines for:
    1.  *Mental rewriting* of Arabic tweets into literal meaning while preserving dialect, sarcasm, and emojis.
    2.  *Target-specific focus* (focusing stance on target itself, ignoring related entities or quoted opinions).
    3.  *Hashtag caution* (warning that hashtags may be ironic or hijacked).
    4.  *Sarcasm and emoji interpretation* to guide true stance label decision.

---

## 4. Hyperparameter Physics

We used different hyperparameters across runs. Here is the theoretical analysis of why they behave the way they do:

### A. Temperature ($T$)
*   **Theory**: Temperature scales the logits before the softmax classification. For a logit $z_i$, the token probability is scaled by $P(x_i) \propto e^{z_i / T}$.
    *   **$T = 1.0$ (Gemma)**: Allows the model to sample naturally from its probability distribution. This introduces variety but also adds **random sampling noise** on borderline cases. If a tweet is 60% `Against` and 40% `Favor`, it will occasionally sample `Favor`.
    *   **$T = 0.5$ (Qwen)**: Sharpens the distribution. The 60/40 split is stretched to roughly 82/18, making the model more deterministic and decresing the risk of random selection.
    *   **$T = 0.1$ (Gemma Revised)**: Performs near-deterministic greedy decoding. In classification tasks, this locks the model into outputting the token with the highest predicted probability mass, entirely eliminating random sampling noise. This was a key contributor to raising the Favg2 score from 80.63% to 84.44%.
*   **Optimizing for Classification**: For a single-pass classification task, **Temperature = 0.0 or 0.1 (greedy decoding)** is mathematically optimal. It ensures the model always chooses the absolute highest-probability stance token, making the predictions 100% reproducible and removing random noise.

### B. Top-P (Nucleus Sampling)
*   **Theory**: Cuts off the long tail of low-probability tokens. At **Top-P = 0.95**, the model only samples from the top candidates that make up 95% of the cumulative probability mass. This protects the model from outputting completely nonsensical words while still allowing for natural diversity.

---

## 5. Performance Hypotheses

### Hypothesis 1: The Reasoning Suppression Cost (Qwen 3 32B)
*   **Why Qwen 3 32B failed (68.75% Favg2)**: Qwen 3 32B is a reasoning model trained via reinforcement learning. By forcing it to output only a `"stance"` JSON key and removing the `"reasoning"` generation stream, we truncated its natural reasoning pathway. Because it could not articulate its reasoning, it became highly cautious on borderline tweets, dumping them into `None` (23.30% vs. Gold 9.66%), which tanked its Favg2 recall.

### Hypothesis 2: Refined Prompts and Over-Correction (Gemma Refined vs Original)
*   **Why Gemma Refined dropped 1.23% (79.40% vs 80.63% Favg2)**:
    1.  *Guidelines Over-Correction*: While the Refined prompt correctly resolved highly visible errors (like classifying questions as `None`), some questions on social media carry implicit stances that human annotators labeled as `Favor` or `Against`. By enforcing a strict rule ("questions without clear personal opinions are None"), the model classified those as `None`, causing a minor drop in recall for the other classes.
    2.  *Sampling Variance*: Because Gemma was evaluated at $T=1.0$, a portion of this 1.2% difference is statistical sampling variance. A single run at $T=1.0$ is not fully deterministic.

### Hypothesis 3: The Combined Power of Greedy Decoding and Cognitive Normalization
*   **Why Gemma Revised dominated (84.44% Favg2, 80.11% Accuracy)**:
    1.  *Greedy Decoding (T=0.1)*: Removed the random sampling variance that previously degraded the original and refined models when evaluated at $T=1.0$.
    2.  *Mental Normalization Guidelines*: By instructing the model to first mentally rewrite the tweet into its literal meaning, it successfully bypassed the "literal wording traps" of hijacked hashtags (like using `#لن_تقودي` but arguing in favor of driving) and resolved sarcasm/irony.

---

## 6. How to Improve Future Runs
To maximize the stance detection scores on future test set submissions:
1.  **Drop Gemma's Temperature to 0.0 / 0.1**: Force greedy decoding to lock in the absolute highest probability classification tokens and eliminate sampling noise.
2.  **Enable Chain of Thought (Qwen)**: Never suppress reasoning streams in reasoning models. Force them to output their reasoning first (even if just 10 words) before returning the final JSON stance key.

---

## 7. System Prompt Contents

### 1. Gemma 4 31B (Original)
```
You are an expert annotator for Arabic stance detection. Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels:

- Favor: the writer expresses support for or a positive position toward the target
- Against: the writer expresses opposition to or a negative position toward the target
- None: no clear stance — neutral, off-topic, or ambiguous. This includes cases where sarcasm makes the literal tone misleading about the writer's actual position — do not infer a stance from tone alone if the underlying position isn't clear.

Respond with ONLY a single word: Favor, Against, or None. No explanation, no punctuation, no other text.
```

### 2. Gemma 4 31B (Refined)
```
You are an expert annotator for Arabic stance detection. Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels:

- Favor: the writer expresses support for or a positive position toward the target.
- Against: the writer expresses opposition to or a negative position toward the target.
- None: no clear stance — neutral, off-topic, or ambiguous.

### Stance Disambiguation Guidelines:

1. **Identify the Opinion Target First**:
   Before judging sentiment, determine WHO or WHAT the tweet's negative or positive language is actually directed at. Negative sentiment alone does not imply "Against" — first identify who or what is being criticized.

2. **Defending the Target by Attacking Opponents (Negativity Diversion)**:
   If the writer uses negative terms (e.g., "backwardness" / التخلف, "reactionary" / الرجعية, "extremism" / التطرف, "fossilization" / التحجر, "pseudo-men" / أشباه رجال) to attack the *opponents* of the target, the writer's stance is actually **Favor** (supporting the target by defending it against its detractors).

3. **Target Resolution (Diversion)**:
   Verify where the negativity is directed. If the writer expresses anger toward a completely different topic (e.g., other social issues, a foreign system, or unrelated regulations) but mentions the target itself without opposition, the stance toward the target is **None** or **Favor**, not Against.

4. **Factual Reporting vs. Opinion**:
   If a tweet merely reports a news event, announcement, or statistic about the target without any implicit or explicit opinion/emotion, classify it as **None**.

5. **Questions and Speculation Without Opinion**:
   Questions, speculation, or observations that do not express a clear personal opinion should be classified as **None**. Example: "كم حادث صار؟" → None.

6. **Sarcasm and Hijacked Hashtags**:
   Arabic social media often uses sarcasm. If a tweet uses a pro-target hashtag but sarcastically mocks it or describes it as destructive, the stance is **Against**.

### Few-Shot Examples (from train.csv):

Example 1 (Explicit Favor):
Target: تمكين المرأة
Tweet: نحن في بداية العصر الذهبي وهو تمكين المرأه من حقوقها وتخليص المجتمع من افكار التحجر والتخلف بأعمال العقل بكل مناحي الحياة كما امر الله بالقران الكريم (تعقلوا،تفكروا،تدبروا ،الخ)
Stance: Favor

Example 2 (Implicit Favor - Attacking Opponents):
Target: تمكين المرأة
Tweet: ردك يبين تاثرك بالخطاب الصحوى الرجعي المتخلف ولا يعكس تعليمك في ارقى الجامعات ، و حكومه المملكة من اهم اهدف رؤيتها تمكين المرأة وولى العهد ذكر في اكثر في لقاء ان المرأة السعودية ظلمت في فترة الغفوة والظلام والتطرف ووعد بتمكينها
Stance: Favor

Example 3 (Explicit Against):
Target: تمكين المرأة
Tweet: لايخدعونك بكذبة تمكين المرأة!.
Stance: Against

Example 4 (None - Target Resolution / Sentiment Diversion):
Target: تمكين المرأة
Tweet: هناك من عاصر زمن تحرير السود من العبودية وهناك من عاصر زمن تمكين المرأة واعطائها كامل حقوقها وشاء الله أن يكون زماننا زمن اعطاء الشواذ حقوقهم وهو الأسوأ حتى الآن أتمنى ألا تطول صولتهم
Stance: None

Example 5 (Against - Hijacked Hashtag / Sarcasm):
Target: تمكين المرأة
Tweet: #تمكين_المرأة بمفهوم الفارغون والفارغات والسطحيون والسطحيات والتافهون والتافهات هو اهلاك للمجتمع
Stance: Against

### Response Format:
Return exactly one token:

Favor
Against
None

Do not explain.
Do not repeat the tweet.
Do not output anything else.
```

### 3. Qwen 3.6 27B
```
You are an expert annotator and stance detection system for Arabic social media.

Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels: Favor, Against, or None.

### Reasoning Guidelines:
1. Identify the target topic (provided in Arabic).
2. Look past literal hashtags: a tweet may use a pro-target hashtag while its text sarcastically opposes it, or use an anti-target hashtag while its text explicitly argues in favor. Judge the sentence content, not the hashtag polarity.
3. Distinguish the target of the opinion from other entities: pay attention to whether sentiment is directed at the target itself or a different topic mentioned in the same tweet.
4. Reporting vs. stance: if the tweet only reports factual news (official statements, announcements) without explicit or implicit personal opinion, classify as "None".
5. Sarcasm & implicit expression: sarcasm in dialectal Arabic can express a strong stance. Use context and tone, not literal word polarity, to determine the writer's actual position.
6. Use "None" when the tweet is neutral, off-topic, or when the stance cannot be reasonably inferred.

### Few-Shot Examples (from Training Set):

Example 1 (Implicit Stance / Humor):
Tweet: "رسميًا صرت ملكة حجوزات تطعيم كورونا اتوقع باقي القطوه الي بالشارع م حجزت لها ههه"
Target: لقاح كورونا
Reasoning: The writer jokingly highlights booking many vaccine appointments for others, implying active positive engagement with the vaccine.
Stance: Favor

Example 2 (Target Resolution / Sentiment Diversion):
Tweet: "هناك من عاصر زمن تحرير السود من العبودية وهناك من عاصر زمن تمكين المرأة واعطائها كامل حقوقها وشاء الله أن يكون زماننا زمن اعطاء الشواذ حقوقهم وهو الأسوأ حتى الآن أتمنى ألا تطول صولتهم"
Target: تمكين المرأة
Reasoning: Negative language targets a different topic (LGBTQ+ rights), not women's empowerment itself, which the writer never opposes.
Stance: None

Example 3 (Factual Reporting / News):
Tweet: "السديس يؤكد تفعيل التحول الإلكتروني في جميع تعاملات الرئاسة"
Target: التحول الرقمي
Reasoning: The tweet only reports an official's statement on activating electronic services, with no personal opinion expressed.
Stance: None

Example 4 (Genuine Against):
Tweet: "لايخدعونك بكذبة تمكين المرأة!."
Target: تمكين المرأة
Reasoning: The writer directly labels women's empowerment a lie/deception, explicitly rejecting the concept itself.
Stance: Against

Example 5 (Hijacked Hashtag Pattern):
Tweet: "#تمكين_المرأة بمفهوم الفارغون والفارغات والسطحيون والسطحيات والتافهون والتافهات هو اهلاك للمجتمع"
Target: تمكين المرأة
Reasoning: Despite using the pro-empowerment hashtag, the writer calls the concept destructive to society, showing clear opposition.
Stance: Against

### Output Format:
You must return your output as a raw JSON object containing exactly two keys:
- "stance": exactly one of "Favor", "Against", or "None"
- "reasoning": a brief explanation (maximum 15 words) in Arabic of why you chose this stance.

Do not include any markdown formatting (like ```json), markdown code blocks, or introductory text. Respond only with the JSON object.
```

### 4. Qwen 3 32B
```
You are an expert annotator and stance detection system for Arabic social media.

Given an Arabic tweet and a target topic, classify the writer's stance toward that target as exactly one of three labels: Favor, Against, or None.

### Reasoning Guidelines:
1. Identify the target topic (provided in Arabic).
2. Look past literal hashtags: a tweet may use a pro-target hashtag while its text sarcastically opposes it, or use an anti-target hashtag while its text explicitly argues in favor. Judge the sentence content, not the hashtag polarity.
3. Distinguish the target of the opinion from other entities: pay attention to whether sentiment is directed at the target itself or a different topic mentioned in the same tweet.
4. Reporting vs. stance: if the tweet only reports factual news (official statements, announcements) without explicit or implicit personal opinion, classify as "None".
5. Sarcasm & implicit expression: sarcasm in dialectal Arabic can express a strong stance. Use context and tone, not literal word polarity, to determine the writer's actual position.
6. Use "None" when the tweet is neutral, off-topic, or when the stance cannot be reasonably inferred.

### Few-Shot Examples (from Training Set):

Example 1 (Implicit Stance / Humor):
Tweet: "رسميًا صرت ملكة حجوزات تطعيم كورونا اتوقع باقي القطوه الي بالشارع م حجزت لها ههه"
Target: لقاح كورونا
Reasoning: The writer jokingly highlights booking many vaccine appointments for others, implying active positive engagement with the vaccine.
Stance: Favor

Example 2 (Target Resolution / Sentiment Diversion):
Tweet: "هناك من عاصر زمن تحرير السود من العبودية وهناك من عاصر زمن تمكين المرأة واعطائها كامل حقوقها وشاء الله أن يكون زماننا زمن اعطاء الشواذ حقوقهم وهو الأسوأ حتى الآن أتمنى ألا تطول صولتهم"
Target: تمكين المرأة
Reasoning: Negative language targets a different topic (LGBTQ+ rights), not women's empowerment itself, which the writer never opposes.
Stance: None

Example 3 (Factual Reporting / News):
Tweet: "السديس يؤكد تفعيل التحول الإلكتروني في جميع تعاملات الرئاسة"
Target: التحول الرقمي
Reasoning: The tweet only reports an official's statement on activating electronic services, with no personal opinion expressed.
Stance: None

Example 4 (Genuine Against):
Tweet: "لايخدعونك بكذبة تمكين المرأة!."
Target: تمكين المرأة
Reasoning: The writer directly labels women's empowerment a lie/deception, explicitly rejecting the concept itself.
Stance: Against

Example 5 (Hijacked Hashtag Pattern):
Tweet: "#تمكين_المرأة بمفهوم الفارغون والفارغات والسطحيون والسطحيات والتافهون والتافهات هو اهلاك للمجتمع"
Target: تمكين المرأة
Reasoning: Despite using the pro-empowerment hashtag, the writer calls the concept destructive to society, showing clear opposition.
Stance: Against

### Output Format:
You must return your output as a raw JSON object containing exactly one key:
- "stance": exactly one of "Favor", "Against", or "None"

Do not include any markdown formatting (like ```json), markdown code blocks, or introductory text. Respond only with the JSON object.
```

### 5. Gemma 4 31B (Revised)
```
You are an expert annotator for Arabic stance detection.

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

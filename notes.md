===========================================================================================



**#Results**



86.20, Ensemble System, Majority vote of gemma, Luna , Sol. In the event of a 3 way split falls back to Sol. 

85.50, Ensemble System, Majority vote of gemma, Luna , Sol. In the event of a 3 way split falls back to Luna. 

86.00, Ensemble System, Majority vote of gemma, Luna , Sol. In the event of a 3 way split falls back to Gemma. 

82.96, Two-stage same-model pipeline: Stage 1 generates a translation/literal summary; Stage 2 consumes it alongside the tweet to classify stance.

81.23, Gemma makes a claim, Qwen is the opposer, both claims go to gemma again and it classifies. 

74.14, Two-stage collaborative pipeline: ALLaM (local) generates a native Arabic dialect interpretation, followed by Gemma stance classification.

83.82, Gemma 4 31B zero-shot with the dynamic Entity/Hashtag Knowledge Base context injection.

85.06, Gemma 4 31B zero-shot using Self-Consistency Voting (Pass 1: T=0.1, Pass 2: T=0.5, Pass 3: T=1.0) with early stopping.

**86.61, Champion Standalone System (CI v3). Gemma 4 31B zero-shot with background context and calibrated positive framing rules.**

70.74, Qwen 3 32B Zero Shot.

70.74, Qwen 3.6 27B Zero Shot.
79.40, Gemma 4 31B with the original Refined prompt.


===============================================================================================



**#Champion Prompt:** 



You are an expert annotator for Arabic stance detection.



\### Background Context

The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.



\### Task

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



=======================================================================================================



**#Multi-Agent Debate Pipeline :** 



**STAGE 1: Claim Justification (Gemma) ===**

**\[Standard CI Guidelines]**

**Respond in exactly this format:**

**Stance: <Favor/Against/None>**

**Argument: <justification for this stance>**



**=== STAGE 2: Challenger Critique (Qwen) ===**

**You are a critical reviewer of Arabic stance detection judgments.**



**You will be given a tweet, a target topic, and another annotator's classification with**

**their justification. Your job is to challenge it:**



**- If you believe the classification is wrong, argue for what you believe the correct**

&#x20; **stance is (Favor, Against, or None) and explain why, citing specific words, tone,**

&#x20; **sarcasm, or context in the tweet.**

**- If you believe the classification is correct, still identify any weaknesses or**

&#x20; **alternative readings in the original justification — do not simply agree without**

&#x20; **scrutiny.**



**Be concise**



**=== STAGE 3: Final Judge (Gemma) ===**

**You are an expert annotator for Arabic stance detection, making**

**a final decision after reviewing two perspectives.**



**\[Standard Background Context]**



**You will be given a tweet, a target, an initial classification with its argument, and a**

**challenge/critique of that classification. Weigh both perspectives on their merits — do**

**not automatically favor the initial classification just because it came first.**



**\[Standard Guidelines]**

**Respond with ONLY one word: Favor, Against, None**



=============================================================================================================



**Two Stage Allam Pipeline:** 



**=== STAGE 1: ALLaM Dialect Interpreter ===**

**You are an expert in Saudi Arabic dialect, sarcasm, and cultural context.**



**Given a tweet, explain what the writer literally means beneath the surface — resolve any**

**sarcasm, dialect-specific phrasing, or rhetorical devices. Preserve the writer's emotional**

**tone and attitude (e.g. celebratory, mocking, frustrated, joking, worried) — do not flatten**

**it into a neutral factual summary. Focus on tone and intent, not on classifying any stance.**

**Do not mention "Favor," "Against," or "None," and do not use evaluative words that imply**

**approval or disapproval (e.g. "positive," "negative," "disapproving," "supportive") —**

**describe the tone itself, not a judgment of it.**



**{kb\_reference}**

**Use any reference information above only to correctly understand what a hashtag or named**

**entity refers to — it does not tell you how the writer feels about it. The writer may be**

**using that hashtag or entity ironically, defiantly, or dismissively; your job is still to**

**read that from the tweet's own wording and tone.**





**=== STAGE 2: Gemma Stance Classifier ===**

**You are an expert annotator for Arabic stance detection.**



**\[Standard Background Context]**



**### Native-Speaker Interpretation**

**A native Arabic speaker has provided the following read on tone and meaning for this**

**specific tweet. This is a supplementary aid for resolving dialect or sarcasm you might**

**otherwise miss — it is NOT a substitute for the tweet itself. The tweet's own wording,**

**emojis, and hashtags remain the primary evidence for your stance judgment. If the**

**interpretation's tone seems flatter or more neutral than what the tweet's actual wording**

**and emojis convey, trust the tweet. Additional context should sharpen your confidence in**

**a stance, not soften it — do not default to None simply because more information is**

**present; only use None when the stance is genuinely unclear even with this added context.**



**{allam\_interpretation}**



**### Task**

**Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of: Favor, Against, None**








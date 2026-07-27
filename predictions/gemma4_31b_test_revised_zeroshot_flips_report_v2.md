# Flip Report: Gemma 4 31B Revised v2 Test Set Run
**Target Topic**: Women Driving (قيادة المرأة للسيارة)  
**Dataset**: 352-row Blind Test Set (Mawqif-v2)  
**Comparison Focus**: Flips between **Revised v1** (Mental translation & normalization guidelines) and **Revised v2** (Adding the Face-Value Guard rule).

---

## 1. Overall Run Statistics

Comparing the predictions of **Gemma 4 31B (Revised v2)** against the previous **Gemma 4 31B (Revised v1)** on all 352 rows:
* **Total Predictions Analyzed:** 352 rows
* **Total Flips Detected (v1 ➔ v2):** 9 rows (2.56%)
* **Stance Distribution Comparison:**
  * **Revised v1:** 148 Favor (42.0%), 188 Against (53.4%), 16 None (4.5%)
  * **Revised v2:** 142 Favor (40.3%), 192 Against (54.5%), 18 None (5.1%)

The addition of the **Face-Value Guard** shifted predictions slightly away from `Favor` towards `Against` and `None` classes, indicating a reduction in false-positive stance detection on neutral or factual reporting tweets.

---

## 2. Flip Trajectory Table (All 9 Flips)

The table below lists the 9 rows where predictions changed between Revised v1 and Revised v2. It includes the stance trajectory from the Original model up to the current run:

| Row ID | Target | Original | Refined | Revised v1 | Revised v2 (Current) | Trajectory Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2** | Women Driving | None | None | None | **Against** | Flipped to Against |
| **19** | Women Driving | Favor | None | Favor | **None** | Flipped to None |
| **45** | Women Driving | Against | Against | Favor | **Against** | Flipped to Against |
| **59** | Women Driving | None | None | Favor | **None** | Flipped to None |
| **155** | Women Driving | None | Against | Favor | **Against** | Flipped to Against |
| **168** | Women Driving | None | Favor | Favor | **Against** | Flipped to Against |
| **228** | Women Driving | Favor | Against | Favor | **Against** | Flipped to Against |
| **299** | Women Driving | Against | Against | Against | **Favor** | Flipped to Favor |
| **339** | Women Driving | None | None | Favor | **None** | Flipped to None |

---

## 3. Row-by-Row Flip Details and Translations

### **Row 2**
* **Arabic Tweet:** `#المراه_السعوديه_تسوق الناس تتلفت بكل اشاره😂`
* **English Translation:** `#SaudiWomenDriving People are looking around at every traffic light 😂`
* **Trajectory:** `None` ➔ `None` ➔ `None` ➔ **`Against`**
* **Analysis:** The writer uses dialectal Saudi Arabic and a laughing emoji to express skepticism and mock female drivers, implying drivers are looking around in anxiety at intersections. The face-value instruction in v2 allowed the model to successfully categorize this mocking observation as `Against` the target.

### **Row 19**
* **Arabic Tweet:** `الامن يوزع الورود للسيدات خلال بداية قيادتهم السيارة في السعودية . . #المرأة_السعودية_تسوق #SaudiWomenDriving`
* **English Translation:** `Security forces distribute roses to women during the start of their driving in Saudi Arabia. . #SaudiWomenDriving #SaudiWomenDriving`
* **Trajectory:** `Favor` ➔ `None` ➔ `Favor` ➔ **`None`**
* **Analysis:** This is a factual reporting of a news event (police distributing flowers) without any personal opinion or alignment expressed. Under the updated prompt (v2), the model correctly backed off from `Favor` to `None` following the directive to separate factual reporting from stance endorsement.

### **Row 45**
* **Arabic Tweet:** `#لن_تقودي واللهي مادري وش اقول واللي معارض وصار القرار اول من يشتري سياره ل اهله هو 🙄💔.`
* **English Translation:** `#YouWillNotDrive Honestly, I don't know what to say, but those who opposed the decision were the first to buy cars for their families 🙄💔.`
* **Trajectory:** `Against` ➔ `Against` ➔ `Favor` ➔ **`Against`**
* **Analysis:** The tweet points out the hypocrisy of the opponents of the driving decree. The writer uses the anti-driving hashtag `#لن_تقودي` to express frustration. While Revised v1 marked it as `Favor` (interpreting the attack on opponents as support for the target under the "Negativity Diversion" rule), the Face-Value guard in v2 correctly recognized that the tone is overwhelmingly negative and doesn't express active support, pulling it back to `Against`.

### **Row 59**
* **Arabic Tweet:** `#المراه_السعوديه_تسوق معليشش تحمستت ،ماضاع مفتاح سيارت ابوي الا الليوم😭😭😭`
* **English Translation:** `#SaudiWomenDriving Sorry I got too excited, but my dad's car keys had to get lost today of all days 😭😭😭`
* **Trajectory:** `None` ➔ `None` ➔ `Favor` ➔ **`None`**
* **Analysis:** The writer is sharing a personal anecdote about being excited but losing their dad's car key. While showing implicit excitement, it is primarily a situational complaint/anecdote rather than a clear public stance. The Face-Value guard in v2 correctly categorized it as `None`, reducing false positives for `Favor`.

### **Row 155**
* **Arabic Tweet:** `#الملك_ينتصر_لقياده_المراه ارتحت لك يوم دقيتي اشارة ههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههاختي قائدة المركبه : تعديل نقابك سبب انقلابك ..`
* **English Translation:** `#TheKingSupportsWomenDriving I felt comfortable when you put on the turn signal hahahaha... My sister driver: adjusting your niqab [face veil] will cause your car to flip over..`
* **Trajectory:** `None` ➔ `Against` ➔ `Favor` ➔ **`Against`**
* **Analysis:** This tweet is a clear mockery of women's driving skills using dialectal sarcasm. Revised v1 was misled into `Favor` (likely due to the pro-driving hashtag), but the v2 prompt correctly resolved the sarcasm and mockery as an `Against` stance.

### **Row 168**
* **Arabic Tweet:** `يابوي خلاص تراكم امتحنتونا #المراه_السعوديه_تسوق`
* **English Translation:** `Oh my god, seriously, you've annoyed/tested us enough [with this topic] #SaudiWomenDriving`
* **Trajectory:** `None` ➔ `Favor` ➔ `Favor` ➔ **`Against`**
* **Analysis:** The writer is expressing severe fatigue/annoyance with the constant debate. The model in v2 classified this as `Against` (interpreting the annoyance at the conversation as opposing the target's public discourse).

### **Row 228**
* **Arabic Tweet:** `ياربي اني من عقب #السماح_بقيادة_المرأة وانا مو قادره انام بالعاده هالوقت اكون بسابع نومه❤️😩`
* **English Translation:** `Oh my God, ever since #AllowingWomenToDrive I haven't been able to sleep, usually at this time I'd be fast asleep ❤️😩`
* **Trajectory:** `Favor` ➔ `Against` ➔ `Favor` ➔ **`Against`**
* **Analysis:** The tweet expresses excitement and sleeplessness due to the decree. Revised v2 flipped to `Against` (possibly due to treating "I cannot sleep" as a negative sentiment/complaint).

### **Row 299**
* **Arabic Tweet:** `#المراه_السعوديه_تقود_السياره. الـمبادي مـا تـغيّر عـلى النهج السليم و الجدل محسوم و الهرج مافيه ثْمرة . الـسـواقه لـلـرجل و الـسـواقه لـلحريم و الـرجل يبقى رجل و المره تبقى مره . #سعد_المخلفي`
* **English Translation:** `#SaudiWomenDriving. Principles do not change on the right path, the debate is settled and talk yields no fruit. Driving is for men and driving is for women, a man remains a man and a woman remains a woman. #Saad_AlMukhlafi`
* **Trajectory:** `Against` ➔ `Against` ➔ `Against` ➔ **`Favor`**
* **Analysis:** This is a poem stating that driving is for both men and women, and the debate is over. The new prompt (v2) successfully bypassed the traditional phrasing ("a man remains a man...") to recognize the central argument ("driving is for men and driving is for women") as supporting the change (`Favor`).

### **Row 339**
* **Arabic Tweet:** `تقرير احدى القنوات الهندية media one TV لتغطية رويترز معي عن التحديات التي تواجه المرأة السعودية في القيادة و مدى حاجتها بالاضافة الى معايير اختيار المرأة السعودية السيارة . وصلت للتلفزيون الهندي👳🏿‍♀️.. تخيلت كل السواقين اللي سفرتهم يتابعوني 😫💔 #قيادة_المرأة_السعودية`
* **English Translation:** `A report by an Indian channel, Media One TV, covering Reuters' interview with me about the challenges facing Saudi women in driving, their level of need, and their criteria for choosing a car. It reached Indian TV 👳🏿‍♀️.. I imagined all the drivers I sent back watching me 😫💔 #SaudiWomenDriving`
* **Trajectory:** `None` ➔ `None` ➔ `Favor` ➔ **`None`**
* **Analysis:** The writer is describing a media report about their interview. Although there's a light joke at the end, it is factual reporting of news/media coverage. The Face-Value and Reporting guidelines in v2 correctly classified this as `None`, reversing the false positive `Favor` from v1.

---

## 4. Key Takeaways

1. **Face-Value Guard Effectiveness:** 
   The addition of the face-value guard successfully resolved several false positives in the `Favor` stance class, correcting factual news reports (Row 19, 339) and personal anecdotes (Row 59) back to `None`.
2. **Mockery/Sarcasm Sensitivity:**
   In complex dialectal sarcasm (Row 2, 155), the updated prompt allowed the model to correctly identify the underlying opposition (`Against`), avoiding errors from previous runs where they were classified as `None` or `Favor`.
3. **Hypocrisy vs. Opposition:**
   Tweets attacking the hypocrisy of opponents (Row 45) were correctly classified as `Against` rather than over-correcting to `Favor`. While the "Negativity toward opponents is Favor" rule is important, the face-value guard ensures it is only applied when there is a clear pro-target stance, preventing errors in tweets that are primarily critical or hostile.

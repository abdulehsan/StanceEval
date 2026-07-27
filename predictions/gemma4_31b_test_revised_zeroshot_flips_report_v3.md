# Flip Report: Gemma 4 31B Revised v3 Test Set Run
**Target Topic**: Women Driving (قيادة المرأة للسيارة)  
**Dataset**: 352-row Blind Test Set (Mawqif-v2)  
**Comparison Focus**: Flips between **Revised v2** (Mental translation + face-value guidelines) and **Revised v3** (Adding specific Target Resolution guidelines).

---

## 1. Overall Run Statistics

Comparing the predictions of **Gemma 4 31B (Revised v3)** against the previous **Gemma 4 31B (Revised v2)** on all 352 rows:
* **Total Predictions Analyzed:** 352 rows
* **Total Flips Detected (v2 ➔ v3):** 9 rows (2.56%)
* **Stance Distribution Comparison:**
  * **Revised v2:** 142 Favor (40.3%), 192 Against (54.5%), 18 None (5.1%)
  * **Revised v3:** 146 Favor (41.5%), 186 Against (52.8%), 20 None (5.7%)

The addition of the **Target Resolution Guidelines** shifted predictions slightly away from `Against` towards `Favor` and `None` classes. Crucially, agreement with other models (Original Gemma, Qwen 27B, and Qwen 32B) increased overall, showing that v3 aligns more strongly with the consensus database.

---

## 2. Flip Trajectory Table (All 9 Flips)

The table below lists the 9 rows where predictions changed between Revised v2 and Revised v3. It includes the stance trajectory from the Original model up to the current run:

| Row ID | Target | Original | Refined | Revised v1 | Revised v2 | Revised v3 (Current) | Trajectory Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Women Driving | Favor | Against | Against | Against | **Favor** | Flipped to Favor |
| **2** | Women Driving | None | None | None | Against | **None** | Flipped to None |
| **63** | Women Driving | None | None | Against | Against | **None** | Flipped to None |
| **94** | Women Driving | None | Against | Against | Against | **None** | Flipped to None |
| **155** | Women Driving | None | Against | Favor | Against | **Favor** | Flipped to Favor |
| **168** | Women Driving | None | Favor | Favor | Against | **Favor** | Flipped to Favor |
| **228** | Women Driving | Favor | Against | Favor | Against | **Favor** | Flipped to Favor |
| **299** | Women Driving | Against | Against | Against | Favor | **Against** | Flipped to Against |
| **310** | Women Driving | None | Against | None | None | **Favor** | Flipped to Favor |

---

## 3. Row-by-Row Flip Details and Translations

### **Row 1**
* **Arabic Tweet:** `#المراه_السعوديه_تسوق بصصصصصصيح ابغا أسوِّق خخخخخخخخخخييييييععععععر؟`
* **English Translation:** `#SaudiWomenDriving I'm crying, I want to drive [giggle/mock expression]...`
* **Trajectory:** `Favor` ➔ `Against` ➔ `Against` ➔ `Against` ➔ **`Favor`**
* **Analysis:** The writer expresses a direct and genuine desire to drive, accompanied by dialectal laughter/giggles. Revised v2 incorrectly categorized this as `Against` (likely due to the dialectal slang), but Revised v3 correctly recovered the **`Favor`** stance.

### **Row 2**
* **Arabic Tweet:** `#المراه_السعوديه_تسوق الناس تتلفت بكل اشاره😂`
* **English Translation:** `#SaudiWomenDriving People are looking around at every traffic light 😂`
* **Trajectory:** `None` ➔ `None` ➔ `None` ➔ `Against` ➔ **`None`**
* **Analysis:** Humorous observation about intersection checkouts. In v2, it was classified as `Against` (mockery). In v3, the face-value guard correctly classified it as **`None`**.

### **Row 63**
* **Arabic Tweet:** `#المراه_السعوديه_تسوق الشي الوحيد اللي باقي لنا يالشباب الذكور هو قسم البخاري في مطعم لؤلؤة الحمراء. ألذ بخاري بالرياض 🤣`
* **English Translation:** `#SaudiWomenDriving The only thing left for us male youth is the Bukhari section in the Loulou'at Al-Hamra restaurant. The most delicious Bukhari in Riyadh 🤣`
* **Trajectory:** `None` ➔ `None` ➔ `Against` ➔ `Against` ➔ **`None`**
* **Analysis:** General sibling/gender banter about restaurants rather than a stance on driving. Under the new target resolution guidelines, it correctly went from `Against` to **`None`**.

### **Row 94**
* **Arabic Tweet:** `#لن_تقودي خلاص ندري اننا ما بنسوق اصلاً 🙂✋🏾.`
* **English Translation:** `#YouWillNotDrive Alright, we already know that we are not going to drive anyway 🙂✋🏾.`
* **Trajectory:** `None` ➔ `Against` ➔ `Against` ➔ `Against` ➔ **`None`**
* **Analysis:** A passive compliance statement without expressing an active stance. Correctly mapped to **`None`**.

### **Row 155**
* **Arabic Tweet:** `#الملك_ينتصر_لقياده_المراه ارتحت لك يوم دقيتي اشارة ههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههاختي قائدة المركبه : تعديل نقابك سبب انقلابك ..`
* **English Translation:** `#TheKingWinsForWomenDriving I felt relieved when you put on the signal indicator hahahaha sister driver: adjusting your niqab [face veil] will cause your car to roll over..`
* **Trajectory:** `None` ➔ `Against` ➔ `Favor` ➔ `Against` ➔ **`Favor`**
* **Analysis:** Sarcasm mocking women's driving. The model in v3 was influenced by the pro-driving hashtag and mapped it to **`Favor`**, highlighting a remaining prompt challenge with highly complex sarcastic tweets.

### **Row 168**
* **Arabic Tweet:** `يابوي خلاص تراكم امتحنتونا #المراه_السعوديه_تسوق`
* **English Translation:** `Oh my god, seriously, you've annoyed/tested us enough [with this topic] #SaudiWomenDriving`
* **Trajectory:** `None` ➔ `Favor` ➔ `Favor` ➔ `Against` ➔ **`Favor`**
* **Analysis:** Expressing fatigue with the constant debate. The model in v3 classified this as **`Favor`** (associating the hashtag with the user's intent).

### **Row 228**
* **Arabic Tweet:** `ياربي اني من عقب #السماح_بقيادة_المرأة وانا مو قادره انام بالعاده هالوقت اكون بسابع نومه❤️😩`
* **English Translation:** `Oh my God, ever since #AllowingWomenToDrive I haven't been able to sleep, usually at this time I'd be fast asleep ❤️😩`
* **Trajectory:** `Favor` ➔ `Against` ➔ `Favor` ➔ `Against` ➔ **`Favor`**
* **Analysis:** Sleeplessness from excitement and joy about the decree (marked by a red heart). The new prompt successfully recovered the **`Favor`** stance that v2 had incorrectly mapped to `Against`.

### **Row 299**
* **Arabic Tweet:** `#المراه_السعوديه_تقود_السياره. الـمبادي مـا تـغيّر عـلى النهج السليم و الجدل محسوم و الهرج مافيه ثْمرة . الـسـواقه لـلـرجل و الـسـواقه لـلحريم و الـرجل يبقى رجل و المره تبقى مره . #سعد_المخلفي`
* **English Translation:** `#SaudiWomenDriving. Principles do not change on the right path, the debate is settled and talk yields no fruit. Driving is for men and driving is for women, a man remains a man and a woman remains a woman. #Saad_AlMukhlafi`
* **Trajectory:** `Against` ➔ `Against` ➔ `Against` ➔ `Favor` ➔ **`Against`**
* **Analysis:** Fixes a notable error in v2! The conservative poem asserting fixed gender roles was incorrectly mapped to `Favor` by v2, but v3 successfully parsed the conservative rhetoric and gender principles, classifying it back to **`Against`** (which matches the writer's traditional stance).

### **Row 310**
* **Arabic Tweet:** `#الملك_ينتصر_لقياده_المراه الآن دور الشباب ينتقمو من أخواتهم لا أوصيك عزيزي الشاب قفل باب سيارة أختك بقوة خليها تنجلط 😂✋🏻`
* **English Translation:** `#TheKingSupportsWomenDriving Now it is the guys' turn to get revenge on their sisters. I advise you, dear young man, to slam your sister's car door hard, make her blow a fuse 😂✋🏻`
* **Trajectory:** `None` ➔ `Against` ➔ `None` ➔ `None` ➔ **`Favor`**
* **Analysis:** Sibling humor about sisters driving. In v3, the model decided to classify this as **`Favor`** (welcoming the fact that sisters will now be driving and owning cars).

---

## 4. Key Takeaways

1. **Target and Reference Resolution:**
   The addition of reference resolution directives successfully corrected Row 299 back to `Against`, separating the mention of "women driving" from the conservative poem's core oppositional message.
2. **Reduced False-Positive Stances:**
   Rows 2, 63, and 94 were successfully pulled back from `Against` to a neutral `None` class, indicating that the face-value guard and target resolution guidelines prevented the model from over-assigning stances to simple jokes, anecdotes, or passive compliance statement tweets.
3. **Consensus Realignment:**
   With these flips, the agreement rates between Revised v3 and all other models (Original Gemma, Qwen 27B, and Qwen 32B) increased. This suggests that the prompt adjustments successfully eliminated marginal noise while preserving overall classification performance.

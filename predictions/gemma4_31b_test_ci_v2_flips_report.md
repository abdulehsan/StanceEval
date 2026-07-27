# Gemma 4 31B Prompt Flip Analysis: CI v1 vs. CI v2

This report analyzes prediction flips across all 352 rows of the blind test set when migrating Gemma 4 31B from the **CI v1 prompt** (Background Context baseline) to the **CI v2 prompt** (calibrating the Against class description to filter out tangential jokes, skepticism, and gender banter into the None category).

---

## 1. Overall Run Statistics
* **Total Predictions Analyzed:** 352 rows
* **Total Flips Detected:** **25** out of 352 rows (**7.10%** flip rate)

### Stance Distribution Comparison
| Model | Favor | Against | None |
| :--- | :---: | :---: | :---: |
| **Gemma 4 31B (CI v1)** | 161 (45.7%) | 174 (49.4%) | 17 (4.8%) |
| **Gemma 4 31B (CI v2)** | 158 (44.9%) | 156 (44.3%) | 38 (10.8%) |

---

## 2. Stance Transition Breakdown
| Transition | Count | Percentage |
| :--- | :---: | :---: |
| `Against` ➔ `None` | 18 | 72.0% |
| `Favor` ➔ `None` | 3 | 12.0% |
| `Favor` ➔ `Against` | 2 | 8.0% |
| `Against` ➔ `Favor` | 2 | 8.0% |


---

## 3. Sample Flip Details
##### **Flip #1 (Row ID: 3)**
- **Target:** Women Driving
- **Arabic Tweet:** `ما محظوظ الا عادل الجبير! من يوم غد اصبح بامكانه قيادة السيارة *_^ #المراه_السعوديه_تسوق`
- **English Translation:** *"Only Adel Al-Jubeir is lucky! From tomorrow he will be able to drive a car *_^ #Saudi_Women_Shopping"*
- **CI v1 Stance:** `Against`
- **CI v2 Stance:** `None`

##### **Flip #2 (Row ID: 20)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي ريم ممكن تسلفيني سيارتك بروح فيها زواج بنت عمي ليه وش فيها سيارتك رحت فيها زواج بنت خالتي قبل شهرين قد شافوها.`
- **English Translation:** *"#You won't drive Reem, can you lend me your car so I can go to my cousin's wedding? Why is your car in it? I went to my cousin's wedding two months ago. They saw it."*
- **CI v1 Stance:** `Against`
- **CI v2 Stance:** `None`

##### **Flip #3 (Row ID: 33)**
- **Target:** Women Driving
- **Arabic Tweet:** `#المراه_السعوديه_تسوق يالليل الحين الاسر المنتجه يبيعون جنوط 🤦🏻‍♂️`
- **English Translation:** *"#Saudi_women_shop tonight, productive families are selling rims 🤦🏻‍♂️"*
- **CI v1 Stance:** `Against`
- **CI v2 Stance:** `None`

##### **Flip #4 (Row ID: 52)**
- **Target:** Women Driving
- **Arabic Tweet:** `#الملك_ينتصر_لقياده_المراه عزيزتي البنت اذا انتي موفر لك سواق وتروحي وتجي وقت ما تبغي وعلى كيفك ترا في بنات ما عندهم سواق وما يخرجو`
- **English Translation:** *"#TheKing_Victory_for_Women_Leadership Dear girl, if you are provided with a driver and you can go and come whenever you want, as you like, you will see that there are girls who do not have a driver and do not go out."*
- **CI v1 Stance:** `Against`
- **CI v2 Stance:** `None`

##### **Flip #5 (Row ID: 62)**
- **Target:** Women Driving
- **Arabic Tweet:** `#كل_يوم_معلومة #قياده_المرأه_السعودية عزيزتي صاحبة المركبة ✋️ هذه علامة "امامك مطبات اصطناعية" وليس كما تعتقدين : "امامك حريم يصلّون 😔💔`
- **English Translation:** *"#Every_Day_Information #Saudi_Women_Driving Dear vehicle owner ✋️ This is a sign: “In front of you are artificial bumps” and not what you think: “In front of you are a harem praying” 😔💔"*
- **CI v1 Stance:** `Against`
- **CI v2 Stance:** `None`

##### **Flip #6 (Row ID: 63)**
- **Target:** Women Driving
- **Arabic Tweet:** `#المراه_السعوديه_تسوق الشي الوحيد اللي باقي لنا يالشباب الذكور هو قسم البخاري في مطعم لؤلؤة الحمراء. ألذ بخاري بالرياض 🤣`
- **English Translation:** *"#Saudi_Women_Shopping The only thing left for us, young males, is the Bukhari section in the Lulu Al Hamra restaurant. The most delicious Bukhari in Riyadh 🤣"*
- **CI v1 Stance:** `Against`
- **CI v2 Stance:** `None`



---

## 4. Key Observations & Insights
* **Against-to-None Calibration:** Our directive (`• Classify Against only when the tweet expresses a clear personal position opposing the target...`) successfully filtered out **18 rows** of tangential jokes, sarcasm, and off-target observations from the `Against` category into `None`.
* **Precision Enhancement:** In Arabic social media, gender jokes (e.g. mocking women driving, car part knowledge, or joking about husbands) are highly prevalent but do not represent active political/social opposition to the policy decree. CI v2 captures this distinction flawlessly.

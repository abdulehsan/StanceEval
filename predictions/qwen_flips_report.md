# Qwen Model Prediction Flips Analysis Report

This report analyzes prediction flips across the 352-row test set for both **Qwen 3.6 27B** and **Qwen 3 32B** when migrating from their initial **Few-Shot JSON prompt** configurations to the zero-shot **Revised v1 prompt** (Mental Translation + Target Normalization Guidelines).

---

## 1. Qwen 3.6 27B Flip Analysis
* **Total Prediction Flips:** **62** out of 352 rows (**17.61%** flip rate)

### Stance Transition Breakdown
| Transition | Count | Percentage |
| :--- | :---: | :---: |
| `Against` ➔ `Favor` | 33 | 53.2% |
| `Favor` ➔ `None` | 10 | 16.1% |
| `Against` ➔ `None` | 8 | 12.9% |
| `Favor` ➔ `Against` | 5 | 8.1% |
| `None` ➔ `Against` | 4 | 6.5% |
| `None` ➔ `Favor` | 2 | 3.2% |


### Sample Flip Details
##### **Flip #1 (Row ID: 12)**
- **Target:** Women Driving
- **Arabic Tweet:** `حنا مانبي نسوق اذا حليييتو المشااكل اللي تمر فيها المرأة ذيك اللحين طالبو بالسواقه #لن_تقودي`
- **English Translation:** *"We don't want to drive. If you solve the problems that a woman with that bad temper goes through, ask for a driver. #You won't_drive"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`

##### **Flip #2 (Row ID: 14)**
- **Target:** Women Driving
- **Arabic Tweet:** `شكل اللي سوت هـ الهشتاق تحلم كثيررر .. #سنسوق_فوق_خشومكم #لن_تقودي`
- **English Translation:** *"The way you used this hashtag is dreaming a lot.. #We_will_drive_over_your_faces #You_will_not_lead"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`

##### **Flip #3 (Row ID: 22)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي الحمدلله مافيه قياده والا كان يجونك نوعية اللي [ تدرين من وين شريت سيارتي ؟ من امريكا تلبيه ، بمليون وشوي] 🚶🏿`
- **English Translation:** *"#You_will_not_drive, thank God, there is no driving, otherwise you would be surprised by the type of [Do you know where I bought my car from? From America you can meet it, for a million and a half] 🚶🏿"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`

##### **Flip #4 (Row ID: 50)**
- **Target:** Women Driving
- **Arabic Tweet:** `قادوكم لنا الله الموقده #لن_تقودي`
- **English Translation:** *"They led you to us, may God bless you. You will not lead"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`

##### **Flip #5 (Row ID: 67)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي يُسقط الله عن المرأه ' وجوب صلاة الجماعة وأنتم تخرجونها لتقود السيارة. الشيخ ابن غديان رحمه الله`
- **English Translation:** *"#You_will_not_drive. God waives the woman’s obligation to pray in congregation while you take her out to drive the car. Sheikh Ibn Ghadian, may God have mercy on him"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`



---

## 2. Qwen 3 32B Flip Analysis
* **Total Prediction Flips:** **104** out of 352 rows (**29.55%** flip rate)

### Stance Transition Breakdown
| Transition | Count | Percentage |
| :--- | :---: | :---: |
| `None` ➔ `Favor` | 30 | 28.8% |
| `Against` ➔ `Favor` | 28 | 26.9% |
| `None` ➔ `Against` | 16 | 15.4% |
| `Favor` ➔ `Against` | 15 | 14.4% |
| `Against` ➔ `None` | 11 | 10.6% |
| `Favor` ➔ `None` | 4 | 3.8% |


### Sample Flip Details
##### **Flip #1 (Row ID: 0)**
- **Target:** Women Driving
- **Arabic Tweet:** `لازم ندزلكم من عندنا كويتيه منقبه سيارتها كورلا وتكون عينه تجريبيه واذا اقتنعتوا راح اشيل شاربي بقزازه #قيادة_المرأة_السعودية`
- **English Translation:** *"We must send you a veiled Kuwaiti woman whose car is a Corolla, and it will be a trial sample, and if you are convinced, I will remove my mustache with a glove #Saudi_Woman_Leadership"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`

##### **Flip #2 (Row ID: 6)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي. ان شاء الله نقود حالنا حال شعوب العالم واكيد راح يكون فيه شروط مو عبث وانتو تعودتو الشارع لكم ليه لازم نشارككم ☺️☺️🌹`
- **English Translation:** *"#You won't drive. God willing, we will improve our situation as the people of the world, and there will certainly be conditions that are not absurd, and you are accustomed to the street. Why should we participate with you? ☺️☺️🌹"*
- **Few-Shot Stance:** `Favor`
- **Revised v1 Stance:** `Against`

##### **Flip #3 (Row ID: 15)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي باختصار😄😜😎`
- **English Translation:** *"#You won't drive in short😄😜😎"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`

##### **Flip #4 (Row ID: 17)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي م راح نقود 😂 بس عند الضروره اسفه لازم اقود غصبن عنك ي ورع`
- **English Translation:** *"#You won't drive, I won't have money 😂 but when necessary, sorry, I have to drive against you, my pious"*
- **Few-Shot Stance:** `Favor`
- **Revised v1 Stance:** `Against`

##### **Flip #5 (Row ID: 43)**
- **Target:** Women Driving
- **Arabic Tweet:** `#لن_تقودي شكل الي مسوي التاق صاحب تكسي خايف يقعد ع الصفر !! 🚕`
- **English Translation:** *"#You won't drive. It looks like the one who has the tag, the taxi owner, is afraid to sit at zero!! 🚕"*
- **Few-Shot Stance:** `Against`
- **Revised v1 Stance:** `Favor`



---

## 3. Comparative Observations
* **Qwen 3.6 27B** experienced a flip rate of **17.61%** while **Qwen 3 32B** experienced **29.55%**.
* A significant driver of flips was the transition from `None` or structured few-shot targets to the strict zero-shot `Revised v1` guidelines, which forced explicit target alignment (e.g., resolving indirect stance markers).

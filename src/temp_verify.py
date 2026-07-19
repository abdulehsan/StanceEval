import pandas as pd
# # # from metrics import per_topic_and_overall_metrics, format_metrics_table

# # # df = pd.read_csv("predictions/results_gemma4_31b_cerebras_v1_prompt_arabic_target.csv", keep_default_na=False)

# # # # Check for parse failures first
# # # print("Parse failures:", (df['parse_success'] == False).sum())
# # # print("Predicted label values:", df['predicted_label'].unique())
# # # print("Gold label values:", df['gold_label'].unique())

# # # results = per_topic_and_overall_metrics(
# # #     df,
# # #     true_col="gold_label",
# # #     pred_col="predicted_label",
# # #     target_col="target",
# # # )
# # # print(format_metrics_table(results))
# # # print(df['target'].value_counts())
# # # we = df[df['target'] == 'Women empowerment']
# # # print(we['gold_label'].value_counts())

# # # df_en = pd.read_csv("predictions/results_gemma4_31b_cerebras_v1_prompt.csv", keep_default_na=False)
# # # print(len(df_en))
# # # print(df_en.columns.tolist())

# # # results_en = per_topic_and_overall_metrics(
# # #     df_en,
# # #     true_col="gold_label",
# # #     pred_col="predicted_label",
# # #     target_col="target",
# # # )
# # # print(format_metrics_table(results_en))

# # # merged = df.merge(df_en, on='row_index', suffixes=('_ar', '_en'))
# # # diff = merged[merged['predicted_label_ar'] != merged['predicted_label_en']]
# # # print(diff[['row_index', 'target_ar', 'gold_label_ar', 'predicted_label_ar', 'predicted_label_en']])
# # # print(merged.columns.tolist())
# # train_df = pd.read_csv("data/train.csv", keep_default_na=False)
# # print(train_df.columns.tolist())
# # print(train_df['target'].unique())

# import pandas as pd
# import json

# RESULTS_PATH = "predictions/results_gemma4_31b_cerebras_train_errmine.csv"
# RAW_LOG_PATH = "qwen_raw_logs/results_gemma4_31b_cerebras_train_errmine_raw.jsonl"

# # Remove bad rows from results CSV
# df = pd.read_csv(RESULTS_PATH, keep_default_na=False)
# bad_indices = {62, 63}
# df = df[~df['row_index'].isin(bad_indices)]
# df.to_csv(RESULTS_PATH, index=False)
# print(f"Removed {len(bad_indices)} bad rows, {len(df)} remain")

# # Remove matching lines from JSONL log
# with open(RAW_LOG_PATH, 'r', encoding='utf-8') as f:
#     lines = [json.loads(l) for l in f if l.strip()]
# lines = [l for l in lines if l['row_index'] not in bad_indices]
# with open(RAW_LOG_PATH, 'w', encoding='utf-8') as f:
#     for l in lines:
#         f.write(json.dumps(l, ensure_ascii=False) + '\n')
# print(f"Cleaned JSONL, {len(lines)} entries remain")

import pandas as pd
from metrics import per_topic_and_overall_metrics

baseline = pd.read_csv("predictions/results_gemma4_31b_cerebras.csv", keep_default_na=False)
fewshot = pd.read_csv("predictions/results_gemma4_31b_cerebras_v4_fewshot.csv", keep_default_na=False)

merged = baseline.merge(fewshot, on='row_index', suffixes=('_base', '_fs'))
covid = merged[merged['target_base'] == 'Covid Vaccine']
diff = covid[covid['predicted_label_base'] != covid['predicted_label_fs']]
print(len(diff), "Covid rows differ")
print(diff[['row_index', 'gold_label_base', 'predicted_label_base', 'predicted_label_fs']])
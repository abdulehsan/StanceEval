import pandas as pd
from metrics import per_topic_and_overall_metrics, format_metrics_table

df = pd.read_csv("predictions/results_gemma4_31b_cerebras_v1_prompt_arabic_target.csv", keep_default_na=False)

# Check for parse failures first
print("Parse failures:", (df['parse_success'] == False).sum())
print("Predicted label values:", df['predicted_label'].unique())
print("Gold label values:", df['gold_label'].unique())

results = per_topic_and_overall_metrics(
    df,
    true_col="gold_label",
    pred_col="predicted_label",
    target_col="target",
)
print(format_metrics_table(results))
print(df['target'].value_counts())
we = df[df['target'] == 'Women empowerment']
print(we['gold_label'].value_counts())

df_en = pd.read_csv("predictions/results_gemma4_31b_cerebras_v1_prompt.csv", keep_default_na=False)
print(len(df_en))
print(df_en.columns.tolist())

results_en = per_topic_and_overall_metrics(
    df_en,
    true_col="gold_label",
    pred_col="predicted_label",
    target_col="target",
)
print(format_metrics_table(results_en))

merged = df.merge(df_en, on='row_index', suffixes=('_ar', '_en'))
diff = merged[merged['predicted_label_ar'] != merged['predicted_label_en']]
print(diff[['row_index', 'target_ar', 'gold_label_ar', 'predicted_label_ar', 'predicted_label_en']])
print(merged.columns.tolist())
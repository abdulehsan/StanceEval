import os
import pandas as pd

WORKSPACE_ROOT = "d:/Abdullah Files/Programmming/python/StanceEval"

def main():
    comparison_path = os.path.join(WORKSPACE_ROOT, "predictions", "all_models_comparison.csv")
    revised_path = os.path.join(WORKSPACE_ROOT, "predictions", "results_gemma4_31b_cerebras_test_revised_zeroshot.csv")

    if not os.path.exists(comparison_path):
        print(f"Error: {comparison_path} does not exist.")
        return
    if not os.path.exists(revised_path):
        print(f"Error: {revised_path} does not exist.")
        return

    # Load dataframes
    comp_df = pd.read_csv(comparison_path, keep_default_na=False)
    rev_df = pd.read_csv(revised_path, keep_default_na=False)

    # Create mapping from row_index to predicted_label
    rev_map = dict(zip(rev_df["row_index"], rev_df["predicted_label"]))

    # Map to new column
    comp_df["gemma_revised_zeroshot_stance"] = comp_df["row_id"].map(rev_map)

    # Re-calculate status
    statuses = []
    agreed_count = 0
    for idx, row in comp_df.iterrows():
        orig = row["gemma_original_stance"]
        refined = row["gemma_refined_stance"]
        revised = row["gemma_revised_zeroshot_stance"]
        qwen_27 = row["qwen_3.6_27b_stance"]
        qwen_32 = row["qwen_3_32b_stance"]

        if orig == refined == revised == qwen_27 == qwen_32:
            statuses.append("Agreed")
            agreed_count += 1
        else:
            statuses.append(
                f"Gemma Orig: {orig} | Gemma Ref: {refined} | Gemma Rev: {revised} | "
                f"Qwen 27B: {qwen_27} | Qwen 32B: {qwen_32}"
            )

    comp_df["status"] = statuses

    # Reorder columns to insert gemma_revised_zeroshot_stance logically after gemma_refined_stance
    cols = list(comp_df.columns)
    # Remove the revised stance column from its current position (end)
    cols.remove("gemma_revised_zeroshot_stance")
    # Insert it right after gemma_refined_stance
    refined_idx = cols.index("gemma_refined_stance")
    cols.insert(refined_idx + 1, "gemma_revised_zeroshot_stance")
    comp_df = comp_df[cols]

    # Save comparison dataframe
    comp_df.to_csv(comparison_path, index=False)
    print(f"Successfully updated {comparison_path}.")
    print(f"Total agreed (all 5 models): {agreed_count} / {len(comp_df)} ({agreed_count / len(comp_df) * 100:.2f}%)")

if __name__ == "__main__":
    main()

import os
import sys
import io
import pandas as pd

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    _ROOT = os.path.dirname(__file__)
    # Go up one level to workspace root from scratch directory
    WORKSPACE_ROOT = os.path.join(_ROOT, "..")
    
    results_path = os.path.join(WORKSPACE_ROOT, "predictions", "results_gemma4_31b_cerebras_zeroshot_arabic_target_temp01.csv")
    dev_path = os.path.join(WORKSPACE_ROOT, "data", "dev.csv")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
        
    results_df = pd.read_csv(results_path, keep_default_na=False)
    dev_df = pd.read_csv(dev_path, keep_default_na=False)
    
    # Map text and target from dev_df
    results_df["row_index"] = results_df["row_index"].astype(int)
    dev_df.index = dev_df.index.astype(int)
    
    results_df["text"] = results_df["row_index"].map(dev_df["text"])
    results_df["true_stance"] = results_df["row_index"].map(dev_df["stance"])
    
    # Find errors
    errors_df = results_df[results_df["predicted_label"] != results_df["true_stance"]]
    
    print(f"Total rows: {len(results_df)}")
    print(f"Total errors: {len(errors_df)}")
    print()
    
    for idx, row in errors_df.iterrows():
        print(f"### Error {idx+1} (Row Index {row['row_index']})")
        print(f"- **Target**: {row['target']}")
        print(f"- **Gold Label**: `{row['true_stance']}`")
        print(f"- **Predicted**: `{row['predicted_label']}`")
        print(f"- **Raw Response**: `{row['raw_response']}`")
        print(f"- **Tweet Text**:\n  > {row['text']}")
        print()

if __name__ == "__main__":
    main()

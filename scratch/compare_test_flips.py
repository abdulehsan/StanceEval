import os
import sys
import io
import pandas as pd

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def main():
    _ROOT = os.path.dirname(__file__)
    WORKSPACE_ROOT = os.path.join(_ROOT, "..")
    
    new_path = os.path.join(WORKSPACE_ROOT, "predictions", "results_gemma4_31b_cerebras_test_revised_zeroshot.csv")
    orig_path = os.path.join(WORKSPACE_ROOT, "predictions", "results_gemma4_31b_cerebras_test.csv")
    ref_path = os.path.join(WORKSPACE_ROOT, "predictions", "results_gemma4_31b_cerebras_test_refined.csv")
    test_path = os.path.join(WORKSPACE_ROOT, "data", "ground_truth.csv")
    
    if not os.path.exists(new_path):
        print("New results file not found yet.")
        return
    
    new_df = pd.read_csv(new_path, keep_default_na=False)
    new_df["row_index"] = new_df["row_index"].astype(int)
    
    completed_indices = set(new_df["row_index"])
    
    # Load comparison files if they exist
    orig_df = pd.read_csv(orig_path, keep_default_na=False) if os.path.exists(orig_path) else None
    ref_df = pd.read_csv(ref_path, keep_default_na=False) if os.path.exists(ref_path) else None
    test_df = pd.read_csv(test_path, keep_default_na=False)
    
    # Align and map texts
    test_df = test_df.rename(columns={"id": "ID", "tweet_text": "text"})
    test_df.columns = test_df.columns.astype(str).str.strip()
    
    if orig_df is not None:
        orig_df["row_index"] = orig_df["row_index"].astype(int)
        orig_map = dict(zip(orig_df["row_index"], orig_df["predicted_label"]))
    else:
        orig_map = {}
        
    if ref_df is not None:
        ref_df["row_index"] = ref_df["row_index"].astype(int)
        ref_map = dict(zip(ref_df["row_index"], ref_df["predicted_label"]))
    else:
        ref_map = {}

    flips = []
    
    for _, row in new_df.iterrows():
        idx = int(row["row_index"])
        new_pred = row["predicted_label"]
        orig_pred = orig_map.get(idx, "N/A")
        ref_pred = ref_map.get(idx, "N/A")
        
        # Check if flipped compared to either previous run
        if new_pred != orig_pred or new_pred != ref_pred:
            tweet_text = test_df.iloc[idx]["text"] if idx < len(test_df) else "N/A"
            flips.append({
                "row_index": idx,
                "text": tweet_text,
                "original": orig_pred,
                "refined": ref_pred,
                "new": new_pred
            })
            
    print(f"Total processed rows: {len(new_df)}")
    print(f"Total flips detected: {len(flips)}")
    print()
    
    for item in flips:
        print(f"### Row Index {item['row_index']}")
        print(f"- **Original (V1)**: `{item['original']}`")
        print(f"- **Refined**: `{item['refined']}`")
        print(f"- **Revised Zero-Shot**: `{item['new']}`")
        print(f"- **Tweet**:\n  > {item['text']}")
        print()

if __name__ == "__main__":
    main()

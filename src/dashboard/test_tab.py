import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_test_tab(
    gt_df: pd.DataFrame, 
    gemma_df: pd.DataFrame | None, 
    anti_df: pd.DataFrame | None,
    qwen_df: pd.DataFrame | None
):
    st.subheader("📋 Blind Test Predictions & Annotation Tool (`ground_truth.csv`)")
    
    st.markdown(
        "This tab loads the test set tweets and aligns them with the predictions of "
        "**Gemma 4 31B (Cerebras)**, **Antigravity (Groq Llama-3.3-70B)**, and **Qwen (Groq Qwen-2.5-32B)**. "
        "You can inspect predictions, compare stance outputs, view Arabic reasoning explanations, and download the annotated dataset."
    )

    # Map predictions by row_index
    gemma_map = {}
    if gemma_df is not None and not gemma_df.empty:
        for _, row in gemma_df.iterrows():
            try:
                idx = int(row["row_index"])
                gemma_map[idx] = row["predicted_label"]
            except Exception:
                pass

    anti_map = {}
    if anti_df is not None and not anti_df.empty:
        for _, row in anti_df.iterrows():
            try:
                idx = int(row["row_index"])
                anti_map[idx] = {
                    "stance": row["antigravity_stance"],
                    "reasoning": row["antigravity_reasoning"]
                }
            except Exception:
                pass

    qwen_map = {}
    if qwen_df is not None and not qwen_df.empty:
        for _, row in qwen_df.iterrows():
            try:
                idx = int(row["row_index"])
                qwen_map[idx] = {
                    "stance": row["qwen_stance"],
                    "reasoning": row["qwen_reasoning"]
                }
            except Exception:
                pass

    combined_rows = []
    for idx, row in gt_df.iterrows():
        gemma_pred = gemma_map.get(idx, "Pending...")
        anti_info = anti_map.get(idx, {"stance": "Pending...", "reasoning": "Pending..."})
        qwen_info = qwen_map.get(idx, {"stance": "Pending...", "reasoning": "Pending..."})
        
        # User stance default to Qwen (highly accurate) or Gemma or None
        default_user_stance = "None"
        if qwen_info["stance"] != "Pending...":
            default_user_stance = qwen_info["stance"]
        elif gemma_pred != "Pending...":
            default_user_stance = gemma_pred
            
        combined_rows.append({
            "Row Index": idx,
            "Tweet ID": row["tweet_id"],
            "Tweet Text": row["text"],
            "Gemma Stance": gemma_pred,
            "Antigravity Stance": anti_info["stance"],
            "Antigravity Reasoning": anti_info["reasoning"],
            "Qwen Stance": qwen_info["stance"],
            "Qwen Reasoning": qwen_info["reasoning"],
            "Your Stance": default_user_stance,
            "Your Reasoning": ""
        })
        
    combined_df = pd.DataFrame(combined_rows)

    # Progress metrics side-by-side (3 columns)
    col_p1, col_p2, col_p3 = st.columns(3)
    total_rows = len(gt_df)
    
    with col_p1:
        gemma_completed = len(gemma_map)
        gemma_pct = (gemma_completed / total_rows) * 100
        st.metric(
            label="Gemma 31B Progress",
            value=f"{gemma_completed} / {total_rows} rows",
            delta=f"{gemma_pct:.1f}% Completed"
        )
        st.progress(gemma_completed / total_rows)
        
    with col_p2:
        anti_completed = len(anti_map)
        anti_pct = (anti_completed / total_rows) * 100
        st.metric(
            label="Antigravity Progress",
            value=f"{anti_completed} / {total_rows} rows",
            delta=f"{anti_pct:.1f}% Completed"
        )
        st.progress(anti_completed / total_rows)
        
    with col_p3:
        qwen_completed = len(qwen_map)
        qwen_pct = (qwen_completed / total_rows) * 100
        st.metric(
            label="Qwen Progress",
            value=f"{qwen_completed} / {total_rows} rows",
            delta=f"{qwen_pct:.1f}% Completed"
        )
        st.progress(qwen_completed / total_rows)

    # Interactive Data Editor
    st.write("---")
    st.write("### ✏️ Inspect & Edit Predictions")
    st.markdown(
        "Double-click any cell in **'Your Stance'** or **'Your Reasoning'** to make manual overrides."
    )

    column_config = {
        "Row Index": st.column_config.NumberColumn("Index", disabled=True),
        "Tweet ID": st.column_config.TextColumn("Tweet ID", disabled=True),
        "Tweet Text": st.column_config.TextColumn("Tweet Text", disabled=True, width="large"),
        "Gemma Stance": st.column_config.TextColumn("Gemma Stance", disabled=True),
        "Antigravity Stance": st.column_config.TextColumn("Antigravity Stance", disabled=True),
        "Antigravity Reasoning": st.column_config.TextColumn("Antigravity Reasoning", disabled=True, width="medium"),
        "Qwen Stance": st.column_config.TextColumn("Qwen Stance", disabled=True),
        "Qwen Reasoning": st.column_config.TextColumn("Qwen Reasoning", disabled=True, width="medium"),
        "Your Stance": st.column_config.SelectboxColumn(
            "Your Stance (Edit)",
            options=["Favor", "Against", "None"],
            required=True
        ),
        "Your Reasoning": st.column_config.TextColumn("Your Reasoning (Edit)", width="medium")
    }

    edited_df = st.data_editor(
        combined_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        key="test_editor"
    )

    # Download and Analytics Dashboard
    st.write("---")
    st.write("### 📊 Distribution & Export")
    
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.write("#### Stance Distribution Comparison")
        
        # Count stances for each model
        gemma_counts = edited_df["Gemma Stance"].value_counts()
        anti_counts = edited_df["Antigravity Stance"].value_counts()
        qwen_counts = edited_df["Qwen Stance"].value_counts()
        user_counts = edited_df["Your Stance"].value_counts()
        
        # Exclude "Pending..."
        for counts in [gemma_counts, anti_counts, qwen_counts]:
            if "Pending..." in counts:
                counts.drop("Pending...", inplace=True)
            
        comp_data = []
        for stance in ["Favor", "Against", "None"]:
            comp_data.append({
                "Stance": stance,
                "Gemma 31B": gemma_counts.get(stance, 0),
                "Antigravity": anti_counts.get(stance, 0),
                "Qwen 27B": qwen_counts.get(stance, 0),
                "Your Stance": user_counts.get(stance, 0)
            })
        comp_df = pd.DataFrame(comp_data).set_index("Stance")
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        comp_df.plot(kind="bar", ax=ax, edgecolor="black", color=["#5D9CEC", "#A0D468", "#4FC1E9", "#FC6E51"])
        ax.set_ylabel("Count")
        ax.set_xlabel("Stance Label")
        ax.set_title("Stance Distribution across Models")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.write("#### Download Complete Report")
        st.markdown(
            "Click the button below to download the annotated dataset as a CSV file. "
            "The file contains: **Tweet Text**, **Gemma Stance**, **Antigravity Stance**, **Antigravity Reasoning**, "
            "**Qwen Stance**, **Qwen Reasoning**, and your custom overrides."
        )
        
        # Clean export dataframe
        export_df = edited_df[[
            "Tweet ID", 
            "Tweet Text", 
            "Gemma Stance", 
            "Antigravity Stance", 
            "Antigravity Reasoning", 
            "Qwen Stance",
            "Qwen Reasoning",
            "Your Stance", 
            "Your Reasoning"
        ]].copy()
        
        csv_data = export_df.to_csv(index=False, encoding="utf-8").encode("utf-8")
        
        st.download_button(
            label="💾 Download CSV Report",
            data=csv_data,
            file_name="stanceeval_all_predictions_annotated.csv",
            mime="text/csv",
            use_container_width=True
        )

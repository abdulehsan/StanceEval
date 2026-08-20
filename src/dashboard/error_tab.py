import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def render_error_tab(results_df: pd.DataFrame, dev_df: pd.DataFrame):
    st.subheader("Model Prediction Error Analysis")
    st.markdown(
        "This tab helps explore why the model made incorrect predictions, "
        "and whether errors are correlated with low annotator confidence, sarcasm, or sentiment mismatch."
    )

    # 1. Join predictions with dev ground truth metadata
    results_df["row_index"] = results_df["row_index"].astype(int)
    dev_df.index = dev_df.index.astype(int)
    
    # Merge on row_index
    merged_df = results_df.merge(dev_df, left_on="row_index", right_index=True, suffixes=("", "_dev"))
    merged_df["correct"] = merged_df["predicted_label"] == merged_df["gold_label"]

    st.write(f"### Overall Correctness")
    total_preds = len(merged_df)
    correct_count = merged_df["correct"].sum()
    accuracy = correct_count / total_preds if total_preds > 0 else 0
    st.info(f"Analyzed **{total_preds}** predictions. Correct: **{correct_count}** ({accuracy*100:.1f}%), Incorrect: **{total_preds - correct_count}** ({(1-accuracy)*100:.1f}%)")

    # 2. Confusion Matrix
    st.write("---")
    st.write("### Confusion Matrix")
    
    # Select target filter
    targets = ["All"] + list(merged_df["target"].unique())
    target_choice = st.selectbox("Filter Confusion Matrix by Target Topic:", options=targets)
    
    plot_df = merged_df if target_choice == "All" else merged_df[merged_df["target"] == target_choice]
    
    if not plot_df.empty:
        labels = sorted(list(set(plot_df["gold_label"].unique()) | set(plot_df["predicted_label"].unique())))
        # Ensure we have standard classes if present
        for std in ["Favor", "Against", "None"]:
            if std in merged_df["gold_label"].unique() and std not in labels:
                labels.append(std)
        labels = sorted(list(set(labels)))
        
        cm = confusion_matrix(plot_df["gold_label"], plot_df["predicted_label"], labels=labels)
        
        col_cm1, col_cm2 = st.columns([2, 1])
        with col_cm1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=ax, edgecolor="black", linewidths=0.5)
            ax.set_title(f"Confusion Matrix ({target_choice})")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Gold (True) Label")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        with col_cm2:
            st.write("**Classification Metrics:**")
            try:
                report = classification_report(plot_df["gold_label"], plot_df["predicted_label"], output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df.loc[labels + ["accuracy", "macro avg"]])
            except Exception as e:
                st.write("Unable to generate metrics for current slice.")
    else:
        st.warning("No data for this target topic.")

    # 3. Confidence vs. Correctness
    st.write("---")
    st.write("### Does the model fail where humans were uncertain?")
    st.markdown(
        "Below, we compare the average annotator confidence scores for **Correct** vs **Incorrect** predictions."
    )

    conf_cols = [c for c in merged_df.columns if "confidence" in c]
    if conf_cols:
        conf_choice = st.selectbox(
            "Select Annotator Confidence Score to plot:",
            options=conf_cols,
            key="err_conf_choice"
        )
        
        # Convert chosen confidence column to float
        merged_df[conf_choice] = pd.to_numeric(merged_df[conf_choice], errors="coerce")
        clean_merged_df = merged_df.dropna(subset=[conf_choice]).copy()
        
        col_err1, col_err2 = st.columns(2)
        with col_err1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=clean_merged_df, x="correct", y=conf_choice, ax=ax, palette="Set1")
            ax.set_title(f"{conf_choice} vs. Model Correctness")
            ax.set_xlabel("Model Prediction Correct?")
            ax.set_ylabel("Annotator Confidence")
            ax.set_xticklabels(["Incorrect (False)", "Correct (True)"])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        with col_err2:
            avg_conf = clean_merged_df.groupby("correct")[conf_choice].mean().reset_index()
            st.write(f"**Average {conf_choice}:**")
            for _, row in avg_conf.iterrows():
                outcome = "Correct (True)" if row["correct"] else "Incorrect (False)"
                st.write(f"- {outcome}: **{row[conf_choice]:.4f}**")
            st.markdown(
                "\n*If average confidence is significantly lower for Incorrect predictions, "
                "it confirms the model is mostly failing on highly ambiguous, borderline, or noisy tweets.*"
            )
            
    # 4. Sentiment and Sarcasm Correlation
    st.write("---")
    st.write("### Sentiment & Sarcasm Correlation with Errors")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.write("#### Error Rates by Sentiment")
        if "sentiment" in merged_df.columns:
            sent_err = merged_df.groupby("sentiment")["correct"].apply(lambda x: (1 - x.mean()) * 100).reset_index()
            sent_err.columns = ["Sentiment", "Error Rate (%)"]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=sent_err, x="Sentiment", y="Error Rate (%)", ax=ax, palette="coolwarm", edgecolor="black")
            ax.set_title("Error Rate (%) by Sentiment Class")
            ax.set_ylabel("Error Rate (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("Sentiment column missing.")
            
    with col_feat2:
        st.write("#### Error Rates on Sarcastic Tweets")
        if "sarcasm" in merged_df.columns:
            sarc_err = merged_df.groupby("sarcasm")["correct"].apply(lambda x: (1 - x.mean()) * 100).reset_index()
            sarc_err.columns = ["Sarcasm", "Error Rate (%)"]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=sarc_err, x="Sarcasm", y="Error Rate (%)", ax=ax, palette="magma", edgecolor="black")
            ax.set_title("Error Rate (%) on Sarcastic vs. Non-sarcastic")
            ax.set_ylabel("Error Rate (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("Sarcasm column missing.")

    # 5. Interactive Explorer
    st.write("---")
    st.write("### Interactive Error Explorer")
    st.markdown("Filter and inspect the exact text of the tweets to analyze model mistakes.")

    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        f_correct = st.selectbox("Prediction Status:", options=["All", "Correct Only", "Incorrect Only"])
    with col_f2:
        f_sarcasm = st.selectbox("Sarcastic:", options=["All", "Yes Only", "No Only"])
    with col_f3:
        f_sentiment = st.selectbox("Sentiment:", options=["All"] + list(merged_df["sentiment"].unique()) if "sentiment" in merged_df.columns else ["All"])

    filtered_explorer = merged_df.copy()
    if f_correct == "Correct Only":
        filtered_explorer = filtered_explorer[filtered_explorer["correct"] == True]
    elif f_correct == "Incorrect Only":
        filtered_explorer = filtered_explorer[filtered_explorer["correct"] == False]
        
    if f_sarcasm == "Yes Only":
        filtered_explorer = filtered_explorer[filtered_explorer["sarcasm"] == "Yes"]
    elif f_sarcasm == "No Only":
        filtered_explorer = filtered_explorer[filtered_explorer["sarcasm"] == "No"]
        
    if f_sentiment != "All" and "sentiment" in filtered_explorer.columns:
        filtered_explorer = filtered_explorer[filtered_explorer["sentiment"] == f_sentiment]

    st.write(f"Showing **{len(filtered_explorer)}** matching tweets:")
    
    # Format and present as table / list
    explorer_display = filtered_explorer[[
        "row_index", "text", "target", "gold_label", "predicted_label", 
        "sentiment", "sarcasm", "stance:confidence"
    ]].copy()
    
    st.dataframe(explorer_display, use_container_width=True)

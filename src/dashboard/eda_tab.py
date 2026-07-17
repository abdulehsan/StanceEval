import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_eda_tab(df: pd.DataFrame, label_suffix: str):
    st.subheader(f"Dataset Overview & Distributions ({label_suffix})")

    # 1. Target & Stance distributions side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Target Topic Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        # Custom premium palette
        colors = sns.color_palette("muted", len(df["target"].unique()))
        df["target"].value_counts().plot(kind="bar", ax=ax, color=colors, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.write("### Stance Label Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = sns.color_palette("pastel", len(df["stance"].unique()))
        df["stance"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "black"})
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 2. Confidence distributions
    st.write("---")
    st.write("### Annotator Confidence Analysis")
    st.markdown(
        "Explore annotator confidence scores across the dataset. The boxplots below show the range, "
        "median, and quartiles of confidence for each class."
    )

    conf_col_mapping = {
        "Stance Confidence": "stance:confidence",
        "Sarcasm Confidence": "sarcasm:confidence",
        "Sentiment Confidence": "sentiment:confidence"
    }

    conf_choice = st.selectbox(
        "Select Confidence Metric:",
        options=list(conf_col_mapping.keys())
    )
    conf_col = conf_col_mapping[conf_choice]

    if conf_col in df.columns:
        # Convert to float safely, handling empty strings if any
        conf_series = pd.to_numeric(df[conf_col], errors="coerce").dropna()
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            # Boxplot grouped by stance
            fig, ax = plt.subplots(figsize=(6, 4))
            # Safe conversion of df copy
            plot_df = df.copy()
            plot_df[conf_col] = pd.to_numeric(plot_df[conf_col], errors="coerce")
            sns.boxplot(data=plot_df.dropna(subset=[conf_col]), x="stance", y=conf_col, ax=ax, palette="Set2")
            ax.set_title(f"{conf_choice} by Stance")
            ax.set_xlabel("Stance Label")
            ax.set_ylabel("Confidence Score")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_c2:
            # General histogram
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(conf_series, bins=15, kde=True, ax=ax, color="skyblue", edgecolor="black")
            ax.set_title(f"Distribution of {conf_choice}")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning(f"Confidence column '{conf_col}' is missing in this split.")

    # 3. Stance vs Sentiment
    st.write("---")
    st.write("### Stance vs. Sentiment Correlation")
    
    if "sentiment" in df.columns:
        # Cross tabulation
        ct = pd.crosstab(df["stance"], df["sentiment"], normalize="index") * 100
        
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={'label': '% of Stance'})
            ax.set_title("How Sentiment correlates with Stance (normalized by stance)")
            ax.set_xlabel("Annotated Sentiment")
            ax.set_ylabel("Annotated Stance")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        with col_s2:
            st.markdown(
                "This heatmap represents what percentage of each **stance** is associated with positive, negative, or neutral sentiment.\n\n"
                "- Typically, **Favor** is strongly positive.\n"
                "- **Against** leans heavily negative.\n"
                "- **None** represents neutral stance, but can have varied sentiments."
            )
    else:
        st.warning("Sentiment column is missing in this split.")

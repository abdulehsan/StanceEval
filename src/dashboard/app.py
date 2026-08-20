"""
StanceEval EDA & Error Analysis Dashboard — Main Entrypoint.
"""

import io
import os
import sys
import pandas as pd
import streamlit as st


# Add parent directory to path to import siblings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from data_utils import load_data
from eda_tab import render_eda_tab
from error_tab import render_error_tab
from test_tab import render_test_tab

st.set_page_config(
    page_title="StanceEval EDA & Error Analysis Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Cache data loading for smooth interaction
@st.cache_data
def load_train_dev_data():
    train_path = os.path.join(_ROOT, "data", "train.csv")
    dev_path = os.path.join(_ROOT, "data", "dev.csv")
    
    train_df = load_data(train_path) if os.path.exists(train_path) else None
    dev_df = load_data(dev_path) if os.path.exists(dev_path) else None
    return train_df, dev_df

def load_test_data():
    gt_path = os.path.join(_ROOT, "data", "ground_truth.csv")
    if os.path.exists(gt_path):
        df = pd.read_csv(gt_path, keep_default_na=False)
        df = df.rename(columns={"id": "ID", "tweet_text": "text"})
        df.columns = df.columns.astype(str).str.strip()
        return df
    return None

def load_test_predictions():
    gemma_path = os.path.join(_ROOT, "predictions", "results_gemma4_31b_cerebras_test.csv")
    anti_path = os.path.join(_ROOT, "predictions", "results_antigravity_test.csv")
    qwen_path = os.path.join(_ROOT, "predictions", "results_qwen_test.csv")
    
    gemma_df = pd.read_csv(gemma_path, keep_default_na=False) if os.path.exists(gemma_path) else None
    anti_df = pd.read_csv(anti_path, keep_default_na=False) if os.path.exists(anti_path) else None
    qwen_df = pd.read_csv(qwen_path, keep_default_na=False) if os.path.exists(qwen_path) else None
    
    return gemma_df, anti_df, qwen_df

def get_available_prediction_files():
    pred_dir = os.path.join(_ROOT, "predictions")
    if not os.path.exists(pred_dir):
        return []
    # return list of CSV prediction files
    return [f for f in os.listdir(pred_dir) if f.endswith(".csv")]

# App Header
st.title("🎯 StanceEval-2026: Arabic Stance Detection")
st.markdown(
    "Interactive Exploratory Data Analysis (EDA) and Model Error Analysis dashboard. "
    "Designed to explore dataset statistics, annotator confidence metrics, sentiment/sarcasm patterns, "
    "and identify where zero-shot or fine-tuned models go wrong."
)

# Load data
train_df, dev_df = load_train_dev_data()

# Navigation Sidebar
st.sidebar.image("https://img.icons8.com/clouds/100/combo-chart.png", width=80)
st.sidebar.title("Navigation")
tab_choice = st.sidebar.radio(
    "Go to page:",
    options=[
        "1. Dataset EDA (train.csv)", 
        "2. Dataset EDA (dev.csv)", 
        "3. Model Error Analysis",
        "4. Test Predictions (ground_truth.csv)"
    ]
)

# Tab 1: Train EDA
if tab_choice == "1. Dataset EDA (train.csv)":
    if train_df is not None:
        render_eda_tab(train_df, "Train Set")
    else:
        st.error("Could not load data/train.csv. Verify the path is correct.")

# Tab 2: Dev EDA
elif tab_choice == "2. Dataset EDA (dev.csv)":
    if dev_df is not None:
        render_eda_tab(dev_df, "Dev Set")
    else:
        st.error("Could not load data/dev.csv. Verify the path is correct.")

# Tab 3: Error Analysis
elif tab_choice == "3. Model Error Analysis":
    pred_files = get_available_prediction_files()
    if not pred_files:
        st.warning("No prediction CSV files found in the `predictions/` directory. Run inference first.")
    else:
        st.sidebar.subheader("Error Analysis Config")

        gemma_full_filename = "results_gemma4_31b_cerebras_full.csv"
        default_idx = pred_files.index(gemma_full_filename) if gemma_full_filename in pred_files else 0

        selected_file = st.sidebar.selectbox(
            "Select Model Predictions:",
            options=pred_files,
            index=default_idx
        )

        # Auto-detect source dataset from filename, but let user override
        auto_source = "train" if "train" in selected_file.lower() else "dev"
        source_choice = st.sidebar.radio(
            "Source ground-truth file (must match predictions):",
            options=["train", "dev"],
            index=0 if auto_source == "train" else 1
        )

        source_df = train_df if source_choice == "train" else dev_df

        st.write(f"📂 Analyzing predictions from: `{selected_file}`")
        st.write(f"📎 Matched against: `{'train.csv' if source_choice == 'train' else 'dev.csv'}`")

        if source_df is None:
            st.error(f"data/{source_choice}.csv could not be loaded.")
        else:
            pred_path = os.path.join(_ROOT, "predictions", selected_file)
            try:
                results_df = pd.read_csv(pred_path, keep_default_na=False)
                if not results_df.empty:
                    render_error_tab(results_df, source_df)
                else:
                    st.warning(f"The selected prediction file `{selected_file}` is empty.")
            except Exception as e:
                st.error(f"Error loading predictions: {e}")

# Tab 4: Test Predictions
elif tab_choice == "4. Test Predictions (ground_truth.csv)":
    gt_df = load_test_data()
    gemma_df, anti_df, qwen_df = load_test_predictions()
    if gt_df is not None:
        render_test_tab(gt_df, gemma_df, anti_df, qwen_df)
    else:
        st.error("Could not load data/ground_truth.csv. Verify the path is correct.")
"""
Streamlit App for Fraud Transaction Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Transaction Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        optimal_model = joblib.load("optimal_model.pkl")
        feature_names = joblib.load("feature_names.pkl")

        return optimal_model, feature_names

    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None


def preprocess_input(input_data, feature_names):
    df = pd.DataFrame([input_data])

    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    df_selected = df[feature_names]

    return df_selected


def main():
    st.markdown(
        '<h1 class="main-header">🔒 Fraud Transaction Detection System</h1>',
        unsafe_allow_html=True
    )

    model, feature_names = load_models()

    if model is None:
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Single Prediction", "Batch Prediction", "Model Information"]
    )

    if page == "Single Prediction":
        single_prediction_page(model, feature_names)

    elif page == "Batch Prediction":
        batch_prediction_page(model, feature_names)

    else:
        model_info_page()


def single_prediction_page(model, feature_names):
    st.header("Single Transaction Prediction")

    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0)
    purchases = st.number_input("Purchases", min_value=0.0, value=0.0)
    cash_advance = st.number_input("Cash Advance", min_value=0.0, value=0.0)
    tenure = st.number_input("Tenure", min_value=0, value=12)

    input_data = {
        "balance": balance,
        "credit_limit": credit_limit,
        "purchases": purchases,
        "cash_advance": cash_advance,
        "tenure": tenure
    }

    if st.button("🔍 Predict Fraud", use_container_width=True):

        try:
            processed_input = preprocess_input(
                input_data, feature_names
            )

            prediction = model.predict(processed_input)[0]
            proba = model.predict_proba(processed_input)[0]

            fraud_prob = proba[1] * 100

            if prediction == 1:
                st.error(f"🚨 FRAUD DETECTED ({fraud_prob:.2f}%)")
            else:
                st.success(f"✅ LEGITIMATE ({100 - fraud_prob:.2f}%)")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def batch_prediction_page(model, feature_names):
    st.header("Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        missing = [c for c in feature_names if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            return

        df_selected = df[feature_names]

        predictions = model.predict(df_selected)
        probas = model.predict_proba(df_selected)[:, 1] * 100

        df["fraud_prediction"] = predictions
        df["fraud_probability"] = probas

        st.dataframe(df.head())

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        )


def model_info_page():
    st.header("Model Information")
    st.write("Fraud Detection using RandomForest / XGBoost")


if __name__ == "__main__":
    main()

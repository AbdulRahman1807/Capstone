"""
Streamlit App for Fraud Transaction Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Transaction Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fraud {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .legitimate {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models and preprocessors"""
    try:
        with open('optimal_model.pkl', 'rb') as f:
            optimal_model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('feature_selector.pkl', 'rb') as f:
            feature_selector = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return optimal_model, scaler, feature_selector, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the pipeline first: {e}")
        return None, None, None, None

def preprocess_input(input_data, feature_names, scaler, feature_selector):
    """Preprocess input data for prediction"""
    # Create DataFrame with all features
    df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in the same order as training
    df_selected = df[feature_names]
    
    # Scale features
    df_scaled = scaler.transform(df_selected)
    
    return df_scaled

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Fraud Transaction Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    optimal_model, scaler, feature_selector, feature_names = load_models()
    
    if optimal_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Single Prediction", "Batch Prediction", "Model Information"])
    
    if page == "Single Prediction":
        single_prediction_page(optimal_model, scaler, feature_selector, feature_names)
    elif page == "Batch Prediction":
        batch_prediction_page(optimal_model, scaler, feature_selector, feature_names)
    else:
        model_info_page()

def single_prediction_page(model, scaler, feature_selector, feature_names):
    """Page for single transaction prediction"""
    st.header("Single Transaction Prediction")
    st.markdown("Enter transaction details below to detect potential fraud.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Information")
        balance = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, step=500.0)
        purchases = st.number_input("Purchases", min_value=0.0, value=0.0, step=100.0)
        oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=0.0, step=100.0)
        installments_purchases = st.number_input("Installments Purchases", min_value=0.0, value=0.0, step=100.0)
        cash_advance = st.number_input("Cash Advance", min_value=0.0, value=0.0, step=100.0)
        payments = st.number_input("Payments", min_value=0.0, value=0.0, step=100.0)
        minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=0.0, step=100.0)
    
    with col2:
        st.subheader("Transaction Frequency")
        balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.5, 0.1)
        purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.0, 0.1)
        oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.0, 0.1)
        purchases_installments_frequency = st.slider("Purchases Installments Frequency", 0.0, 1.0, 0.0, 0.1)
        cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0, 0.1)
        prc_full_payment = st.slider("Percentage Full Payment", 0.0, 1.0, 0.0, 0.1)
        
        st.subheader("Transaction Counts")
        cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0, value=0, step=1)
        purchases_trx = st.number_input("Purchases Transactions", min_value=0, value=0, step=1)
        
        st.subheader("Other Information")
        tenure = st.number_input("Tenure (months)", min_value=0, value=12, step=1)
    
    # Create input dictionary
    input_data = {
        'balance': balance,
        'balance_frequency': balance_frequency,
        'purchases': purchases,
        'oneoff_purchases': oneoff_purchases,
        'installments_purchases': installments_purchases,
        'cash_advance': cash_advance,
        'purchases_frequency': purchases_frequency,
        'oneoff_purchases_frequency': oneoff_purchases_frequency,
        'purchases_installments_frequency': purchases_installments_frequency,
        'cash_advance_frequency': cash_advance_frequency,
        'cash_advance_trx': cash_advance_trx,
        'purchases_trx': purchases_trx,
        'credit_limit': credit_limit,
        'payments': payments,
        'minimum_payments': minimum_payments,
        'prc_full_payment': prc_full_payment,
        'tenure': tenure
    }
    
    # Prediction button
    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
        try:
            # Preprocess input
            processed_input = preprocess_input(input_data, feature_names, scaler, feature_selector)
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fraud_prob = prediction_proba[1] * 100
                legitimate_prob = prediction_proba[0] * 100
                
                st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
                st.metric("Legitimate Probability", f"{legitimate_prob:.2f}%")
            
            with col2:
                if prediction == 1:
                    st.error("🚨 FRAUD DETECTED")
                    st.markdown('<div class="prediction-box fraud">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ This transaction has been flagged as FRAUDULENT")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                    st.markdown('<div class="prediction-box legitimate">', unsafe_allow_html=True)
                    st.markdown("### ✓ This transaction appears to be LEGITIMATE")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                # Confidence level
                confidence = max(fraud_prob, legitimate_prob)
                st.metric("Confidence Level", f"{confidence:.2f}%")
                
                # Risk level
                if fraud_prob > 70:
                    risk_level = "🔴 HIGH RISK"
                elif fraud_prob > 40:
                    risk_level = "🟡 MEDIUM RISK"
                else:
                    risk_level = "🟢 LOW RISK"
                
                st.markdown(f"**Risk Level:** {risk_level}")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("---")
                st.subheader("Top Contributing Factors")
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(feature_importance_df.set_index('Feature'))
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def batch_prediction_page(model, scaler, feature_selector, feature_names):
    """Page for batch prediction"""
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with multiple transactions to predict fraud in bulk.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} transactions found.")
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Predict All Transactions", type="primary"):
                # Check if required columns exist
                missing_cols = [col for col in feature_names if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.info("Please ensure your CSV contains all required features.")
                else:
                    # Preprocess
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Select features
                    df_selected = df[feature_names]
                    
                    # Scale
                    status_text.text("Scaling features...")
                    progress_bar.progress(30)
                    df_scaled = scaler.transform(df_selected)
                    
                    # Predict
                    status_text.text("Making predictions...")
                    progress_bar.progress(60)
                    predictions = model.predict(df_scaled)
                    prediction_probas = model.predict_proba(df_scaled)[:, 1]
                    
                    # Add predictions to dataframe
                    df['fraud_prediction'] = predictions
                    df['fraud_probability'] = prediction_probas * 100
                    df['status'] = df['fraud_prediction'].apply(lambda x: 'FRAUD' if x == 1 else 'LEGITIMATE')
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        st.metric("Fraudulent", df['fraud_prediction'].sum())
                    with col3:
                        st.metric("Legitimate", (df['fraud_prediction'] == 0).sum())
                    with col4:
                        st.metric("Fraud Rate", f"{(df['fraud_prediction'].mean() * 100):.2f}%")
                    
                    # Results table
                    st.dataframe(df[['fraud_prediction', 'fraud_probability', 'status'] + feature_names[:5]])
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_info_page():
    """Page showing model information"""
    st.header("Model Information")
    
    st.subheader("About the Model")
    st.markdown("""
    This fraud detection system uses machine learning to identify potentially fraudulent transactions.
    
    **Algorithms Used:**
    - Random Forest Classifier
    - XGBoost Classifier
    
    **Optimization Techniques:**
    - Max Depth Tuning
    - Learning Rate Tuning
    - GridSearchCV for hyperparameter optimization
    - Feature Selection using SelectKBest
    
    **Model Pipeline:**
    1. Data Preprocessing (handling missing values, date features)
    2. Feature Selection (SelectKBest with f_classif)
    3. Feature Scaling (StandardScaler)
    4. Train-Test Split (80-20)
    5. Model Training with Hyperparameter Tuning
    6. Model Evaluation and Selection
    """)
    
    # Try to load model comparison results
    try:
        results_df = pd.read_csv('model_comparison_results.csv')
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        st.subheader("Performance Metrics")
        metric = st.selectbox("Select metric to visualize", 
                             ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test ROC-AUC'])
        
        fig_data = results_df[['Model', metric]].set_index('Model')
        st.bar_chart(fig_data)
    
    except FileNotFoundError:
        st.info("Model comparison results not found. Please run the pipeline first.")

if __name__ == "__main__":
    main()

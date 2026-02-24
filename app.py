import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("Fraud Detection App")

st.write("""
This Streamlit app loads the pre-trained optimal model and selected features from the pickled files.
You can input feature values to get fraud predictions. The app uses the top features selected during training.
""")

# Load the saved model and feature names
try:
    best_model = joblib.load("optimal_model.pkl")
    best_features = joblib.load("feature_names.pkl")
    st.success("Model and features loaded successfully!")
except FileNotFoundError:
    st.error("Pickled files not found. Please ensure 'optimal_model.pkl' and 'feature_names.pkl' are in the directory.")
    st.stop()

# Display the features used
st.subheader("Features Used by the Model")
st.write(best_features)

# Single Prediction Section
st.header("Single Instance Prediction")
st.write("Enter values for each feature below:")

with st.form("prediction_form"):
    inputs = {}
    for feature in best_features:
        # Assuming all features are numeric, use number_input with default value
        inputs[feature] = st.number_input(
            feature,
            value=0.0,
            help="Enter a numeric value for this feature."
        )
    
    submit_button = st.form_submit_button("Predict Fraud")

if submit_button:
    # Create DataFrame from inputs
    input_df = pd.DataFrame([inputs])
    
    # Predict class and probability
    prediction = best_model.predict(input_df)[0]
    probability = best_model.predict_proba(input_df)[0][1]
    
    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"**Fraud Detected!** Probability: {probability:.2f}")
    else:
        st.success(f"**Not Fraud.** Probability of Fraud: {probability:.2f}")
    
    # Visualize probability
    fig, ax = plt.subplots(figsize=(6, 1))
    sns.barplot(x=[probability, 1 - probability], y=["Fraud", "Not Fraud"], ax=ax, palette="coolwarm")
    ax.set_title("Fraud Probability")
    ax.set_xlabel("Probability")
    st.pyplot(fig)
    plt.close(fig)

# Batch Prediction Section
st.header("Batch Prediction")
st.write("Upload a CSV file with the required features for batch predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    
    # Check if all required features are present
    missing_features = [f for f in best_features if f not in batch_df.columns]
    if missing_features:
        st.error(f"Missing features in uploaded file: {missing_features}")
    else:
        # Select only the required features
        batch_X = batch_df[best_features]
        
        # Predict
        batch_predictions = best_model.predict(batch_X)
        batch_probabilities = best_model.predict_proba(batch_X)[:, 1]
        
        # Add to DataFrame
        batch_df['Predicted_Fraud'] = batch_predictions
        batch_df['Fraud_Probability'] = batch_probabilities
        
        # Display results
        st.subheader("Batch Prediction Results")
        st.dataframe(batch_df)
        
        # Download button
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
        
        # Visualize distribution
        st.subheader("Distribution of Predictions")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x=batch_predictions, ax=ax2)
        ax2.set_title("Predicted Fraud Distribution")
        ax2.set_xlabel("Predicted Class (0: Not Fraud, 1: Fraud)")
        st.pyplot(fig2)
        plt.close(fig2)

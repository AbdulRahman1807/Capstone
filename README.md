# Fraud Transaction Detection System

A comprehensive machine learning project for detecting fraudulent transactions using Random Forest and XGBoost algorithms.

## Project Overview

**Domain:** Financial Security  
**Algorithms:** Random Forest, XGBoost  
**Optimization Techniques:**
- Max Depth tuning
- Learning Rate tuning
- GridSearchCV
- Feature Selection

## Project Structure

```
capstone/
├── Stori_Data_Challenge_2021..csv    # Dataset
├── fraud_detection_pipeline.py      # Main ML pipeline
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/capstone
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Run the ML Pipeline

First, run the main pipeline to train models and generate pickle files:

```bash
python fraud_detection_pipeline.py
```

This will:
- Load and preprocess the data
- Perform feature selection
- Scale the features
- Split data into train/test sets
- Train baseline Random Forest and XGBoost models
- Perform hyperparameter tuning using GridSearchCV
- Compare models and select the optimal one
- Save models and preprocessors as pickle files

**Generated Files:**
- `optimal_model.pkl` - Best performing model
- `rf_baseline_model.pkl` - Baseline Random Forest model
- `rf_tuned_model.pkl` - Tuned Random Forest model
- `xgb_baseline_model.pkl` - Baseline XGBoost model
- `xgb_tuned_model.pkl` - Tuned XGBoost model
- `scaler.pkl` - Feature scaler
- `feature_selector.pkl` - Feature selector
- `feature_names.pkl` - Selected feature names
- `model_comparison_results.csv` - Model performance comparison

### Step 2: Launch the Streamlit App

After running the pipeline, launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Streamlit App Features

### 1. Single Prediction
- Enter transaction details manually
- Get real-time fraud prediction
- View probability scores and risk levels
- See top contributing factors

### 2. Batch Prediction
- Upload a CSV file with multiple transactions
- Get predictions for all transactions at once
- Download results as CSV

### 3. Model Information
- View model performance metrics
- Compare different models
- Understand the pipeline

## Dataset Information

The dataset contains the following features:
- Customer information (ID, activation date, tenure)
- Financial metrics (balance, credit limit, purchases, payments)
- Transaction frequencies
- Transaction counts
- Target variable: `fraud` (0 = legitimate, 1 = fraudulent)

## Model Pipeline Details

1. **Data Preprocessing:**
   - Handle missing values (median imputation)
   - Convert date columns to numeric features
   - Remove customer IDs

2. **Feature Selection:**
   - SelectKBest with f_classif
   - Selects top 15 most important features

3. **Feature Scaling:**
   - StandardScaler for normalization

4. **Model Training:**
   - Baseline models with default parameters
   - Hyperparameter tuning with GridSearchCV
   - 5-fold cross-validation

5. **Model Evaluation:**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC Score
   - Confusion Matrix

## Hyperparameter Tuning

### Random Forest:
- `n_estimators`: [100, 200]
- `max_depth`: [5, 10, 15, 20]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]

### XGBoost:
- `n_estimators`: [100, 200]
- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.1, 0.2]
- `subsample`: [0.8, 1.0]

## Results

The pipeline automatically compares all models and selects the best one based on ROC-AUC score. Results are saved in `model_comparison_results.csv`.

## Troubleshooting

1. **FileNotFoundError when running app.py:**
   - Make sure you've run `fraud_detection_pipeline.py` first
   - Check that all pickle files are in the same directory

2. **Memory issues:**
   - Reduce the number of features selected
   - Reduce GridSearchCV parameter grid size
   - Use a smaller dataset for testing

3. **Import errors:**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ recommended)

## Author

Created for Final Review Project - Fraud Transaction Detection

## License

This project is for educational purposes.

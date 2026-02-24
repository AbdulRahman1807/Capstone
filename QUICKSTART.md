# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
```bash
python fraud_detection_pipeline.py
```

**Expected Output:**
- Data loading and preprocessing messages
- Feature selection results
- Model training progress
- Hyperparameter tuning progress (this may take several minutes)
- Model comparison results
- Saved pickle files

**Time:** Approximately 10-30 minutes depending on your system

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 4. Use the App

#### Single Prediction:
1. Navigate to "Single Prediction" page
2. Fill in the transaction details
3. Click "Predict Fraud"
4. View the results

#### Batch Prediction:
1. Navigate to "Batch Prediction" page
2. Upload a CSV file with transaction data
3. Ensure your CSV has the required columns (see below)
4. Click "Predict All Transactions"
5. Download results

## Required CSV Columns for Batch Prediction

Your CSV file should include these columns:
- balance
- balance_frequency
- purchases
- oneoff_purchases
- installments_purchases
- cash_advance
- purchases_frequency
- oneoff_purchases_frequency
- purchases_installments_frequency
- cash_advance_frequency
- cash_advance_trx
- purchases_trx
- credit_limit
- payments
- minimum_payments
- prc_full_payment
- tenure

**Note:** The pipeline will automatically select the most important features, so your CSV may have additional columns, but it must include at least the selected features.

## Troubleshooting

### Issue: "Model files not found"
**Solution:** Run `python fraud_detection_pipeline.py` first to generate the required pickle files.

### Issue: GridSearchCV taking too long
**Solution:** Reduce the parameter grid in `fraud_detection_pipeline.py`:
- Reduce `n_estimators` options
- Reduce `max_depth` options
- Reduce CV folds from 5 to 3

### Issue: Memory errors
**Solution:** 
- Close other applications
- Reduce dataset size for testing
- Reduce number of features selected

### Issue: Streamlit app not loading
**Solution:**
- Check that all pickle files are in the same directory as `app.py`
- Verify Streamlit is installed: `pip install streamlit`
- Check for error messages in the terminal

## Example Usage

### Example 1: Single Transaction
```
Balance: 5000
Credit Limit: 10000
Purchases: 2000
... (fill in other fields)
```

### Example 2: Batch CSV Format
```csv
balance,credit_limit,purchases,payments,tenure,...
5000,10000,2000,1500,12,...
3000,8000,1000,800,12,...
```

## Next Steps

After running the pipeline:
1. Review `model_comparison_results.csv` to see which model performed best
2. Check the feature importance in the pipeline output
3. Experiment with different hyperparameters
4. Try different feature selection methods

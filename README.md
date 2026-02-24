# 💳 Fraud Detection System  
### Transaction Risk Assessment using Machine Learning  

---

## ✅ Project Overview

This project builds and deploys a **Fraud Detection System** using Machine Learning models optimized for **high recall** (fraud detection priority).

The system:

✔ Trains multiple ML models  
✔ Performs hyperparameter tuning  
✔ Selects the best model based on Recall  
✔ Saves production artifacts (`optimal_model.pkl`)  
✔ Deploys via a black-themed Streamlit enterprise dashboard  

---

## 🎯 Business Objective

Fraud detection is a **high-recall classification problem**.

- ❌ False Negative → Fraud missed → Financial loss  
- ⚠ False Positive → Transaction investigated  

Therefore, we optimize for:

✔ **Recall (Fraud class)**  
✔ ROC-AUC  

---

## 📊 Dataset

**File Used:**
```
Stori_Data_Challenge_2021..csv
```

**Target Column:**
```
fraud
(0 = Not Fraud, 1 = Fraud)
```

### Data Preprocessing

✔ Dropped ID & date columns  
✔ Median imputation for missing values  
✔ Correlation analysis  
✔ Stratified Train/Test split (80/20)  

---

## ⚙️ Model Pipeline

---

### 1️⃣ Baseline Models

- 🌲 Random Forest  
- ⚡ XGBoost (with class imbalance handling)

---

### 2️⃣ Hyperparameter Tuning

Using:

```
GridSearchCV
scoring = "recall"
cv = 5
```

#### Random Forest Grid

```
max_depth: [3, 5, 8, 12]
n_estimators: [100, 200]
min_samples_split: [2, 5]
```

#### XGBoost Grid

```
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.1, 0.2]
n_estimators: [100, 200]
```

---

### 3️⃣ Feature Selection

✔ Extracted feature importance from tuned RF  
✔ Selected Top 10 features  
✔ Retrained reduced model  
✔ Compared performance  

---

### 4️⃣ Model Comparison

All models evaluated on:

✔ Recall  
✔ ROC-AUC  

Final model selected based on:

```
Highest Recall Score
```

---

## 🏆 Final Artifacts

Generated automatically after training:

```
optimal_model.pkl
feature_names.pkl
```

These are used in the Streamlit deployment.

---

## 🚀 Streamlit Dashboard

### UI Features

✔ Black enterprise theme  
✔ Manual transaction input  
✔ Predefined risk simulation profiles  
✔ Risk score visualization  
✔ Dynamic fraud classification  
✔ Feature importance chart  
✔ Risk progress bar  

---

## 🎯 Risk Scoring Logic

Probability threshold:

```
0.15
```

Risk classification:

| Fraud Score | Risk Level |
|-------------|------------|
| ≤ 8         | LOW RISK |
| 8 – 15      | SUSPICIOUS |
| > 15        | HIGH RISK |

Fraud score is calculated as:

```
probability * 100
```

---

## 📈 Training Visualizations

Generated and saved inside `/plots`:

✔ Class distribution  
✔ Correlation heatmap  
✔ Feature importance  
✔ Confusion matrix  
✔ Model comparison  

---

## 🖥️ How To Run

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Train Model

```
python train_model.py
```

This generates:

```
optimal_model.pkl
feature_names.pkl
```

### 3️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 📦 Project Structure

```
.
│
├── app.py
├── train_model.py
├── Stori_Data_Challenge_2021..csv
├── optimal_model.pkl
├── feature_names.pkl
├── requirements.txt
│
└── plots/
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    └── confusion_matrix.png
```

---

## 🧠 Technical Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  
- Joblib  
- Streamlit  

---

## 🔍 Why Recall Over Accuracy?

In fraud detection:

```
False Negative > False Positive
```

Missing fraud is more costly than investigating a normal transaction.

Therefore:

✔ Recall is prioritized  
✔ Balanced class weights used  
✔ scale_pos_weight applied in XGBoost  

---

## 🏁 Final Result

✔ Automated model selection  
✔ Optimized for fraud detection recall  
✔ Enterprise-level dashboard  
✔ Real-time scoring  
✔ Explainable risk drivers  

---

## 👤 Author

Abdul Rahman  
B.Tech Artificial Intelligence & Data Science  

---

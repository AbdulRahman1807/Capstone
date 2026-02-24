# IMPORT STATEMENTS
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Create plots folder
os.makedirs("plots", exist_ok=True)


# 1. LOAD DATA
df = pd.read_csv("Stori_Data_Challenge_2021..csv")

print("Dataset Shape:", df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("\nFraud Distribution:\n", df['fraud'].value_counts())
print("Fraud Percentage:", df['fraud'].mean() * 100)


# Class Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x=df['fraud'])
plt.title("Fraud Class Distribution")
plt.savefig("plots/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# 2. DATA CLEANING
df = df.drop(columns=['cust_id', 'activated_date', 'last_payment_date'])

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])


# FILLING MISSING VALUES
df.fillna(df.median(numeric_only=True), inplace=True)


# 3. CORRELATION ANALYSIS
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# 4. FEATURE / TARGET SPLIT
X = df.drop('fraud', axis=1)
y = df['fraud']


# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Fraud %:", y_train.mean() * 100)
print("Test Fraud %:", y_test.mean() * 100)


# 6. BASELINE RANDOM FOREST
rf_baseline = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

rf_baseline.fit(X_train, y_train)

y_pred_rf = rf_baseline.predict(X_test)
y_proba_rf = rf_baseline.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Baseline ===")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))


# 7. BASELINE XGBOOST
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_baseline = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight
)

xgb_baseline.fit(X_train, y_train)

y_pred_xgb = xgb_baseline.predict(X_test)
y_proba_xgb = xgb_baseline.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Baseline ===")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))


# 8. RANDOM FOREST TUNING
rf_param_grid = {
    'max_depth': [3, 5, 8, 12],
    'n_estimators': [100, 200],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    rf_param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nBest RF Params:", rf_grid.best_params_)
print("Best RF CV Recall:", rf_grid.best_score_)

rf_tuned = rf_grid.best_estimator_

y_pred_rf_tuned = rf_tuned.predict(X_test)
y_proba_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Tuned ===")
print(classification_report(y_test, y_pred_rf_tuned))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf_tuned))


# 9. XGBOOST TUNING
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200]
}

xgb_grid = GridSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    ),
    xgb_param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("\nBest XGB Params:", xgb_grid.best_params_)
print("Best XGB CV Recall:", xgb_grid.best_score_)

xgb_tuned = xgb_grid.best_estimator_

y_pred_xgb_tuned = xgb_tuned.predict(X_test)
y_proba_xgb_tuned = xgb_tuned.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Tuned ===")
print(classification_report(y_test, y_pred_xgb_tuned))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb_tuned))


# 10. FEATURE IMPORTANCE
importances = rf_tuned.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# Feature Importance Plot
plt.figure(figsize=(8, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df.head(10)
)
plt.title("Top 10 Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


# 11. FEATURE SELECTION
top_features = feature_importance_df.head(10)["Feature"].tolist()

X_train_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]

rf_reduced = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=rf_grid.best_params_['max_depth'],
    n_estimators=rf_grid.best_params_['n_estimators'],
    min_samples_split=rf_grid.best_params_['min_samples_split']
)

rf_reduced.fit(X_train_reduced, y_train)

y_pred_reduced = rf_reduced.predict(X_test_reduced)
y_proba_reduced = rf_reduced.predict_proba(X_test_reduced)[:, 1]

print("\n=== Random Forest with Top 10 Features ===")
print(classification_report(y_test, y_pred_reduced))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_reduced))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_reduced)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - RF Reduced")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# 12. FINAL MODEL COMPARISON
def get_recall(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)['1']['recall']

models_summary = [
    {"name": "RF Baseline", "model": rf_baseline,
     "recall": get_recall(y_test, y_pred_rf),
     "roc_auc": roc_auc_score(y_test, y_proba_rf),
     "features": X.columns.tolist()},
     
    {"name": "XGB Baseline", "model": xgb_baseline,
     "recall": get_recall(y_test, y_pred_xgb),
     "roc_auc": roc_auc_score(y_test, y_proba_xgb),
     "features": X.columns.tolist()},
     
    {"name": "RF Tuned", "model": rf_tuned,
     "recall": get_recall(y_test, y_pred_rf_tuned),
     "roc_auc": roc_auc_score(y_test, y_proba_rf_tuned),
     "features": X.columns.tolist()},
     
    {"name": "XGB Tuned", "model": xgb_tuned,
     "recall": get_recall(y_test, y_pred_xgb_tuned),
     "roc_auc": roc_auc_score(y_test, y_proba_xgb_tuned),
     "features": X.columns.tolist()},
     
    {"name": "RF Reduced (Top 10)", "model": rf_reduced,
     "recall": get_recall(y_test, y_pred_reduced),
     "roc_auc": roc_auc_score(y_test, y_proba_reduced),
     "features": top_features}
]

comparison_df = pd.DataFrame(models_summary)

print("\n=== MODEL COMPARISON TABLE ===")
print(comparison_df[["name", "recall", "roc_auc"]])


# 13. SELECT BEST MODEL
best_index = comparison_df["recall"].idxmax()
best_entry = comparison_df.loc[best_index]

best_model = best_entry["model"]
best_model_name = best_entry["name"]
best_features = best_entry["features"]

print("\nSelected Best Model:", best_model_name)
print("Best Recall:", best_entry["recall"])
print("Best ROC-AUC:", best_entry["roc_auc"])


# 14. SAVE BEST MODEL
joblib.dump(best_model, "optimal_model.pkl")
joblib.dump(best_features, "feature_names.pkl")

print("\nBest model saved successfully.")
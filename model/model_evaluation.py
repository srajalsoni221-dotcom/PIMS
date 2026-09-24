import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

DELAY_DATA = "D:/New Project/Predictive_Project_Monitoring/data/cleaned/ml_delay_dataset.csv"
COST_DATA = "D:/New Project/Predictive_Project_Monitoring/data/cleaned/ml_cost_dataset.csv"

DELAY_MODEL = "delay_prediction_model.joblib"
COST_MODEL = "cost_overrun_prediction_model.joblib"


# ============================================================
# FUNCTION: EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# DELAY MODEL
# ============================================================

print("\nLoading Delay Dataset...")

delay_df = pd.read_csv(DELAY_DATA)

X_delay = delay_df.drop(columns=["delay_flag"])
y_delay = delay_df["delay_flag"]

X_train_delay, X_test_delay, y_train_delay, y_test_delay = train_test_split(
    X_delay,
    y_delay,
    test_size=0.20,
    random_state=42,
    stratify=y_delay
)

delay_model = joblib.load(DELAY_MODEL)

delay_result = evaluate_model(
    delay_model,
    X_test_delay,
    y_test_delay,
    "DELAY PREDICTION MODEL"
)


# ============================================================
# COST OVERRUN MODEL
# ============================================================

print("\nLoading Cost Overrun Dataset...")

cost_df = pd.read_csv(COST_DATA)

X_cost = cost_df.drop(columns=["cost_overrun_flag"])
y_cost = cost_df["cost_overrun_flag"]

X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(
    X_cost,
    y_cost,
    test_size=0.20,
    random_state=42,
    stratify=y_cost
)

cost_model = joblib.load(COST_MODEL)

cost_result = evaluate_model(
    cost_model,
    X_test_cost,
    y_test_cost,
    "COST OVERRUN PREDICTION MODEL"
)


# ============================================================
# COMPARISON TABLE
# ============================================================

results = pd.DataFrame([
    delay_result,
    cost_result
])

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "../data/cleaned/model_evaluation_results.csv"

results.to_csv(output_path, index=False)

print("\nEvaluation results saved to:")
print(output_path)

print("\nModel Evaluation Completed Successfully!")
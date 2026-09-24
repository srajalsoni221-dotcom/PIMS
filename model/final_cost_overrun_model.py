import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "../data/cleaned/ml_cost_dataset.csv"
MODEL_PATH = "cost_overrun_prediction_model.joblib"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Cost Overrun Dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["cost_overrun_flag"])
y = df["cost_overrun_flag"]


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "original_cost",
    "expenditure",
    "physical_progress",
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month"
]

categorical_features = [
    "sector",
    "ministry",
    "agency"
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Final Extra Trees Model...")


# ============================================================
# FINAL MODEL
# ============================================================

model = ExtraTreesClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# TRAIN
# ============================================================

pipeline.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("FINAL COST OVERRUN MODEL")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nFinal model saved successfully:")
print(MODEL_PATH)
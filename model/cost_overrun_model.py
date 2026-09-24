import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

input_file = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned\ml_cost_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("COST OVERRUN PREDICTION MODEL")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ---------------------------------------------------------
# 2. Features and Target
# ---------------------------------------------------------

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

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["cost_overrun_flag"]


# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# 5. Random Forest Model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ---------------------------------------------------------
# 6. Complete Pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------------
# 7. Train Model
# ---------------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ---------------------------------------------------------
# 8. Prediction
# ---------------------------------------------------------

y_pred = pipeline.predict(X_test)


# ---------------------------------------------------------
# 9. Model Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Cost Overrun",
            "Cost Overrun"
        ]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ---------------------------------------------------------
# 10. Save Model
# ---------------------------------------------------------

model_folder = r"D:\New Project\Predictive_Project_Monitoring\model"

model_file = os.path.join(
    model_folder,
    "cost_overrun_prediction_model.joblib"
)

joblib.dump(
    pipeline,
    model_file
)

print("\nModel saved successfully:")
print(model_file)


# ---------------------------------------------------------
# 11. Final Message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COST OVERRUN MODEL COMPLETED")
print("=" * 60)
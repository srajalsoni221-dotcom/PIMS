import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# 1. Load ML Datasets
# ---------------------------------------------------------

base_path = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned"

delay_file = os.path.join(
    base_path,
    "ml_delay_dataset.csv"
)

cost_file = os.path.join(
    base_path,
    "ml_cost_dataset.csv"
)

delay_df = pd.read_csv(delay_file)
cost_df = pd.read_csv(cost_file)


print("=" * 60)
print("TRAIN / TEST SPLIT AND PREPROCESSING")
print("=" * 60)


# ---------------------------------------------------------
# 2. Define Features
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


# =========================================================
# 3. DELAY MODEL DATA
# =========================================================

X_delay = delay_df[feature_columns]
y_delay = delay_df["delay_flag"]


print("\nDelay model:")
print("X shape:", X_delay.shape)
print("y shape:", y_delay.shape)


# ---------------------------------------------------------
# 4. Train-Test Split for Delay Model
# ---------------------------------------------------------

X_delay_train, X_delay_test, y_delay_train, y_delay_test = train_test_split(
    X_delay,
    y_delay,
    test_size=0.20,
    random_state=42,
    stratify=y_delay
)


print("\nDelay training data:")
print(X_delay_train.shape)

print("Delay testing data:")
print(X_delay_test.shape)


# =========================================================
# 5. COST OVERRUN MODEL DATA
# =========================================================

X_cost = cost_df[feature_columns]
y_cost = cost_df["cost_overrun_flag"]


print("\nCost overrun model:")
print("X shape:", X_cost.shape)
print("y shape:", y_cost.shape)


# ---------------------------------------------------------
# 6. Train-Test Split for Cost Model
# ---------------------------------------------------------

X_cost_train, X_cost_test, y_cost_train, y_cost_test = train_test_split(
    X_cost,
    y_cost,
    test_size=0.20,
    random_state=42,
    stratify=y_cost
)


print("\nCost training data:")
print(X_cost_train.shape)

print("Cost testing data:")
print(X_cost_test.shape)


# =========================================================
# 7. PREPROCESSING PIPELINE
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
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
# 8. Fit Preprocessor ONLY on Training Data
# ---------------------------------------------------------

X_delay_train_processed = preprocessor.fit_transform(
    X_delay_train
)

X_delay_test_processed = preprocessor.transform(
    X_delay_test
)


print("\nDelay preprocessing:")
print(
    "Training processed shape:",
    X_delay_train_processed.shape
)

print(
    "Testing processed shape:",
    X_delay_test_processed.shape
)


# ---------------------------------------------------------
# 9. Cost Model Preprocessing
# ---------------------------------------------------------

cost_preprocessor = ColumnTransformer(
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


X_cost_train_processed = cost_preprocessor.fit_transform(
    X_cost_train
)

X_cost_test_processed = cost_preprocessor.transform(
    X_cost_test
)


print("\nCost preprocessing:")
print(
    "Training processed shape:",
    X_cost_train_processed.shape
)

print(
    "Testing processed shape:",
    X_cost_test_processed.shape
)


# ---------------------------------------------------------
# 10. Final Target Distribution Check
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)


print("\nDelay training target:")
print(y_delay_train.value_counts())

print("\nDelay testing target:")
print(y_delay_test.value_counts())


print("\nCost training target:")
print(y_cost_train.value_counts())

print("\nCost testing target:")
print(y_cost_test.value_counts())


# ---------------------------------------------------------
# 11. Final Message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN / TEST PREPARATION COMPLETED")
print("=" * 60)

print("\nData is ready for ML model training.")
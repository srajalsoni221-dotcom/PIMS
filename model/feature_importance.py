import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

base_path = r"D:\New Project\Predictive_Project_Monitoring"

model_path = os.path.join(
    base_path,
    "model"
)

data_path = os.path.join(
    base_path,
    "data",
    "cleaned"
)


# ---------------------------------------------------------
# 2. Load Models
# ---------------------------------------------------------

delay_model_file = os.path.join(
    model_path,
    "delay_prediction_model.joblib"
)

cost_model_file = os.path.join(
    model_path,
    "cost_overrun_prediction_model.joblib"
)

delay_model = joblib.load(delay_model_file)
cost_model = joblib.load(cost_model_file)


# ---------------------------------------------------------
# 3. Feature Names
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


# ---------------------------------------------------------
# 4. Get Processed Feature Names
# ---------------------------------------------------------

def get_feature_names(pipeline):

    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    return feature_names


# ---------------------------------------------------------
# 5. Get Feature Importance
# ---------------------------------------------------------

def get_importance(pipeline):

    model = pipeline.named_steps["model"]

    feature_names = get_feature_names(pipeline)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    return importance_df


# ---------------------------------------------------------
# 6. Delay Feature Importance
# ---------------------------------------------------------

print("=" * 60)
print("DELAY MODEL FEATURE IMPORTANCE")
print("=" * 60)

delay_importance = get_importance(delay_model)

print("\nTop 20 important features:")

print(
    delay_importance.head(20).to_string(index=False)
)


# ---------------------------------------------------------
# 7. Save Delay Importance
# ---------------------------------------------------------

delay_output = os.path.join(
    data_path,
    "delay_feature_importance.csv"
)

delay_importance.to_csv(
    delay_output,
    index=False
)


# ---------------------------------------------------------
# 8. Cost Overrun Feature Importance
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COST OVERRUN MODEL FEATURE IMPORTANCE")
print("=" * 60)

cost_importance = get_importance(cost_model)

print("\nTop 20 important features:")

print(
    cost_importance.head(20).to_string(index=False)
)


# ---------------------------------------------------------
# 9. Save Cost Importance
# ---------------------------------------------------------

cost_output = os.path.join(
    data_path,
    "cost_feature_importance.csv"
)

cost_importance.to_csv(
    cost_output,
    index=False
)


# ---------------------------------------------------------
# 10. Delay Feature Importance Graph
# ---------------------------------------------------------

top_delay = delay_importance.head(15).sort_values(
    "importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_delay["feature"],
    top_delay["importance"]
)

plt.title("Top Features for Delay Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 11. Cost Feature Importance Graph
# ---------------------------------------------------------

top_cost = cost_importance.head(15).sort_values(
    "importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_cost["feature"],
    top_cost["importance"]
)

plt.title("Top Features for Cost Overrun Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 12. Final Message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 60)

print("\nSaved files:")

print(delay_output)
print(cost_output)
import pandas as pd
import os

# ---------------------------------------------------------
# 1. Load Feature-Engineered Dataset
# ---------------------------------------------------------

input_file = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned\projects_features.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("ML DATA PREPARATION")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. Select Features
# ---------------------------------------------------------

feature_columns = [
    "original_cost",
    "expenditure",
    "physical_progress",
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month",
    "sector",
    "ministry",
    "agency"
]

# ---------------------------------------------------------
# 3. Select Delay Dataset
# ---------------------------------------------------------

delay_columns = feature_columns + ["delay_flag"]

delay_df = df[delay_columns].copy()

# Keep only projects with known delay outcome
delay_df = delay_df.dropna(subset=["delay_flag"])

print("\nDelay dataset shape before missing-value handling:")
print(delay_df.shape)

# ---------------------------------------------------------
# 4. Handle Missing Numerical Values
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

for column in numeric_features:
    delay_df[column] = delay_df[column].fillna(
        delay_df[column].median()
    )

# ---------------------------------------------------------
# 5. Handle Missing Categorical Values
# ---------------------------------------------------------

categorical_features = [
    "sector",
    "ministry",
    "agency"
]

for column in categorical_features:
    delay_df[column] = delay_df[column].fillna("Unknown")

# ---------------------------------------------------------
# 6. Convert Target to Integer
# ---------------------------------------------------------

delay_df["delay_flag"] = delay_df["delay_flag"].astype(int)

# ---------------------------------------------------------
# 7. Select Cost Overrun Dataset
# ---------------------------------------------------------

cost_columns = feature_columns + ["cost_overrun_flag"]

cost_df = df[cost_columns].copy()

# Keep only projects with known cost outcome
cost_df = cost_df.dropna(
    subset=["cost_overrun_flag"]
)

print("\nCost dataset shape before missing-value handling:")
print(cost_df.shape)

# ---------------------------------------------------------
# 8. Handle Missing Numerical Values
# ---------------------------------------------------------

for column in numeric_features:
    cost_df[column] = cost_df[column].fillna(
        cost_df[column].median()
    )

# ---------------------------------------------------------
# 9. Handle Missing Categorical Values
# ---------------------------------------------------------

for column in categorical_features:
    cost_df[column] = cost_df[column].fillna("Unknown")

# ---------------------------------------------------------
# 10. Convert Target to Integer
# ---------------------------------------------------------

cost_df["cost_overrun_flag"] = (
    cost_df["cost_overrun_flag"].astype(int)
)

# ---------------------------------------------------------
# 11. Create Output Folder
# ---------------------------------------------------------

output_folder = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned"

# ---------------------------------------------------------
# 12. Save ML Datasets
# ---------------------------------------------------------

delay_output = os.path.join(
    output_folder,
    "ml_delay_dataset.csv"
)

cost_output = os.path.join(
    output_folder,
    "ml_cost_dataset.csv"
)

delay_df.to_csv(
    delay_output,
    index=False
)

cost_df.to_csv(
    cost_output,
    index=False
)

# ---------------------------------------------------------
# 13. Final Checks
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL ML DATASET CHECK")
print("=" * 60)

print("\nDelay dataset:")
print(delay_df.shape)

print("\nDelay target distribution:")
print(delay_df["delay_flag"].value_counts())

print("\nDelay dataset missing values:")
print(delay_df.isnull().sum().sum())

print("\nCost overrun dataset:")
print(cost_df.shape)

print("\nCost overrun target distribution:")
print(cost_df["cost_overrun_flag"].value_counts())

print("\nCost dataset missing values:")
print(cost_df.isnull().sum().sum())

print("\nFiles saved:")
print(delay_output)
print(cost_output)

print("\nML datasets prepared successfully!")
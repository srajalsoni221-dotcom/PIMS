import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

input_file = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned\projects_cleaned.csv"

output_file = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned\projects_features.csv"


# ---------------------------------------------------------
# 2. Check input file
# ---------------------------------------------------------

if not os.path.exists(input_file):
    print("ERROR: Cleaned CSV file not found.")
    print("Expected location:")
    print(input_file)
    exit()


# ---------------------------------------------------------
# 3. Read cleaned dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Cleaned dataset loaded successfully.")
print("Original shape:", df.shape)


# ---------------------------------------------------------
# 4. Convert date columns
# ---------------------------------------------------------

date_columns = [
    "original_completion",
    "revised_completion",
    "sanction_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# 5. Project Duration
# ---------------------------------------------------------

df["project_duration_days"] = (
    df["original_completion"] - df["sanction_date"]
).dt.days

df["project_duration_months"] = (
    df["project_duration_days"] / 30.44
)


# ---------------------------------------------------------
# 6. Expenditure Ratio
# ---------------------------------------------------------

df["expenditure_ratio"] = np.where(
    df["original_cost"] > 0,
    df["expenditure"] / df["original_cost"],
    np.nan
)


# ---------------------------------------------------------
# 7. Progress per Month
# ---------------------------------------------------------

df["progress_per_month"] = np.where(
    df["project_duration_months"] > 0,
    df["physical_progress"] / df["project_duration_months"],
    np.nan
)


# ---------------------------------------------------------
# 10. Date inconsistency helper
# ---------------------------------------------------------

df["completion_date_difference_days"] = (
    df["revised_completion"] -
    df["original_completion"]
).dt.days


# ---------------------------------------------------------
# 11. Save feature-engineered dataset
# ---------------------------------------------------------

df.to_csv(output_file, index=False)


# ---------------------------------------------------------
# 12. Display result
# ---------------------------------------------------------

print("\nFeature engineering completed successfully.")

print("New dataset shape:", df.shape)

print("\nNew features created:")

new_features = [
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month",
    "completion_date_difference_days"
]

for feature in new_features:
    print("-", feature)

print("\nOutput file:")
print(output_file)

print("\nFirst 5 rows of new features:")
print(df[new_features].head())

# ---------------------------------------------------------
# 13. Feature Quality Check
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("FEATURE QUALITY CHECK")
print("=" * 50)

print("\nMissing values in new features:")

for feature in new_features:
    missing = df[feature].isna().sum()
    print(f"{feature}: {missing}")

print("\nNegative values check:")

duration_features = [
    "project_duration_days",
    "project_duration_months"
]

for feature in duration_features:
    negative = (df[feature] < 0).sum()
    print(f"{feature}: {negative}")

print("\nFeature statistics:")

print(
    df[new_features].describe()
)

# ---------------------------------------------------------
# 14. Investigate Missing and Negative Durations
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("DURATION INVESTIGATION")
print("=" * 50)

# Rows where project duration is missing
missing_duration = df[
    df["project_duration_days"].isna()
]

print("\nProjects with missing project duration:")
print("Count:", len(missing_duration))

print(
    missing_duration[
        [
            "project_code",
            "project_name",
            "sanction_date",
            "original_completion"
        ]
    ].head(10)
)


# Rows where project duration is negative
negative_duration = df[
    df["project_duration_days"] < 0
]

print("\nProjects with negative project duration:")
print("Count:", len(negative_duration))

print(
    negative_duration[
        [
            "project_code",
            "project_name",
            "sanction_date",
            "original_completion",
            "project_duration_days"
        ]
    ]
)

# ---------------------------------------------------------
# 15. Delay Target Creation
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("DELAY TARGET ANALYSIS")
print("=" * 50)

# Calculate actual delay in days
df["delay_days"] = (
    df["revised_completion"] -
    df["original_completion"]
).dt.days

# Create delay flag
# 1 = Delayed
# 0 = Not Delayed
# NaN = Cannot determine because revised completion is missing

df["delay_flag"] = np.where(
    df["delay_days"].notna(),
    (df["delay_days"] > 0).astype(int),
    np.nan
)

print("\nDelay target created.")

print("\nDelay days missing:")
print(df["delay_days"].isna().sum())

print("\nDelay flag distribution:")
print(df["delay_flag"].value_counts(dropna=False))

print("\nDelay statistics:")
print(df["delay_days"].describe())

# Save updated feature dataset
df.to_csv(output_file, index=False)

print("\nUpdated feature dataset saved:")
print(output_file)

# ---------------------------------------------------------
# 16. Cost Overrun Target Creation
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("COST OVERRUN TARGET ANALYSIS")
print("=" * 50)

# Calculate additional/revised cost
df["cost_overrun"] = (
    df["revised_cost"] - df["original_cost"]
)

# Calculate percentage cost overrun
df["cost_overrun_percent"] = np.where(
    df["original_cost"] > 0,
    (df["cost_overrun"] / df["original_cost"]) * 100,
    np.nan
)

# Create cost overrun flag
# 1 = Cost overrun
# 0 = No cost overrun
# NaN = Revised cost unavailable

df["cost_overrun_flag"] = np.where(
    df["revised_cost"].notna(),
    (df["revised_cost"] > df["original_cost"]).astype(int),
    np.nan
)

print("\nCost overrun target created.")

print("\nCost overrun missing:")
print(df["cost_overrun"].isna().sum())

print("\nCost overrun flag distribution:")
print(df["cost_overrun_flag"].value_counts(dropna=False))

print("\nCost overrun percentage statistics:")
print(df["cost_overrun_percent"].describe())

# Save updated dataset
df.to_csv(output_file, index=False)

print("\nUpdated feature dataset saved:")
print(output_file)

# ---------------------------------------------------------
# 17. Final Feature and Target Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL FEATURE AND TARGET SUMMARY")
print("=" * 60)

print("\nTotal projects:")
print(len(df))

print("\nDelay target:")
print("Known delay outcomes:", df["delay_flag"].notna().sum())
print("Delayed:", (df["delay_flag"] == 1).sum())
print("Not delayed:", (df["delay_flag"] == 0).sum())

print("\nCost overrun target:")
print("Known cost outcomes:", df["cost_overrun_flag"].notna().sum())
print("Cost overrun:", (df["cost_overrun_flag"] == 1).sum())
print("No cost overrun:", (df["cost_overrun_flag"] == 0).sum())

print("\nImportant feature missing values:")

feature_columns = [
    "original_cost",
    "expenditure",
    "physical_progress",
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month"
]

for feature in feature_columns:
    print(f"{feature}: {df[feature].isna().sum()}")

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

# Final save
df.to_csv(output_file, index=False)

print("\nFinal feature-engineered dataset saved successfully.")
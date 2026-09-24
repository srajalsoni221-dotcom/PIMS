import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. Load Feature-Engineered Dataset
# ---------------------------------------------------------

file_path = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned\projects_features.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. Basic Information
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 3. Basic Statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

numeric_columns = [
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress",
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month",
    "delay_days",
    "cost_overrun_percent"
]

print(df[numeric_columns].describe())

# ---------------------------------------------------------
# 4. Delay Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DELAY ANALYSIS")
print("=" * 60)

print("\nDelay flag counts:")
print(df["delay_flag"].value_counts(dropna=False))

print("\nDelayed projects:")
print((df["delay_flag"] == 1).sum())

print("\nProjects without delay:")
print((df["delay_flag"] == 0).sum())

print("\nUnknown delay outcome:")
print(df["delay_flag"].isna().sum())

# ---------------------------------------------------------
# 5. Cost Overrun Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COST OVERRUN ANALYSIS")
print("=" * 60)

print("\nCost overrun flag counts:")
print(df["cost_overrun_flag"].value_counts(dropna=False))

print("\nProjects with cost overrun:")
print((df["cost_overrun_flag"] == 1).sum())

print("\nProjects without cost overrun:")
print((df["cost_overrun_flag"] == 0).sum())

print("\nUnknown cost outcome:")
print(df["cost_overrun_flag"].isna().sum())

# ---------------------------------------------------------
# 6. Sector Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SECTOR ANALYSIS")
print("=" * 60)

sector_counts = df["sector"].value_counts()

print("\nProjects by sector:")
print(sector_counts)

# ---------------------------------------------------------
# 7. Delays by Sector
# ---------------------------------------------------------

delay_sector = (
    df.dropna(subset=["delay_flag"])
    .groupby("sector")["delay_flag"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

delay_sector["delay_percentage"] = delay_sector["mean"] * 100

print("\nDelay analysis by sector:")
print(delay_sector)

# ---------------------------------------------------------
# 8. Cost Overrun by Sector
# ---------------------------------------------------------

cost_sector = (
    df.dropna(subset=["cost_overrun_flag"])
    .groupby("sector")["cost_overrun_flag"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

cost_sector["overrun_percentage"] = cost_sector["mean"] * 100

print("\nCost overrun analysis by sector:")
print(cost_sector)

# ---------------------------------------------------------
# 9. High Expenditure + Low Progress Projects
# ---------------------------------------------------------

problematic_projects = df[
    (df["physical_progress"] <= 30) &
    (df["expenditure"] > df["original_cost"] * 0.70)
]

print("\n" + "=" * 60)
print("HIGH EXPENDITURE + LOW PROGRESS")
print("=" * 60)

print("\nNumber of potentially problematic projects:")
print(len(problematic_projects))

if len(problematic_projects) > 0:
    print("\nPotentially problematic projects:")
    print(
        problematic_projects[
            [
                "project_code",
                "project_name",
                "original_cost",
                "expenditure",
                "physical_progress"
            ]
        ].head(20)
    )

# ---------------------------------------------------------
# 10. Extreme Cost Overrun
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EXTREME COST OVERRUN")
print("=" * 60)

extreme_cost = df[
    df["cost_overrun_percent"] > 100
].sort_values(
    "cost_overrun_percent",
    ascending=False
)

print("\nProjects with cost overrun greater than 100%:")
print(len(extreme_cost))

if len(extreme_cost) > 0:
    print(
        extreme_cost[
            [
                "project_code",
                "project_name",
                "original_cost",
                "revised_cost",
                "cost_overrun_percent"
            ]
        ].head(20)
    )

# ---------------------------------------------------------
# 11. Save EDA Tables
# ---------------------------------------------------------

output_folder = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned"

sector_counts.to_csv(
    os.path.join(output_folder, "sector_counts.csv")
)

delay_sector.to_csv(
    os.path.join(output_folder, "delay_by_sector.csv")
)

cost_sector.to_csv(
    os.path.join(output_folder, "cost_overrun_by_sector.csv")
)

print("\nEDA summary tables saved successfully.")

# ---------------------------------------------------------
# 12. Graph 1 - Projects by Sector
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sector_counts.plot(kind="bar")

plt.title("Number of Projects by Sector")
plt.xlabel("Sector")
plt.ylabel("Number of Projects")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 13. Graph 2 - Delay Distribution
# ---------------------------------------------------------

delay_data = df["delay_days"].dropna()

plt.figure(figsize=(10, 6))

plt.hist(delay_data, bins=30)

plt.title("Distribution of Project Delay")
plt.xlabel("Delay Days")
plt.ylabel("Number of Projects")

plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 14. Graph 3 - Physical Progress
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(df["physical_progress"], bins=20)

plt.title("Distribution of Physical Progress")
plt.xlabel("Physical Progress (%)")
plt.ylabel("Number of Projects")

plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 15. Graph 4 - Expenditure Ratio
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["expenditure_ratio"],
    bins=30
)

plt.title("Distribution of Expenditure Ratio")
plt.xlabel("Expenditure / Original Cost")
plt.ylabel("Number of Projects")

plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 16. Graph 5 - Expenditure vs Physical Progress
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["expenditure"],
    df["physical_progress"],
    alpha=0.6
)

plt.title("Expenditure vs Physical Progress")
plt.xlabel("Expenditure")
plt.ylabel("Physical Progress (%)")

plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 17. Graph 6 - Correlation Heatmap
# ---------------------------------------------------------

correlation_columns = [
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress",
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month",
    "delay_days",
    "cost_overrun_percent"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 18. Final Message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)
import pandas as pd
import os

# ==============================
# 1. File Paths
# ==============================

input_file = r"D:/New Project/Predictive_Project_Monitoring/data/raw/Projects_Report.xlsx"

output_file = r"D:\New Project\Predictive_Project_Monitoring\data\cleaned\projects_cleaned.csv"


# ==============================
# 2. Read Excel File
# ==============================

df = pd.read_excel(input_file, header=2)

print("\nOriginal Dataset Shape:")
print(df.shape)


# ==============================
# 3. Rename Columns
# ==============================

df.columns = [
    "sr_no",
    "sector",
    "ministry",
    "agency",
    "project_code",
    "project_name",
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress",
    "original_completion",
    "revised_completion",
    "sanction_date"
]


# ==============================
# 4. Convert Dates
# ==============================

date_columns = [
    "original_completion",
    "revised_completion",
    "sanction_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        dayfirst=True,
        errors="coerce"
    )


# ==============================
# 5. Handle Zero Revised Cost
# ==============================

# 0 means revised cost is unavailable
df["revised_cost"] = df["revised_cost"].replace(0, pd.NA)


# ==============================
# 6. Create Date Consistency Flags
# ==============================

df["date_inconsistency_flag"] = 0

# Revised completion earlier than original completion
condition_1 = (
    df["revised_completion"].notna()
    & (df["revised_completion"] < df["original_completion"])
)

# Original completion earlier than sanction date
condition_2 = (
    df["sanction_date"].notna()
    & (df["original_completion"] < df["sanction_date"])
)

df.loc[condition_1 | condition_2, "date_inconsistency_flag"] = 1


# ==============================
# 7. Create Data Quality Flags
# ==============================

df["missing_revised_cost_flag"] = (
    df["revised_cost"].isna().astype(int)
)

df["missing_revised_completion_flag"] = (
    df["revised_completion"].isna().astype(int)
)

df["missing_sanction_date_flag"] = (
    df["sanction_date"].isna().astype(int)
)


# ==============================
# 8. Save Cleaned Dataset
# ==============================

# Create folder if it does not exist
os.makedirs(
    r"D:\New Project\Predictive_Project_Monitoring\data\cleaned",
    exist_ok=True
)

df.to_csv(
    output_file,
    index=False
)


# ==============================
# 9. Final Report
# ==============================

print("\nCleaning Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Revised Cost:")
print(df["revised_cost"].isna().sum())

print("\nDate Inconsistency Projects:")
print(df["date_inconsistency_flag"].sum())

print("\nOutput File:")
print(output_file)


#D:/New Project/Predictive_Project_Monitoring/data/raw/Projects_Report.xlsx
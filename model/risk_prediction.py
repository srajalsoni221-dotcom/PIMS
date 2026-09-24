import pandas as pd
import joblib
import os

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "projects_features.csv"
)

DELAY_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "ml_delay_dataset.csv"
)

COST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "ml_cost_dataset.csv"
)

DELAY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "delay_prediction_model.joblib"
)

COST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "cost_overrun_prediction_model.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "project_risk_predictions.csv"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "risk_summary.csv"
)

# ============================================================
# LOAD ORIGINAL FEATURE DATA
# ============================================================

print("Loading original project data...")

features_df = pd.read_csv(FEATURES_PATH)

print("Original project dataset:", features_df.shape)

if "project_code" not in features_df.columns:
    raise ValueError(
        "project_code not found in projects_features.csv"
    )

# ============================================================
# LOAD ML DATASETS
# ============================================================

print("\nLoading delay ML dataset...")

delay_df = pd.read_csv(DELAY_DATA_PATH)

print("Delay dataset:", delay_df.shape)

print("\nLoading cost ML dataset...")

cost_df = pd.read_csv(COST_DATA_PATH)

print("Cost dataset:", cost_df.shape)

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading Delay Model...")

delay_model = joblib.load(DELAY_MODEL_PATH)

print("Delay model loaded.")

print("\nLoading Cost Overrun Model...")

cost_model = joblib.load(COST_MODEL_PATH)

print("Cost overrun model loaded.")

# ============================================================
# DELAY PREDICTION
# ============================================================

print("\nGenerating delay predictions...")

delay_target = "delay_flag"

X_delay = delay_df.drop(
    columns=[delay_target]
)

delay_predictions = delay_model.predict(X_delay)

delay_probabilities = delay_model.predict_proba(
    X_delay
)[:, 1]

print("Delay predictions completed.")

# ============================================================
# COST PREDICTION
# ============================================================

print("\nGenerating cost overrun predictions...")

cost_target = "cost_overrun_flag"

X_cost = cost_df.drop(
    columns=[cost_target]
)

cost_predictions = cost_model.predict(X_cost)

cost_probabilities = cost_model.predict_proba(
    X_cost
)[:, 1]

print("Cost predictions completed.")

# ============================================================
# CREATE PROJECT INFORMATION
# ============================================================

project_columns = [
    "project_code",
    "project_name",
    "sector",
    "ministry",
    "agency",
    "original_cost",
    "revised_cost",
    "expenditure",
    "physical_progress",
    "original_completion",
    "revised_completion",
    "sanction_date"
]

available_columns = [
    col for col in project_columns
    if col in features_df.columns
]

project_info = features_df[
    available_columns
].copy()

# ============================================================
# ADD DELAY PREDICTIONS
# ============================================================

# Delay dataset contains 528 projects.
# We use its original row order to identify the
# corresponding projects from projects_features.csv.

delay_project_codes = []

for i in range(len(delay_df)):

    project_code = delay_df.iloc[i].get(
        "project_code",
        None
    )

    if project_code is not None:
        delay_project_codes.append(project_code)

# If project_code was removed from ML dataset,
# recover the projects using target alignment.

if len(delay_project_codes) == 0:

    delay_mask = features_df["delay_flag"].notna()

    delay_projects = features_df[
        delay_mask
    ].copy()

else:

    delay_projects = project_info[
        project_info["project_code"].isin(
            delay_project_codes
        )
    ].copy()

# ============================================================
# SAFETY CHECK
# ============================================================

if len(delay_projects) != len(delay_predictions):

    print(
        "\nWarning: Delay project count does not match."
    )

    print(
        "Delay projects:",
        len(delay_projects)
    )

    print(
        "Delay predictions:",
        len(delay_predictions)
    )

    # Use target mask because project_code was removed
    delay_projects = features_df[
        features_df["delay_flag"].notna()
    ].copy()

# ============================================================
# ATTACH DELAY RESULTS
# ============================================================

delay_projects = delay_projects.reset_index(
    drop=True
)

delay_projects["delay_prediction"] = (
    delay_predictions
)

delay_projects["delay_probability"] = (
    delay_probabilities
)

# ============================================================
# COST PROJECTS
# ============================================================

cost_mask = features_df[
    "cost_overrun_flag"
].notna()

cost_projects = features_df[
    cost_mask
].copy()

cost_projects = cost_projects.reset_index(
    drop=True
)

# ============================================================
# SAFETY CHECK
# ============================================================

if len(cost_projects) != len(cost_predictions):

    raise ValueError(
        "Cost project count does not match cost predictions."
    )

# ============================================================
# ATTACH COST RESULTS
# ============================================================

cost_projects["cost_overrun_prediction"] = (
    cost_predictions
)

cost_projects["cost_overrun_probability"] = (
    cost_probabilities
)

# ============================================================
# MERGE USING PROJECT CODE
# ============================================================

print("\nMatching delay and cost predictions...")

delay_result = delay_projects[
    [
        "project_code",
        "delay_prediction",
        "delay_probability"
    ]
]

cost_result = cost_projects[
    [
        "project_code",
        "cost_overrun_prediction",
        "cost_overrun_probability"
    ]
]

result = pd.merge(
    delay_result,
    cost_result,
    on="project_code",
    how="inner"
)

print(
    "Successfully matched projects:",
    len(result)
)

# ============================================================
# ADD PROJECT INFORMATION
# ============================================================

project_info_unique = project_info.drop_duplicates(
    subset=["project_code"]
)

result = pd.merge(
    project_info_unique,
    result,
    on="project_code",
    how="inner"
)

# ============================================================
# OVERALL RISK SCORE
# ============================================================

result["overall_risk_score"] = (
    result["delay_probability"] * 0.50
    +
    result["cost_overrun_probability"] * 0.50
)

result["overall_risk_percentage"] = (
    result["overall_risk_score"] * 100
)

# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score >= 0.70:
        return "High Risk"

    elif score >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


result["overall_risk_level"] = (
    result["overall_risk_score"]
    .apply(get_risk_level)
)

# ============================================================
# SORT BY RISK
# ============================================================

result = result.sort_values(
    by="overall_risk_score",
    ascending=False
)

# ============================================================
# SAVE RESULTS
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# RISK SUMMARY
# ============================================================

risk_summary = (
    result["overall_risk_level"]
    .value_counts()
    .reset_index()
)

risk_summary.columns = [
    "risk_level",
    "project_count"
]

risk_summary.to_csv(
    SUMMARY_PATH,
    index=False
)

# ============================================================
# DISPLAY TOP 10
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 HIGH-RISK PROJECTS")
print("=" * 80)

display_columns = [
    "project_code",
    "project_name",
    "sector",
    "original_cost",
    "expenditure",
    "physical_progress",
    "delay_probability",
    "cost_overrun_probability",
    "overall_risk_percentage",
    "overall_risk_level"
]

display_columns = [
    col for col in display_columns
    if col in result.columns
]

print(
    result[display_columns]
    .head(10)
    .to_string(index=False)
)

# ============================================================
# RISK SUMMARY OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("RISK SUMMARY")
print("=" * 80)

print(
    risk_summary.to_string(index=False)
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("RISK PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nDetailed results:")
print(OUTPUT_PATH)

print("\nRisk summary:")
print(SUMMARY_PATH)
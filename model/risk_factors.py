import os
import numpy as np
import pandas as pd
import joblib
import shap

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ============================================================
# FILE PATHS
# ============================================================

FEATURE_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "projects_features.csv"
)

RISK_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "project_risk_predictions.csv"
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
    "project_risk_factors.csv"
)

# ============================================================
# MODEL FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "original_cost",
    "expenditure",
    "physical_progress",
    "project_duration_days",
    "project_duration_months",
    "expenditure_ratio",
    "progress_per_month"
]

CATEGORICAL_FEATURES = [
    "sector",
    "ministry",
    "agency"
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PROJECT RISK FACTOR ANALYSIS")
print("=" * 75)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature-engineered project data...")

features_df = pd.read_csv(FEATURE_DATA_PATH)

print("Feature dataset:", features_df.shape)

print("\nLoading ML risk predictions...")

risk_df = pd.read_csv(RISK_DATA_PATH)

print("Risk prediction dataset:", risk_df.shape)

# ============================================================
# VALIDATION
# ============================================================

if "project_code" not in features_df.columns:
    raise ValueError(
        "project_code not found in projects_features.csv"
    )

if "project_code" not in risk_df.columns:
    raise ValueError(
        "project_code not found in project_risk_predictions.csv"
    )

missing_columns = [
    col for col in MODEL_FEATURES
    if col not in features_df.columns
]

if missing_columns:
    print("\nMissing model features:")

    for col in missing_columns:
        print("-", col)

    raise ValueError(
        "Required model features are missing."
    )

print("\nAll required model features found.")

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading Delay Model...")

delay_pipeline = joblib.load(
    DELAY_MODEL_PATH
)

print("Delay model loaded successfully.")

print("\nLoading Cost Overrun Model...")

cost_pipeline = joblib.load(
    COST_MODEL_PATH
)

print("Cost overrun model loaded successfully.")

# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_model_data(data):

    X = data[MODEL_FEATURES].copy()

    for col in NUMERIC_FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        X[col] = X[col].fillna(
            X[col].median()
        )

    for col in CATEGORICAL_FEATURES:

        X[col] = (
            X[col]
            .fillna("Unknown")
            .astype(str)
        )

    return X


# ============================================================
# SHAP CALCULATION
# ============================================================

def calculate_shap(pipeline, data, model_name):

    print("\n" + "-" * 60)
    print("Calculating SHAP for:", model_name)
    print("-" * 60)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    print("ML model:", type(model).__name__)

    X = prepare_model_data(data)

    # --------------------------------------------------------
    # Transform data
    # --------------------------------------------------------

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray()
    else:
        X_dense = np.asarray(X_transformed)

    # IMPORTANT:
    # Force pure numeric float64 matrix
    X_dense = np.asarray(
        X_dense,
        dtype=np.float64
    )

    print(
        "Transformed feature shape:",
        X_dense.shape
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print(
        "Feature names:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    print("Creating TreeExplainer...")

    explainer = shap.TreeExplainer(
        model
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_dense
    )

    # --------------------------------------------------------
    # SHAP version compatibility
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        # Binary classification
        # Class 1 = risk

        shap_array = np.asarray(
            shap_values[1],
            dtype=np.float64
        )

    else:

        shap_array = np.asarray(
            shap_values,
            dtype=np.float64
        )

        # Possible format:
        # samples x features x classes

        if shap_array.ndim == 3:

            shap_array = shap_array[:, :, 1]

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if shap_array.ndim != 2:

        raise ValueError(
            f"Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )

    if shap_array.shape[0] != len(data):

        raise ValueError(
            "SHAP sample count does not match dataset."
        )

    if shap_array.shape[1] != len(feature_names):

        raise ValueError(
            "SHAP feature count does not match feature names."
        )

    print(
        "SHAP calculation successful."
    )

    return shap_array, feature_names


# ============================================================
# CALCULATE DELAY SHAP
# ============================================================

print("\n" + "=" * 75)
print("DELAY RISK FACTOR ANALYSIS")
print("=" * 75)

delay_shap, delay_feature_names = calculate_shap(
    delay_pipeline,
    features_df,
    "Delay Model"
)

# ============================================================
# CALCULATE COST SHAP
# ============================================================

print("\n" + "=" * 75)
print("COST OVERRUN RISK FACTOR ANALYSIS")
print("=" * 75)

cost_shap, cost_feature_names = calculate_shap(
    cost_pipeline,
    features_df,
    "Cost Overrun Model"
)

# ============================================================
# CLEAN FEATURE NAME
# ============================================================

def clean_feature_name(name):

    name = str(name)

    if name.startswith("numeric__"):
        return name.replace(
            "numeric__",
            ""
        )

    if name.startswith("categorical__"):
        return name.replace(
            "categorical__",
            ""
        )

    return name


delay_clean_names = [
    clean_feature_name(name)
    for name in delay_feature_names
]

cost_clean_names = [
    clean_feature_name(name)
    for name in cost_feature_names
]

# ============================================================
# PROJECT EXPLANATIONS
# ============================================================

results = []

print("\nGenerating project-level explanations...")

# ============================================================
# LOOP
# ============================================================

for i in range(len(features_df)):

    project = features_df.iloc[i]

    project_code = project["project_code"]

    matching = risk_df[
        risk_df["project_code"].astype(str)
        == str(project_code)
    ]

    if matching.empty:
        continue

    prediction = matching.iloc[0]

    # ========================================================
    # DELAY FACTORS
    # ========================================================

    delay_scores = np.abs(
        delay_shap[i]
    )

    delay_indices = np.argsort(
        delay_scores
    )[::-1][:5]

    delay_factors = []

    for idx in delay_indices:

        score = float(
            delay_shap[i][idx]
        )

        if abs(score) < 0.001:
            continue

        factor = delay_clean_names[idx]

        if score > 0:
            direction = "increases delay risk"
        else:
            direction = "reduces delay risk"

        delay_factors.append(
            f"{factor}: {direction}"
        )

    # ========================================================
    # COST FACTORS
    # ========================================================

    cost_scores = np.abs(
        cost_shap[i]
    )

    cost_indices = np.argsort(
        cost_scores
    )[::-1][:5]

    cost_factors = []

    for idx in cost_indices:

        score = float(
            cost_shap[i][idx]
        )

        if abs(score) < 0.001:
            continue

        factor = cost_clean_names[idx]

        if score > 0:
            direction = "increases cost-overrun risk"
        else:
            direction = "reduces cost-overrun risk"

        cost_factors.append(
            f"{factor}: {direction}"
        )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "project_code":
            project_code,

        "project_name":
            project["project_name"],

        "sector":
            project["sector"],

        "physical_progress":
            project["physical_progress"],

        "expenditure":
            project["expenditure"],

        "delay_probability":
            prediction[
                "delay_probability"
            ],

        "cost_overrun_probability":
            prediction[
                "cost_overrun_probability"
            ],

        "overall_risk_percentage":
            prediction[
                "overall_risk_percentage"
            ],

        "overall_risk_level":
            prediction[
                "overall_risk_level"
            ],

        "top_delay_factors":
            " | ".join(
                delay_factors
            ),

        "top_cost_overrun_factors":
            " | ".join(
                cost_factors
            )
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

risk_factors_df = pd.DataFrame(
    results
)

# ============================================================
# SAVE
# ============================================================

risk_factors_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("RISK FACTOR ANALYSIS RESULT")
print("=" * 75)

print(
    "Projects explained:",
    len(risk_factors_df)
)

if len(risk_factors_df) > 0:

    print("\nSample results:\n")

    print(
        risk_factors_df[
            [
                "project_code",
                "project_name",
                "overall_risk_level",
                "top_delay_factors",
                "top_cost_overrun_factors"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "No projects were matched."
    )

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)

print(
    "RISK FACTOR ANALYSIS COMPLETED SUCCESSFULLY"
)

print("\nSaved file:")

print(
    OUTPUT_PATH
)

print("=" * 75)
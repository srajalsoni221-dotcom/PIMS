import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "ml_delay_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "delay_prediction_model.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "shap_delay_importance.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

target = "delay_flag"

X = df.drop(columns=[target])

print("Dataset loaded.")
print("Shape:", X.shape)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained delay model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# GET PREPROCESSOR AND MODEL
# ============================================================

preprocessor = model.named_steps["preprocessor"]
estimator = model.named_steps["model"]

print("\nPreprocessor found successfully.")
print("ML model found successfully.")
print("ML model type:", type(estimator).__name__)

# ============================================================
# TRANSFORM DATA
# ============================================================

print("\nTransforming data...")

X_transformed = preprocessor.transform(X)

# SHAP currently needs a numeric dense array
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# Force numeric NumPy format
X_transformed = np.asarray(
    X_transformed,
    dtype=np.float64
)

feature_names = preprocessor.get_feature_names_out()

print("Transformed features:", len(feature_names))
print("Transformed data type:", X_transformed.dtype)
print("Transformed data shape:", X_transformed.shape)

# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(estimator)

shap_values = explainer.shap_values(X_transformed)

print("SHAP values calculated successfully.")

# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):

    if len(shap_values) == 2:
        shap_values_class = shap_values[1]
    else:
        shap_values_class = shap_values[0]

elif isinstance(shap_values, np.ndarray):

    if shap_values.ndim == 2:
        shap_values_class = shap_values

    elif shap_values.ndim == 3:
        shap_values_class = shap_values[:, :, 1]

    else:
        raise ValueError(
            f"Unexpected SHAP array shape: {shap_values.shape}"
        )

else:
    raise ValueError("Unexpected SHAP output format.")

# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(shap_values_class).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)

# ============================================================
# DISPLAY TOP 20
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 SHAP FEATURES - DELAY MODEL")
print("=" * 70)

print(
    shap_importance.head(20).to_string(index=False)
)

# ============================================================
# SAVE RESULTS
# ============================================================

shap_importance.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSHAP importance saved to:")
print(OUTPUT_PATH)

# ============================================================
# GRAPH
# ============================================================

top_features = shap_importance.head(15).sort_values(
    by="mean_abs_shap"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Top 15 Features Affecting Delay Prediction")

plt.tight_layout()

plt.show()

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED")
print("=" * 70)
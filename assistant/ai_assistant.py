import streamlit as st
import pandas as pd
import os


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RISK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "project_risk_predictions.csv"
)

FACTOR_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "project_risk_factors.csv"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Project Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    risk_data = pd.read_csv(RISK_FILE)
    factor_data = pd.read_csv(FACTOR_FILE)

    # Convert project codes to string
    risk_data["project_code"] = (
        risk_data["project_code"]
        .astype(str)
        .str.strip()
    )

    factor_data["project_code"] = (
        factor_data["project_code"]
        .astype(str)
        .str.strip()
    )

    return risk_data, factor_data


# =========================================================
# LOAD FILES
# =========================================================

try:

    risk_data, factor_data = load_data()

except Exception as e:

    st.error("Unable to load project data.")

    st.code(str(e))

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Project Monitoring Assistant")

st.write(
    "This assistant explains infrastructure project risk "
    "using actual machine learning predictions and "
    "model-identified risk factors."
)


st.divider()


# =========================================================
# PROJECT SELECTION
# =========================================================

st.subheader("📋 Select Infrastructure Project")

project_codes = risk_data["project_code"].tolist()

selected_code = st.selectbox(
    "Project Code",
    project_codes
)


# =========================================================
# FIND SELECTED PROJECT
# =========================================================

selected_project = risk_data[
    risk_data["project_code"] == str(selected_code)
]

selected_factors = factor_data[
    factor_data["project_code"] == str(selected_code)
]


if selected_project.empty:

    st.error("Selected project was not found.")

    st.stop()


project = selected_project.iloc[0]


# =========================================================
# PROJECT INFORMATION
# =========================================================

project_name = project.get(
    "project_name",
    "Unknown Project"
)

sector = project.get(
    "sector",
    "Unknown"
)

risk_level = project.get(
    "overall_risk_level",
    "Unknown"
)

risk_percentage = project.get(
    "overall_risk_percentage",
    "N/A"
)

delay_prediction = project.get(
    "delay_prediction",
    "N/A"
)

cost_prediction = project.get(
    "cost_overrun_prediction",
    "N/A"
)


# =========================================================
# CONVERT ML PREDICTIONS
# =========================================================

def prediction_text(value):

    try:

        value = int(float(value))

        if value == 1:
            return "Yes"

        elif value == 0:
            return "No"

    except:

        pass

    return str(value)


delay_text = prediction_text(
    delay_prediction
)

cost_text = prediction_text(
    cost_prediction
)


# =========================================================
# PROJECT TITLE
# =========================================================

st.subheader(
    f"📁 {project_name}"
)

st.caption(
    f"Project Code: {selected_code} | Sector: {sector}"
)


# =========================================================
# ML SUMMARY
# =========================================================

st.subheader("📊 ML Prediction Summary")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Overall Risk",
        risk_level
    )


with col2:

    st.metric(
        "Risk Percentage",
        f"{risk_percentage}%"
    )


with col3:

    st.metric(
        "Delay Predicted",
        delay_text
    )


with col4:

    st.metric(
        "Cost Overrun Predicted",
        cost_text
    )


# =========================================================
# RISK FACTORS
# =========================================================

st.divider()

st.subheader(
    "🔎 Why is this Project Risky?"
)


# Get risk factors from project_risk_factors.csv

delay_factors = ""
cost_factors = ""

if not selected_factors.empty:

    factor_row = selected_factors.iloc[0]

    delay_factors = factor_row.get(
        "top_delay_factors",
        ""
    )

    cost_factors = factor_row.get(
        "top_cost_overrun_factors",
        ""
    )


# =========================================================
# DELAY RISK FACTORS
# =========================================================

if (
    pd.notna(delay_factors)
    and str(delay_factors).strip() != ""
):

    st.markdown(
        "### ⏱️ Top Delay Risk Factors"
    )

    delay_list = str(
        delay_factors
    ).split(" | ")

    for i, factor in enumerate(
        delay_list[:5],
        start=1
    ):

        st.info(
            f"**{i}.** {factor}"
        )


# =========================================================
# COST RISK FACTORS
# =========================================================

if (
    pd.notna(cost_factors)
    and str(cost_factors).strip() != ""
):

    st.markdown(
        "### 💰 Top Cost-Overrun Risk Factors"
    )

    cost_list = str(
        cost_factors
    ).split(" | ")

    for i, factor in enumerate(
        cost_list[:5],
        start=1
    ):

        st.info(
            f"**{i}.** {factor}"
        )


# =========================================================
# NO FACTORS AVAILABLE
# =========================================================

if (
    (
        pd.isna(delay_factors)
        or str(delay_factors).strip() == ""
    )
    and
    (
        pd.isna(cost_factors)
        or str(cost_factors).strip() == ""
    )
):

    st.info(
        "No risk-factor information is available "
        "for this project."
    )


# =========================================================
# AI ASSISTANT
# =========================================================

st.divider()

st.subheader(
    "💬 Ask About This Project"
)

st.write(
    "Ask questions about risk, delay, cost overrun "
    "or the major risk factors."
)


question = st.text_input(
    "Enter your question",
    placeholder="Why is this project risky?"
)


# =========================================================
# ASSISTANT RESPONSE
# =========================================================

if question:

    q = question.lower().strip()

    st.markdown(
        "### 🤖 Assistant Response"
    )


    # =====================================================
    # TOP 5 RISK / RISK FACTORS
    # =====================================================

    if (
        "top 5" in q
        or "top five" in q
        or "risk factor" in q
        or "risk factors" in q
        or "major factor" in q
        or "major factors" in q
    ):

        st.write(
            "Based on the model's risk-factor analysis, "
            "the important factors for this project are:"
        )


        if (
            pd.notna(delay_factors)
            and str(delay_factors).strip() != ""
        ):

            st.markdown(
                "**⏱️ Delay-related factors:**"
            )

            delay_list = str(
                delay_factors
            ).split(" | ")

            for factor in delay_list[:5]:

                st.write(
                    f"• {factor}"
                )


        if (
            pd.notna(cost_factors)
            and str(cost_factors).strip() != ""
        ):

            st.markdown(
                "**💰 Cost-overrun factors:**"
            )

            cost_list = str(
                cost_factors
            ).split(" | ")

            for factor in cost_list[:5]:

                st.write(
                    f"• {factor}"
                )


    # =====================================================
    # WHY RISKY
    # =====================================================

    elif (
        "why" in q
        or "risky" in q
        or "risk" in q
    ):

        st.write(
            f"The ML model classifies this project as "
            f"**{risk_level}** with an overall risk "
            f"percentage of **{risk_percentage}%**."
        )

        st.write(
            f"The model predicts delay: **{delay_text}**."
        )

        st.write(
            f"The model predicts cost overrun: "
            f"**{cost_text}**."
        )


        if (
            pd.notna(delay_factors)
            and str(delay_factors).strip() != ""
        ):

            st.markdown(
                "**Important delay-related factors:**"
            )

            delay_list = str(
                delay_factors
            ).split(" | ")

            for factor in delay_list[:5]:

                st.write(
                    f"• {factor}"
                )


        if (
            pd.notna(cost_factors)
            and str(cost_factors).strip() != ""
        ):

            st.markdown(
                "**Important cost-related factors:**"
            )

            cost_list = str(
                cost_factors
            ).split(" | ")

            for factor in cost_list[:5]:

                st.write(
                    f"• {factor}"
                )


    # =====================================================
    # DELAY QUESTIONS
    # =====================================================

    elif (
        "delay" in q
        or "delayed" in q
        or "late" in q
        or "completion" in q
    ):

        if delay_text == "Yes":

            st.warning(
                "⚠️ The ML model predicts that this "
                "project is likely to face a delay."
            )

        elif delay_text == "No":

            st.success(
                "✅ The ML model does not predict a "
                "delay for this project."
            )

        else:

            st.write(
                f"Delay prediction from the ML model: "
                f"**{delay_text}**."
            )


        if (
            pd.notna(delay_factors)
            and str(delay_factors).strip() != ""
        ):

            st.markdown(
                "**Model-identified delay factors:**"
            )

            delay_list = str(
                delay_factors
            ).split(" | ")

            for factor in delay_list[:5]:

                st.write(
                    f"• {factor}"
                )


    # =====================================================
    # COST QUESTIONS
    # =====================================================

    elif (
        "cost" in q
        or "overrun" in q
        or "expensive" in q
        or "budget" in q
    ):

        if cost_text == "Yes":

            st.warning(
                "⚠️ The ML model predicts that this "
                "project may experience a cost overrun."
            )

        elif cost_text == "No":

            st.success(
                "✅ The ML model does not predict a "
                "cost overrun for this project."
            )

        else:

            st.write(
                f"Cost-overrun prediction from the ML model: "
                f"**{cost_text}**."
            )


        if (
            pd.notna(cost_factors)
            and str(cost_factors).strip() != ""
        ):

            st.markdown(
                "**Model-identified cost-overrun factors:**"
            )

            cost_list = str(
                cost_factors
            ).split(" | ")

            for factor in cost_list[:5]:

                st.write(
                    f"• {factor}"
                )


    # =====================================================
    # RISK LEVEL QUESTION
    # =====================================================

    elif (
        "level" in q
        or "status" in q
        or "condition" in q
    ):

        st.write(
            f"The current ML-based overall risk level "
            f"is **{risk_level}**."
        )

        st.write(
            f"Overall risk percentage: "
            f"**{risk_percentage}%**."
        )


    # =====================================================
    # UNKNOWN QUESTION
    # =====================================================

    else:

        st.info(
            "I can currently answer questions about "
            "this project's:\n\n"
            "• Overall risk\n"
            "• Delay prediction\n"
            "• Cost-overrun prediction\n"
            "• Top risk factors\n"
            "• Reasons behind the predicted risk"
        )


# =========================================================
# DATA SOURCE NOTE
# =========================================================

st.divider()

st.caption(
    "⚙️ This assistant uses actual ML predictions and "
    "model-identified risk factors from the project "
    "monitoring system. It does not replace the ML model."
)
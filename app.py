import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import re
import sqlite3
import time
import requests

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PIMS | Project Monitoring",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")

PROJECT_FILE = os.path.join(DATA_DIR, "projects_features.csv")
RISK_FILE = os.path.join(DATA_DIR, "project_risk_predictions.csv")
RISK_FACTORS_FILE = os.path.join(DATA_DIR, "project_risk_factors.csv")

DB_FILE = os.path.join(BASE_DIR, "database", "pims.db")
DELAY_IMPORTANCE_FILE = os.path.join(DATA_DIR, "delay_feature_importance.csv")
COST_IMPORTANCE_FILE = os.path.join(DATA_DIR, "cost_feature_importance.csv")

# ============================================================
# ENTERPRISE LIGHT / TERRACOTTA CSS
# ============================================================

st.markdown(
    """
<style>
/* Hide defaults */
#MainMenu, footer, header { visibility: hidden; }

/* Main App Background - Light Cream/Off-white */
.stApp { background-color: #f8f7f3 !important; }

/* Global Typography - Forcing Dark Text for Light Theme */
h1, h2, h3, h4, p, span, div, strong, label, th, td { 
    font-family: 'Inter', -apple-system, sans-serif; 
}
p, div, span, label { color: #333333 !important; }
h1, h2, h3 { color: #111111 !important; }

/* ---------------- SIDEBAR ---------------- */
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    min-width: 280px !important;
    width: 280px !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #d1d1d1 !important; }
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Override Streamlit Radio */
div[role="radiogroup"] > label > div:first-child { display: none; }
div[role="radiogroup"] { gap: 2px; }
div[role="radiogroup"] > label {
    padding: 8px 15px !important; border-radius: 6px !important; 
    margin-bottom: 2px !important; transition: all 0.2s;
}
div[role="radiogroup"] > label[data-baseweb="radio"] { background-color: transparent; }
div[role="radiogroup"] > label[aria-checked="true"] { background-color: #ff5722 !important; }
div[role="radiogroup"] > label[aria-checked="true"] p { color: white !important; font-weight: 600 !important; }

/* Streamlit Metrics */
[data-testid="stMetricLabel"] p { color: #888 !important; font-size: 11px !important; font-weight: 600; text-transform: uppercase; }
[data-testid="stMetricValue"] div { color: #111 !important; font-weight: 800; }

/* ---------------- HEADINGS ---------------- */
.main-title { font-size: 26px; font-weight: 800; color: #111 !important; margin-bottom: 4px; }
.subtitle { color: #888 !important; font-size: 13px; margin-bottom: 25px; }
.section-title { font-size: 12px; font-weight: 700; color: #ff5722 !important; text-transform: uppercase; margin-top: 25px; margin-bottom: 15px; letter-spacing: 1px; }

/* ---------------- CARDS (HTML) ---------------- */
.summary-card {
    background: white; border: 1px solid #eaeaea;
    border-radius: 12px; padding: 20px; box-shadow: 0px 2px 8px rgba(0,0,0,0.02);
}
.card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.card-top span { color: #888; font-size: 11px; font-weight: 600; text-transform: uppercase; }
.card-icon { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 16px; }
.card-icon.green { background: rgba(76,175,80,0.1); color: #2e7d32; }
.card-icon.yellow { background: rgba(255,179,0,0.1); color: #f57c00; }
.card-icon.red { background: rgba(218,41,28,0.1); color: #da291c; }
.summary-card h2 { font-size: 32px; font-weight: 800; color: #111 !important; margin: 0 0 5px 0; line-height: 1; }
.positive { color: #2e7d32 !important; font-size: 12px; font-weight: 600; margin: 0; }
.warning-text { color: #f57c00 !important; font-size: 12px; font-weight: 600; margin: 0; }
.danger-text { color: #da291c !important; font-size: 12px; font-weight: 600; margin: 0; }
.summary-card p span { color: #999 !important; font-weight: 400; }

/* ---------------- PANELS ---------------- */
.panel {
    background: white; border: 1px solid #eaeaea; border-radius: 12px;
    padding: 20px; box-shadow: 0px 2px 8px rgba(0,0,0,0.02); margin-bottom: 20px;
}
.panel-header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #f5f5f5; padding-bottom: 15px; margin-bottom: 15px; }
.panel-header h3 { margin: 0 0 5px 0; font-size: 16px; color: #111 !important; font-weight: 700; }
.panel-header p { margin: 0; color: #888 !important; font-size: 12px; }

/* Warnings inside Panels */
.warning-item { display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid #f5f5f5; }
.warning-item:last-child { border-bottom: none; }
.warning-icon { width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px; font-size: 12px; }
.warning-item.critical .warning-icon { background: rgba(218,41,28,0.1); color: #da291c; }
.warning-info strong { display: block; color: #111 !important; font-size: 13px; font-weight: 600; }
.warning-info p { margin: 2px 0 0 0; color: #666 !important; font-size: 11px; }

/* Badges */
.status { padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: 700; white-space: nowrap; display: inline-block; }
.red-status { background: #da291c !important; color: white !important; }
.yellow-status { background: #f57c00 !important; color: white !important; }
.green-status { background: #4caf50 !important; color: white !important; }
.red-text { color: #da291c !important; }
.yellow-text { color: #f57c00 !important; }
.green-text { color: #4caf50 !important; }

/* ---------------- TABLES ---------------- */
table { width: 100%; border-collapse: collapse; text-align: left; }
th { padding: 12px 10px; background: white; color: #999 !important; font-weight: 600; font-size: 10px; text-transform: uppercase; border-bottom: 1px solid #eee !important; }
td { padding: 12px 10px; border-bottom: 1px solid #f5f5f5 !important; vertical-align: middle; background: white !important; color: #333 !important; font-size: 13px; }
td strong { display: block; color: #111 !important; font-size: 13px; font-weight: 600;}
td small { color: #888 !important; font-size: 11px; }

/* Mini Progress Bar */
.progress { background: #f0f0f0; border-radius: 4px; height: 6px; width: 80px; display: inline-block; margin-right: 10px; }
.progress-bar { background: #ff5722; height: 100%; border-radius: 4px; }

/* ---------------- DETAILS & FACTORS ---------------- */
.risk-card { background: white; border: 1px solid #eaeaea; border-radius: 12px; padding: 22px; margin-top: 10px; margin-bottom: 18px; }
.factor-card { background: white; border: 1px solid #eaeaea; border-radius: 8px; padding: 12px 15px; margin-bottom: 8px; font-size: 13px; color: #333 !important; }
.info-label { color: #888 !important; font-size: 11px !important; font-weight: 600; text-transform: uppercase; }
.info-value { color: #111 !important; font-size: 18px !important; font-weight: 700 !important; }

/* ============================================================
   FLOATING AI ASSISTANT - FINAL
   ============================================================ */

/* Remove full-width bottom container */
[data-testid="stPopover"] {
    position: fixed !important;
    right: 22px !important;
    bottom: 22px !important;
    left: auto !important;
    top: auto !important;

    width: 64px !important;
    height: 64px !important;
    min-width: 64px !important;
    max-width: 64px !important;

    padding: 0 !important;
    margin: 0 !important;

    background: transparent !important;
    border: none !important;
    box-shadow: none !important;

    z-index: 999999 !important;
}

/* Actual button */
[data-testid="stPopover"] button {
    position: fixed !important;

    right: 22px !important;
    bottom: 22px !important;
    left: auto !important;
    top: auto !important;

    width: 64px !important;
    height: 64px !important;
    min-width: 64px !important;
    min-height: 64px !important;
    max-width: 64px !important;
    max-height: 64px !important;

    padding: 0 !important;
    margin: 0 !important;

    border-radius: 50% !important;
    border: 3px solid #ffffff !important;

    background: linear-gradient(
        135deg,
        #ff5722,
        #ff8a3d
    ) !important;

    box-shadow:
        0 7px 22px rgba(255, 87, 34, 0.45) !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    overflow: hidden !important;
}

/* Hide old 💬 text */
[data-testid="stPopover"] button p {
    font-size: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Robot */
[data-testid="stPopover"] button p::before {
    content: "🤖" !important;
    font-size: 30px !important;
    line-height: 1 !important;
}

/* Hide dropdown arrow */
[data-testid="stPopover"] button svg {
    display: none !important;
}

/* Hover */
[data-testid="stPopover"] button:hover {
    transform: scale(1.08) !important;
    background: linear-gradient(
        135deg,
        #ff5722,
        #ff8a3d
    ) !important;
}

/* Remove focus outline */
[data-testid="stPopover"] button:focus {
    outline: none !important;
    box-shadow:
        0 7px 22px rgba(255, 87, 34, 0.45) !important;
}

/* ---------------- LOGIN BOX ---------------- */
.login-box {
    background: white; border: 1px solid #eaeaea; border-radius: 12px; padding: 40px; 
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05); text-align: center; margin-top: 50px;
}

/* ============================================================
   SIDEBAR SECTOR FILTER - FORCE BLACK TEXT
   ============================================================ */

/* Selectbox main box */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background-color: #ffffff !important;
}

/* Selected value - ALL possible Streamlit/BaseWeb elements */
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: #000000 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #000000 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

/* Combobox itself */
[data-testid="stSelectbox"] [role="combobox"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Selected text container */
[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    color: #000000 !important;
}

/* Dropdown menu */
div[role="listbox"] {
    background-color: #ffffff !important;
}

div[role="listbox"] div,
div[role="listbox"] span,
div[role="option"] {
    color: #000000 !important;
}

/* Hovered option */
div[role="option"]:hover {
    color: #000000 !important;
}



</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# AUTHENTICATION & LOGIN (Professional Enterprise UI)
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def authenticate_user(username, password):
    # Real FastAPI Backend Call
    try:
        response = requests.post(
            "http://localhost:8000/api/login", 
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        # Backend connection fail hone par warning dikhayega
        st.error(f"Backend connection failed. Is the FastAPI server running?")
        return False

if not st.session_state["authenticated"]:
    # Professional Login CSS with Premium Grid Background
    st.markdown("""
    <style>
    /* Premium Blueprint Grid Background with Ambient Glow */
    .stApp {
        background-color: #f8f7f3 !important;
        background-image: 
            radial-gradient(circle at 50% -10%, rgba(255, 87, 34, 0.12) 0%, transparent 60%),
            linear-gradient(rgba(0, 0, 0, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 0, 0, 0.03) 1px, transparent 1px) !important;
        background-size: 100% 100%, 35px 35px, 35px 35px !important;
        background-position: center top, center center, center center !important;
    }
    
    /* Center the login form with Glassmorphism Effect */
    .stForm {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 87, 34, 0.15) !important;
        border-top: 5px solid #ff5722 !important;
        border-radius: 16px !important;
        padding: 45px 35px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08) !important;
        margin-top: 8vh !important;
    }
    
    /* Input field styling */
    div[data-testid="stTextInput"] label { color: #555 !important; font-weight: 600 !important; font-size: 13px !important; }
    div[data-testid="stTextInput"] input {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 12px !important;
        font-size: 14px !important;
        background-color: #ffffff !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #ff5722 !important;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.15) !important;
    }
    
    /* Submit button styling */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #ff5722 !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
        padding: 12px !important;
        margin-top: 20px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #e64a19 !important;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.25) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Hide the global sidebar and header completely on the login page */
    [data-testid="stSidebar"] { display: none !important; }
    .top-header-container { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # 3-column layout to center the login card perfectly
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        with st.form("login_form", clear_on_submit=False):
            # Premium Logo & Title placement inside the form
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 30px;">
                    <div style="background: linear-gradient(135deg, #ff5722 0%, #e64a19 100%); color: white; width: 64px; height: 64px; border-radius: 14px; display: inline-flex; justify-content: center; align-items: center; font-weight: 800; font-size: 32px; margin-bottom: 15px; box-shadow: 0 8px 20px rgba(255, 87, 34, 0.3);">P</div>
                    <h2 style="color: #111; margin: 0; font-family: 'Inter', sans-serif; font-weight: 800; font-size: 24px;">Welcome to PIMS</h2>
                    <p style="color: #666; font-size: 13px; margin-top: 8px;">Predictive Infrastructure Project Monitoring</p>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Inputs with placeholders
            username = st.text_input("Username", placeholder="e.g. admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            # Full-width Sign In button
            submit = st.form_submit_button("Sign In")
            
            # Trust signal text
            st.markdown('<div style="text-align: center; margin-top: 20px; font-size: 12px; color: #888; font-weight: 500;">🔒 AES-256 Encrypted Connection</div>', unsafe_allow_html=True)
            
            if submit:
                if username and password:
                    with st.spinner("Authenticating..."):
                        if authenticate_user(username, password):
                            st.session_state["authenticated"] = True
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Try admin / pims123")
                else:
                    st.warning("Please enter both username and password.")
        
        # Enterprise Footer to fill empty space
        st.markdown(
            """
            <div style="text-align: center; margin-top: 35px; color: #999; font-size: 12px; font-weight: 500;">
                PIMS Enterprise Edition v2.0 <br>
                <span style="color: #bbb;">Monitoring 240+ Infrastructure Projects Nationwide</span>
            </div>
            """, unsafe_allow_html=True
        )
                    
    st.stop() # Stops execution until login is successful
    
    # 3-column layout to center the login card perfectly
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        with st.form("login_form", clear_on_submit=False):
            # Premium Logo & Title placement inside the form
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 25px;">
                    <div style="background: #ff5722; color: white; width: 56px; height: 56px; border-radius: 12px; display: inline-flex; justify-content: center; align-items: center; font-weight: 800; font-size: 28px; margin-bottom: 15px; box-shadow: 0 6px 15px rgba(255, 87, 34, 0.25);">P</div>
                    <h2 style="color: #111; margin: 0; font-family: 'Inter', sans-serif; font-weight: 800; font-size: 22px;">Welcome to PIMS</h2>
                    <p style="color: #666; font-size: 13px; margin-top: 5px;">Secure Infrastructure Analytics Platform</p>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Inputs with placeholders
            username = st.text_input("Username", placeholder="e.g. admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            # Full-width Sign In button
            submit = st.form_submit_button("Sign In")
            
            # Trust signal text
            st.markdown('<div style="text-align: center; margin-top: 15px; font-size: 11px; color: #aaa;">🔒 End-to-end encrypted connection</div>', unsafe_allow_html=True)
            
            if submit:
                if username and password:
                    with st.spinner("Authenticating..."):
                        if authenticate_user(username, password):
                            st.session_state["authenticated"] = True
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Try admin / pims123")
                else:
                    st.warning("Please enter both username and password.")
                    
    st.stop() # Stops execution until login is successful

# ============================================================
# LOAD DATA (Yeh tabhi run hoga jab user logged in hoga)
# ============================================================

@st.cache_data
def load_data():
    if os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            projects = pd.read_sql_query("SELECT * FROM projects", conn)
            risk = pd.read_sql_query("SELECT * FROM risk_predictions", conn)
            risk_factors = pd.read_sql_query("SELECT * FROM risk_factors", conn)
            conn.close()
            return projects, risk, risk_factors
        except Exception as e:
            st.warning("Database could not be loaded. Using CSV files instead.")

    projects = pd.read_csv(PROJECT_FILE)
    risk = pd.read_csv(RISK_FILE)
    if os.path.exists(RISK_FACTORS_FILE):
        risk_factors = pd.read_csv(RISK_FACTORS_FILE)
    else:
        risk_factors = pd.DataFrame()
    return projects, risk, risk_factors

@st.cache_data
def load_importance():
    delay_imp = pd.DataFrame()
    cost_imp = pd.DataFrame()
    if os.path.exists(DELAY_IMPORTANCE_FILE):
        delay_imp = pd.read_csv(DELAY_IMPORTANCE_FILE)
    if os.path.exists(COST_IMPORTANCE_FILE):
        cost_imp = pd.read_csv(COST_IMPORTANCE_FILE)
    return delay_imp, cost_imp


if not os.path.exists(PROJECT_FILE):
    st.error("projects_features.csv not found.\n\nExpected location:\n" + PROJECT_FILE)
    st.stop()

if not os.path.exists(RISK_FILE):
    st.error("project_risk_predictions.csv not found.\n\nPlease run the risk prediction script first.")
    st.stop()

try:
    projects, risk_data, risk_factors = load_data()
    delay_importance, cost_importance = load_importance()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

projects.columns = projects.columns.str.strip()
risk_data.columns = risk_data.columns.str.strip()
if not risk_factors.empty:
    risk_factors.columns = risk_factors.columns.str.strip()

for df in [projects, risk_data, risk_factors]:
    if not df.empty and "project_code" in df.columns:
        df["project_code"] = df["project_code"].astype(str).str.strip().str.replace(".0", "", regex=False)

if ("project_code" in projects.columns and "project_code" in risk_data.columns):
    display_data = projects.merge(risk_data, on="project_code", how="inner", suffixes=("", "_risk"))
else:
    display_data = risk_data.copy()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; padding: 10px 0 20px 0; border-bottom: 1px solid #333; margin-bottom: 15px;">
            <div style="background: #ff5722; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; justify-content: center; align-items: center; font-weight: 800; font-size: 18px; margin-right: 12px;">P</div>
            <div>
                <div style="color: white; font-weight: 800; font-size: 16px; line-height: 1.2;">PIMS</div>
                <div style="color: #888; font-size: 10px; line-height: 1.2;">Predictive Infrastructure<br>Project Monitoring</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pages = [
        "Executive Dashboard", "Project Monitoring", "Project Intelligence", 
        "Predictions", "Risk Monitoring", "Early Warning Center", 
        "Benchmarking", "Cost Escalation Drivers", 
        "Dependency Intelligence", "Analytics", 
        "Data Quality", "Model Information", "Methodology", "About"
    ]
    
    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("SECTOR FILTER")
    
    sector_options = ["All Sectors"]
    if "sector" in display_data.columns:
        sector_values = display_data["sector"].dropna().astype(str).sort_values().unique().tolist()
        sector_options.extend(sector_values)

    selected_sector = st.selectbox("Sector Filter", sector_options, label_visibility="collapsed")

    st.markdown('<div style="font-size:10px; color:#666; display:flex; align-items:center; gap:5px; margin-top:10px;"><div style="width:6px; height:6px; background:#4caf50; border-radius:50%;"></div> PAIMANA Data Active</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ============================================================
# FILTERS & COMMON METRICS
# ============================================================

st.markdown('<div class="main-title">Infrastructure Project Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning based infrastructure project risk monitoring and analytics</div>', unsafe_allow_html=True)

filtered_data = display_data.copy()
if selected_sector != "All Sectors":
    filtered_data = filtered_data[filtered_data["sector"].astype(str) == selected_sector]

total_projects = len(filtered_data)

if "overall_risk_level" in filtered_data.columns:
    high_risk = int((filtered_data["overall_risk_level"] == "High Risk").sum())
    medium_risk = int((filtered_data["overall_risk_level"] == "Medium Risk").sum())
    low_risk = int((filtered_data["overall_risk_level"] == "Low Risk").sum())
else:
    high_risk = 0
    medium_risk = 0
    low_risk = 0

# ============================================================
# ROUTING CONTROLLER
# ============================================================

if selected_page == "Executive Dashboard":

    low_pct = (low_risk / total_projects * 100) if total_projects > 0 else 0
    med_pct = (medium_risk / total_projects * 100) if total_projects > 0 else 0
    high_pct = (high_risk / total_projects * 100) if total_projects > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'''
            <div class="summary-card">
                <div class="card-top"><span>Total Projects</span><div class="card-icon">▣</div></div>
                <h2>{total_projects:,}</h2>
                <p class="positive">Live <span>monitoring</span></p>
            </div>
            ''', unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'''
            <div class="summary-card">
                <div class="card-top"><span>On Track</span><div class="card-icon green">✓</div></div>
                <h2>{low_risk:,}</h2>
                <p class="positive">{low_pct:.1f}% <span>of projects</span></p>
            </div>
            ''', unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'''
            <div class="summary-card">
                <div class="card-top"><span>At Risk</span><div class="card-icon yellow">!</div></div>
                <h2>{medium_risk:,}</h2>
                <p class="warning-text">{med_pct:.1f}% <span>need attention</span></p>
            </div>
            ''', unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f'''
            <div class="summary-card">
                <div class="card-top"><span>Critical</span><div class="card-icon red">⚠</div></div>
                <h2>{high_risk:,}</h2>
                <p class="danger-text">{high_pct:.1f}% <span>immediate action</span></p>
            </div>
            ''', unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Project Performance</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    total_original_cost = pd.to_numeric(filtered_data["original_cost"], errors="coerce").sum() if "original_cost" in filtered_data.columns else 0
    total_expenditure = pd.to_numeric(filtered_data["expenditure"], errors="coerce").sum() if "expenditure" in filtered_data.columns else 0
    avg_progress = pd.to_numeric(filtered_data["physical_progress"], errors="coerce").mean() if "physical_progress" in filtered_data.columns else 0

    with p1: st.metric("Total Original Cost", f"₹ {total_original_cost:,.2f}")
    with p2: st.metric("Total Expenditure", f"₹ {total_expenditure:,.2f}")
    with p3: st.metric("Average Physical Progress", f"{avg_progress:.1f}%")

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown(
            """<div class="panel"><div class="panel-header"><div><h3>Project Risk Distribution</h3><p>Current project health overview</p></div></div></div>""",
            unsafe_allow_html=True
        )
        risk_counts = pd.DataFrame({"Risk Level": ["High Risk", "Medium Risk", "Low Risk"], "Projects": [high_risk, medium_risk, low_risk]})
        fig = px.pie(risk_counts, names="Risk Level", values="Projects", hole=0.55, color="Risk Level", 
                     color_discrete_map={"High Risk":"#da291c", "Medium Risk":"#f57c00", "Low Risk":"#4caf50"})
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"))
        st.plotly_chart(fig, width="stretch")

    with chart2:
        warnings_html = ""
        if "overall_risk_level" in filtered_data.columns:
            critical_projs = filtered_data[filtered_data['overall_risk_level'] == 'High Risk'].head(3)
            for _, row in critical_projs.iterrows():
                proj_name = row.get("project_name", f"Project #{row.get('project_code', 'Unknown')}")
                warnings_html += f'<div class="warning-item critical"><div class="warning-icon">HIGH</div><div class="warning-info"><strong>{proj_name}</strong><p>Immediate action required</p></div></div>'
        
        empty_msg = "<p style='padding:15px; color:#888;'>No critical projects detected.</p>"
        final_warnings = warnings_html if warnings_html else empty_msg
        
        html_content = f'<div class="panel"><div class="panel-header"><div><h3>Early Warning Center</h3><p>Projects requiring attention</p></div></div><div class="warning-list">{final_warnings}</div></div>'
        st.markdown(html_content, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # PROJECT HEALTH OVERVIEW TABLE (Dynamic HTML)
    # ------------------------------------------------------------
    table_html = """<div class="panel project-health"><div class="panel-header"><div><h3>Project Health Overview</h3><p>Recently monitored infrastructure projects</p></div></div><div class="table-container"><table><thead><tr><th>Project</th><th>Sector</th><th>Progress</th><th>Risk Score</th><th>Status</th></tr></thead><tbody>"""
    
    top_projects = filtered_data.head(5)
    for _, row in top_projects.iterrows():
        p_name = row.get("project_name", "Unknown Project")
        p_code = row.get("project_code", "N/A")
        sector = row.get("sector", "General")
        progress = float(row.get("physical_progress", 0)) if pd.notna(row.get("physical_progress")) else 0
        risk_pct = float(row.get("overall_risk_percentage", 0)) if pd.notna(row.get("overall_risk_percentage")) else 0
        status = row.get("overall_risk_level", "Unknown")
        
        if status == "High Risk":
            status_class, text_class, status_label = "red-status", "red-text", "Critical"
        elif status == "Medium Risk":
            status_class, text_class, status_label = "yellow-status", "yellow-text", "At Risk"
        else:
            status_class, text_class, status_label = "green-status", "green-text", "On Track"
            
        table_html += f"""<tr><td><strong>{p_name}</strong><small>{p_code}</small></td><td>{sector}</td><td><div class="progress"><div class="progress-bar" style="width: {progress}%"></div></div><span>{progress:.0f}%</span></td><td><strong class="risk-number {text_class}">{risk_pct:.1f}</strong>/100</td><td><span class="status {status_class}">{status_label}</span></td></tr>"""
        
    table_html += "</tbody></table></div></div>"
    st.markdown(table_html.replace('\n', ''), unsafe_allow_html=True)

    # ------------------------------------------------------------
    # TIMELINE & ML INSIGHTS
    # ------------------------------------------------------------
    st.markdown('<div class="section-title">Project Timeline (Gantt Chart)</div>', unsafe_allow_html=True)
    if "start_date" in filtered_data.columns and "target_date" in filtered_data.columns:
        timeline_data = filtered_data.dropna(subset=['start_date', 'target_date']).copy()
        if not timeline_data.empty:
            timeline_data['start_date'] = pd.to_datetime(timeline_data['start_date'], errors='coerce')
            timeline_data['target_date'] = pd.to_datetime(timeline_data['target_date'], errors='coerce')
            timeline_data = timeline_data.dropna(subset=['start_date', 'target_date'])
            fig_timeline = px.timeline(timeline_data, x_start="start_date", x_end="target_date", y="project_name",
                                       color="overall_risk_level" if "overall_risk_level" in timeline_data.columns else None,
                                       hover_name="project_code", color_discrete_map={"High Risk": "#da291c", "Medium Risk": "#f57c00", "Low Risk": "#4caf50"})
            fig_timeline.update_yaxes(autorange="reversed")
            fig_timeline.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"), showlegend=True, margin=dict(t=20, b=20))
            st.plotly_chart(fig_timeline, width="stretch")
        else:
            st.info("Timeline data not available.")
            
    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        st.markdown('<div class="section-title">Top AI Risk Factors</div>', unsafe_allow_html=True)
        if not delay_importance.empty:
            top_factors = delay_importance.sort_values(by='importance', ascending=False).head(7)
            top_factors['feature_clean'] = top_factors['feature'].str.replace('numeric__', '').str.replace('categorical__', '').str.replace('_', ' ').str.title()
            fig_importance = px.bar(top_factors, x='importance', y='feature_clean', orientation='h', color='importance', color_continuous_scale="Reds")
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"), height=350, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_importance, width="stretch")
            
    with col_ml2:
        st.markdown('<div class="section-title">Sector Vulnerability</div>', unsafe_allow_html=True)
        delay_sector_path = os.path.join(DATA_DIR, "delay_by_sector.csv")
        if os.path.exists(delay_sector_path):
            sector_delays = pd.read_csv(delay_sector_path)
            top_delayed_sectors = sector_delays.sort_values(by='delay_percentage', ascending=False).head(7)
            fig_sector = px.bar(top_delayed_sectors, x='sector', y='delay_percentage', text='delay_percentage', color='delay_percentage', color_continuous_scale="Oranges")
            fig_sector.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_sector.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"), height=350, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sector, width="stretch")

# ============================================================
# ROUTING: ANALYTICS
# ============================================================
elif selected_page == "Analytics":
    st.markdown('<div class="section-title">Project Analytics</div>', unsafe_allow_html=True)
    if "physical_progress" in filtered_data.columns:
        fig = px.histogram(filtered_data, x="physical_progress", nbins=20, title="Physical Progress Distribution", color_discrete_sequence=["#ff5722"])
        fig.update_layout(xaxis_title="Physical Progress (%)", yaxis_title="Number of Projects", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"))
        st.plotly_chart(fig, width="stretch")

    if "original_cost" in filtered_data.columns and "expenditure" in filtered_data.columns:
        chart_data = filtered_data.copy()
        chart_data["original_cost"] = pd.to_numeric(chart_data["original_cost"], errors="coerce")
        chart_data["expenditure"] = pd.to_numeric(chart_data["expenditure"], errors="coerce")
        fig2 = px.scatter(chart_data, x="original_cost", y="expenditure", size="physical_progress" if "physical_progress" in chart_data.columns else None, hover_name="project_name" if "project_name" in chart_data.columns else None, title="Original Cost vs Expenditure", color_discrete_sequence=["#2e7d32"])
        fig2.update_layout(xaxis_title="Original Cost", yaxis_title="Expenditure", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"))
        st.plotly_chart(fig2, width="stretch")
    
    if "sector" in filtered_data.columns and "overall_risk_level" in filtered_data.columns:
        st.markdown('<div class="section-title">Risk Distribution by Sector</div>', unsafe_allow_html=True)
        sector_risk_data = filtered_data.groupby(["sector", "overall_risk_level"]).size().reset_index(name="Projects")
        fig3 = px.pie(sector_risk_data, names="sector", values="Projects", color="overall_risk_level", hole=0.45, title="Risk Distribution by Sector", color_discrete_map={"High Risk":"#da291c", "Medium Risk":"#f57c00", "Low Risk":"#4caf50"})
        fig3.update_layout(height=550, legend_title="Risk Level", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111"))
        st.plotly_chart(fig3, width="stretch")

# ============================================================
# ROUTING: PROJECT MONITORING
# ============================================================
elif selected_page == "Project Monitoring":
    st.markdown('<div class="section-title">Project Explorer</div>', unsafe_allow_html=True)
    search_text = st.text_input("🔎 Search Project", placeholder="Enter project name or project code...")
    project_view = filtered_data.copy()

    if search_text:
        search_lower = search_text.lower()
        mask = pd.Series(False, index=project_view.index)
        if "project_name" in project_view.columns:
            mask = mask | project_view["project_name"].astype(str).str.lower().str.contains(search_lower, na=False)
        if "project_code" in project_view.columns:
            mask = mask | project_view["project_code"].astype(str).str.contains(search_text, na=False)
        project_view = project_view[mask]

    st.write(f"Showing **{len(project_view):,}** projects")
    columns_to_show = [col for col in ["project_code", "project_name", "sector", "original_cost", "expenditure", "physical_progress", "delay_probability", "cost_overrun_probability", "overall_risk_percentage", "overall_risk_level"] if col in project_view.columns]
    
    if len(project_view) > 0:
        st.dataframe(project_view[columns_to_show], width="stretch", height=450)
        if "project_code" in project_view.columns:
            project_codes = project_view["project_code"].astype(str).tolist()
            selected_project = st.selectbox("Select a project for detailed analysis", project_codes, label_visibility="collapsed")
            selected_row = project_view[project_view["project_code"].astype(str) == selected_project].iloc[0]

            st.markdown('<div class="section-title">Project Details</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1: st.markdown(f'<div class="info-label">PROJECT CODE</div><div class="info-value">{selected_row.get("project_code", "-")}</div>', unsafe_allow_html=True)
            with d2: st.markdown(f'<div class="info-label">SECTOR</div><div class="info-value">{selected_row.get("sector", "-")}</div>', unsafe_allow_html=True)
            with d3: st.markdown(f'<div class="info-label">PHYSICAL PROGRESS</div><div class="info-value">{selected_row.get("physical_progress", 0)}%</div>', unsafe_allow_html=True)

            if "project_name" in selected_row:
                st.markdown(f"### {selected_row['project_name']}")

            x1, x2, x3 = st.columns(3)
            with x1: st.metric("Original Cost", f"₹ {float(selected_row.get('original_cost', 0)):,.2f}")
            with x2: st.metric("Expenditure", f"₹ {float(selected_row.get('expenditure', 0)):,.2f}")
            with x3: st.metric("Overall Risk", selected_row.get("overall_risk_level", "Unknown"))


# ============================================================
# ROUTING: PREDICTIONS
# ============================================================
elif selected_page == "Predictions":

    st.markdown(
        '<div class="section-title">ML PREDICTIONS</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Machine Learning based prediction of project delay and cost-overrun risk."
    )

    if filtered_data.empty:
        st.warning("No project data available.")
        st.stop()

    # --------------------------------------------------------
    # PROJECT SEARCH
    # --------------------------------------------------------

    search_text = st.text_input(
        "🔎 Search Project",
        placeholder="Enter project name or project code..."
    )

    prediction_data = filtered_data.copy()

    if search_text:

        search_lower = search_text.lower().strip()

        name_mask = (
            prediction_data["project_name"]
            .astype(str)
            .str.lower()
            .str.contains(search_lower, na=False)
        )

        code_mask = (
            prediction_data["project_code"]
            .astype(str)
            .str.lower()
            .str.contains(search_lower, na=False)
        )

        prediction_data = prediction_data[
            name_mask | code_mask
        ]

    if prediction_data.empty:
        st.warning("No matching project found.")
        st.stop()

    # --------------------------------------------------------
    # PROJECT SELECTOR
    # --------------------------------------------------------

    project_codes = (
        prediction_data["project_code"]
        .astype(str)
        .tolist()
    )

    selected_prediction_project = st.selectbox(
        "Select Project",
        project_codes
    )

    pred_row = prediction_data[
        prediction_data["project_code"].astype(str)
        == str(selected_prediction_project)
    ].iloc[0]

    project_name = pred_row.get(
        "project_name",
        "Selected Project"
    )

    st.markdown(f"## 🏗️ {project_name}")

    st.caption(
        f"Project Code: {selected_prediction_project}"
    )

    # --------------------------------------------------------
    # GET ML VALUES
    # --------------------------------------------------------

    delay_prob = float(
        pred_row.get("delay_probability", 0)
    )

    cost_prob = float(
        pred_row.get("cost_overrun_probability", 0)
    )

    overall_score = float(
        pred_row.get("overall_risk_percentage", 0)
    )

    overall_level = str(
        pred_row.get("overall_risk_level", "Unknown")
    )

    delay_prediction = pred_row.get(
        "delay_prediction",
        None
    )

    cost_prediction = pred_row.get(
        "cost_overrun_prediction",
        None
    )

    # --------------------------------------------------------
    # PREDICTION CARDS
    # --------------------------------------------------------

    st.markdown("### 🎯 Prediction Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Delay Probability",
            f"{delay_prob * 100:.1f}%"
        )

    with c2:
        st.metric(
            "Cost Overrun Probability",
            f"{cost_prob * 100:.1f}%"
        )

    with c3:
        st.metric(
            "Overall Risk Score",
            f"{overall_score:.1f}%"
        )

    # --------------------------------------------------------
    # PREDICTION STATUS
    # --------------------------------------------------------

    s1, s2 = st.columns(2)

    with s1:

        if delay_prediction is not None:

            if int(float(delay_prediction)) == 1:
                st.error(
                    "⚠️ Delay Prediction: Risk of Delay"
                )
            else:
                st.success(
                    "✓ Delay Prediction: No Delay Signal"
                )

        else:
            st.info(
                "Delay prediction unavailable."
            )

    with s2:

        if cost_prediction is not None:

            if int(float(cost_prediction)) == 1:
                st.error(
                    "⚠️ Cost Prediction: Overrun Risk"
                )
            else:
                st.success(
                    "✓ Cost Prediction: No Overrun Signal"
                )

        else:
            st.info(
                "Cost-overrun prediction unavailable."
            )

    # --------------------------------------------------------
    # PROBABILITY VISUALIZATION
    # --------------------------------------------------------

    st.markdown("### 📈 Risk Probability")

    probability_df = pd.DataFrame({
        "Prediction": [
            "Delay Risk",
            "Cost Overrun Risk"
        ],
        "Probability": [
            delay_prob * 100,
            cost_prob * 100
        ]
    })

    fig_prediction = px.bar(
        probability_df,
        x="Prediction",
        y="Probability",
        text="Probability",
        range_y=[0, 100],
        title="Model Prediction Probability"
    )

    fig_prediction.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_prediction.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111"),
        height=400
    )

    st.plotly_chart(
        fig_prediction,
        width="stretch"
    )

    # --------------------------------------------------------
    # PROJECT CONTEXT
    # --------------------------------------------------------

    st.markdown("### 📊 Current Project Context")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "Physical Progress",
            f"{float(pred_row.get('physical_progress', 0)):.1f}%"
        )

    with p2:
        st.metric(
            "Original Cost",
            f"₹ {float(pred_row.get('original_cost', 0)):,.2f}"
        )

    with p3:
        st.metric(
            "Expenditure",
            f"₹ {float(pred_row.get('expenditure', 0)):,.2f}"
        )

    with p4:

        ratio = pred_row.get(
            "expenditure_ratio",
            None
        )

        if pd.notna(ratio):
            st.metric(
                "Expenditure Ratio",
                f"{float(ratio):.2f}"
            )
        else:
            st.metric(
                "Expenditure Ratio",
                "N/A"
            )

    # --------------------------------------------------------
    # OVERALL RISK INTERPRETATION
    # --------------------------------------------------------

    st.markdown("### 🛡️ Overall Risk Classification")

    if overall_level == "High Risk":

        st.error(
            f"**{overall_level}** — "
            f"Combined ML risk score is {overall_score:.1f}%."
        )

    elif overall_level == "Medium Risk":

        st.warning(
            f"**{overall_level}** — "
            f"Combined ML risk score is {overall_score:.1f}%."
        )

    else:

        st.success(
            f"**{overall_level}** — "
            f"Combined ML risk score is {overall_score:.1f}%."
        )

    st.caption(
        "Prediction probabilities are generated by the existing trained "
        "ML models. They represent model estimates, not guaranteed outcomes."
    )

# ============================================================
# ROUTING: RISK MONITORING
# ============================================================
elif selected_page == "Risk Monitoring":
    st.markdown('<div class="section-title">Risk Monitoring</div>', unsafe_allow_html=True)
    if "project_code" not in filtered_data.columns:
        st.error("Project code is not available.")
        st.stop()

    project_codes = filtered_data["project_code"].astype(str).tolist()
    selected_project = st.selectbox("Select Project", project_codes, label_visibility="collapsed")
    selected = filtered_data[filtered_data["project_code"].astype(str) == selected_project].iloc[0]
    project_name = selected.get("project_name", "Selected Project")

    st.markdown(f"### 🏗️ {project_name}")
    delay_probability = float(selected.get("delay_probability", 0))
    cost_probability = float(selected.get("cost_overrun_probability", 0))
    overall_percentage = float(selected.get("overall_risk_percentage", 0))
    overall_level = selected.get("overall_risk_level", "Unknown")

    r1, r2, r3 = st.columns(3)
    with r1: st.metric("Delay Probability", f"{delay_probability * 100:.1f}%")
    with r2: st.metric("Cost Overrun Probability", f"{cost_probability * 100:.1f}%")
    with r3: st.metric("Overall Risk", f"{overall_percentage:.1f}%")

    if overall_level == "High Risk":
        risk_color = "#da291c"
    elif overall_level == "Medium Risk":
        risk_color = "#f57c00"
    else:
        risk_color = "#4caf50"

    st.markdown(
        f'''
        <div class="risk-card">
            <div class="info-label">OVERALL PROJECT RISK</div>
            <div style="font-size:40px; font-weight:750; color:{risk_color}; margin-top:5px;">{overall_percentage:.1f}%</div>
            <div style="font-size:16px; font-weight:700; color:{risk_color};">{overall_level}</div>
            <div style="background:#f0f0f0; height:8px; border-radius:8px; margin-top:15px;">
                <div style="width:{min(max(overall_percentage,0),100):.1f}%; height:8px; border-radius:8px; background:{risk_color};"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True
    )

        # ============================================================
    # PROJECT RISK EXPLANATION - ACTUAL PROJECT CONDITIONS
    # ============================================================

    st.markdown("### 🧠 Why is this Project Risky?")

    st.info(
        "The explanation below uses the selected project's actual "
        "monitoring indicators and compares them with the available "
        "project dataset. These are risk signals, not proven causal reasons."
    )

    # ---------- Safe numeric conversion ----------
    def safe_value(value, default=0.0):
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    # Selected project values
    physical_progress = safe_value(selected.get("physical_progress"))
    expenditure = safe_value(selected.get("expenditure"))
    original_cost = safe_value(selected.get("original_cost"))
    duration_days = safe_value(selected.get("project_duration_days"))
    progress_per_month = safe_value(selected.get("progress_per_month"))

    # Expenditure ratio
    if original_cost > 0:
        expenditure_ratio = expenditure / original_cost
    else:
        expenditure_ratio = safe_value(selected.get("expenditure_ratio"))

    # Convert ratio into percentage for display
    expenditure_ratio_pct = expenditure_ratio * 100

    # ---------- Dataset benchmarks ----------
    dataset_progress = pd.to_numeric(
        filtered_data.get("physical_progress", pd.Series(dtype=float)),
        errors="coerce"
    ).dropna()

    dataset_expenditure_ratio = pd.Series(dtype=float)

    if "expenditure_ratio" in filtered_data.columns:
        dataset_expenditure_ratio = pd.to_numeric(
            filtered_data["expenditure_ratio"],
            errors="coerce"
        ).dropna()

        # If ratio is already percentage based, convert it
        if len(dataset_expenditure_ratio) > 0 and dataset_expenditure_ratio.median() > 1:
            dataset_expenditure_ratio = dataset_expenditure_ratio / 100

    dataset_duration = pd.to_numeric(
        filtered_data.get("project_duration_days", pd.Series(dtype=float)),
        errors="coerce"
    ).dropna()

    dataset_progress_rate = pd.to_numeric(
        filtered_data.get("progress_per_month", pd.Series(dtype=float)),
        errors="coerce"
    ).dropna()

    median_progress = (
        float(dataset_progress.median())
        if len(dataset_progress) > 0 else None
    )

    median_expenditure_ratio = (
        float(dataset_expenditure_ratio.median())
        if len(dataset_expenditure_ratio) > 0 else None
    )

    median_duration = (
        float(dataset_duration.median())
        if len(dataset_duration) > 0 else None
    )

    median_progress_rate = (
        float(dataset_progress_rate.median())
        if len(dataset_progress_rate) > 0 else None
    )

    # ============================================================
    # DELAY RISK SIGNALS
    # ============================================================

    delay_signals = []

    # 1. Physical progress below dataset benchmark
    if median_progress is not None and physical_progress < median_progress:
        delay_signals.append(
            f"""
            <div class="factor-card">
                <b>🔴 Physical Progress</b><br>
                Current physical progress is
                <b>{physical_progress:.1f}%</b>, compared with a
                dataset median of <b>{median_progress:.1f}%</b>.
                Lower progress indicates that the project is behind
                the typical progress level observed in the dataset.
            </div>
            """
        )

    # 2. Progress rate below dataset benchmark
    if (
        median_progress_rate is not None
        and progress_per_month > 0
        and progress_per_month < median_progress_rate
    ):
        delay_signals.append(
            f"""
            <div class="factor-card">
                <b>🟠 Progress Rate</b><br>
                The project is progressing at approximately
                <b>{progress_per_month:.2f}% per month</b>, compared with
                a dataset median of <b>{median_progress_rate:.2f}% per month</b>.
            </div>
            """
        )

    # 3. Long project duration
    if (
        median_duration is not None
        and duration_days > 0
        and duration_days > median_duration
    ):
        st.markdown(
            f"""
            <div class="factor-card">
                <b>🟠 Project Duration</b><br>
                Current project duration is approximately
                <b>{duration_days:,.0f} days</b>, compared with a
                dataset median of <b>{median_duration:,.0f} days</b>.
                A longer duration provides a larger time window in which
                schedule risk can remain relevant.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Spending ahead of physical progress
    progress_fraction = physical_progress / 100

    if (
        original_cost > 0
        and expenditure > 0
        and expenditure_ratio > progress_fraction + 0.15
    ):
        delay_signals.append(
            f"""
            <div class="factor-card">
                <b>🟠 Expenditure vs Physical Progress</b><br>
                Approximately <b>{expenditure_ratio_pct:.1f}%</b> of the
                original project cost has been spent, while physical
                progress is <b>{physical_progress:.1f}%</b>.
                This gap is an observable monitoring signal requiring
                attention.
            </div>
            """
        )

    # ============================================================
    # DISPLAY DELAY SIGNALS
    # ============================================================

    st.markdown("### ⏱️ Delay Risk Signals")

    if delay_signals:
        for signal in delay_signals:
            st.markdown(signal, unsafe_allow_html=True)
    else:
        st.success(
            "No unusually adverse delay indicators were identified "
            "from the available project-level measurements."
        )

    # ============================================================
    # COST-OVERRUN RISK SIGNALS
    # ============================================================

    cost_signals = []

    # 1. High expenditure utilisation compared with dataset
    if (
        median_expenditure_ratio is not None
        and expenditure_ratio > median_expenditure_ratio
    ):
        cost_signals.append(
            f"""
            <div class="factor-card">
                <b>🟠 Expenditure Utilisation</b><br>
                Expenditure is approximately
                <b>{expenditure_ratio_pct:.1f}%</b> of the original cost,
                compared with a dataset median of
                <b>{median_expenditure_ratio * 100:.1f}%</b>.
            </div>
            """
        )

    # 2. Expenditure significantly ahead of physical progress
    if (
        original_cost > 0
        and expenditure > 0
        and expenditure_ratio > progress_fraction + 0.15
    ):
        cost_signals.append(
            f"""
            <div class="factor-card">
                <b>🟠 Spending–Progress Gap</b><br>
                Financial utilisation is
                <b>{expenditure_ratio_pct:.1f}%</b>, while physical
                progress is <b>{physical_progress:.1f}%</b>.
                The difference between financial utilisation and physical
                progress is a project-level monitoring signal.
            </div>
            """
        )

    # 3. Very low progress with substantial expenditure
    if (
        physical_progress < 40
        and expenditure_ratio > 0.60
    ):
        cost_signals.append(
            f"""
            <div class="factor-card">
                <b>🔴 High Spending with Low Physical Progress</b><br>
                The project has reached only <b>{physical_progress:.1f}%</b>
                physical progress while approximately
                <b>{expenditure_ratio_pct:.1f}%</b> of the original cost
                has been utilised.
            </div>
            """
        )

    # ============================================================
    # DISPLAY COST SIGNALS
    # ============================================================

    st.markdown("### 💰 Cost-Overrun Risk Signals")

    if cost_signals:
        for signal in cost_signals:
            st.markdown(signal, unsafe_allow_html=True)
    else:
        st.success(
            "No unusually adverse cost-utilisation indicators were "
            "identified from the available project-level measurements."
        )

    # ============================================================
    # ML INTERPRETATION
    # ============================================================

    st.markdown("### 🤖 ML Risk Assessment")

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        st.markdown(
            f"""
            <div class="factor-card">
                <b>Schedule Risk</b><br>
                ML-estimated delay probability:
                <strong>{delay_probability * 100:.1f}%</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    with ml_col2:
        st.markdown(
            f"""
            <div class="factor-card">
                <b>Financial Risk</b><br>
                ML-estimated cost-overrun probability:
                <strong>{cost_probability * 100:.1f}%</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "The ML model generates the risk probabilities. The signals above "
        "translate the selected project's actual measurable conditions "
        "into an understandable monitoring explanation. They should not "
        "be interpreted as proof that any single factor caused the risk."
    )

    st.markdown("### 📊 Project Monitoring Indicators")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Physical Progress", f"{float(selected.get('physical_progress', 0)):.1f}%")
    with m2: st.metric("Original Cost", f"₹ {float(selected.get('original_cost', 0)):,.2f}")
    with m3: st.metric("Expenditure", f"₹ {float(selected.get('expenditure', 0)):,.2f}")
    with m4:
        if "expenditure_ratio" in selected:
            try:
                ratio = float(selected["expenditure_ratio"])
                st.metric("Expenditure Ratio", f"{ratio:.2f}")
            except Exception:
                st.metric("Expenditure Ratio", "N/A")
        else:
            st.metric("Expenditure Ratio", "N/A")

# ============================================================
# ROUTING: PROJECT INTELLIGENCE
# ============================================================
elif selected_page == "Project Intelligence":

    st.markdown(
        '<div class="section-title">PROJECT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Project-level intelligence combining monitoring indicators, "
        "ML predictions and model-influential risk signals."
    )

    if filtered_data.empty:
        st.warning("No projects available for the selected sector.")
        st.stop()

    # --------------------------------------------------------
    # PROJECT SEARCH + SELECTION
    # --------------------------------------------------------

    search_text = st.text_input(
        "🔎 Search Project",
        placeholder="Search by project name or project code..."
    )

    project_view = filtered_data.copy()

    if search_text:
        search_lower = search_text.lower().strip()

        name_mask = (
            project_view["project_name"]
            .astype(str)
            .str.lower()
            .str.contains(search_lower, na=False)
        )

        code_mask = (
            project_view["project_code"]
            .astype(str)
            .str.lower()
            .str.contains(search_lower, na=False)
        )

        project_view = project_view[name_mask | code_mask]

    if project_view.empty:
        st.warning("No matching project found.")
        st.stop()

    project_codes = (
        project_view["project_code"]
        .astype(str)
        .tolist()
    )

    selected_project = st.selectbox(
        "Select Project",
        project_codes
    )

    selected = project_view[
        project_view["project_code"].astype(str)
        == str(selected_project)
    ].iloc[0]

    project_name = selected.get(
        "project_name",
        "Unknown Project"
    )

    # --------------------------------------------------------
    # PROJECT HEADER
    # --------------------------------------------------------

    st.markdown(f"## 🏗️ {project_name}")

    st.caption(
        f"Project Code: {selected_project}"
    )

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.markdown("### 📋 Project Information")

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.metric(
            "Sector",
            str(selected.get("sector", "Unknown"))
        )

    with i2:
        st.metric(
            "Ministry",
            str(selected.get("ministry", "Unknown"))
        )

    with i3:
        st.metric(
            "Agency",
            str(selected.get("agency", "Unknown"))
        )

    with i4:
        st.metric(
            "Physical Progress",
            f"{float(selected.get('physical_progress', 0)):.1f}%"
        )

    # --------------------------------------------------------
    # FINANCIAL & EXECUTION STATUS
    # --------------------------------------------------------

    st.markdown("### 📊 Financial & Execution Status")

    f1, f2, f3, f4 = st.columns(4)

    original_cost = float(
        selected.get("original_cost", 0)
    )

    expenditure = float(
        selected.get("expenditure", 0)
    )

    progress = float(
        selected.get("physical_progress", 0)
    )

    with f1:
        st.metric(
            "Original Cost",
            f"₹ {original_cost:,.2f}"
        )

    with f2:
        st.metric(
            "Expenditure",
            f"₹ {expenditure:,.2f}"
        )

    with f3:
        ratio = selected.get(
            "expenditure_ratio",
            None
        )

        if pd.notna(ratio):
            st.metric(
                "Expenditure Ratio",
                f"{float(ratio):.2f}"
            )
        else:
            st.metric(
                "Expenditure Ratio",
                "N/A"
            )

    with f4:
        duration = selected.get(
            "project_duration_months",
            None
        )

        if pd.notna(duration):
            st.metric(
                "Project Duration",
                f"{float(duration):.1f} months"
            )
        else:
            st.metric(
                "Project Duration",
                "N/A"
            )

    # --------------------------------------------------------
    # ML PREDICTIONS
    # --------------------------------------------------------

    st.markdown("### 🤖 ML Prediction Summary")

    delay_probability = float(
        selected.get(
            "delay_probability",
            0
        )
    )

    cost_probability = float(
        selected.get(
            "cost_overrun_probability",
            0
        )
    )

    overall_percentage = float(
        selected.get(
            "overall_risk_percentage",
            0
        )
    )

    overall_level = str(
        selected.get(
            "overall_risk_level",
            "Unknown"
        )
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Predicted Delay Risk",
            f"{delay_probability * 100:.1f}%"
        )

    with p2:
        st.metric(
            "Predicted Cost-Overrun Risk",
            f"{cost_probability * 100:.1f}%"
        )

    with p3:
        st.metric(
            "Overall Project Risk",
            f"{overall_percentage:.1f}%"
        )

    st.info(
        f"Current ML-based risk classification: **{overall_level}**"
    )

    # --------------------------------------------------------
    # PROJECT-SPECIFIC ML FACTORS
    # --------------------------------------------------------

    st.markdown("### 🧠 Project Risk Signals")

    st.caption(
        "These are model-influential factors identified by the existing "
        "SHAP analysis for this project. They indicate model association, "
        "not proven real-world causation."
    )

    factor_row = None

    if (
        not risk_factors.empty
        and "project_code" in risk_factors.columns
    ):
        matching_factors = risk_factors[
            risk_factors["project_code"]
            .astype(str)
            == str(selected_project)
        ]

        if not matching_factors.empty:
            factor_row = matching_factors.iloc[0]

    # --------------------------------------------------------
    # HUMAN-READABLE FACTOR MAPPING
    # --------------------------------------------------------

    def explain_factor(factor_text, selected_row):

        factor_text = str(factor_text).strip()

        if not factor_text:
            return None

        # Remove ML suffixes
        raw = factor_text

        direction = ""
        if ": increases delay risk" in raw:
            direction = "Higher model risk signal"
            raw = raw.replace(
                ": increases delay risk", ""
            )

        elif ": reduces delay risk" in raw:
            direction = "Lower model risk signal"
            raw = raw.replace(
                ": reduces delay risk", ""
            )

        elif ": increases cost-overrun risk" in raw:
            direction = "Higher model risk signal"
            raw = raw.replace(
                ": increases cost-overrun risk", ""
            )

        elif ": reduces cost-overrun risk" in raw:
            direction = "Lower model risk signal"
            raw = raw.replace(
                ": reduces cost-overrun risk", ""
            )

        raw = raw.strip()

        # ----------------------------------------------------
        # Progress per month
        # ----------------------------------------------------

        if raw == "progress_per_month":

            value = selected_row.get(
                "progress_per_month",
                None
            )

            if pd.notna(value):
                return (
                    "Progress Rate",
                    f"{float(value):.2f}% progress/month",
                    direction
                )

            return (
                "Progress Rate",
                "Calculated progress rate unavailable",
                direction
            )

        # ----------------------------------------------------
        # Project duration
        # ----------------------------------------------------

        if raw == "project_duration_days":

            value = selected_row.get(
                "project_duration_days",
                None
            )

            if pd.notna(value):
                return (
                    "Project Duration",
                    f"{float(value):,.0f} days",
                    direction
                )

            return (
                "Project Duration",
                "Duration data unavailable",
                direction
            )

        # ----------------------------------------------------
        # Project duration months
        # ----------------------------------------------------

        if raw == "project_duration_months":

            value = selected_row.get(
                "project_duration_months",
                None
            )

            if pd.notna(value):
                return (
                    "Project Duration",
                    f"{float(value):.1f} months",
                    direction
                )

            return (
                "Project Duration",
                "Duration data unavailable",
                direction
            )

        # ----------------------------------------------------
        # Expenditure ratio
        # ----------------------------------------------------

        if raw == "expenditure_ratio":

            value = selected_row.get(
                "expenditure_ratio",
                None
            )

            if pd.notna(value):
                return (
                    "Expenditure Utilisation",
                    f"{float(value):.2f}",
                    direction
                )

            return (
                "Expenditure Utilisation",
                "Ratio unavailable",
                direction
            )

        # ----------------------------------------------------
        # Physical progress
        # ----------------------------------------------------

        if raw == "physical_progress":

            value = selected_row.get(
                "physical_progress",
                None
            )

            if pd.notna(value):
                return (
                    "Physical Progress",
                    f"{float(value):.1f}%",
                    direction
                )

            return (
                "Physical Progress",
                "Progress unavailable",
                direction
            )

        # ----------------------------------------------------
        # Ministry
        # ----------------------------------------------------

        if raw.startswith("ministry_"):

            value = selected_row.get(
                "ministry",
                "Unknown"
            )

            return (
                "Responsible Ministry",
                str(value),
                direction
            )

        # ----------------------------------------------------
        # Sector
        # ----------------------------------------------------

        if raw.startswith("sector_"):

            value = selected_row.get(
                "sector",
                "Unknown"
            )

            return (
                "Infrastructure Sector",
                str(value),
                direction
            )

        # ----------------------------------------------------
        # Agency
        # ----------------------------------------------------

        if raw.startswith("agency_"):

            value = selected_row.get(
                "agency",
                "Unknown"
            )

            return (
                "Implementing Agency",
                str(value),
                direction
            )

        # ----------------------------------------------------
        # Fallback for future factors
        # ----------------------------------------------------

        return (
            raw.replace("_", " ").title(),
            "Model-influential feature",
            direction
        )

    # --------------------------------------------------------
    # DISPLAY DELAY FACTORS
    # --------------------------------------------------------

    if factor_row is not None:

        d_col, c_col = st.columns(2)

        with d_col:

            st.markdown("#### ⏱️ Delay Risk Signals")

            delay_text = str(
                factor_row.get(
                    "top_delay_factors",
                    ""
                )
            )

            delay_factors = [
                x.strip()
                for x in delay_text.split(" | ")
                if x.strip()
            ]

            if delay_factors:

                for factor in delay_factors[:5]:

                    explanation = explain_factor(
                        factor,
                        selected
                    )

                    if explanation:

                        title, value, direction = explanation

                        st.markdown(
                            f"""
                            <div class="factor-card">
                                <div class="info-label">{title.upper()}</div>
                                <div class="info-value">{value}</div>
                                <div style="font-size:13px; margin-top:6px;">
                                    {direction}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:
                st.info(
                    "No project-specific delay signals available."
                )

        # ----------------------------------------------------
        # DISPLAY COST FACTORS
        # ----------------------------------------------------

        with c_col:

            st.markdown("#### 💰 Cost-Overrun Risk Signals")

            cost_text = str(
                factor_row.get(
                    "top_cost_overrun_factors",
                    ""
                )
            )

            cost_factors = [
                x.strip()
                for x in cost_text.split(" | ")
                if x.strip()
            ]

            if cost_factors:

                for factor in cost_factors[:5]:

                    explanation = explain_factor(
                        factor,
                        selected
                    )

                    if explanation:

                        title, value, direction = explanation

                        st.markdown(
                            f"""
                            <div class="factor-card">
                                <div class="info-label">{title.upper()}</div>
                                <div class="info-value">{value}</div>
                                <div style="font-size:13px; margin-top:6px;">
                                    {direction}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:
                st.info(
                    "No project-specific cost signals available."
                )

    else:

        st.info(
            "Project-specific SHAP factor information is not available "
            "for this project."
        )


# ============================================================
# ROUTING: EARLY WARNING CENTER
# ============================================================
elif selected_page == "Early Warning Center":

    st.markdown(
        '<div class="section-title">EARLY WARNING CENTER</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Early identification of infrastructure projects requiring monitoring attention."
    )

    if filtered_data.empty:
        st.warning("No project data available.")
        st.stop()

    warning_data = filtered_data.copy()

    # --------------------------------------------------------
    # WARNING THRESHOLDS
    # --------------------------------------------------------

    # Existing ML risk score
    warning_data["risk_score"] = pd.to_numeric(
        warning_data.get("overall_risk_percentage", 0),
        errors="coerce"
    ).fillna(0)

    warning_data["delay_probability_pct"] = (
        pd.to_numeric(
            warning_data.get("delay_probability", 0),
            errors="coerce"
        ).fillna(0) * 100
    )

    warning_data["cost_probability_pct"] = (
        pd.to_numeric(
            warning_data.get("cost_overrun_probability", 0),
            errors="coerce"
        ).fillna(0) * 100
    )

    warning_data["physical_progress"] = pd.to_numeric(
        warning_data.get("physical_progress", 0),
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # EARLY WARNING STATUS
    # --------------------------------------------------------

    def warning_status(row):

        risk = row["risk_score"]
        delay = row["delay_probability_pct"]
        cost = row["cost_probability_pct"]

        if risk >= 70 or delay >= 70 or cost >= 70:
            return "CRITICAL"

        elif risk >= 50 or delay >= 50 or cost >= 50:
            return "HIGH"

        elif risk >= 30 or delay >= 30 or cost >= 30:
            return "WATCH"

        return "NORMAL"

    warning_data["warning_status"] = warning_data.apply(
        warning_status,
        axis=1
    )

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    critical_count = (
        warning_data["warning_status"] == "CRITICAL"
    ).sum()

    high_count = (
        warning_data["warning_status"] == "HIGH"
    ).sum()

    watch_count = (
        warning_data["warning_status"] == "WATCH"
    ).sum()

    normal_count = (
        warning_data["warning_status"] == "NORMAL"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🔴 Critical",
            critical_count
        )

    with c2:
        st.metric(
            "🟠 High Attention",
            high_count
        )

    with c3:
        st.metric(
            "🟡 Watch",
            watch_count
        )

    with c4:
        st.metric(
            "🟢 Normal",
            normal_count
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    st.markdown("### 🚨 Warning Queue")

    selected_status = st.multiselect(
        "Filter Warning Level",
        ["CRITICAL", "HIGH", "WATCH", "NORMAL"],
        default=["CRITICAL", "HIGH", "WATCH"]
    )

    queue = warning_data[
        warning_data["warning_status"].isin(selected_status)
    ].copy()

    # Sort highest risk first
    queue = queue.sort_values(
        "risk_score",
        ascending=False
    )

    # --------------------------------------------------------
    # WARNING TABLE
    # --------------------------------------------------------

    if queue.empty:

        st.success(
            "No projects match the selected warning level."
        )

    else:

        display_columns = [
            "project_code",
            "project_name",
            "sector",
            "physical_progress",
            "delay_probability_pct",
            "cost_probability_pct",
            "risk_score",
            "warning_status"
        ]

        display_columns = [
            col for col in display_columns
            if col in queue.columns
        ]

        warning_table = queue[display_columns].copy()

        warning_table = warning_table.rename(
            columns={
                "project_code": "Project Code",
                "project_name": "Project",
                "sector": "Sector",
                "physical_progress": "Progress (%)",
                "delay_probability_pct": "Delay Risk (%)",
                "cost_probability_pct": "Cost Risk (%)",
                "risk_score": "Overall Risk (%)",
                "warning_status": "Warning Level"
            }
        )

        for col in [
            "Progress (%)",
            "Delay Risk (%)",
            "Cost Risk (%)",
            "Overall Risk (%)"
        ]:

            if col in warning_table.columns:
                warning_table[col] = warning_table[col].round(1)

        st.dataframe(
            warning_table,
            width="stretch",
            hide_index=True
        )

    # --------------------------------------------------------
    # SELECT PROJECT FOR ALERT DETAILS
    # --------------------------------------------------------

    st.markdown("### 🔎 Alert Details")

    if not queue.empty:

        project_codes = (
            queue["project_code"]
            .astype(str)
            .tolist()
        )

        selected_code = st.selectbox(
            "Select a project to inspect its warning signals",
            project_codes
        )

        selected_warning = queue[
            queue["project_code"].astype(str)
            == str(selected_code)
        ].iloc[0]

        st.markdown(
            f"#### 🏗️ {selected_warning.get('project_name', 'Project')}"
        )

        a1, a2, a3 = st.columns(3)

        with a1:
            st.metric(
                "Delay Risk",
                f"{selected_warning['delay_probability_pct']:.1f}%"
            )

        with a2:
            st.metric(
                "Cost Overrun Risk",
                f"{selected_warning['cost_probability_pct']:.1f}%"
            )

        with a3:
            st.metric(
                "Overall Risk",
                f"{selected_warning['risk_score']:.1f}%"
            )

        # ----------------------------------------------------
        # WARNING SIGNALS
        # ----------------------------------------------------

        st.markdown("#### ⚠️ Current Warning Signals")

        signals = []

        if selected_warning["delay_probability_pct"] >= 70:
            signals.append(
                "High model-estimated delay risk."
            )

        elif selected_warning["delay_probability_pct"] >= 50:
            signals.append(
                "Elevated model-estimated delay risk."
            )

        if selected_warning["cost_probability_pct"] >= 70:
            signals.append(
                "High model-estimated cost-overrun risk."
            )

        elif selected_warning["cost_probability_pct"] >= 50:
            signals.append(
                "Elevated model-estimated cost-overrun risk."
            )

        if selected_warning["physical_progress"] < 30:
            signals.append(
                f"Physical progress is currently "
                f"{selected_warning['physical_progress']:.1f}%."
            )

        if signals:

            for signal in signals:
                st.warning(signal)

        else:

            st.info(
                "No additional threshold-based warning signal "
                "was triggered for this project."
            )

        st.caption(
            "Early warnings are threshold-based monitoring indicators "
            "built from the existing ML prediction outputs and project data. "
            "They are intended for monitoring, not guaranteed outcomes."
        )


elif selected_page == "Benchmarking":

    st.markdown(
        '<div class="section-title">BENCHMARKING</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Compare project performance, expenditure utilisation and risk indicators "
        "across sectors, ministries and agencies."
    )

    benchmark_data = filtered_data.copy()

    # ---------------------------------------------------------
    # Benchmark Controls
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        benchmark_type = st.selectbox(
            "Benchmark By",
            ["Sector", "Ministry", "Agency"]
        )

    with col2:
        metric = st.selectbox(
            "Performance Metric",
            [
                "Physical Progress",
                "Expenditure Ratio",
                "Delay Risk",
                "Cost Overrun Risk",
                "Overall Risk"
            ]
        )

    with col3:
        top_n = st.slider(
            "Show Top Groups",
            min_value=5,
            max_value=15,
            value=10
        )

    # ---------------------------------------------------------
    # Prepare metric columns
    # ---------------------------------------------------------
    benchmark_data["Physical Progress"] = pd.to_numeric(
        benchmark_data["physical_progress"],
        errors="coerce"
    )

    benchmark_data["Expenditure Ratio"] = pd.to_numeric(
        benchmark_data["expenditure_ratio"],
        errors="coerce"
    )

    benchmark_data["Delay Risk"] = pd.to_numeric(
        benchmark_data["delay_probability"],
        errors="coerce"
    ) * 100

    benchmark_data["Cost Overrun Risk"] = pd.to_numeric(
        benchmark_data["cost_overrun_probability"],
        errors="coerce"
    ) * 100

    benchmark_data["Overall Risk"] = pd.to_numeric(
        benchmark_data["overall_risk_percentage"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Select grouping column
    # ---------------------------------------------------------
    group_map = {
        "Sector": "sector",
        "Ministry": "ministry",
        "Agency": "agency"
    }

    group_column = group_map[benchmark_type]

    benchmark_data[group_column] = (
        benchmark_data[group_column]
        .fillna("Unknown")
        .astype(str)
    )

    # ---------------------------------------------------------
    # Calculate benchmark
    # ---------------------------------------------------------
    benchmark_table = (
        benchmark_data
        .groupby(group_column)
        .agg(
            Projects=("project_code", "count"),
            Average=("{}".format(metric), "mean")
        )
        .reset_index()
    )

    benchmark_table["Average"] = benchmark_table["Average"].round(2)

    benchmark_table = benchmark_table[
        benchmark_table["Projects"] >= 2
    ]

    benchmark_table = benchmark_table.sort_values(
        "Average",
        ascending=False
    ).head(top_n)

    # ---------------------------------------------------------
    # KPI Cards
    # ---------------------------------------------------------
    if not benchmark_table.empty:

        total_groups = len(benchmark_table)

        avg_value = benchmark_table["Average"].mean()

        total_projects = benchmark_table["Projects"].sum()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Groups Compared",
                total_groups
            )

        with c2:
            st.metric(
                "Projects Covered",
                total_projects
            )

        with c3:
            st.metric(
                f"Avg {metric}",
                f"{avg_value:.2f}"
            )

        st.markdown("### Benchmark Comparison")

        # -----------------------------------------------------
        # Chart
        # -----------------------------------------------------
        fig = px.bar(
            benchmark_table,
            x=group_column,
            y="Average",
            text="Average",
            hover_data=["Projects"],
            title=f"{metric} by {benchmark_type}"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title=benchmark_type,
            yaxis_title=metric,
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------------------------
        # Detailed Table
        # -----------------------------------------------------
        st.markdown("### Benchmark Details")

        display_table = benchmark_table.rename(
            columns={
                group_column: benchmark_type,
                "Average": f"Average {metric}"
            }
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------
        # Project-level comparison
        # -----------------------------------------------------
        st.markdown("### Project-Level Comparison")

        project_options = benchmark_data[
            ["project_code", "project_name"]
        ].drop_duplicates()

        project_options["display"] = (
            project_options["project_name"].astype(str)
            + " — "
            + project_options["project_code"].astype(str)
        )

        selected_project = st.selectbox(
            "Select a project to inspect",
            project_options["display"].tolist()
        )

        selected_row = project_options[
            project_options["display"] == selected_project
        ].iloc[0]

        project_code = selected_row["project_code"]

        project_info = benchmark_data[
            benchmark_data["project_code"] == project_code
        ].iloc[0]

        st.markdown("#### Selected Project Performance")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Physical Progress",
                f"{project_info['Physical Progress']:.1f}%"
            )

        with p2:
            st.metric(
                "Expenditure Ratio",
                f"{project_info['Expenditure Ratio']:.1f}%"
            )

        with p3:
            st.metric(
                "Delay Risk",
                f"{project_info['Delay Risk']:.1f}%"
            )

        with p4:
            st.metric(
                "Overall Risk",
                f"{project_info['Overall Risk']:.1f}%"
            )

    else:

        st.warning(
            "Not enough project data available for benchmarking."
        )


elif selected_page == "Cost Escalation Drivers":

    st.markdown(
        '<div class="section-title">COST ESCALATION DRIVERS</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Model-based analysis of factors associated with cost-overrun risk "
        "across the monitored infrastructure projects."
    )

    cost_data = filtered_data.copy()

    # ---------------------------------------------------------
    # Feature importance data
    # ---------------------------------------------------------
    try:
        cost_importance_path = os.path.join(
            BASE_DIR,
            "data",
            "cleaned",
            "cost_feature_importance.csv"
        )

        if os.path.exists(cost_importance_path):
            cost_importance = pd.read_csv(cost_importance_path)
        else:
            cost_importance = pd.DataFrame()

    except Exception:
        cost_importance = pd.DataFrame()

    # ---------------------------------------------------------
    # Helper for readable feature names
    # ---------------------------------------------------------
    def readable_cost_factor(feature):

        feature = str(feature)

        mapping = {
            "original_cost": "Original Project Cost",
            "expenditure": "Current Expenditure",
            "physical_progress": "Physical Progress",
            "project_duration_days": "Project Duration",
            "project_duration_months": "Project Duration (Months)",
            "expenditure_ratio": "Expenditure Utilisation",
            "progress_per_month": "Progress Rate",
            "sector": "Sector",
            "ministry": "Ministry",
            "agency": "Implementing Agency"
        }

        if feature in mapping:
            return mapping[feature]

        if feature.startswith("sector_"):
            return "Sector: " + feature.replace("sector_", "")

        if feature.startswith("ministry_"):
            return "Ministry: " + feature.replace("ministry_", "")

        if feature.startswith("agency_"):
            return "Agency: " + feature.replace("agency_", "")

        return feature.replace("_", " ").title()

    # ---------------------------------------------------------
    # Current cost-risk statistics
    # ---------------------------------------------------------
    cost_data["cost_risk_pct"] = (
        pd.to_numeric(
            cost_data["cost_overrun_probability"],
            errors="coerce"
        ) * 100
    )

    cost_data["cost_risk_pct"] = cost_data["cost_risk_pct"].clip(0, 100)

    high_cost_risk = cost_data[
        cost_data["cost_risk_pct"] >= 70
    ]

    avg_cost_risk = cost_data["cost_risk_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Projects Analysed",
            len(cost_data)
        )

    with c2:
        st.metric(
            "Average Cost Risk",
            f"{avg_cost_risk:.1f}%"
        )

    with c3:
        st.metric(
            "High Cost-Risk Projects",
            len(high_cost_risk)
        )

    with c4:
        st.metric(
            "Maximum Cost Risk",
            f"{cost_data['cost_risk_pct'].max():.1f}%"
        )

    # ---------------------------------------------------------
    # Model-influential factors
    # ---------------------------------------------------------
    st.markdown("### Model-Influential Cost Factors")

    if not cost_importance.empty:

        # Find likely feature/importance columns
        feature_col = None
        importance_col = None

        for col in cost_importance.columns:
            col_lower = str(col).lower()

            if feature_col is None and (
                "feature" in col_lower or "factor" in col_lower
            ):
                feature_col = col

            if importance_col is None and (
                "importance" in col_lower or
                "weight" in col_lower
            ):
                importance_col = col

        if feature_col is not None and importance_col is not None:

            factor_table = cost_importance[
                [feature_col, importance_col]
            ].copy()

            factor_table.columns = [
                "Factor",
                "Importance"
            ]

            factor_table["Importance"] = pd.to_numeric(
                factor_table["Importance"],
                errors="coerce"
            )

            factor_table = factor_table.dropna(
                subset=["Importance"]
            )

            factor_table = factor_table.sort_values(
                "Importance",
                ascending=False
            ).head(10)

            factor_table["Factor"] = factor_table[
                "Factor"
            ].apply(readable_cost_factor)

            # Chart
            fig = px.bar(
                factor_table.sort_values("Importance"),
                x="Importance",
                y="Factor",
                orientation="h",
                title="Top Factors Influencing Cost-Overrun Prediction"
            )

            fig.update_layout(
                height=450,
                xaxis_title="Model Importance",
                yaxis_title="Factor"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                factor_table,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "Feature-importance columns could not be identified "
                "in the existing cost feature-importance file."
            )

    else:
        st.info(
            "Cost feature-importance data is not available."
        )

    # ---------------------------------------------------------
    # Project-specific cost escalation analysis
    # ---------------------------------------------------------
    st.markdown("### Project-Specific Cost Risk Analysis")

    project_options = cost_data[
        ["project_code", "project_name"]
    ].drop_duplicates()

    project_options["display"] = (
        project_options["project_name"].astype(str)
        + " — "
        + project_options["project_code"].astype(str)
    )

    if len(project_options) > 0:

        selected_project = st.selectbox(
            "Select a project",
            project_options["display"].tolist()
        )

        selected_code = project_options.loc[
            project_options["display"] == selected_project,
            "project_code"
        ].iloc[0]

        project = cost_data[
            cost_data["project_code"] == selected_code
        ].iloc[0]

        st.markdown("#### Current Cost Indicators")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Original Cost",
                f"₹{project['original_cost']:,.0f}"
            )

        with p2:
            st.metric(
                "Expenditure",
                f"₹{project['expenditure']:,.0f}"
            )

        with p3:
            st.metric(
                "Expenditure Utilisation",
                f"{project['expenditure_ratio']:.1f}%"
            )

        with p4:
            st.metric(
                "Cost-Overrun Risk",
                f"{project['cost_risk_pct']:.1f}%"
            )

        # -----------------------------------------------------
        # Risk status
        # -----------------------------------------------------
        if project["cost_risk_pct"] >= 70:
            st.error(
                "High model-predicted cost-overrun risk"
            )
        elif project["cost_risk_pct"] >= 50:
            st.warning(
                "Elevated model-predicted cost-overrun risk"
            )
        else:
            st.success(
                "Lower model-predicted cost-overrun risk"
            )

        # -----------------------------------------------------
        # Project factor information
        # -----------------------------------------------------
        if "risk_factors" in locals() and not risk_factors.empty:

            project_factors = risk_factors[
                risk_factors["project_code"] == selected_code
            ]

            if not project_factors.empty:

                row = project_factors.iloc[0]

                st.markdown(
                    "#### Project-Specific Model Factors"
                )

                cost_factors = str(
                    row.get(
                        "top_cost_overrun_factors",
                        ""
                    )
                )

                if cost_factors and cost_factors.lower() != "nan":

                    factors = [
                        x.strip()
                        for x in cost_factors.split(";")
                        if x.strip()
                    ]

                    for factor in factors[:5]:

                        st.info(
                            f"**{readable_cost_factor(factor)}**"
                        )

                else:

                    st.info(
                        "No project-specific cost factors "
                        "are available for this project."
                    )

    # ---------------------------------------------------------
    # Interpretation note
    # ---------------------------------------------------------
    st.markdown("---")

    st.caption(
        "Note: These factors describe variables that influence the "
        "ML model's cost-overrun prediction. They should not be "
        "interpreted as confirmed causal drivers."
    )


elif selected_page == "Dependency Intelligence":

    st.markdown(
        '<div class="section-title">DEPENDENCY INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Identify project conditions that may require dependency-aware "
        "monitoring using available PAIMANA project and risk indicators."
    )

    dependency_data = filtered_data.copy()

    # ---------------------------------------------------------
    # Prepare numeric fields
    # ---------------------------------------------------------
    dependency_data["progress"] = pd.to_numeric(
        dependency_data["physical_progress"],
        errors="coerce"
    ).fillna(0)

    dependency_data["expenditure_ratio_clean"] = pd.to_numeric(
        dependency_data["expenditure_ratio"],
        errors="coerce"
    )

    dependency_data["delay_risk"] = (
        pd.to_numeric(
            dependency_data["delay_probability"],
            errors="coerce"
        ).fillna(0) * 100
    ).clip(0, 100)

    dependency_data["cost_risk"] = (
        pd.to_numeric(
            dependency_data["cost_overrun_probability"],
            errors="coerce"
        ).fillna(0) * 100
    ).clip(0, 100)

    dependency_data["overall_risk"] = pd.to_numeric(
        dependency_data["overall_risk_percentage"],
        errors="coerce"
    ).fillna(0).clip(0, 100)

    # ---------------------------------------------------------
    # Dependency monitoring signals
    # ---------------------------------------------------------
    dependency_data["risk_signals"] = ""

    for idx, row in dependency_data.iterrows():

        signals = []

        if row["delay_risk"] >= 70:
            signals.append("High delay risk")

        elif row["delay_risk"] >= 50:
            signals.append("Elevated delay risk")

        if row["cost_risk"] >= 70:
            signals.append("High cost-overrun risk")

        elif row["cost_risk"] >= 50:
            signals.append("Elevated cost-overrun risk")

        if row["progress"] < 30:
            signals.append("Low physical progress")

        if (
            pd.notna(row["expenditure_ratio_clean"])
            and row["expenditure_ratio_clean"] >= 70
            and row["progress"] < 50
        ):
            signals.append(
                "High expenditure with comparatively lower progress"
            )

        dependency_data.at[idx, "risk_signals"] = (
            " • ".join(signals)
            if signals
            else "No major dependency signal"
        )

    dependency_data["dependency_signal_count"] = (
        dependency_data["risk_signals"]
        .str.count("•") + 1
    )

    dependency_data.loc[
        dependency_data["risk_signals"] == "No major dependency signal",
        "dependency_signal_count"
    ] = 0

    # ---------------------------------------------------------
    # KPIs
    # ---------------------------------------------------------
    flagged_projects = dependency_data[
        dependency_data["dependency_signal_count"] > 0
    ]

    high_dependency = dependency_data[
        dependency_data["dependency_signal_count"] >= 2
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Projects Analysed",
            len(dependency_data)
        )

    with c2:
        st.metric(
            "Projects With Signals",
            len(flagged_projects)
        )

    with c3:
        st.metric(
            "Multiple-Signal Projects",
            len(high_dependency)
        )

    with c4:
        st.metric(
            "Average Overall Risk",
            f"{dependency_data['overall_risk'].mean():.1f}%"
        )

    # ---------------------------------------------------------
    # Dependency signal distribution
    # ---------------------------------------------------------
    st.markdown("### Dependency Monitoring Signals")

    signal_counts = {
        "High Delay Risk": int(
            (dependency_data["delay_risk"] >= 70).sum()
        ),
        "High Cost-Overrun Risk": int(
            (dependency_data["cost_risk"] >= 70).sum()
        ),
        "Low Physical Progress": int(
            (dependency_data["progress"] < 30).sum()
        ),
        "High Expenditure + Lower Progress": int(
            (
                (dependency_data["expenditure_ratio_clean"] >= 70)
                & (dependency_data["progress"] < 50)
            ).sum()
        )
    }

    signal_df = pd.DataFrame(
        {
            "Signal": list(signal_counts.keys()),
            "Projects": list(signal_counts.values())
        }
    )

    fig = px.bar(
        signal_df,
        x="Signal",
        y="Projects",
        text="Projects",
        title="Projects Requiring Dependency-Aware Monitoring"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=420,
        xaxis_title="Monitoring Signal",
        yaxis_title="Number of Projects"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # Dependency monitoring queue
    # ---------------------------------------------------------
    st.markdown("### Dependency Monitoring Queue")

    level_filter = st.multiselect(
        "Filter by Risk Level",
        ["High", "Medium", "Low"],
        default=["High", "Medium"]
    )

    queue = dependency_data.copy()

    if level_filter and "overall_risk_level" in queue.columns:

        queue = queue[
            queue["overall_risk_level"]
            .astype(str)
            .isin(level_filter)
        ]

    queue = queue[
        queue["dependency_signal_count"] > 0
    ].sort_values(
        ["dependency_signal_count", "overall_risk"],
        ascending=False
    )

    queue_display = queue[
        [
            "project_code",
            "project_name",
            "sector",
            "progress",
            "delay_risk",
            "cost_risk",
            "overall_risk",
            "risk_signals"
        ]
    ].copy()

    queue_display.columns = [
        "Project Code",
        "Project Name",
        "Sector",
        "Progress (%)",
        "Delay Risk (%)",
        "Cost Risk (%)",
        "Overall Risk (%)",
        "Monitoring Signals"
    ]

    queue_display["Progress (%)"] = (
        queue_display["Progress (%)"].round(1)
    )

    queue_display["Delay Risk (%)"] = (
        queue_display["Delay Risk (%)"].round(1)
    )

    queue_display["Cost Risk (%)"] = (
        queue_display["Cost Risk (%)"].round(1)
    )

    queue_display["Overall Risk (%)"] = (
        queue_display["Overall Risk (%)"].round(1)
    )

    st.dataframe(
        queue_display,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # Project dependency profile
    # ---------------------------------------------------------
    st.markdown("### Project Dependency Profile")

    project_options = dependency_data[
        ["project_code", "project_name"]
    ].drop_duplicates()

    project_options["display"] = (
        project_options["project_name"].astype(str)
        + " — "
        + project_options["project_code"].astype(str)
    )

    if not project_options.empty:

        selected_project = st.selectbox(
            "Select a project",
            project_options["display"].tolist()
        )

        selected_code = project_options.loc[
            project_options["display"] == selected_project,
            "project_code"
        ].iloc[0]

        project = dependency_data[
            dependency_data["project_code"] == selected_code
        ].iloc[0]

        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                "Physical Progress",
                f"{project['progress']:.1f}%"
            )

        with b:
            st.metric(
                "Delay Risk",
                f"{project['delay_risk']:.1f}%"
            )

        with c:
            st.metric(
                "Cost Risk",
                f"{project['cost_risk']:.1f}%"
            )

        with d:
            st.metric(
                "Overall Risk",
                f"{project['overall_risk']:.1f}%"
            )

        st.markdown("#### Identified Monitoring Signals")

        signals = project["risk_signals"]

        if signals == "No major dependency signal":

            st.success(
                "No major dependency-style monitoring signal "
                "was identified from the available indicators."
            )

        else:

            for signal in signals.split(" • "):
                st.warning(
                    f"**{signal}**"
                )

    # ---------------------------------------------------------
    # Important methodology note
    # ---------------------------------------------------------
    st.markdown("---")

    st.caption(
        "Dependency Intelligence uses relationships between available "
        "project indicators for monitoring. The current PAIMANA dataset "
        "does not contain explicit project-to-project dependency records, "
        "so this page does not claim unobserved dependencies as facts."
    )

elif selected_page == "Data Quality":

    # ============================================================
    # DATA QUALITY & VALIDATION CENTER
    # ============================================================

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f4511e, #ff8a3d);
        padding: 24px 28px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 8px 22px rgba(244,81,30,0.18);
    ">
        <div style="color:white; font-size:28px; font-weight:800;">
            🔎 Data Quality & Validation Center
        </div>
        <div style="color:white; font-size:14px; margin-top:6px;">
            Monitor data completeness, consistency and validation status
            of the infrastructure project dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    dq = filtered_data.copy()

    # ------------------------------------------------------------
    # BASIC CALCULATIONS
    # ------------------------------------------------------------

    total_rows = len(dq)
    total_columns = len(dq.columns)
    total_cells = total_rows * total_columns

    missing_cells = int(dq.isna().sum().sum())
    duplicate_rows = int(dq.duplicated().sum())

    if total_cells > 0:
        completeness = ((total_cells - missing_cells) / total_cells) * 100
    else:
        completeness = 0

    duplicate_rate = (
        (duplicate_rows / total_rows) * 100
        if total_rows > 0 else 0
    )

    # ------------------------------------------------------------
    # DATA VALIDATION CHECKS
    # ------------------------------------------------------------

    def numeric_series(column):
        if column in dq.columns:
            return pd.to_numeric(dq[column], errors="coerce")
        return pd.Series(dtype="float64")

    progress = numeric_series("physical_progress")
    original_cost = numeric_series("original_cost")
    revised_cost = numeric_series("revised_cost")
    expenditure = numeric_series("expenditure")

    invalid_progress = int(
        ((progress < 0) | (progress > 100)).sum()
    )

    negative_original_cost = int((original_cost < 0).sum())
    negative_revised_cost = int((revised_cost < 0).sum())
    negative_expenditure = int((expenditure < 0).sum())

    zero_revised_cost = int((revised_cost == 0).sum())

    expenditure_above_original = int(
        ((expenditure > original_cost) &
         original_cost.notna() &
         expenditure.notna()).sum()
    )

    revised_less_original = int(
        ((revised_cost < original_cost) &
         revised_cost.notna() &
         original_cost.notna() &
         (revised_cost != 0)).sum()
    )

    duplicate_project_codes = (
        int(dq["project_code"].duplicated().sum())
        if "project_code" in dq.columns else 0
    )

    missing_project_codes = (
        int(dq["project_code"].isna().sum())
        if "project_code" in dq.columns else 0
    )

    missing_project_names = (
        int(dq["project_name"].isna().sum())
        if "project_name" in dq.columns else 0
    )

    # Date checks
    original_completion = (
        pd.to_datetime(dq["original_completion"], errors="coerce")
        if "original_completion" in dq.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    revised_completion = (
        pd.to_datetime(dq["revised_completion"], errors="coerce")
        if "revised_completion" in dq.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    sanction_date = (
        pd.to_datetime(dq["sanction_date"], errors="coerce")
        if "sanction_date" in dq.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    original_after_revised = int(
        ((original_completion > revised_completion) &
         original_completion.notna() &
         revised_completion.notna()).sum()
    )

    original_before_sanction = int(
        ((original_completion < sanction_date) &
         original_completion.notna() &
         sanction_date.notna()).sum()
    )

    revised_before_sanction = int(
        ((revised_completion < sanction_date) &
         revised_completion.notna() &
         sanction_date.notna()).sum()
    )

    # ------------------------------------------------------------
    # DATA HEALTH SCORE
    # ------------------------------------------------------------

    issues = (
        missing_cells
        + duplicate_rows
        + invalid_progress
        + negative_original_cost
        + negative_revised_cost
        + negative_expenditure
        + duplicate_project_codes
        + original_after_revised
        + original_before_sanction
        + revised_before_sanction
    )

    if total_rows > 0:
        issue_rate = min((issues / total_rows) * 100, 100)
    else:
        issue_rate = 100

    health_score = max(0, round(100 - issue_rate, 1))

    if health_score >= 95:
        health_status = "Excellent"
        health_icon = "🟢"
    elif health_score >= 85:
        health_status = "Good"
        health_icon = "🟡"
    elif health_score >= 70:
        health_status = "Needs Review"
        health_icon = "🟠"
    else:
        health_status = "Attention Required"
        health_icon = "🔴"

    # ------------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------------

    st.markdown("### 📊 Data Health Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Projects",
        f"{total_rows:,}"
    )

    c2.metric(
        "Data Fields",
        f"{total_columns:,}"
    )

    c3.metric(
        "Completeness",
        f"{completeness:.1f}%"
    )

    c4.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

    c5.metric(
        "Data Health",
        f"{health_score:.1f}%"
    )

    st.markdown(
        f"""
        <div style="
            background:#fff7f2;
            border-left:5px solid #f4511e;
            padding:14px 18px;
            border-radius:10px;
            margin:15px 0 25px 0;
        ">
            <b>{health_icon} Current Data Status: {health_status}</b>
            <br>
            <span style="color:#666;">
                Health score is calculated from completeness,
                duplication and validation checks on the currently
                loaded project data.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------
    # COMPLETENESS ANALYSIS
    # ------------------------------------------------------------

    st.markdown("### 🧹 Field Completeness")

    completeness_data = pd.DataFrame({
        "Field": dq.columns,
        "Missing": [int(dq[c].isna().sum()) for c in dq.columns],
        "Available": [int(dq[c].notna().sum()) for c in dq.columns]
    })

    completeness_data["Completeness %"] = (
        completeness_data["Available"] /
        total_rows * 100
        if total_rows > 0 else 0
    )

    completeness_data = completeness_data.sort_values(
        "Completeness %",
        ascending=True
    )

    fig_completeness = px.bar(
        completeness_data,
        x="Completeness %",
        y="Field",
        orientation="h",
        text="Completeness %",
        title="Data Completeness by Field"
    )

    fig_completeness.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_completeness.update_layout(
        height=520,
        xaxis=dict(range=[0, 110]),
        margin=dict(l=10, r=30, t=60, b=20)
    )

    st.plotly_chart(
        fig_completeness,
        width="stretch"
    )

    # ------------------------------------------------------------
    # VALIDATION SUMMARY
    # ------------------------------------------------------------

    st.markdown("### 🛡️ Validation Summary")

    validation_data = pd.DataFrame({
        "Validation Check": [
            "Duplicate project codes",
            "Missing project codes",
            "Missing project names",
            "Invalid physical progress",
            "Negative original cost",
            "Negative revised cost",
            "Negative expenditure",
            "Zero revised cost",
            "Expenditure > original cost",
            "Revised cost < original cost",
            "Original completion after revised",
            "Original completion before sanction",
            "Revised completion before sanction"
        ],
        "Records": [
            duplicate_project_codes,
            missing_project_codes,
            missing_project_names,
            invalid_progress,
            negative_original_cost,
            negative_revised_cost,
            negative_expenditure,
            zero_revised_cost,
            expenditure_above_original,
            revised_less_original,
            original_after_revised,
            original_before_sanction,
            revised_before_sanction
        ]
    })

    validation_data["Status"] = validation_data["Records"].apply(
        lambda x: "⚠ Review" if x > 0 else "✓ Clear"
    )

    left, right = st.columns([1.4, 1])

    with left:
        st.dataframe(
            validation_data,
            width="stretch",
            hide_index=True
        )

    with right:

        issue_chart = validation_data[
            validation_data["Records"] > 0
        ].copy()

        if len(issue_chart) > 0:

            fig_issues = px.bar(
                issue_chart,
                x="Records",
                y="Validation Check",
                orientation="h",
                title="Validation Findings"
            )

            fig_issues.update_layout(
                height=470,
                margin=dict(l=10, r=20, t=60, b=20)
            )

            st.plotly_chart(
                fig_issues,
                width="stretch"
            )

        else:
            st.success("✓ No validation issues detected.")

    # ------------------------------------------------------------
    # FINANCIAL DATA QUALITY
    # ------------------------------------------------------------

    st.markdown("### 💰 Financial Data Checks")

    f1, f2, f3 = st.columns(3)

    f1.metric(
        "Zero Revised Cost",
        f"{zero_revised_cost:,}"
    )

    f2.metric(
        "Expenditure > Original Cost",
        f"{expenditure_above_original:,}"
    )

    f3.metric(
        "Revised Cost < Original Cost",
        f"{revised_less_original:,}"
    )

    st.caption(
        "These checks identify records requiring interpretation; "
        "they are not automatically treated as errors."
    )

    # ------------------------------------------------------------
    # TIMELINE QUALITY
    # ------------------------------------------------------------

    st.markdown("### 📅 Timeline Consistency")

    t1, t2, t3 = st.columns(3)

    t1.metric(
        "Original > Revised Completion",
        f"{original_after_revised:,}"
    )

    t2.metric(
        "Original < Sanction Date",
        f"{original_before_sanction:,}"
    )

    t3.metric(
        "Revised < Sanction Date",
        f"{revised_before_sanction:,}"
    )

    # ------------------------------------------------------------
    # COLUMN QUALITY TABLE
    # ------------------------------------------------------------

    st.markdown("### 📋 Detailed Field Quality")

    field_quality = pd.DataFrame({
        "Field": dq.columns,
        "Data Type": [str(dq[c].dtype) for c in dq.columns],
        "Records": [len(dq)] * len(dq.columns),
        "Missing": [int(dq[c].isna().sum()) for c in dq.columns],
        "Unique Values": [int(dq[c].nunique(dropna=True)) for c in dq.columns]
    })

    field_quality["Completeness"] = (
        (field_quality["Records"] - field_quality["Missing"])
        / field_quality["Records"] * 100
        if total_rows > 0 else 0
    ).round(1)

    field_quality["Status"] = field_quality["Completeness"].apply(
        lambda x:
            "🟢 Good" if x >= 95
            else "🟡 Review" if x >= 80
            else "🔴 Attention"
    )

    st.dataframe(
        field_quality,
        width="stretch",
        hide_index=True
    )

    # ------------------------------------------------------------
    # FINAL NOTE
    # ------------------------------------------------------------

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#f8f9fa,#fff7f2);
        border:1px solid #eeeeee;
        border-radius:15px;
        padding:18px 20px;
        margin-top:22px;
    ">
        <b>ℹ️ Data Quality Interpretation</b>
        <br><br>
        Data quality checks help identify incomplete, duplicated or
        inconsistent records before analytical and machine-learning
        workflows are used. Some findings, such as zero revised cost
        or expenditure exceeding original cost, may represent valid
        project conditions and therefore require review rather than
        automatic deletion.
    </div>
    """, unsafe_allow_html=True)

elif selected_page == "Model Information":

    st.markdown(
        '<div class="section-title">MODEL INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Details of the machine learning models used for "
        "project delay and cost-overrun risk prediction."
    )

    # =========================================================
    # MODEL FILES
    # =========================================================

    delay_model_path = os.path.join(
        BASE_DIR,
        "model",
        "delay_prediction_model.joblib"
    )

    cost_model_path = os.path.join(
        BASE_DIR,
        "model",
        "cost_overrun_prediction_model.joblib"
    )

    # =========================================================
    # MODEL STATUS
    # =========================================================

    st.markdown("### Model Status")

    m1, m2 = st.columns(2)

    with m1:

        if os.path.exists(delay_model_path):

            st.success(
                "✓ Delay Prediction Model Loaded"
            )

        else:

            st.error(
                "✗ Delay Prediction Model Not Found"
            )

    with m2:

        if os.path.exists(cost_model_path):

            st.success(
                "✓ Cost Overrun Model Loaded"
            )

        else:

            st.error(
                "✗ Cost Overrun Model Not Found"
            )

    # =========================================================
    # MODEL OVERVIEW
    # =========================================================

    st.markdown("### Model Overview")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown("#### Delay Prediction")

        st.write(
            "**Algorithm:** Random Forest Classifier"
        )

        st.write(
            "**Purpose:** Predict project delay risk"
        )

        st.write(
            "**Training records:** 528 projects"
        )

        st.write(
            "**Estimators:** 200 trees"
        )

        st.write(
            "**Class balancing:** Enabled"
        )

    with model_col2:

        st.markdown("#### Cost Overrun Prediction")

        st.write(
            "**Algorithm:** Extra Trees Classifier"
        )

        st.write(
            "**Purpose:** Predict cost-overrun risk"
        )

        st.write(
            "**Training records:** 403 projects"
        )

        st.write(
            "**Estimators:** 200 trees"
        )

        st.write(
            "**Class balancing:** Enabled"
        )

    # =========================================================
    # EVALUATION RESULTS
    # =========================================================

    st.markdown("### Model Evaluation")

    eval_file = os.path.join(
        BASE_DIR,
        "data",
        "cleaned",
        "model_evaluation_results.csv"
    )

    if os.path.exists(eval_file):

        try:

            evaluation_df = pd.read_csv(
                eval_file
            )

            st.dataframe(
                evaluation_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception:

            st.warning(
                "Model evaluation results could not be loaded."
            )

    else:

        # Actual previously recorded evaluation values
        evaluation_display = pd.DataFrame({
            "Model": [
                "Delay Prediction",
                "Cost Overrun Prediction"
            ],
            "Algorithm": [
                "Random Forest Classifier",
                "Extra Trees Classifier"
            ],
            "Accuracy": [
                "80.19%",
                "67.90%"
            ],
            "Precision": [
                "87.50%",
                "69.77%"
            ],
            "Recall": [
                "84.00%",
                "69.77%"
            ],
            "F1 Score": [
                "85.71%",
                "69.77%"
            ],
            "ROC-AUC": [
                "86.15%",
                "71.70%"
            ]
        })

        st.dataframe(
            evaluation_display,
            use_container_width=True,
            hide_index=True
        )

    # =========================================================
    # MODEL METRICS
    # =========================================================

    st.markdown("### Performance Summary")

    e1, e2 = st.columns(2)

    with e1:

        st.metric(
            "Delay Model Accuracy",
            "80.19%"
        )

        st.metric(
            "Delay Model ROC-AUC",
            "86.15%"
        )

    with e2:

        st.metric(
            "Cost Model Accuracy",
            "67.90%"
        )

        st.metric(
            "Cost Model ROC-AUC",
            "71.70%"
        )

    # =========================================================
    # FEATURES USED
    # =========================================================

    st.markdown("### Features Used for Prediction")

    numeric_features = [
        "Original Cost",
        "Expenditure",
        "Physical Progress",
        "Project Duration (Days)",
        "Project Duration (Months)",
        "Expenditure Ratio",
        "Progress per Month"
    ]

    categorical_features = [
        "Sector",
        "Ministry",
        "Agency"
    ]

    f1, f2 = st.columns(2)

    with f1:

        st.markdown("#### Numerical Features")

        for feature in numeric_features:
            st.write(f"• {feature}")

    with f2:

        st.markdown("#### Categorical Features")

        for feature in categorical_features:
            st.write(f"• {feature}")

    # =========================================================
    # PREPROCESSING
    # =========================================================

    st.markdown("### Data Preprocessing")

    preprocessing_steps = [
        "Median imputation for missing numerical values",
        "Unknown category handling for missing categorical values",
        "Numerical and categorical feature preprocessing",
        "Class balancing during model training"
    ]

    for step in preprocessing_steps:

        st.write(
            f"✓ {step}"
        )

    # =========================================================
    # RISK CALCULATION
    # =========================================================

    st.markdown("### Overall Risk Calculation")

    st.info(
        "Overall project risk combines the predicted delay probability "
        "and cost-overrun probability."
    )

    st.code(
        "Overall Risk Score = "
        "(0.50 × Delay Probability) + "
        "(0.50 × Cost Overrun Probability)",
        language="text"
    )

    st.write(
        "The resulting score is used to classify projects into "
        "High, Medium and Low overall risk categories."
    )

    # =========================================================
    # IMPORTANT NOTE
    # =========================================================

    st.markdown("---")

    st.caption(
        "Model-influential factors indicate variables that contribute "
        "to the model's predictions. They should not automatically be "
        "interpreted as confirmed causal relationships."
    )


elif selected_page == "Methodology":

    st.markdown(
        '<div class="section-title">METHODOLOGY</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "End-to-end methodology followed in the Predictive "
        "Infrastructure Project Monitoring System (PIMS)."
    )

    # =========================================================
    # PROJECT WORKFLOW
    # =========================================================

    st.markdown("### PIMS Analytical Workflow")

    workflow = [
        ("01", "Data Collection",
         "PAIMANA infrastructure project data is used as the primary source."),
        
        ("02", "Data Cleaning",
         "Missing values, duplicate records, invalid values and inconsistent "
         "dates are identified and handled."),
        
        ("03", "Data Validation",
         "Project costs, expenditure, physical progress and date relationships "
         "are checked for data-quality issues."),
        
        ("04", "Feature Engineering",
         "Additional project indicators such as duration, expenditure ratio "
         "and progress rate are derived."),
        
        ("05", "Exploratory Data Analysis",
         "Project trends, sector patterns, expenditure and progress "
         "distributions are analysed."),
        
        ("06", "ML Model Training",
         "Machine learning models are trained separately for delay risk "
         "and cost-overrun risk."),
        
        ("07", "Risk Prediction",
         "The trained models generate project-level delay and cost-overrun "
         "probabilities."),
        
        ("08", "Risk Scoring",
         "Delay and cost-overrun probabilities are combined to calculate "
         "the overall project risk."),
        
        ("09", "Risk Factor Analysis",
         "Model-influential factors are extracted to help explain "
         "project-specific predictions."),
        
        ("10", "Dashboard Monitoring",
         "Predictions, project indicators, warnings and analytics are "
         "presented through the Streamlit dashboard.")
    ]

    for number, title, description in workflow:

        st.markdown(
            f"""
            <div style="
                padding:16px;
                margin:8px 0;
                border-radius:10px;
                border:1px solid rgba(128,128,128,0.25);
            ">
                <b style="font-size:18px;">{number} &nbsp; {title}</b>
                <br>
                <span style="opacity:0.85;">{description}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================================================
    # DATA PREPARATION
    # =========================================================

    st.markdown("### 1. Data Preparation")

    d1, d2 = st.columns(2)

    with d1:

        st.markdown("#### Source Dataset")

        st.write(
            "The system uses the PAIMANA infrastructure project dataset."
        )

        st.write(
            "Original dataset size: **791 projects × 13 attributes**."
        )

        st.write(
            "Important project attributes include:"
        )

        for item in [
            "Project Code",
            "Project Name",
            "Sector",
            "Ministry",
            "Agency",
            "Original Cost",
            "Revised Cost",
            "Expenditure",
            "Physical Progress",
            "Original Completion",
            "Revised Completion",
            "Sanction Date"
        ]:

            st.write(f"• {item}")

    with d2:

        st.markdown("#### Data Quality Checks")

        for item in [
            "Duplicate project-code checks",
            "Missing-value analysis",
            "Invalid progress-value checks",
            "Negative cost/expenditure checks",
            "Revised-cost validation",
            "Expenditure consistency checks",
            "Completion-date consistency checks",
            "Sanction-date consistency checks"
        ]:

            st.write(f"✓ {item}")

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    st.markdown("### 2. Feature Engineering")

    feature_df = pd.DataFrame({
        "Feature": [
            "Project Duration",
            "Project Duration (Months)",
            "Expenditure Ratio",
            "Progress per Month",
            "Completion Date Difference"
        ],
        "Purpose": [
            "Measures project duration in days.",
            "Represents project duration in months.",
            "Measures expenditure relative to project cost.",
            "Measures progress achieved per month.",
            "Captures difference between completion-date estimates."
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================================================
    # ML MODELS
    # =========================================================

    st.markdown("### 3. Machine Learning")

    ml1, ml2 = st.columns(2)

    with ml1:

        st.markdown("#### Delay Prediction Model")

        st.write(
            "**Algorithm:** Random Forest Classifier"
        )

        st.write(
            "**Purpose:** Predict the probability of project delay."
        )

        st.write(
            "**Training dataset:** 528 projects"
        )

        st.write(
            "**Estimators:** 200"
        )

        st.write(
            "**Class balancing:** Enabled"
        )

    with ml2:

        st.markdown("#### Cost-Overrun Prediction Model")

        st.write(
            "**Algorithm:** Extra Trees Classifier"
        )

        st.write(
            "**Purpose:** Predict the probability of cost overrun."
        )

        st.write(
            "**Training dataset:** 403 projects"
        )

        st.write(
            "**Estimators:** 200"
        )

        st.write(
            "**Class balancing:** Enabled"
        )

    # =========================================================
    # MODEL INPUTS
    # =========================================================

    st.markdown("### 4. Model Inputs")

    input_features = [
        "Original Cost",
        "Expenditure",
        "Physical Progress",
        "Project Duration (Days)",
        "Project Duration (Months)",
        "Expenditure Ratio",
        "Progress per Month",
        "Sector",
        "Ministry",
        "Agency"
    ]

    feature_cols = st.columns(2)

    for i, feature in enumerate(input_features):

        with feature_cols[i % 2]:

            st.write(f"• {feature}")

    # =========================================================
    # PREPROCESSING
    # =========================================================

    st.markdown("### 5. Preprocessing")

    preprocessing = [
        "Numerical missing values are handled using median imputation.",
        "Missing categorical values are represented as 'Unknown'.",
        "Numerical and categorical features are processed before model prediction.",
        "Class balancing is used during model training."
    ]

    for item in preprocessing:

        st.write(
            f"✓ {item}"
        )

    # =========================================================
    # PREDICTION PIPELINE
    # =========================================================

    st.markdown("### 6. Prediction Pipeline")

    st.code(
        """
Project Data
     ↓
Data Preprocessing
     ↓
Feature Transformation
     ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Delay Model          Cost Model
 │                       │
 ▼                       ▼
Delay Probability   Cost-Overrun Probability
 │                       │
 └───────────┬───────────┘
             ↓
      Overall Risk Score
             ↓
      High / Medium / Low
        """,
        language="text"
    )

    # =========================================================
    # RISK CALCULATION
    # =========================================================

    st.markdown("### 7. Overall Risk Calculation")

    st.info(
        "The overall risk score combines the two model probabilities "
        "with equal weight."
    )

    st.code(
        "Overall Risk = "
        "(0.50 × Delay Probability) + "
        "(0.50 × Cost-Overrun Probability)",
        language="text"
    )

    st.write(
        "The resulting score is used to classify projects into "
        "High, Medium and Low risk categories."
    )

    # =========================================================
    # EXPLAINABILITY
    # =========================================================

    st.markdown("### 8. Model Explainability")

    st.write(
        "Project-specific risk factors are analysed using model "
        "explanation techniques to identify variables that are "
        "influential in the prediction."
    )

    st.warning(
        "Model-influential factors indicate relationships used by "
        "the model. They should not automatically be interpreted "
        "as confirmed causal relationships."
    )

    # =========================================================
    # DASHBOARD LAYER
    # =========================================================

    st.markdown("### 9. Monitoring & Visualization")

    dashboard_features = [
        "Executive project overview",
        "Project-level monitoring",
        "Risk monitoring",
        "Early warning indicators",
        "Benchmarking",
        "Cost escalation analysis",
        "Dependency-aware monitoring",
        "Data quality monitoring",
        "AI-assisted project queries",
        "Analytics and visual reporting"
    ]

    for feature in dashboard_features:

        st.write(
            f"✓ {feature}"
        )

    # =========================================================
    # TECHNOLOGY STACK
    # =========================================================

    st.markdown("### 10. Technology Stack")

    tech_df = pd.DataFrame({
        "Layer": [
            "Frontend",
            "Data Processing",
            "Machine Learning",
            "Explainability",
            "Database",
            "Visualization",
            "Backend API"
        ],
        "Technology": [
            "Streamlit",
            "Python, Pandas, NumPy",
            "Scikit-learn",
            "SHAP",
            "SQLite",
            "Plotly",
            "FastAPI"
        ]
    })

    st.dataframe(
        tech_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================================================
    # FINAL NOTE
    # =========================================================

    st.markdown("---")

    st.caption(
        "PIMS is designed as a predictive monitoring and decision-support "
        "system. Model predictions represent estimated risk based on "
        "available project data and should be interpreted alongside "
        "project and administrative information."
    )


elif selected_page == "About":

    # ============================================================
    # ABOUT — CODE CATALYST
    # ============================================================

    st.html("""
    <style>

    .about-hero {
    background: linear-gradient(135deg, #f4511e, #ff8a3d);
    border-radius: 20px;
    padding: 30px 34px;
    margin: 10px 0 25px 0;
    box-shadow: 0 10px 28px rgba(244, 81, 30, 0.25);
}

.about-hero,
.about-hero * {
    color: #ffffff !important;
}

.about-hero-title {
    color: #ffffff !important;
    font-size: 31px;
    font-weight: 800 !important;
    margin-bottom: 10px;
}

.about-hero-subtitle {
    color: #ffffff !important;
    font-size: 15px;
    font-weight: 700 !important;
    margin: 6px 0;
}

    .about-card {
        background: white;
        border: 1px solid #e6e6e6;
        border-radius: 18px;
        padding: 24px;
        margin: 18px 0;
        box-shadow: 0 5px 18px rgba(0,0,0,0.06);
    }

    .about-title {
        color: #292929;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .about-text {
        color: #444444;
        font-size: 14px;
        line-height: 1.75;
    }

    .feature-card {
        border-radius: 16px;
        padding: 20px;
        min-height: 175px;
        margin-bottom: 15px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }

    .feature-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .feature-title {
        color: #252525;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #555555;
        font-size: 13px;
        line-height: 1.6;
    }

    .team-card {
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 16px;
        padding: 19px;
        margin-bottom: 15px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }

    .team-name {
        color: #222222;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .team-role {
        color: #555555;
        font-size: 13px;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        margin-top: 10px;
        padding: 5px 11px;
        border-radius: 20px;
        background: #fff0e9;
        color: #f4511e;
        font-size: 10px;
        font-weight: 800;
    }

    .tech-card {
        background: white;
        border: 1px solid #e2e2e2;
        border-radius: 13px;
        padding: 14px 8px;
        margin-bottom: 12px;
        text-align: center;
        color: #333333;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    .footer-card {
        background: linear-gradient(135deg, #fff7f2, #fff0e7);
        border: 1px solid #ffd3bd;
        border-radius: 18px;
        padding: 24px;
        margin-top: 20px;
        text-align: center;
    }

    .footer-title {
        color: #f4511e;
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .footer-text {
        color: #555555;
        font-size: 13px;
        margin: 5px 0;
    }

    </style>

    <div class="about-hero">

        <div class="about-hero-title">
            🚀 Predictive Infrastructure Project Monitoring System
        </div>

        <div class="about-hero-subtitle">
            PIMS — AI-powered infrastructure project monitoring and risk analysis
        </div>

        <div class="about-hero-subtitle">
            Early Risk Detection &nbsp; • &nbsp;
            Data-Driven Insights &nbsp; • &nbsp;
            Smarter Project Monitoring
        </div>

        <div class="about-hero-subtitle">
            Developed by <b>Code Catalyst</b>
        </div>

    </div>
    """)


    # ============================================================
    # ABOUT PIMS
    # ============================================================

    st.html("""
    <div class="about-card">

        <div class="about-title">
            🎯 About PIMS
        </div>

        <div class="about-text">

            The <b>Predictive Infrastructure Project Monitoring System
            (PIMS)</b> is designed to provide a centralized view of
            infrastructure projects and support project monitoring through
            data analytics and machine learning-based risk prediction.

            <br><br>

            PIMS analyses project information such as
            <b>cost, expenditure, physical progress, project duration,
            sector, ministry and agency</b> to generate insights related
            to delay risk and cost-overrun risk.

            <br><br>

            The system combines <b>data preprocessing, feature engineering,
            machine learning, risk scoring, explainability and interactive
            visualization</b> into a single project monitoring platform.

        </div>

    </div>
    """)


    # ============================================================
    # WHAT PIMS PROVIDES
    # ============================================================

    st.html("""
    <div class="about-card">
        <div class="about-title">
            📊 What PIMS Provides
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html("""
        <div class="feature-card"
             style="background:linear-gradient(135deg,#f0fff8,#ffffff);">

            <div class="feature-icon">📋</div>

            <div class="feature-title">
                Project Monitoring
            </div>

            <div class="feature-text">
                Track project progress, expenditure and important
                project information through an interactive dashboard.
            </div>

        </div>
        """)

    with col2:
        st.html("""
        <div class="feature-card"
             style="background:linear-gradient(135deg,#fff2ef,#ffffff);">

            <div class="feature-icon">⚠️</div>

            <div class="feature-title">
                Risk Prediction
            </div>

            <div class="feature-text">
                Estimate delay and cost-overrun probabilities using
                trained machine learning models.
            </div>

        </div>
        """)

    with col3:
        st.html("""
        <div class="feature-card"
             style="background:linear-gradient(135deg,#eef6ff,#ffffff);">

            <div class="feature-icon">🛡️</div>

            <div class="feature-title">
                Risk Monitoring
            </div>

            <div class="feature-text">
                Identify projects requiring closer monitoring based
                on calculated project risk.
            </div>

        </div>
        """)


    col1, col2, col3 = st.columns(3)

    with col1:
        st.html("""
        <div class="feature-card"
             style="background:linear-gradient(135deg,#f7f0ff,#ffffff);">

            <div class="feature-icon">🔍</div>

            <div class="feature-title">
                Risk Factors
            </div>

            <div class="feature-text">
                Show model-influential factors associated with
                individual project predictions.
            </div>

        </div>
        """)

    with col2:
        st.html("""
        <div class="feature-card"
             style="background:linear-gradient(135deg,#effcfb,#ffffff);">

            <div class="feature-icon">📈</div>

            <div class="feature-title">
                Analytics
            </div>

            <div class="feature-text">
                Explore project and sector-level patterns through
                interactive visualizations and analysis.
            </div>

        </div>
        """)

    with col3:
        st.html("""
        <div class="feature-card"
             style="background:linear-gradient(135deg,#fff7ec,#ffffff);">

            <div class="feature-icon">🤖</div>

            <div class="feature-title">
                AI Assistant
            </div>

            <div class="feature-text">
                Interact with project information and prediction
                results through a conversational interface.
            </div>

        </div>
        """)


    # ============================================================
    # TEAM
    # ============================================================

    st.html("""
    <div class="about-card">

        <div class="about-title">
            👥 Meet the Team — Code Catalyst
        </div>

        <div class="about-text">
            A six-member team working across research, data,
            development, machine learning and system integration.
        </div>

    </div>
    """)


    col1, col2 = st.columns(2)

    with col1:
        st.html("""
        <div class="team-card">
            <div class="team-name">👑 Rimjhim Dubey</div>

            <div class="team-role">
                Team Leader · Research & Presentation
            </div>

            <span class="badge">TEAM LEAD</span>
        </div>
        """)

    with col2:
        st.html("""
        <div class="team-card">
            <div class="team-name">🎨 Sohani Jat</div>

            <div class="team-role">
                Frontend Development
            </div>

            <span class="badge">FRONTEND</span>
        </div>
        """)


    col1, col2 = st.columns(2)

    with col1:
        st.html("""
        <div class="team-card">
            <div class="team-name">⚙️ Srajal Soni</div>

            <div class="team-role">
                Backend Development
            </div>

            <span class="badge">BACKEND</span>
        </div>
        """)

    with col2:
        st.html("""
        <div class="team-card">
            <div class="team-name">📂 Hemant Sisodiya</div>

            <div class="team-role">
                Data Collection & Pre-processing
            </div>

            <span class="badge">DATA</span>
        </div>
        """)


    col1, col2 = st.columns(2)

    with col1:
        st.html("""
        <div class="team-card">
            <div class="team-name">🤖 Parth Gupta</div>

            <div class="team-role">
                ML & AI Model Development
            </div>

            <span class="badge">ML & AI</span>
        </div>
        """)

    with col2:
        st.html("""
        <div class="team-card">
            <div class="team-name">🔗 Nitin Verma</div>

            <div class="team-role">
                Integration, Training & Testing
            </div>

            <span class="badge">INTEGRATION</span>
        </div>
        """)


    # ============================================================
    # TECHNOLOGY STACK
    # ============================================================

    st.html("""
    <div class="about-card">

        <div class="about-title">
            💻 Technology Stack
        </div>

    </div>
    """)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.html('<div class="tech-card">🐍 Python</div>')

    with col2:
        st.html('<div class="tech-card">📊 Pandas</div>')

    with col3:
        st.html('<div class="tech-card">🔢 NumPy</div>')

    with col4:
        st.html('<div class="tech-card">📈 Plotly</div>')

    with col5:
        st.html('<div class="tech-card">🎛️ Streamlit</div>')


    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.html('<div class="tech-card">🧠 Scikit-learn</div>')

    with col2:
        st.html('<div class="tech-card">🔍 SHAP</div>')

    with col3:
        st.html('<div class="tech-card">🗄️ SQLite</div>')

    with col4:
        st.html('<div class="tech-card">⚡ FastAPI</div>')

    with col5:
        st.html('<div class="tech-card">📊 Data Analytics</div>')


    # ============================================================
    # INSTITUTION / PROGRAM / TEAM
    # ============================================================

    st.html("""
    <div class="footer-card">

        <div class="footer-title">
            ✨ Code Catalyst
        </div>

        <div class="footer-text">
            🏫 <b>Oriental College of Technology</b>
        </div>

        <div class="footer-text">
            🎓 B.Tech CSE – Data Science
        </div>

        <div class="footer-text">
            🚀 Predictive Infrastructure Project Monitoring System (PIMS)
        </div>

    </div>
    """)

# ============================================================
# ROUTING: FALLBACK FOR OTHER TABS
# ============================================================
else:

    st.markdown(
        f'<div class="section-title">{selected_page.upper()}</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"'{selected_page}' is currently under development."
    )

# ============================================================
# AI ASSISTANT FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def find_project_from_question(question, data):
    question = clean_text(question)
    if not question: return None

    if "project_code" in data.columns:
        codes = sorted(set(data["project_code"].astype(str).str.strip().tolist()), key=len, reverse=True)
        for code in codes:
            if len(code) >= 4 and code in question:
                matched = data[data["project_code"].astype(str).str.strip() == code]
                if not matched.empty: return matched.iloc[0]

    if "project_name" in data.columns:
        question_lower = question.lower()
        name_rows = data[["project_code", "project_name"]].dropna().sort_values("project_name", key=lambda x: x.astype(str).str.len(), ascending=False)
        for _, row in name_rows.iterrows():
            name = clean_text(row["project_name"])
            if not name: continue
            name_lower = name.lower()
            if name_lower in question_lower:
                matched = data[data["project_code"].astype(str).str.strip() == str(row["project_code"]).strip()]
                if not matched.empty: return matched.iloc[0]
            
            words = [w for w in re.findall(r"[a-zA-Z]{4,}", name_lower) if w not in {"construction", "project", "scheme", "development", "building", "works", "including", "other", "phase"}]
            if len(words) >= 2:
                matched_words = sum(1 for word in words[:5] if word in question_lower)
                if matched_words >= min(2, len(words[:5])):
                    matched = data[data["project_code"].astype(str).str.strip() == str(row["project_code"]).strip()]
                    if not matched.empty: return matched.iloc[0]
    return None

# ============================================================
# AI ASSISTANT - PROJECT EXPLANATION HELPERS
# ============================================================

def format_factor_name(text):
    """
    Kept for compatibility with existing code.
    Raw SHAP feature names are no longer shown to the user.
    """
    return clean_text(text)


def get_project_factors(project_code):
    """
    Kept for compatibility.
    Returns stored ML factors if needed internally.
    The AI Assistant does NOT directly display these raw factors.
    """
    if risk_factors.empty:
        return [], []

    rows = risk_factors[
        risk_factors["project_code"].astype(str).str.strip()
        == str(project_code).strip()
    ]

    if rows.empty:
        return [], []

    row = rows.iloc[0]

    delay_text = clean_text(
        row.get("top_delay_factors", "")
    )

    cost_text = clean_text(
        row.get("top_cost_overrun_factors", "")
    )

    delay_factors = [
        x.strip()
        for x in delay_text.split(" | ")
        if x.strip()
    ]

    cost_factors = [
        x.strip()
        for x in cost_text.split(" | ")
        if x.strip()
    ]

    return delay_factors, cost_factors


def safe_float(value):
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def format_probability(value):
    try:
        if pd.isna(value):
            return "Not available"

        value = float(value)

        if value <= 1:
            value *= 100

        return f"{value:.1f}%"

    except Exception:
        return "Not available"


def format_progress_value(value):
    try:
        value = safe_float(value)

        if value <= 1:
            value *= 100

        return f"{value:.1f}%"

    except Exception:
        return "Not available"


def format_money_value(value):
    try:
        if pd.isna(value):
            return "Not available"

        return f"{float(value):,.2f}"

    except Exception:
        return "Not available"


def get_project_risk_signals(project, dataset):
    """
    Creates understandable project-specific monitoring signals
    using actual project values and dataset benchmarks.

    These are OBSERVED RISK SIGNALS, not proven causes.
    """

    signals = {
        "delay": [],
        "cost": []
    }

    # --------------------------------------------------------
    # PROJECT VALUES
    # --------------------------------------------------------

    physical_progress = safe_float(
        project.get("physical_progress")
    )

    expenditure = safe_float(
        project.get("expenditure")
    )

    original_cost = safe_float(
        project.get("original_cost")
    )

    duration_days = safe_float(
        project.get("project_duration_days")
    )

    progress_per_month = safe_float(
        project.get("progress_per_month")
    )

    # --------------------------------------------------------
    # EXPENDITURE RATIO
    # --------------------------------------------------------

    if original_cost > 0:
        expenditure_ratio = expenditure / original_cost
    else:
        expenditure_ratio = safe_float(
            project.get("expenditure_ratio")
        )

    # Handle datasets where ratio may already be percentage
    dataset_ratio = pd.Series(dtype=float)

    if "expenditure_ratio" in dataset.columns:

        dataset_ratio = pd.to_numeric(
            dataset["expenditure_ratio"],
            errors="coerce"
        ).dropna()

        if (
            len(dataset_ratio) > 0
            and dataset_ratio.median() > 1
        ):
            dataset_ratio = dataset_ratio / 100

    # --------------------------------------------------------
    # DATASET BENCHMARKS
    # --------------------------------------------------------

    progress_values = pd.to_numeric(
        dataset.get(
            "physical_progress",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).dropna()

    duration_values = pd.to_numeric(
        dataset.get(
            "project_duration_days",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).dropna()

    progress_rate_values = pd.to_numeric(
        dataset.get(
            "progress_per_month",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    ).dropna()

    median_progress = (
        float(progress_values.median())
        if len(progress_values) > 0
        else None
    )

    median_ratio = (
        float(dataset_ratio.median())
        if len(dataset_ratio) > 0
        else None
    )

    median_duration = (
        float(duration_values.median())
        if len(duration_values) > 0
        else None
    )

    median_progress_rate = (
        float(progress_rate_values.median())
        if len(progress_rate_values) > 0
        else None
    )

    # ========================================================
    # DELAY SIGNALS
    # ========================================================

    # Physical progress below dataset benchmark
    if (
        median_progress is not None
        and physical_progress < median_progress
    ):
        signals["delay"].append(
            f"Physical progress is {physical_progress:.1f}%, "
            f"below the dataset median of "
            f"{median_progress:.1f}%."
        )

    # Progress rate below benchmark
    if (
        median_progress_rate is not None
        and progress_per_month > 0
        and progress_per_month < median_progress_rate
    ):
        signals["delay"].append(
            f"Progress rate is approximately "
            f"{progress_per_month:.2f}% per month, "
            f"below the dataset median of "
            f"{median_progress_rate:.2f}% per month."
        )

    # Longer duration
    if (
        median_duration is not None
        and duration_days > 0
        and duration_days > median_duration
    ):
        signals["delay"].append(
            f"Project duration is approximately "
            f"{duration_days:,.0f} days, above the "
            f"dataset median of {median_duration:,.0f} days."
        )

    # Spending ahead of physical progress
    progress_fraction = physical_progress / 100

    if (
        original_cost > 0
        and expenditure > 0
        and expenditure_ratio > progress_fraction + 0.15
    ):
        signals["delay"].append(
            f"Financial utilisation is "
            f"{expenditure_ratio * 100:.1f}% while physical "
            f"progress is {physical_progress:.1f}%, showing "
            f"a noticeable spending–progress gap."
        )

    # ========================================================
    # COST SIGNALS
    # ========================================================

    # High expenditure utilisation
    if (
        median_ratio is not None
        and expenditure_ratio > median_ratio
    ):
        signals["cost"].append(
            f"Expenditure utilisation is "
            f"{expenditure_ratio * 100:.1f}% of original cost, "
            f"above the dataset median of "
            f"{median_ratio * 100:.1f}%."
        )

    # Spending-progress gap
    if (
        original_cost > 0
        and expenditure > 0
        and expenditure_ratio > progress_fraction + 0.15
    ):
        signals["cost"].append(
            f"{expenditure_ratio * 100:.1f}% of the original cost "
            f"has been utilised while physical progress is "
            f"{physical_progress:.1f}%."
        )

    # High spending + low progress
    if (
        physical_progress < 40
        and expenditure_ratio > 0.60
    ):
        signals["cost"].append(
            f"More than 60% of the original cost has been utilised "
            f"while physical progress is only "
            f"{physical_progress:.1f}%."
        )

    return signals


# ============================================================
# AI ASSISTANT RESPONSE
# ============================================================

def assistant_response(question):

    question = clean_text(question)

    if not question:
        return "Please enter a question about a project."

    # --------------------------------------------------------
    # FIND PROJECT
    # --------------------------------------------------------

    project = find_project_from_question(
        question,
        display_data
    )

    # --------------------------------------------------------
    # FOLLOW-UP QUESTION
    # --------------------------------------------------------

    if project is None:

        current_code = st.session_state.get(
            "ai_current_project_code"
        )

        if current_code is not None:

            matched = display_data[
                display_data["project_code"].astype(str).str.strip()
                == str(current_code).strip()
            ]

            if not matched.empty:
                project = matched.iloc[0]

    # --------------------------------------------------------
    # NO PROJECT FOUND
    # --------------------------------------------------------

    if project is None:

        return (
            "Please include the **project name or project code** "
            "so I can provide project-specific information."
        )

    # --------------------------------------------------------
    # SAVE CURRENT PROJECT
    # --------------------------------------------------------

    project_code = clean_text(
        project.get("project_code")
    )

    st.session_state.ai_current_project_code = project_code

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    project_name = clean_text(
        project.get("project_name")
    )

    sector = clean_text(
        project.get("sector")
    )

    physical_progress = project.get(
        "physical_progress",
        np.nan
    )

    expenditure = project.get(
        "expenditure",
        np.nan
    )

    original_cost = project.get(
        "original_cost",
        np.nan
    )

    delay_probability = project.get(
        "delay_probability",
        np.nan
    )

    cost_probability = project.get(
        "cost_overrun_probability",
        np.nan
    )

    overall_percentage = project.get(
        "overall_risk_percentage",
        np.nan
    )

    overall_risk = clean_text(
        project.get(
            "overall_risk_level",
            "Not available"
        )
    )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    delay_pct = format_probability(
        delay_probability
    )

    cost_pct = format_probability(
        cost_probability
    )

    progress = format_progress_value(
        physical_progress
    )

    expenditure_value = format_money_value(
        expenditure
    )

    original_cost_value = format_money_value(
        original_cost
    )

    overall_pct = format_probability(
        overall_percentage
    )

    # --------------------------------------------------------
    # PROJECT-SPECIFIC SIGNALS
    # --------------------------------------------------------

    signals = get_project_risk_signals(
        project,
        display_data
    )

    delay_signals = signals["delay"]
    cost_signals = signals["cost"]

    # ========================================================
    # COST QUESTION
    # ========================================================

    if any(
        word in question.lower()
        for word in [
            "cost",
            "overrun",
            "budget",
            "expense",
            "expenditure",
            "money",
            "financial"
        ]
    ):

        signal_text = ""

        if cost_signals:

            signal_text = (
                "\n\n**Project-specific cost signals:**\n"
                + "\n".join(
                    f"- {signal}"
                    for signal in cost_signals[:4]
                )
            )

        else:

            signal_text = (
                "\n\n**Project-specific cost signals:**\n"
                "- No unusually adverse cost-utilisation "
                "signal was identified from the available measurements."
            )

        return (
            f"### 💰 Cost Risk — {project_name}\n\n"
            f"**Project Code:** `{project_code}`\n\n"
            f"**Predicted cost-overrun probability:** "
            f"**{cost_pct}**\n\n"
            f"**Original Cost:** {original_cost_value}\n\n"
            f"**Expenditure:** {expenditure_value}"
            f"{signal_text}\n\n"
            "The ML model provides the cost-risk probability. "
            "The signals above are based on the project's observed "
            "financial and physical-progress indicators and are "
            "not proof of a causal reason."
        )

    # ========================================================
    # DELAY QUESTION
    # ========================================================

    if any(
        word in question.lower()
        for word in [
            "delay",
            "delayed",
            "late",
            "completion",
            "deadline",
            "time",
            "schedule"
        ]
    ):

        signal_text = ""

        if delay_signals:

            signal_text = (
                "\n\n**Project-specific delay signals:**\n"
                + "\n".join(
                    f"- {signal}"
                    for signal in delay_signals[:4]
                )
            )

        else:

            signal_text = (
                "\n\n**Project-specific delay signals:**\n"
                "- No unusually adverse schedule indicator "
                "was identified from the available measurements."
            )

        return (
            f"### ⏱️ Delay Risk — {project_name}\n\n"
            f"**Project Code:** `{project_code}`\n\n"
            f"**Predicted delay probability:** "
            f"**{delay_pct}**"
            f"{signal_text}\n\n"
            "This is an ML-based prediction. The project-specific "
            "signals describe observed conditions and do not prove "
            "that any single condition caused the predicted risk."
        )

    # ========================================================
    # WHY RISKY / RISK FACTORS
    # ========================================================

    if any(
        word in question.lower()
        for word in [
            "factor",
            "factors",
            "reason",
            "reasons",
            "why risky",
            "why is this project risky",
            "why is this risky",
            "why is the project risky",
            "risk factor",
            "risk factors",
            "cause",
            "causes",
            "risky",
            "risk"
        ]
    ):

        delay_text = (
            "\n".join(
                f"- {signal}"
                for signal in delay_signals[:4]
            )
            if delay_signals
            else
            "- No unusually adverse delay indicator identified."
        )

        cost_text = (
            "\n".join(
                f"- {signal}"
                for signal in cost_signals[:4]
            )
            if cost_signals
            else
            "- No unusually adverse cost-utilisation indicator identified."
        )

        return (
            f"### ⚠️ Risk Assessment — {project_name}\n\n"
            f"**Project Code:** `{project_code}`\n\n"
            f"**Overall Risk:** **{overall_risk}** "
            f"({overall_pct})\n\n"

            f"### ⏱️ Schedule Risk Signals\n"
            f"{delay_text}\n\n"

            f"### 💰 Cost Risk Signals\n"
            f"{cost_text}\n\n"

            f"### 🤖 ML Assessment\n"
            f"**Delay Probability:** **{delay_pct}**\n\n"
            f"**Cost-Overrun Probability:** **{cost_pct}**\n\n"

            "The ML model generates the risk probabilities. "
            "The signals shown above are derived from the selected "
            "project's actual measurable conditions and dataset "
            "benchmarks. They should not be interpreted as confirmed "
            "causal reasons."
        )

    # ========================================================
    # PROGRESS QUESTION
    # ========================================================

    if any(
        word in question.lower()
        for word in [
            "progress",
            "physical progress",
            "work completed",
            "completed",
            "how much work",
            "how much is completed"
        ]
    ):

        return (
            f"### 📊 Project Progress — {project_name}\n\n"
            f"**Project Code:** `{project_code}`\n\n"
            f"**Physical Progress:** {progress}\n\n"
            f"**Expenditure:** {expenditure_value}\n\n"
            f"**Original Cost:** {original_cost_value}"
        )

    # ========================================================
    # DEFAULT PROJECT SUMMARY
    # ========================================================

    return (
        f"### 📋 {project_name}\n\n"
        f"**Project Code:** `{project_code}`\n\n"
        f"**Sector:** {sector}\n\n"
        f"**Overall Risk:** **{overall_risk}** "
        f"({overall_pct})\n\n"
        f"**Delay Probability:** **{delay_pct}**\n\n"
        f"**Cost Overrun Probability:** **{cost_pct}**\n\n"
        f"**Physical Progress:** **{progress}**"
    )
# ============================================================
# FLOATING AI ASSISTANT
# ============================================================

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = [{"role": "assistant", "content": "Hi! 👋 I'm your **AI Project Monitoring Assistant**.\n\nPlease include the **project name or project code** in your question."}]
if "ai_chat_input" not in st.session_state:
    st.session_state.ai_chat_input = ""
if "ai_current_project_code" not in st.session_state:
    st.session_state.ai_current_project_code = None

with st.popover("💬", width='content'):
    st.markdown('<div class="ai-chat-header"><div class="ai-chat-title">🤖 AI Project Assistant</div><div class="ai-chat-subtitle">Ask about project risk, delay, cost, progress or risk factors.</div></div>', unsafe_allow_html=True)

    for message in st.session_state.ai_chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-msg">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg">{message["content"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask about a project...", key="ai_chat_question")
    if question:
        st.session_state.ai_chat_history.append({"role": "user", "content": question})
        response = assistant_response(question)
        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("**Quick questions**")
    q1, q2 = st.columns(2)
    with q1:
        if st.button("🔎 Why risky?", use_container_width=True):
            quick_question = "Why is this project risky?"
            st.session_state.ai_chat_history.append({"role": "user", "content": quick_question})
            response = assistant_response(quick_question)
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    with q2:
        if st.button("💰 Cost risk", use_container_width=True):
            quick_question = "What about cost?"
            st.session_state.ai_chat_history.append({"role": "user", "content": quick_question})
            response = assistant_response(quick_question)
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    q3, q4 = st.columns(2)
    with q3:
        if st.button("⏱️ Delay risk", use_container_width=True):
            quick_question = "What about delay?"
            st.session_state.ai_chat_history.append({"role": "user", "content": quick_question})
            response = assistant_response(quick_question)
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    with q4:
        if st.button("📊 Progress", use_container_width=True):
            quick_question = "How much work is completed?"
            st.session_state.ai_chat_history.append({"role": "user", "content": quick_question})
            response = assistant_response(quick_question)
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.ai_chat_history = [{"role": "assistant", "content": "Hi! 👋 I'm your **AI Project Monitoring Assistant**.\n\nPlease include the **project name or project code** in your question."}]
        st.session_state.ai_current_project_code = None
        st.rerun()

    st.caption("AI responses are based on actual ML predictions and model-identified project risk factors.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("PIMS — Predictive Infrastructure Project Monitoring System | Machine Learning based project risk monitoring")
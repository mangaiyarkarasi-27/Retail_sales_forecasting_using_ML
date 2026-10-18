"""
dashboard/app.py
----------------
E-IDSS: Intelligent Retail Decision Support System
A unified, interactive software application integrating decision-time-safe 
direct multi-horizon sales forecasting, game-theoretic explainability, 
distribution-free conformal uncertainty calibration, and discrete-event 
continuous-review inventory replenishment simulation.

Launch command:
    streamlit run dashboard/app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ----------------- PATH CONFIGURATION -----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
JOURNAL_ROOT = os.path.join(PROJECT_ROOT, "journal_extension")

# Add directories to sys.path for internal imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import series service and modular view renderers
import series_service
from views.home_view import render_home_view
from views.forecasting_view import render_forecasting_view
from views.inventory_view import render_inventory_view
from views.explainability_view import render_explainability_view
from views.uncertainty_view import render_uncertainty_view
from views.decision_support_view import render_decision_support_view
from views.evaluation_view import render_evaluation_view
from views.validation_view import render_validation_view

# ----------------- STREAMLIT CONFIGURATION -----------------
st.set_page_config(
    page_title="E-IDSS | Intelligent Retail Decision Support System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM STYLING -----------------
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .reportview-container {
        background: #f8fafc;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
    }
    .metric-card.critical { border-left-color: #ef4444; }
    .metric-card.warning  { border-left-color: #f59e0b; }
    .metric-card.success  { border-left-color: #10b981; }
    .metric-card.info     { border-left-color: #3b82f6; }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.15rem 0;
        letter-spacing: -0.03em;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tables and Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SAFE ASSET LOADERS -----------------
@st.cache_data
def load_csv(path: str):
    """Safely loads a CSV into a pandas DataFrame."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error loading CSV {path}: {e}")
    return None

def load_txt(path: str) -> str:
    """Safely loads a text or markdown file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {path}: {e}"
    return f"File not found at {path}."

def load_image(path: str):
    """Safely loads an image file."""
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception as e:
            st.error(f"Error loading image {path}: {e}")
    return None

helpers = {
    "load_csv": load_csv,
    "load_txt": load_txt,
    "load_image": load_image
}

# ----------------- SESSION STATE SYNCHRONIZATION -----------------
if "active_store" not in st.session_state:
    st.session_state["active_store"] = 1

if "active_item" not in st.session_state:
    st.session_state["active_item"] = 1

active_store = st.session_state["active_store"]
active_item = st.session_state["active_item"]

# ----------------- SIDEBAR WORKSPACE CONTROLS -----------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
    <span style="font-size:1.8rem;">🛒</span>
    <div>
        <div style="font-size:1.15rem; font-weight:800; color:#0f172a; line-height:1.2;">E-IDSS Platform</div>
        <div style="font-size:0.75rem; color:#64748b; font-weight:500;">Retail Decision Support</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Global Active Series Workspace Selector
st.sidebar.markdown("### 🎯 Global Series Focus")
sb_store = st.sidebar.selectbox(
    "Active Store:",
    options=list(range(1, 11)),
    index=active_store - 1,
    format_func=lambda x: f"Store {x} (Regional Center)",
    key="sb_global_store"
)
if sb_store != active_store:
    st.session_state["active_store"] = sb_store
    st.rerun()

sb_item = st.sidebar.selectbox(
    "Active Item:",
    options=list(range(1, 51)),
    index=active_item - 1,
    format_func=lambda x: f"Item {x} (SKU #{x:02d})",
    key="sb_global_item"
)
if sb_item != active_item:
    st.session_state["active_item"] = sb_item
    st.rerun()

# Series Mini Snapshot Badge in Sidebar
prof = series_service.get_series_profile(JOURNAL_ROOT, active_store, active_item)
dec = series_service.get_series_decision_record(PROJECT_ROOT, active_store, active_item)
p_rank = dec.get("Priority_Rank", 1)
d_score = dec.get("Decision_Score", 50.0)
sbc_c = prof.get("sbc_category", "Smooth")
m_sales = prof.get("mean_sales", 0.0)

st.sidebar.markdown(f"""
<div style="background:#f1f5f9; border-radius:8px; padding:0.65rem 0.8rem; margin:0.4rem 0 1rem 0; font-size:0.78rem; line-height:1.4; color:#334155;">
    <b>Store {active_store} · Item {active_item}</b> <span style="float:right; color:#0284c7; font-weight:700;">Rank #{p_rank}</span><br>
    Profile: <b>{sbc_c}</b> ({m_sales:.1f} u/d)<br>
    MCDA Score: <b>{d_score:.1f} / 100</b>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation Menu
st.sidebar.markdown("### 🧭 Platform Navigation")
NAV_OPTIONS = [
    "1. Home",
    "2. Demand Forecasting",
    "3. Inventory Intelligence",
    "4. Explainable AI",
    "5. Uncertainty and Risk Analysis",
    "6. Decision Support and Recommendations",
    "7. Performance Evaluation",
    "8. Research Validation"
]

# Synchronize with home page shortcut buttons
default_nav_idx = 0
if "nav_selection" in st.session_state and st.session_state["nav_selection"] in NAV_OPTIONS:
    default_nav_idx = NAV_OPTIONS.index(st.session_state["nav_selection"])

selected_nav = st.sidebar.radio(
    "Select System Section:",
    options=NAV_OPTIONS,
    index=default_nav_idx,
    key="main_nav_radio"
)
st.session_state["nav_selection"] = selected_nav

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem; color:#64748b; line-height:1.5;'>
    <b>System Scope:</b> 500 Retail Time Series<br>
    <b>Observations:</b> 913,000 Historical Records<br>
    <b>Simulation:</b> Discrete-Event (365 Days)<br>
    <b>Status:</b> All Results Audited & Frozen
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN VIEW DISPATCHER -----------------
if selected_nav == "1. Home":
    render_home_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "2. Demand Forecasting":
    render_forecasting_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "3. Inventory Intelligence":
    render_inventory_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "4. Explainable AI":
    render_explainability_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "5. Uncertainty and Risk Analysis":
    render_uncertainty_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "6. Decision Support and Recommendations":
    render_decision_support_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "7. Performance Evaluation":
    render_evaluation_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

elif selected_nav == "8. Research Validation":
    render_validation_view(PROJECT_ROOT, JOURNAL_ROOT, helpers)

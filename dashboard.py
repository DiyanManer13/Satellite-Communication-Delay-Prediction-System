import pickle
import time
import warnings
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from datetime import datetime
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

HAS_MAP = True
try:
    from streamlit_folium import st_folium
    import folium
except ImportError:
    st_folium = None
    folium = None
    HAS_MAP = False

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
MODEL_PKL   = BASE_DIR / "model.pkl"
DATASET_CSV = BASE_DIR / "dataset.csv"

st.set_page_config(
    page_title="Satellite Link Delay Simulator",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: #f5f8fa !important;
    color: #0f1419 !important;
}
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e1e8ed !important;
    width: 380px !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Sidebar text & content visibility ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #0f1419 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ── */
header[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e1e8ed !important;
}
footer { display: none !important; }

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p, label {
    color: #536471 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Radio + Input text styling in sidebar ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] div,
[data-testid="stRadio"] span,
[data-testid="stRadio"] p,
[data-testid="stRadio"] .css-1wlv8u7,
[data-testid="stRadio"] .css-k1ih3n,
[data-testid="stRadio"] .css-1szy77x,
[data-testid="stRadio"] .stRadio > label,
.stRadio label p,
.stRadio span,
div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] p,
/* Catch-all: every p, span, div inside sidebar uses Inter */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] * {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

[data-testid="stRadio"] label p,
div[role="radiogroup"] label p {
    color: #0f1419 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] label {
    color: #0f1419 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 13px !important;
}

/* ── Number input field styling ── */
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid #e1e8ed !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
    color: #0f1419 !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: #1d9bf0 !important;
    box-shadow: 0 0 0 2px rgba(29, 155, 240, 0.1) !important;
}

/* ── Map container & iframe styling ── */
[data-testid="stIFrame"] {
    border-radius: 12px !important;
    border: 1px solid #e1e8ed !important;
    overflow: hidden !important;
}

/* ── Alert/Info/Warning styling ── */
.stAlert, [data-testid="stAlert"] {
    border-left: 4px solid #1d9bf0 !important;
    border-radius: 8px !important;
}

.stWarning, [data-testid="stWarning"] {
    border-left: 4px solid #f5a623 !important;
    border-radius: 8px !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: #e1e8ed !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #1d9bf0 !important;
}

/* ── Column spacing in sidebar ── */
[data-testid="stSidebar"] [data-testid="column"] {
    padding: 0 0.5rem !important;
}

/* ── Button ── */
div.stButton > button {
    background: #1d9bf0 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: background 0.2s, transform 0.1s !important;
    box-shadow: 0 2px 8px rgba(29,155,240,0.25) !important;
}
div.stButton > button:hover {
    background: #1a8cd8 !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #e1e8ed !important;
    color: #0f1419 !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #e1e8ed !important;
    margin: 1rem 0 !important;
}

/* ── Metric card ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e1e8ed;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card:hover {
    border-color: #1d9bf0;
    box-shadow: 0 4px 16px rgba(29,155,240,0.12);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 11px;
    font-weight: 700;
    color: #536471;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #0f1419;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-unit {
    font-size: 13px;
    font-weight: 400;
    color: #536471;
    margin-left: 4px;
}
.metric-sub {
    font-size: 12px;
    color: #8899a6;
    margin-top: 6px;
}

/* ── Hero prediction card ── */
.hero-card {
    background: linear-gradient(135deg, #1d9bf0 0%, #0a6ebd 100%);
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(29,155,240,0.25);
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    color: rgba(255,255,255,0.75);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-value {
    font-size: 4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1;
}
.hero-unit {
    font-size: 1.4rem;
    font-weight: 300;
    color: rgba(255,255,255,0.75);
    margin-left: 6px;
}
.hero-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin-top: 10px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.35);
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Section heading ── */
.section-head {
    font-size: 17px;
    font-weight: 800;
    color: #0f1419;
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e1e8ed;
}

/* ── Breakdown bar ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.bar-label {
    font-size: 12px;
    color: #536471;
    width: 110px;
    flex-shrink: 0;
    font-weight: 600;
}
.bar-track {
    flex: 1;
    height: 8px;
    background: #e1e8ed;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}
.bar-val {
    font-size: 12px;
    font-weight: 700;
    color: #0f1419;
    width: 85px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    font-size: 20px;
    font-weight: 800;
    color: #0f1419;
    margin-bottom: 2px;
    letter-spacing: -0.03em;
}
.sidebar-tagline {
    font-size: 11px;
    color: #8899a6;
    margin-bottom: 1.2rem;
    font-weight: 400;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f5f8fa;
    border: 1px solid #e1e8ed;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    color: #536471;
    margin-right: 6px;
    margin-bottom: 10px;
}
.dot-ok   { width:7px; height:7px; border-radius:50%; background:#00ba7c; display:inline-block; }
.dot-err  { width:7px; height:7px; border-radius:50%; background:#f4212e; display:inline-block; }
.dot-warn { width:7px; height:7px; border-radius:50%; background:#ffad1f; display:inline-block; }

/* ── Orbit badge ── */
.orbit-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.orbit-LEO { background: #e8f5fd; color: #1d9bf0; border: 1px solid #b3e0fc; }
.orbit-MEO { background: #f3e8fd; color: #7856ff; border: 1px solid #d4b3fc; }
.orbit-GEO { background: #fff8e1; color: #f5a623; border: 1px solid #ffe082; }

/* ── Sidebar section card ── */
.sb-section {
    background: #f5f8fa;
    border: 1px solid #e1e8ed;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.sb-section-title {
    font-size: 11px;
    font-weight: 800;
    color: #0f1419;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Sidebar info row ── */
.sb-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.sb-info-label {
    font-size: 11px;
    color: #536471;
    font-weight: 500;
}
.sb-info-val {
    font-size: 12px;
    color: #0f1419;
    font-weight: 700;
}

/* ── ML testing steps ── */
.ml-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
    color: #0f1419;
    background: #f5f8fa;
    border: 1px solid #e1e8ed;
    transition: all 0.3s;
}
.ml-step.done {
    background: #e8f5fd;
    border-color: #b3e0fc;
    color: #1d9bf0;
}
.ml-step.active {
    background: #fff8e1;
    border-color: #ffe082;
    color: #f5a623;
}
.ml-step-icon { font-size: 15px; }

/* ── Chart wrapper ── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e1e8ed;
    border-radius: 16px;
    padding: 1.2rem 1.2rem 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.chart-title {
    font-size: 14px;
    font-weight: 700;
    color: #0f1419;
    margin-bottom: 0.4rem;
}
.chart-sub {
    font-size: 11px;
    color: #8899a6;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


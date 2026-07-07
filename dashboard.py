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
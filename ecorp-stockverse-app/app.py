# --- START OF FILE app.py ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, confusion_matrix
from PIL import Image, ImageDraw
from io import BytesIO
import json
import traceback
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans

# --- Page Config (ONCE) ---
# Changed title as requested for Welcome Page
st.set_page_config(
    page_title="E-Corp StockVerse", # Changed title here
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME CONSTANTS ---
# E-Corp / Dark Army Theme Colors (Primary for Welcome)
ECORP_BLUE = "#0047AB" # E-Corp Corporate Blue
ECORP_GREY = "#A9A9A9" # E-Corp Subtle Grey
DA_GREEN = "#00FF00"   # Dark Army / Mr. Robot Green
DA_BLACK = "#000000"   # Pure Black Background
DA_RED_ACCENT = "#D30000" # Used for binary pattern (from AIB theme, fits well)

# AIB Theme Colors
AIB_RED = "#D30000"
AIB_BLACK = "#1A1A1A"
AIB_WHITE = "#FFFFFF"
AIB_GOLD = "#FFD700"

# Squid Game Theme Colors
SQ_NEON_PINK = "#FF0087"
SQ_NEON_GREEN = "#00FF5F"
SQ_DARK_BG = "#000000" # Use pure black for background

# Money Heist Theme Colors
MH_RED = "#FF0000"
MH_BLACK = "#000000" # Use pure black for background
MH_WHITE = "#FFFFFF"
MH_GOLD = "#FFD700"


# --- Base64 encoded binary pattern for Welcome page background ---
# Re-using the red pattern as it fits the glitchy/warning vibe well with E-Corp Blue/DA Green
BINARY_PATTERN_RED_BG = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNEMzAwMDAiIGZpbGwtb3BhY2l0eT0iMC4wNyI+PHBhdGggZD0iTTM2IDBjLTUuNTIzIDAtMTAgNC40NzctMTAgMTBzNC40NzcgMTAgMTAgMTAgMTAtNC40NzctMTAtMTAtNC40NzctMTAtMTAtMTB6bTAgMzBjLTUuNTIzIDAtMTAgNC40NzctMTAgMTBzNC40NzcgMTAgMTAgMTAgMTAtNC40NzctMTAtMTAtNC40NzctMTAtMTAtMTB6TTUgMGMtNS41MjMgMC0xMCA0LjQ3Ny0xMCAxMHM0LjQ3NyAxMCAxMCAxMCAxMC00LjQ3NyAxMC0xMFMxMC41MjMgMCA1IDB6TTUgMzBjLTUuNTIzIDAtMTAgNC40NzctMTAgMTBzNC40NzcgMTAgMTAgMTAgMTAtNC40NzctMTAtMTAtNC40NzctMTAtMTAtMTB6Ii8+PC9nPjwvZz48L3N2Zz4="

# --- Base64 Dark Army Mask (simple text art representation for VT323 font) ---
# Using text characters that look like eyes/mask in VT323 font
DARK_ARMY_MASK_TEXT = r"/\_\_\/" # Text representation of a mask
DARK_ARMY_MASK_TEXT_LARGE = r"""
  /\
 (__)
 /\/\
 /  \
""" # Multi-line text art


# --- SVGs (Used ONLY on their respective module pages, NOT welcome) ---
# These SVGs are defined as f-strings. Literal CSS braces { and } inside their <style> blocks
# must be doubled to {{ and }} to be preserved after f-string formatting.
# Kept existing SVGs for module headers as they were already page-specific.
AIB_CARDS_SVG_HEADER = f"""
<svg viewBox="0 0 100 55" xmlns="http://www.w3.org/2000/svg" width="50" height="50">
    <style>
        .aib-h-card {{ transition: transform 0.2s ease-out; filter: drop-shadow(0 1px 1px rgba(0,0,0,0.2)); }}
        .aib-h-card:hover {{ transform: translateY(-2px) scale(1.1); filter: drop-shadow(0 2px 4px {AIB_GOLD}80);}}
        @keyframes card-sway1-h {{ 0%,100% {{{{transform:rotate(-2deg);}}}} 50% {{{{transform:rotate(1deg);}}}} }}
        @keyframes card-sway2-h {{ 0%,100% {{{{transform:rotate(2deg);}}}} 50% {{{{transform:rotate(-1deg);}}}} }}
    </style>
    <text class="aib-h-card" x="5" y="35" font-family="Arial" font-size="28" fill="{AIB_RED}" style="animation: card-sway1-h 8s ease-in-out infinite;">♦️</text>
    <text class="aib-h-card" x="30" y="40" font-family="Arial" font-size="28" fill="{AIB_BLACK}" style="animation: card-sway2-h 7s ease-in-out infinite 0.5s;">♠️</text>
    <text class="aib-h-card" x="55" y="35" font-family="Arial" font-size="28" fill="{AIB_BLACK}" style="animation: card-sway1-h 9s ease-in-out infinite 1s;">♣️</text>
    <text class="aib-h-card" x="80" y="40" font-family="Arial" font-size="28" fill="{AIB_RED}" style="animation: card-sway2-h 6s ease-in-out infinite 1.5s;">♥️</text>
</svg>
"""

SQUIDGAME_SVG_HEADER = f"""
<svg viewBox="0 0 100 55" xmlns="http://www.w3.org/2000/svg" width="50" height="50">
    <style>
        .sg-h-shape {{ transition: transform 0.2s ease-out; }}
        .sg-h-shape:hover {{ transform: scale(1.12); filter: drop-shadow(0 0 4px {SQ_NEON_GREEN}); }}
        @keyframes sg-pulse-glow-h {{ 0%,100% {{{{ opacity:0.7; transform: scale(0.98);}}}} 50% {{{{ opacity:1; transform: scale(1.02); filter: drop-shadow(0 0 6px {SQ_NEON_PINK});}}}} }}
    </style>
    <circle class="sg-h-shape" cx="20" cy="28" r="11" fill="none" stroke="{SQ_NEON_PINK}" stroke-width="2.5" style="animation: sg-pulse-glow-h 3s ease-in-out infinite;"/>
    <polygon class="sg-h-shape" points="50,13 64,40 36,40" fill="none" stroke="{SQ_NEON_GREEN}" stroke-width="2.5" style="animation: sg-pulse-glow-h 3s ease-in-out infinite 0.5s;"/>
    <rect class="sg-h-shape" x="72" y="17" width="18" height="18" fill="none" stroke="{SQ_NEON_PINK}" stroke-width="2.5" style="animation: sg-pulse-glow-h 3s ease-in-out infinite 1s;"/>
</svg>
"""

MONEYHEIST_SVG_HEADER = f"""
<svg viewBox="0 0 100 55" xmlns="http://www.w3.org/2000/svg" width="50" height="50">
    <style>
        .mh-h-shape {{ transition: transform 0.2s ease-out; }}
        .mh-h-shape:hover {{ transform: scale(1.12); filter: drop-shadow(0 0 4px {MH_GOLD}); }}
        @keyframes mh-pulse-glow-h {{ 0%,100% {{{{ opacity:0.7; transform: scale(0.98);}}}} 50% {{{{ opacity:1; transform: scale(1.02); filter: drop-shadow(0 0 6px {MH_RED});}}}} }}
    </style>
    <circle class="mh-h-shape" cx="20" cy="28" r="11" fill="none" stroke="{MH_RED}" stroke-width="2.5" style="animation: mh-pulse-glow-h 3s ease-in-out infinite;"/>
    <polygon class="mh-h-shape" points="50,13 64,40 36,40" fill="none" stroke="{MH_GOLD}" stroke-width="2.5" style="animation: mh-pulse-glow-h 3s ease-in-out infinite 0.5s;"/>
    <rect class="mh-h-shape" x="72" y="17" width="18" height="18" fill="none" stroke="{MH_RED}" stroke-width="2.5" style="animation: mh-pulse-glow-h 3s ease-in-out infinite 1s;"/>
</svg>
"""

# --- Global CSS for base styling ---
APP_BASE_CSS = f"""
<style>
/* Base styles overridden by specific page styles */
body {{
    color: white;
    background-color: #000000 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    line-height: 1.6;
    transition: background-color 0.5s ease; /* Smooth background transition between pages */
}}
[data-testid="stSidebar"][aria-expanded="true"] {{
    display: block !important;
    transition: all 0.3s ease;
}}
.main .block-container {{
    padding: 1.5rem 2rem 1rem 2rem !important; max-width: 100% !important;
}}
.stButton > button {{
    border-radius: 8px; padding: 0.8rem 1.5rem; transition: all 0.3s ease-out;
    width: 100%; margin-bottom: 10px; font-weight: 600; font-size: 1.0em;
    letter-spacing: 1px; text-transform: uppercase; cursor: pointer;
    border: 2px solid rgba(255,255,255,0.2); background-color: rgba(0,0,0,0.3);
    color: white; position: relative; overflow: hidden; z-index: 1;
}}
.stButton > button:hover {{
    transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.4);
}}
.stButton > button:active {{
    transform: translateY(0); box-shadow: none;
}}

/* Add a hover background effect */
.stButton > button::before {{
    content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: rgba(255,255,255,0.1); transition: all 0.4s ease-in-out; z-index: -1;
}}
.stButton > button:hover::before {{ left: 0; }}


.stSelectbox > div > div {{ border-radius: 8px; background-color: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.2); color: white; }}
.stTextInput > div > div > input {{ border-radius: 8px; background-color: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.2); color: white; }}
.stTextInput > label, .stSelectbox > label, .stRadio > label {{ color: #B0B0B0; font-size: 0.9em; margin-bottom: 5px; display: block; }}


/* Common Card Styles */
.hover-card {{
    background-color: rgba(28,28,30,0.85); border-radius: 12px; padding: 20px; margin: 15px 0;
    border: 1px solid #383838; transition: all 0.3s ease-out;
    backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);
    position: relative; overflow: hidden;
}}
.hover-card:hover {{
    transform: translateY(-5px); box-shadow: 0 12px 25px rgba(0,0,0,0.4);
    border-color: #555;
}}
.dashboard-section {{
    margin:30px 0; padding:25px; border-radius:15px; background-color:rgba(24,24,26,0.75);
    border:1px solid #3a3a3c; transition:all 0.3s ease; backdrop-filter:blur(6px); -webkit-backdrop-filter: blur(6px);
}}
.dashboard-section:hover {{ background-color:rgba(30,30,32,0.85); box-shadow:0 8px 20px rgba(0,0,0,0.4); }}

/* Common Metric Card Styles */
.metrics-container {{ display:flex; flex-wrap:wrap; gap:20px; margin-bottom:30px; }}
.metric-card {{
    flex:1; min-width:200px; background-color:rgba(30,30,32,0.9); border-radius:12px;
    padding:20px; text-align:center; transition:all 0.3s ease-out; border:1px solid #404042;
    border-left-width: 6px; position: relative; overflow: hidden;
}}
.metric-card:hover {{ transform:translateY(-6px) scale(1.02); box-shadow:0 10px 22px rgba(0,0,0,0.4); }}
.metric-value {{ font-size:1.8em; font-weight:700; margin:12px 0 6px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }}
.metric-title {{ color:#c0c0c0; font-size:0.85em; margin-bottom:6px; text-transform:uppercase; letter-spacing: 0.8px; }}
.metric-subtext {{ color:#999; font-size:0.8em; }}

/* Scrollbar Styling */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: #18181A; border-radius: 5px; }}
::-webkit-scrollbar-thumb {{ background: #555; border-radius: 5px; transition: background 0.3s ease; }}
::-webkit-scrollbar-thumb:hover {{ background: #777; }}

/* Plotly Tooltip Styling */
.js-plotly-plot .plotly .modebar {{ background-color: transparent !important; }}
.js-plotly-plot .plotly .cursor-crosshair {{ cursor: crosshair; }}


/* Hide Streamlit header/footer - Can be overridden per page */
header {{ display: none !important; }}
footer {{ display: none !important; }}

/* Neon Glow Effect Utility Class */
.neon-glow {{
    text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px currentColor, 0 0 20px currentColor, 0 0 25px currentColor, 0 0 30px currentColor, 0 0 35px currentColor;
}}
/* Pulsing Animation */
@keyframes pulse-glow {{{{
    0%, 100% {{{{ opacity: 1; text-shadow: 0 0 5px #fff, 0 0 10px currentColor; }}}}
    50% {{{{ opacity: 0.8; text-shadow: 0 0 8px #fff, 0 0 15px currentColor, 0 0 20px currentColor; }}}}
}}}}
/* Flicker Animation */
@keyframes flicker {{{{
    0%, 18%, 22%, 25%, 53%, 57%, 100% {{{{ text-shadow: 0 0 4px #fff, 0 0 8px currentColor, 0 0 12px currentColor; opacity: 1; }}}}
    20%, 24%, 55% {{{{ text-shadow: none; opacity: 0.7; }}}}
}}}}

/* Add padding around Plotly charts */
.stPlotlyChart {{ padding: 10px; background-color: rgba(28,28,30,0.7); border-radius: 10px; margin: 15px 0; border: 1px solid #404042; }}


</style>
"""

# --- COLOR HELPER FUNCTION ---
def hex_to_rgba_string_with_alpha(hex_color_input, alpha_hex):
    named_color_map = {
        "red": "#FF0000", "green": "#008000", "blue": "#0000FF",
        "white": "#FFFFFF", "black": "#000000", "yellow": "#FFFF00",
        "gold": "#FFD700", "pink": "#FF0087" # Added common names
    }
    hex_color = named_color_map.get(hex_color_input.lower(), hex_color_input)
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        return 'rgba(128,128,128,0.5)'
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        alpha_float = int(alpha_hex, 16) / 255.0
        return f'rgba({r},{g},{b},{alpha_float:.2f})'
    except ValueError:
        return 'rgba(128,128,128,0.5)'

# --- Helper Functions (load_stock_data, calculate_rsi, prepare_ml_data, train_and_predict, create_prediction_summary - remain unchanged as they were correct) ---
# Ensure these functions are exactly as they were, only adding imports or minor logging fixes if necessary.
# The logic should not be altered.

@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty: return None, None, f"No data for '{ticker}'. Invalid or delisted?"
        data.columns = [col.upper() for col in data.columns]
        data.dropna(subset=['CLOSE'], inplace=True)
        if data.empty: return None, None, f"No valid 'CLOSE' price for '{ticker}'."
        # Robust fillna using bfill then ffill after ensuring CLOSE is clean
        data.ffill(inplace=True); data.bfill(inplace=True)
        # Ensure numeric types and fill remaining NAs robustly
        for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE']: data[col] = pd.to_numeric(data[col], errors='coerce')
        data['VOLUME'] = pd.to_numeric(data['VOLUME'], errors='coerce').fillna(0).astype(np.int64)
        data.dropna(subset=['OPEN', 'HIGH', 'LOW', 'CLOSE'], inplace=True)
        if data.empty: return None, None, f"Data for '{ticker}' empty after cleaning."
        info = stock.info
        return data, info, None
    except Exception as e:
        error_msg = f"Error fetching '{ticker}': {e}"
        if "Too many requests" in str(e): error_msg = "API Limit: Too many yfinance requests."
        elif "No data found" in str(e) or "delisted" in str(e) or "failed" in str(e).lower():
             error_msg = f"Invalid Ticker/No Data for '{ticker}'."
        # st.error(f"Error loading data: {error_msg}") # Removed direct error display from function
        return None, None, error_msg

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0); loss = -delta.where(delta < 0, 0).fillna(0)
    # Use .mean() for simple average if EWM is problematic with short data
    if len(gain) < window: avg_gain = gain.mean(); avg_loss = loss.mean()
    else: avg_gain = gain.ewm(com=window - 1, min_periods=window).mean(); avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi_val = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    # Use ffill then bfill for final NA handling
    rsi_series = pd.Series(rsi_val, index=prices.index).fillna(method='ffill').fillna(method='bfill').fillna(50) # fill any remaining NA with 50
    return rsi_series


def prepare_ml_data(df):
    min_len = 60
    if df is None or len(df) < min_len:
        return None,None,None,None,None,None,None, f"Need at least {min_len} days of data, got {len(df) if df is not None else 0}."
    df_p = df.copy(); df_p.sort_index(inplace=True)

    # Basic OHLCV features first, ensure they are numeric and filled
    base_features = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
    for col in base_features:
         if col in df_p.columns: df_p[col] = pd.to_numeric(df_p[col], errors='coerce')
         else: df_p[col] = np.nan # Ensure missing base cols are nan

    df_p.dropna(subset=['CLOSE'], inplace=True)
    if df_p.empty: return None,None,None,None,None,None,None, f"No valid CLOSE data after cleaning."

    # Forward fill / backward fill any remaining NAs after numeric conversion
    df_p.ffill(inplace=True); df_p.bfill(inplace=True)
    if df_p.isnull().values.any():
         # Fallback fill for any remaining NAs (shouldn't happen if ffill/bfill is robust, but belt-and-suspenders)
         for col in df_p.columns:
              if df_p[col].dtype in ['float64', 'int64']:
                   df_p[col].fillna(df_p[col].mean(), inplace=True)
              else:
                   df_p[col].fillna(df_p[col].mode()[0] if not df_p[col].mode().empty else 'Missing', inplace=True)


    win = [5,10,20,50]; feats = []
    for w in win:
        # Ensure window is not larger than data length
        valid_w = min(w, len(df_p) - 1) if len(df_p) > 1 else 1
        if valid_w <= 0: continue # Avoid errors on very short data

        df_p[f'MA{w}'] = df_p['CLOSE'].rolling(valid_w,min_periods=max(1,valid_w//2)).mean()
        df_p[f'Std{w}'] = df_p['CLOSE'].rolling(valid_w,min_periods=max(1,valid_w//2)).std()
        if valid_w > 0: # Diff requires valid_w > 0
             df_p[f'Mom{w}'] = df_p['CLOSE'].diff(valid_w)
             feats.extend([f'MA{w}',f'Std{w}',f'Mom{w}'])
        else:
             # Handle case where valid_w is 0 (shouldn't happen with min_len check, but safety)
             df_p[f'MA{w}'] = df_p['CLOSE']
             df_p[f'Std{w}'] = 0
             df_p[f'Mom{w}'] = 0
             feats.extend([f'MA{w}',f'Std{w}',f'Mom{w}'])

    # Volatility calculation needs pct_change and rolling, check length
    if len(df_p) > 10:
         df_p['Vol10'] = df_p['CLOSE'].pct_change().rolling(10,min_periods=5).std()*np.sqrt(252)
         feats.append('Vol10')
    else:
         df_p['Vol10'] = df_p['CLOSE'].pct_change().std() * np.sqrt(252) * 10 # Fallback for short data
         feats.append('Vol10')

    # RSI calculation
    if len(df_p) > 14:
         df_p['RSI14'] = calculate_rsi(df_p['CLOSE'],14)
         feats.append('RSI14')
    else:
        df_p['RSI14'] = 50 # Default RSI for very short data
        feats.append('RSI14')

    # Final fillna after feature creation
    df_p.bfill(inplace=True); df_p.ffill(inplace=True)
    # Ensure no inf values accidentally introduced by calculations
    df_p.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_p.fillna(method='ffill', inplace=True); df_p.fillna(method='bfill', inplace=True)
    df_p.fillna(df_p.mean(), inplace=True) # Final fallback for columns that are all NA or inf


    if len(df_p) < 20: return None,None,None,None,None,None,None, f"Data too short ({len(df_p)}) after feature engineering and cleaning."

    sft = 5 # Predict 5 days ahead
    # Check if there are enough data points for the shift
    if len(df_p) < sft + 1:
         return None,None,None,None,None,None,None, f"Data too short ({len(df_p)}) for {sft}-day target variable."

    df_p['Tgt_Reg'] = df_p['CLOSE'].shift(-sft)
    df_p['Tgt_Cls'] = (df_p['Tgt_Reg'] > df_p['CLOSE']).astype(int)

    all_f = [c for c in base_features+feats if c in df_p.columns and df_p[c].dtype in [np.float64, np.int64]] # Select only numeric features
    df_p.dropna(subset=['Tgt_Reg','Tgt_Cls'], inplace=True)

    if df_p.empty or len(df_p) < 2: return None,None,None,None,None,None,None, "No data available after creating targets or too few points."

    X=df_p[all_f]; yR=df_p['Tgt_Reg']; yC=df_p['Tgt_Cls']

    # Ensure X and y index align after dropna
    common_index = X.index.intersection(yR.index).intersection(yC.index)
    if common_index.empty: return None,None,None,None,None,None,None, "No common index after feature/target alignment."
    X = X.loc[common_index]; yR = yR.loc[common_index]; yC = yC.loc[common_index]

    corr_df=X.copy(); corr_df['Price_5d_Fwd']=yR; corr_df['Trend_5d_Fwd']=yC

    try:
        # Ensure enough data for split. Need at least 2 points in train and 1 in test.
        if len(X) < 3: return None,None,None,None,None,None,None, f"Data too short ({len(X)}) for train/test split (minimum 3 needed)."

        t_frac=0.2; # Use 20% for test
        # Ensure test set has at least 1 sample, and train at least 2 for splitting
        if len(X) * t_frac < 1: t_frac = max(0.1, 1/len(X)) if len(X) > 0 else 0.2 # Ensure at least 1 test sample
        train_size = len(X) - int(len(X)*t_frac)
        if train_size < 2: train_size = 2 # Ensure minimum train size
        if train_size >= len(X): train_size = len(X) - 1 # Ensure there's at least one test sample if possible

        Xtr, Xte = X.iloc[:train_size], X.iloc[train_size:]
        yRtr, yRte = yR.iloc[:train_size], yR.iloc[train_size:]
        yCtr, yCte = yC.iloc[:train_size], yC.iloc[train_size:]


        if Xtr.empty or Xte.empty or yRtr.empty or yRte.empty or yCtr.empty or yCte.empty:
             return None,None,None,None,None,None,None,f"Empty train/test sets after splitting ({len(X)} total)."

    except Exception as e: return None,None,None,None,None,None,None,f"Split error: {e}."

    scl=StandardScaler();
    try:
        Xtr_s=scl.fit_transform(Xtr);
    except ValueError as e:
         return None,None,None,None,None,None,None, f"Scaler fit_transform error on train data: {e}. Data might contain NaN or inf values."

    try:
        Xte_s=scl.transform(Xte);
    except ValueError as e:
         return None,None,None,None,None,None,None, f"Scaler transform error on test data: {e}. Data might contain NaN or inf values."

    if X.empty: return None,None,None,None,None,None,None,"X empty pre-latest."

    # Get the latest data point and scale it
    latest_f = X.iloc[[-1]].values # Use [[-1]] to preserve shape (1, num_features)
    try:
        latest_d_s = scl.transform(latest_f)
    except ValueError as e:
         return None,None,None,None,None,None,None, f"Scaler transform error on latest data: {e}. Latest features might contain NaN or inf values."

    return Xtr_s,Xte_s,yRtr,yRte,yCtr,yCte,latest_d_s,corr_df

def train_and_predict(X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, latest_data):
    res={}; mdls={}
    if any(x is None for x in [X_train,X_test,y_reg_train,y_cls_train,latest_data]): return None,None,"ML input missing."
    if X_train.shape[0]==0 or X_test.shape[0]==0: return None,None,"Empty train/test set."

    # Ensure targets are pandas Series for .iloc and .values
    y_reg_train = pd.Series(y_reg_train, index=X_train.index if isinstance(X_train, pd.DataFrame) else range(X_train.shape[0])) if not isinstance(y_reg_train, pd.Series) else y_reg_train
    y_reg_test = pd.Series(y_reg_test, index=X_test.index if isinstance(X_test, pd.DataFrame) else range(X_test.shape[0])) if not isinstance(y_reg_test, pd.Series) else y_reg_test
    y_cls_train = pd.Series(y_cls_train, index=X_train.index if isinstance(X_train, pd.DataFrame) else range(X_train.shape[0])) if not isinstance(y_cls_train, pd.Series) else y_cls_train
    y_cls_test = pd.Series(y_cls_test, index=X_test.index if isinstance(X_test, pd.DataFrame) else range(X_test.shape[0])) if not isinstance(y_cls_test, pd.Series) else y_cls_test


    try:
        # Regression
        reg = LinearRegression(); reg.fit(X_train,y_reg_train); mdls['regression']=reg
        yRp = reg.predict(X_test)
        # Handle cases where y_reg_test is empty or contains NaNs
        valid_reg_indices = ~np.isnan(y_reg_test.values)
        if np.sum(valid_reg_indices) > 0:
             rmse=np.sqrt(mean_squared_error(y_reg_test.values[valid_reg_indices],yRp[valid_reg_indices]))
             r2s=r2_score(y_reg_test.values[valid_reg_indices],yRp[valid_reg_indices])
             resid = y_reg_test.values[valid_reg_indices] - yRp[valid_reg_indices]
             p_std = np.std(resid) if len(resid)>1 else 0
        else:
            rmse = float('nan'); r2s = float('nan'); p_std = 0; resid = np.array([])

        nxt_p = reg.predict(latest_data)[0] if latest_data is not None and latest_data.shape[0] > 0 else np.nan
        # Ensure prediction is not NaN before calculating CI
        if np.isnan(nxt_p):
             lower_bound = np.nan; upper_bound = np.nan
        else:
             lower_bound = nxt_p - 1.96*p_std
             upper_bound = nxt_p + 1.96*p_std

        res['regression']={'prediction':float(nxt_p),'lower_bound':float(lower_bound),'upper_bound':float(upper_bound),
                           'rmse':float(rmse),'r2':float(r2s),'y_true':y_reg_test.values.tolist(),'y_pred':yRp.tolist(),'residuals':resid.tolist()}

        # Classification
        # Check for single class *in the training data* which is what the model trains on
        unique_classes = np.unique(y_cls_train)
        if len(unique_classes) > 1:
            cls=LogisticRegression(random_state=42,class_weight='balanced',max_iter=1000,solver='liblinear')
            cls.fit(X_train,y_cls_train); mdls['classification']=cls
            yCp=cls.predict(X_test);
            yCprob=cls.predict_proba(X_test)[:,1] if hasattr(cls, 'predict_proba') else np.full(len(y_cls_test), 0.5) # Handle no predict_proba
            # Handle cases where y_cls_test is empty or single class
            valid_cls_indices = ~np.isnan(y_cls_test.values)
            if np.sum(valid_cls_indices) > 0 and len(np.unique(y_cls_test.values[valid_cls_indices])) > 1:
                 acc=accuracy_score(y_cls_test.values[valid_cls_indices],yCp[valid_cls_indices])
                 confm=confusion_matrix(y_cls_test.values[valid_cls_indices],yCp[valid_cls_indices],labels=[0,1])
            else:
                 acc = float('nan'); confm = np.array([[0,0],[0,0]]);
                 if np.sum(valid_cls_indices) > 0: # If data exists but only one class in test set
                      confm = confusion_matrix(y_cls_test.values[valid_cls_indices],yCp[valid_cls_indices],labels=np.unique(y_cls_test.values[valid_cls_indices]))
                      # Need to pad confusion matrix if needed
                      if confm.shape != (2,2):
                           padded_confm = np.zeros((2,2), dtype=int)
                           for i, true_label in enumerate(np.unique(y_cls_test.values[valid_cls_indices])):
                                for j, pred_label in enumerate(np.unique(yCp[valid_cls_indices])):
                                     if true_label in [0,1] and pred_label in [0,1]:
                                          padded_confm[true_label][pred_label] = confm[i][j]
                           confm = padded_confm


            trend_pr = cls.predict_proba(latest_data)[0][1] if latest_data is not None and latest_data.shape[0] > 0 and hasattr(cls, 'predict_proba') else 0.5
            trend_pred_c = 1 if trend_pr >=0.5 else 0

            res['classification']={'trend_numeric':int(trend_pred_c), 'trend':"UP" if trend_pred_c==1 else "DOWN",
                                   'probability':float(trend_pr),'accuracy':float(acc) if acc is not None else None,'confusion_matrix':confm.tolist(),
                                   'y_true':y_cls_test.values.tolist(),'y_pred':yCp.tolist(),'y_proba':yCprob.tolist()}
        else:
            # Handle single class in training data
            maj_c = unique_classes[0] if unique_classes.size > 0 else 0
            st.warning(f"LogReg Warn: Single class in train ({maj_c}). Predicting majority.")
            # Predict majority class for test set
            yCp_majority = np.full(len(y_cls_test), maj_c)
            valid_cls_indices = ~np.isnan(y_cls_test.values)
            if np.sum(valid_cls_indices) > 0 and len(np.unique(y_cls_test.values[valid_cls_indices])) > 1:
                # Can still calculate accuracy/confusion matrix if test set has > 1 class
                 acc=accuracy_score(y_cls_test.values[valid_cls_indices],yCp_majority[valid_cls_indices])
                 confm=confusion_matrix(y_cls_test.values[valid_cls_indices],yCp_majority[valid_cls_indices],labels=[0,1] if len(np.unique(y_cls_test.values[valid_cls_indices])) > 1 else np.unique(y_cls_test.values[valid_cls_indices]))
                 if confm.shape != (2,2): # Pad confusion matrix if needed
                      padded_confm = np.zeros((2,2), dtype=int)
                      unique_test_labels = np.unique(y_cls_test.values[valid_cls_indices])
                      unique_pred_labels = np.unique(yCp_majority[valid_cls_indices])
                      for i, true_label in enumerate(unique_test_labels):
                           for j, pred_label in enumerate(unique_pred_labels):
                                if true_label in [0,1] and pred_label in [0,1]:
                                     padded_confm[true_label][pred_label] = confm[i][j]
                      confm = padded_confm
            else:
                 acc = float('nan'); confm = np.array([[0,0],[0,0]]);


            res['classification']={'trend_numeric':int(maj_c),'trend':"UP" if maj_c==1 else "DOWN",'probability':float(1.0 if maj_c==1 else 0.0),
                                   'accuracy':float(acc) if acc is not None else None,'confusion_matrix':confm.tolist(), 'y_true':y_cls_test.values.tolist(),
                                   'y_pred':yCp_majority.tolist(), 'y_proba':np.full(len(y_cls_test),1.0 if maj_c==1 else 0.0).tolist()}
            mdls['classification']=None # Don't store the model if it couldn't learn

        # K-means Clustering
        try:
            n_clusters = 2 # Default to 2 clusters (Bullish/Bearish)
            if X_train.shape[0] >= n_clusters: # Ensure enough data points for the number of clusters
                 kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                 kmeans.fit(X_train)
                 mdls['kmeans'] = kmeans

                 train_clusters = kmeans.predict(X_train)
                 test_clusters = kmeans.predict(X_test)
                 latest_cluster = kmeans.predict(latest_data)[0] if latest_data is not None and latest_data.shape[0] > 0 else 0 # Default if no latest data

                 train_df = pd.DataFrame(X_train, columns=[f'Feature_{i}' for i in range(X_train.shape[1])])
                 train_df['Cluster'] = train_clusters
                 train_df['Target_Value'] = y_reg_train.values

                 cluster_stats = train_df.groupby('Cluster').agg(Mean_Value=('Target_Value', 'mean'), Std_Value=('Target_Value', 'std'), Count=('Target_Value', 'count')).reset_index()
                 cluster_stats[['Mean_Value', 'Std_Value']] = cluster_stats[['Mean_Value', 'Std_Value']].fillna(0) # Fill NaNs in stats if any

                 latest_cluster_stats = cluster_stats.loc[cluster_stats['Cluster'] == latest_cluster]
                 latest_cluster_mean = latest_cluster_stats['Mean_Value'].values[0] if not latest_cluster_stats.empty else 0
                 latest_cluster_std = latest_cluster_stats['Std_Value'].values[0] if not latest_cluster_stats.empty else 0
                 latest_cluster_count = latest_cluster_stats['Count'].values[0] if not latest_cluster_stats.empty else 0


                 res['kmeans'] = {'train_clusters': train_clusters.tolist(),'test_clusters': test_clusters.tolist(),'latest_cluster': int(latest_cluster),
                    'cluster_stats': cluster_stats.to_dict('records'),'latest_cluster_mean': float(latest_cluster_mean),
                    'latest_cluster_std': float(latest_cluster_std),'latest_cluster_count': int(latest_cluster_count),'n_clusters': n_clusters}
            else:
                st.warning(f"K-means Warn: Not enough data points ({X_train.shape[0]}) to form {n_clusters} clusters. Skipping K-means.")
                res['kmeans'] = {} # Indicate kmeans was skipped

        except Exception as e:
            st.warning(f"K-means clustering error: {e}")
            traceback.print_exc() # Print K-means specific error for debugging
            res['kmeans'] = {} # Indicate kmeans failed

        return res,mdls,None
    except Exception as e:
        st.error(f"ML Training/Prediction Error: {e}");
        traceback.print_exc(); # Print main ML error for debugging
        return None,None,f"ML Error: {e}"

def create_prediction_summary(ticker, last_price, ml_results):
    # Ensure inputs are valid
    last_price = float(last_price) if pd.notna(last_price) else np.nan

    summary = {"Ticker": ticker, "Last_Price": f"{last_price:.2f}" if pd.notna(last_price) else "N/A", "Pred_Date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")}
    if ml_results:
        reg_res = ml_results.get('regression', {})
        cls_res = ml_results.get('classification', {})
        summary.update({
            "Pred_Price": f"{reg_res.get('prediction', np.nan):.2f}" if pd.notna(reg_res.get('prediction', np.nan)) else "N/A",
            "Price_Lower_CI": f"{reg_res.get('lower_bound', np.nan):.2f}" if pd.notna(reg_res.get('lower_bound', np.nan)) else "N/A",
            "Price_Upper_CI": f"{reg_res.get('upper_bound', np.nan):.2f}" if pd.notna(reg_res.get('upper_bound', np.nan)) else "N/A",
            "Pred_Trend": cls_res.get('trend', "N/A"),
            "Trend_Prob_Up": f"{cls_res.get('probability', np.nan):.3f}" if pd.notna(cls_res.get('probability', np.nan)) else "N/A"
        })
    else:
        summary.update({k: "N/A" for k in ["Pred_Price", "Price_Lower_CI", "Price_Upper_CI", "Pred_Trend", "Trend_Prob_Up"]})
    return summary

# --- Themed Display Functions (Enhanced UI/Styling) ---
def display_stock_charts_themed(data, ticker, theme_colors):
    if data is None or data.empty or 'CLOSE' not in data.columns or data['CLOSE'].isnull().all():
        # st.warning(f"No valid data to display stock charts for {ticker}.") # removed to avoid clutter
        return None, None
    df_c = data.copy()
    # Ensure MA and RSI calculations are robust to initial NaNs
    df_c['MA20'] = df_c['CLOSE'].rolling(window=20, min_periods=1).mean()
    df_c['MA50'] = df_c['CLOSE'].rolling(window=50, min_periods=1).mean()
    df_c['RSI14'] = calculate_rsi(df_c['CLOSE'], 14)

    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                          row_heights=[0.7, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    # Candlestick chart (only if enough data)
    if len(df_c) >= 2 and not df_c[['OPEN', 'HIGH', 'LOW', 'CLOSE']].isnull().all().all():
        fig_p.add_trace(go.Candlestick(x=df_c.index,open=df_c['OPEN'],high=df_c['HIGH'],low=df_c['LOW'],close=df_c['CLOSE'],
                                     name='Price', increasing_line_color=theme_colors.get('price_inc', '#00CC96'),
                                     decreasing_line_color=theme_colors.get('price_dec', '#EF553B'), opacity=0.9),row=1,col=1)

    # Moving Averages
    if not df_c['MA20'].isnull().all():
        fig_p.add_trace(go.Scatter(x=df_c.index,y=df_c['MA20'],name='MA 20',line=dict(color=theme_colors.get('ma20', '#636EFA'),width=1.5, dash='dot')),row=1,col=1)
    if not df_c['MA50'].isnull().all():
        fig_p.add_trace(go.Scatter(x=df_c.index,y=df_c['MA50'],name='MA 50',line=dict(color=theme_colors.get('ma50', '#EF553B'),width=1.5)),row=1,col=1)

    # RSI plot
    if 'RSI14' in df_c and not df_c['RSI14'].isnull().all():
        fig_p.add_trace(go.Scatter(x=df_c.index,y=df_c['RSI14'],name='RSI 14',line=dict(color=theme_colors.get('rsi_line', '#29B6F6'),width=1.2)),row=2,col=1)
        fig_p.add_hline(y=70,line_dash="dash",line_color=theme_colors.get('rsi_over', theme_colors.get('price_dec', '#EF553B')),opacity=0.7,row=2,col=1)
        fig_p.add_hline(y=30,line_dash="dash",line_color=theme_colors.get('rsi_under', theme_colors.get('price_inc', '#00CC96')),opacity=0.7,row=2,col=1)
        fig_p.update_yaxes(title_text="RSI", range=[0,100], row=2,col=1, title_font_size=11, tickfont_size=10, gridcolor='#444')

    fig_p.update_layout(title={'text':f"{ticker} Price Chart & RSI",'font':{'size':18,'color':theme_colors.get('title','#FFFFFF')},'x':0.5, 'xanchor':'center'},
                      template="plotly_dark", xaxis_rangeslider_visible=False, height=500, hovermode="x unified",
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1, font_size=9),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)', margin=dict(t=60,l=50,r=30,b=40))
    fig_p.update_yaxes(title_text="Price", row=1,col=1, title_font_size=11, tickfont_size=10, gridcolor='#444')
    fig_p.update_xaxes(showgrid=False, zeroline=False, tickfont_size=10)


    # Volume chart
    fig_v = None
    if 'VOLUME' in df_c and not df_c['VOLUME'].isnull().all() and 'OPEN' in df_c and 'CLOSE' in df_c \
        and len(df_c) >= 2 and not df_c['OPEN'].isnull().all() and not df_c['CLOSE'].isnull().all():
        vol_bar_colors = np.where(df_c['CLOSE']>=df_c['OPEN'], theme_colors.get('price_inc', '#00CC96'), theme_colors.get('price_dec', '#EF553B'))
        fig_v = go.Figure(data=[go.Bar(x=df_c.index,y=df_c['VOLUME'],marker_color=vol_bar_colors,name='Volume',opacity=0.75)])
        df_c['VolumeMA20'] = df_c['VOLUME'].rolling(window=20, min_periods=1).mean()
        if not df_c['VolumeMA20'].isnull().all():
             fig_v.add_trace(go.Scatter(x=df_c.index,y=df_c['VolumeMA20'],name='Vol MA 20',line=dict(color=theme_colors.get('ma50', '#EF553B'),width=1.2,dash='dashdot')))
        fig_v.update_layout(title={'text':f"{ticker} Trading Volume",'font':{'size':16,'color':theme_colors.get('title','#FFFFFF')},'x':0.5, 'xanchor':'center'},
                          template="plotly_dark",height=250,hovermode="x unified",showlegend=False,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)',margin=dict(t=50,l=50,r=30,b=40))
        fig_v.update_yaxes(title_text="Volume", title_font_size=11, tickfont_size=10, gridcolor='#444')
        fig_v.update_xaxes(showgrid=False, zeroline=False, tickfont_size=10)

    return fig_p, fig_v

def plot_returns_distribution(data, theme_colors):
    if data is None or data.empty or 'CLOSE' not in data.columns or data['CLOSE'].pct_change().dropna().empty:
        return None
    returns = data['CLOSE'].pct_change().dropna() * 100
    fig = px.histogram(returns, nbins=60, title="Daily Returns Distribution (%)", opacity=0.85, marginal="box")
    fig.update_layout(template="plotly_dark", showlegend=False, height=350, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(28,28,30,0.85)', title_font_color=theme_colors.get('title', AIB_WHITE),
                      title_x=0.5, xaxis_title="Daily Return (%)", yaxis_title="Frequency", margin=dict(t=50,l=40,r=20,b=40))
    fig.update_traces(marker_color=theme_colors.get('accent', AIB_GOLD), marker_line_color=AIB_BLACK, marker_line_width=0.5)
    return fig

def plot_rolling_volatility(data, theme_colors, window=21):
    if data is None or data.empty or 'CLOSE' not in data.columns:
        return None
    volatility_series = data['CLOSE'].pct_change().rolling(window=window).std() * np.sqrt(252) * 100
    if volatility_series.dropna().empty: return None
    fig = px.area(x=volatility_series.index, y=volatility_series, title=f"{window}-Day Rolling Annualized Volatility (%)", labels={'y':'Volatility (%)'})
    fig.update_layout(template="plotly_dark", showlegend=False, height=350, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(28,28,30,0.85)', title_font_color=theme_colors.get('title', AIB_WHITE),
                      title_x=0.5, xaxis_title=None, yaxis_title="Volatility (%)", margin=dict(t=50,l=40,r=20,b=40))
    accent_color = theme_colors.get('accent', AIB_GOLD)
    rgba_color = hex_to_rgba_string_with_alpha(accent_color, "40") # 25% alpha
    rgba_line = hex_to_rgba_string_with_alpha(accent_color, "FF")
    fig.update_traces(line=dict(color=rgba_line, width=2), fillcolor=rgba_color)
    return fig

def plot_feature_correlation_heatmap(data_df, theme_colors):
    if data_df is None or data_df.empty: return None
    # Ensure necessary columns exist before attempting calculations
    if 'CLOSE' not in data_df.columns or data_df['CLOSE'].isnull().all(): return None # Need CLOSE price

    df_copy = data_df.copy()
    # Ensure MA columns are calculated if not present
    if 'MA20' not in df_copy.columns: df_copy['MA20'] = df_copy['CLOSE'].rolling(min(20, len(df_copy)), min_periods=min(1, len(df_copy))).mean()
    if 'MA50' not in df_copy.columns: df_copy['MA50'] = df_copy['CLOSE'].rolling(min(50, len(df_copy)), min_periods=min(1, len(df_copy))).mean()

    features_to_check = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'MA20', 'MA50']

    valid_features = [f for f in features_to_check if f in df_copy.columns and not df_copy[f].isnull().all()]
    if not valid_features or len(valid_features) < 2: return None

    df_corr = df_copy[valid_features].corr()
    if df_corr.empty: return None

    fig = go.Figure(data=go.Heatmap(z=df_corr.values, x=df_corr.columns, y=df_corr.columns,
                                  colorscale='IceFire', text=df_corr.values, texttemplate="%{text:.2f}", hoverongaps=False,
                                  zmin=-1, zmax=1))
    fig.update_layout(title="Key Feature Correlation Matrix", template="plotly_dark", height=400,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)',
                      title_font_color=theme_colors.get('title', AIB_WHITE), title_x=0.5,
                      xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange='reversed',
                      margin=dict(t=50,l=60,r=20,b=60))
    return fig

def display_price_prediction(last_price, prediction_data, ticker, theme_colors, historical_data_df):
    # Ensure historical data is usable
    if historical_data_df is None or historical_data_df.empty or 'CLOSE' not in historical_data_df.columns or historical_data_df['CLOSE'].isnull().all():
        st.warning("Historical data not available or invalid for price prediction plot.")
        return

    pred_reg = prediction_data.get('regression', {})
    next_p = pred_reg.get('prediction', np.nan)
    lower_b = pred_reg.get('lower_bound', np.nan)
    upper_b = pred_reg.get('upper_bound', np.nan)
    rmse = pred_reg.get('rmse', float('nan')); r2 = pred_reg.get('r2', float('nan'))

    # Ensure last_price is a valid number before calculating percentage change
    last_price_float = float(last_price) if pd.notna(last_price) else np.nan
    pct_chg = ((next_p - last_price_float) / last_price_float) * 100 if pd.notna(next_p) and pd.notna(last_price_float) and last_price_float != 0 else 0

    # Use last valid index from historical data
    last_actual_date = historical_data_df.index[-1]
    last_actual_val = historical_data_df['CLOSE'].iloc[-1]

    # Use recent history for plotting, ensure it's not empty
    recent_history = historical_data_df['CLOSE'].iloc[-60:].dropna()
    if recent_history.empty: recent_history = historical_data_df['CLOSE'].dropna(); st.info("Using all available historical data for prediction plot.")
    if recent_history.empty: st.warning("No valid historical price data to plot prediction."); return

    fig = go.Figure()

    # Add historical price line
    fig.add_trace(go.Scatter(x=recent_history.index, y=recent_history, mode='lines', name='Historical Price', line=dict(color=theme_colors.get('historical_line', '#999999'), width=2)))

    # Add marker for last actual price
    fig.add_trace(go.Scatter(x=[last_actual_date], y=[last_actual_val], mode='markers', name='Last Actual Price', marker_symbol='diamond', marker_size=10, marker_color=theme_colors.get('actual_price_marker', SQ_NEON_PINK)))

    # Add prediction line and marker (only if prediction is a valid number)
    if pd.notna(next_p):
        pred_date = pd.to_datetime(last_actual_date) + timedelta(days=5)
        fig.add_trace(go.Scatter(x=[last_actual_date, pred_date], y=[last_actual_val, next_p], mode='lines', name='Prediction Path', line=dict(color=theme_colors.get('prediction_path_line', SQ_NEON_GREEN), width=1.5, dash='dash'), showlegend=False))
        fig.add_trace(go.Scatter(x=[pred_date], y=[next_p], mode='markers', name='Predicted Price', marker_symbol='star', marker_size=12, marker_color=theme_colors.get('pred_marker', SQ_NEON_PINK)))

        # Add Confidence Interval (only if bounds are valid numbers)
        if pd.notna(lower_b) and pd.notna(upper_b):
             fig.add_trace(go.Scatter(x=[pred_date, pred_date], y=[lower_b, upper_b], mode='lines', name='95% CI', line=dict(width=6, color=theme_colors.get('ci', SQ_NEON_GREEN)), opacity=0.7))


    fig.update_layout(title_text=f"{ticker} 5-Day Price Prediction", template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)', legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1, font_size=10), margin=dict(t=60,b=40,l=50,r=20), xaxis_title="Date", yaxis_title="Price")
    x_range_start = recent_history.index[0] if not recent_history.empty else last_actual_date - timedelta(days=7)
    x_range_end = (pred_date + timedelta(days=2)) if pd.notna(next_p) else (last_actual_date + timedelta(days=7))
    fig.update_xaxes(range=[x_range_start, x_range_end])

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    pred_val_color = theme_colors.get('up', SQ_NEON_GREEN) if pct_chg >=0 else theme_colors.get('down', '#FF3030')
    with col1:
        pred_price_str = f"${next_p:.2f}" if pd.notna(next_p) else "N/A"
        pct_chg_str = f"{pct_chg:+.1f}%" if pd.notna(next_p) and pd.notna(last_price_float) else "N/A"
        st.markdown(f"<div class='metric-card {'up' if pct_chg >=0 else 'down'}'><div class='metric-title'>Pred. Price (5 Days)</div><div class='metric-value' style='color:{pred_val_color}'>{pred_price_str}</div><div class='metric-subtext' style='color:{pred_val_color}'>{pct_chg_str}</div></div>", unsafe_allow_html=True)
    with col2:
        ci_str = f"${lower_b:.2f} - ${upper_b:.2f}" if pd.notna(lower_b) and pd.notna(upper_b) else "N/A"
        st.markdown(f"<div class='metric-card neutral'><div class='metric-title'>95% CI</div><div class='metric-value' style='font-size:1.1em; color:white;'>{ci_str}</div></div>", unsafe_allow_html=True)
    with col3:
        rmse_str = f"{rmse:.2f}" if pd.notna(rmse) else "N/A"
        r2_str = f"{r2:.2f}" if pd.notna(r2) else "N/A"
        st.markdown(f"<div class='metric-card neutral'><div class='metric-title'>RMSE / R²</div><div class='metric-value' style='font-size:1.1em; color:white;'>{rmse_str} / {r2_str}</div></div>", unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)


def display_trend_prediction(prediction_data, theme_colors):
    pred_cls = prediction_data.get('classification',{})
    trend = pred_cls.get('trend','N/A'); prob = pred_cls.get('probability',np.nan)
    acc = pred_cls.get('accuracy'); conf_mx = pred_cls.get('confusion_matrix')

    if isinstance(prob, (np.number, float)): prob = float(prob)
    else: prob = np.nan # Ensure prob is a float or NaN

    prob_pct = prob * 100 if pd.notna(prob) else 50 # Default to 50% if probability is NaN

    gauge_bar_color = theme_colors.get('up', SQ_NEON_GREEN) if trend == "UP" else theme_colors.get('down', '#FF3030')
    icon_color = gauge_bar_color if trend != 'N/A' else "#888888"
    icon = "⬆️" if trend == "UP" else "⬇️" if trend == "DOWN" else "➖"
    prob_str = f"{prob_pct:.1f}%" if pd.notna(prob) else "N/A"

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown(f""" <div class='metric-card' style='border-left-color: {icon_color}; height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div class='metric-title' style='color: {theme_colors.get('title', SQ_NEON_PINK)};'>Predicted Trend (5 Days)</div>
            <div style='font-size: 3.5em; margin: 5px 0; color:{icon_color};'>{icon}</div>
            <div style='font-size: 1.3em; color:{icon_color}; font-weight: bold;'>{trend}</div> </div>""", unsafe_allow_html=True)
    with col2:
        down_base_color_hex = theme_colors.get('down', '#FF3030')
        up_base_color_hex = theme_colors.get('up', SQ_NEON_GREEN)
        alpha_hex_for_steps = "50" # Approx 30% alpha
        down_step_color_rgba = hex_to_rgba_string_with_alpha(down_base_color_hex, alpha_hex_for_steps)
        up_step_color_rgba = hex_to_rgba_string_with_alpha(up_base_color_hex, alpha_hex_for_steps)

        fig_g = go.Figure(go.Indicator(mode="gauge+number",value=prob_pct, title={'text':"Trend Confidence",'font_size':12},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':gauge_bar_color},
                   'steps':[{'range':[0,50],'color': down_step_color_rgba}, {'range':[50,100],'color': up_step_color_rgba}],
                   'threshold' : {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': prob_pct if pd.notna(prob) else 50}
                   }, number={'suffix': "%"}))
        fig_g.update_layout(height=200, margin=dict(t=30,b=30,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_g, use_container_width=True)

    if acc is not None and conf_mx is not None:
        with st.expander("View Trend Model Performance Details", expanded=False):
            acc_str = f"{acc:.2%}" if pd.notna(acc) else "N/A"
            st.markdown(f"<p style='color:{theme_colors.get('title', SQ_NEON_PINK)};'><b>Model Accuracy:</b> <span style='color:white;'>{acc_str}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{theme_colors.get('title', SQ_NEON_PINK)};'><b>Confusion Matrix:</b></p>", unsafe_allow_html=True)
            # Ensure confusion matrix is a list of lists as expected
            if isinstance(conf_mx, list) and len(conf_mx) == 2 and all(isinstance(row, list) and len(row) == 2 for row in conf_mx):
                conf_mx_np = np.array(conf_mx)
                conf_mx_labels = ['Predicted Down', 'Predicted Up']; conf_mx_index = ['Actual Down', 'Actual Up']
                conf_mx_df = pd.DataFrame(conf_mx_np, index=conf_mx_index, columns=conf_mx_labels)
                down_cm_rgba = hex_to_rgba_string_with_alpha(theme_colors.get('down', '#FF3030'), '4D') # 30% alpha
                up_cm_rgba = hex_to_rgba_string_with_alpha(theme_colors.get('up', SQ_NEON_GREEN), 'B3') # 70% alpha
                cm_colorscale = [[0, down_cm_rgba], [1, up_cm_rgba]] # Using Red/Green scale
                fig_cm = px.imshow(conf_mx_df, text_auto=True, aspect="auto", labels=dict(x="Predicted Trend", y="Actual Trend", color="Count"), color_continuous_scale=cm_colorscale)
                fig_cm.update_layout(title_text='Confusion Matrix', title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,30,0.8)', font_color='white', height=300, margin=dict(t=40, b=10, l=10, r=10))
                fig_cm.update_xaxes(side="bottom"); st.plotly_chart(fig_cm, use_container_width=True)
            else: st.text(f"Confusion Matrix data invalid:\n{conf_mx}")


def display_company_info(info, theme_accent_color, theme_key_color):
    if not info: return
    st.markdown(f"<div class='hover-card' style='border-left: 3px solid {theme_accent_color};'>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{theme_accent_color}; margin-bottom: 8px;'>Company Profile</h4>", unsafe_allow_html=True)
    name = info.get('longName', info.get('shortName', 'N/A')); symbol = info.get('symbol', 'N/A')
    st.markdown(f"<p style='color:{theme_key_color}; font-size:1.1em; margin-bottom:5px;'>{name} ({symbol})</p>", unsafe_allow_html=True)
    metrics = {'Sector': info.get('sector', 'N/A'), 'Industry': info.get('industry', 'N/A'),
               'Market Cap': f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else 'N/A',
               '52W High': f"${info.get('fiftyTwoWeekHigh', np.nan):.2f}" if pd.notna(info.get('fiftyTwoWeekHigh')) else 'N/A',
               '52W Low': f"${info.get('fiftyTwoWeekLow', np.nan):.2f}" if pd.notna(info.get('fiftyTwoWeekLow')) else 'N/A',
               'P/E Ratio': f"{info.get('trailingPE', np.nan):.2f}" if pd.notna(info.get('trailingPE')) else 'N/A'}
    cols = st.columns(2)
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % 2]: st.markdown(f"<p style='color:#888; font-size:0.8em; margin:0;'>{key}</p><p style='color:white; font-size:0.9em; margin:0 0 8px 0;'>{value}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def generate_prediction_verdict(current_price, prediction_data, theme_accent_color, theme_up_color):
    # Ensure inputs are valid
    current_price_float = float(current_price) if pd.notna(current_price) else np.nan

    if not prediction_data or pd.isna(current_price_float): return ""

    reg_data = prediction_data.get('regression', {})
    cls_data = prediction_data.get('classification', {})

    pred_price = reg_data.get('prediction', np.nan)
    trend = cls_data.get('trend', 'N/A')
    prob = cls_data.get('probability', np.nan)

    # Ensure prediction results are valid numbers before displaying
    pred_price_float = float(pred_price) if pd.notna(pred_price) else np.nan
    prob_float = float(prob) if pd.notna(prob) else np.nan

    price_chg = ((pred_price_float - current_price_float) / current_price_float) * 100 if pd.notna(pred_price_float) and pd.notna(current_price_float) and current_price_float != 0 else np.nan

    price_chg_color = theme_up_color if price_chg >=0 else '#FF3030' if pd.notna(price_chg) else '#888'

    current_price_str = f"${current_price_float:.2f}" if pd.notna(current_price_float) else "N/A"
    pred_price_str = f"${pred_price_float:.2f}" if pd.notna(pred_price_float) else "N/A"
    price_chg_str = f"({price_chg:+.1f}%)" if pd.notna(price_chg) else ""
    trend_prob_str = f"({prob_float*100:.1f}% conf.)" if pd.notna(prob_float) else ""
    trend_color = theme_up_color if trend == "UP" else '#FF3030' if trend == "DOWN" else '#888'

    verdict_html = f""" <div class='hover-card' style='border-left: 3px solid {theme_accent_color}; margin-bottom: 20px;'>
        <h4 style='color:{theme_accent_color}; margin-bottom: 12px;'>PREDICTION VERDICT (5 DAYS AHEAD)</h4>
        <div style='display: flex; justify-content: space-between; align-items: center;'> <div style='flex: 1;'>
                <p style='color:#AAA; font-size:0.8em; margin:0;'>Current Price</p>
                <p style='color:white; font-size:1.2em; margin:0 0 8px 0;'>{current_price_str}</p>
                <p style='color:#AAA; font-size:0.8em; margin:0;'>Predicted Price</p>
                <p style='color:{price_chg_color}; font-size:1.2em; margin:0 0 8px 0;'>{pred_price_str} <span style='font-size:0.8em;'>{price_chg_str}</span></p>
            </div> <div style='flex: 1; text-align: right;'>
                <p style='color:#AAA; font-size:0.8em; margin:0;'>Predicted Trend</p>
                <p style='color:{trend_color}; font-size:1.2em; margin:0 0 8px 0;'>{trend} <span style='font-size:0.8em;'>{trend_prob_str}</span></p>
            </div></div></div>"""
    return verdict_html

def plot_seasonal_decomposition(data, theme_colors):
    try:
        if data is None or data.empty or 'CLOSE' not in data.columns :
            # st.warning("No valid data for seasonal decomposition.") # removed to avoid clutter
            return None

        cleaned_close = data['CLOSE'].dropna()
        if len(cleaned_close) < 60: # Min length for seasonal_decompose is usually recommended > period*2
            # st.warning(f"Not enough non-NA data points ({len(cleaned_close)}) for seasonal decomposition (min 60 recommended).") # removed to avoid clutter
            return None

        # Determine a suitable period dynamically or use a fixed value
        period_val = 30 # Default period for daily data (approx a month)
        if len(cleaned_close) < period_val * 2 + 2: # Ensure enough data points > 2*period + 2
             period_val = max(2, len(cleaned_close) // 4) # Fallback period if data is short
             if len(cleaned_close) < period_val * 2 + 2:
                  # st.warning(f"Calculated period ({period_val}) for seasonal decomposition is too short or data is too short after dropping NAs ({len(cleaned_close)}).") # removed to avoid clutter
                  return None

        # Handle potential frequency issues for seasonal_decompose
        try:
             decomposition = seasonal_decompose(cleaned_close, period=period_val, model='additive', extrapolate_trend='freq')
        except Exception as e:
             # This can happen if data frequency isn't uniform or period is wrong
             st.warning(f"Seasonal decomposition failed with period {period_val}. Attempting decomposition without specifying period.")
             try:
                  decomposition = seasonal_decompose(cleaned_close, model='additive', extrapolate_trend='freq')
             except Exception as e2:
                  st.error(f"Seasonal decomposition error even without period: {str(e2)}. Data might be too short or irregular.")
                  return None


        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))

        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed', line=dict(color=theme_colors.get('title', AIB_WHITE), width=1.5)), row=1, col=1)
        if decomposition.trend is not None and not decomposition.trend.isnull().all():
            fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend', line=dict(color=theme_colors.get('accent', AIB_GOLD), width=2)), row=2, col=1)
        if decomposition.seasonal is not None and not decomposition.seasonal.isnull().all():
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal', line=dict(color=theme_colors.get('up', AIB_RED), width=1.5)), row=3, col=1)
        if decomposition.resid is not None and not decomposition.resid.isnull().all():
            fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual', line=dict(color=theme_colors.get('down', '#505050'), width=1)), row=4, col=1)

        fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(28,28,30,0.85)', showlegend=False, margin=dict(t=50,l=40,r=20,b=40))
        return fig
    except Exception as e:
        # st.warning(f"Seasonal decomposition error: {str(e)}. Data might be too short or lack seasonality.") # Avoid clutter
        return None


def plot_autocorrelation(data, theme_colors):
    try:
        if data is None or data.empty or 'CLOSE' not in data.columns or data['CLOSE'].dropna().empty:
            return None

        cleaned_close = data['CLOSE'].dropna()
        if len(cleaned_close) < 20:
            # st.warning("Not enough data points for autocorrelation plot after dropping NAs (min 20 needed).") # Avoid clutter
            return None

        nlags = min(40, len(cleaned_close) // 2 - 1)
        if nlags < 1:
            # st.warning("Calculated nlags is too small for autocorrelation plot.") # Avoid clutter
            return None

        acf = sm.tsa.stattools.acf(cleaned_close, nlags=nlags, fft=True)
        pacf = sm.tsa.stattools.pacf(cleaned_close, nlags=nlags, method='ols')

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)'))
        fig.add_trace(go.Bar(x=np.arange(len(acf)), y=acf, name='ACF', marker_color=theme_colors.get('accent', AIB_GOLD), opacity=0.8), row=1, col=1)
        fig.add_trace(go.Bar(x=np.arange(len(pacf)), y=pacf, name='PACF', marker_color=theme_colors.get('up', AIB_RED), opacity=0.8), row=2, col=1)

        # Confidence intervals (Approximate)
        ci = 1.96/np.sqrt(len(cleaned_close))
        for r_val in [1,2]:
            fig.add_hline(y=ci, line_dash="dash", line_color="grey", opacity=0.7, row=r_val, col=1)
            fig.add_hline(y=-ci, line_dash="dash", line_color="grey", opacity=0.7, row=r_val, col=1)

        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(28,28,30,0.85)', showlegend=False, margin=dict(t=50,l=40,r=20,b=40))
        return fig
    except Exception as e:
        # st.warning(f"Autocorrelation plot error: {str(e)}") # Avoid clutter
        return None

def plot_heatmap_advanced(data, theme_colors):
    try:
        if data is None or data.empty: return None
        data_copy = data.copy()
        if 'CLOSE' not in data_copy.columns or data_copy['CLOSE'].isnull().all(): return None # Need CLOSE price

        # Ensure numeric columns are handled
        numeric_cols = data_copy.select_dtypes(include=np.number).columns.tolist()
        data_copy = data_copy[numeric_cols] # Work only with numeric data

        # Calculate advanced features, being mindful of length requirements
        if len(data_copy) > 1:
            data_copy['Returns'] = data_copy['CLOSE'].pct_change()
            data_copy['Log_Returns'] = np.log(data_copy['CLOSE']/data_copy['CLOSE'].shift(1))
        if len(data_copy) >= 20:
            data_copy['MA20'] = data_copy['CLOSE'].rolling(window=20, min_periods=1).mean()
            data_copy['Volatility_20D'] = data_copy['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252) if 'Returns' in data_copy.columns else np.nan
        if len(data_copy) >= 50:
             data_copy['MA50'] = data_copy['CLOSE'].rolling(window=50, min_periods=1).mean()
        if len(data_copy) >= 14:
            data_copy['RSI_14D'] = calculate_rsi(data_copy['CLOSE'], 14)


        features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'Returns', 'Log_Returns', 'MA20', 'MA50', 'Volatility_20D', 'RSI_14D']
        # Filter for features that exist and are not entirely null
        valid_features = [f for f in features if f in data_copy.columns and not data_copy[f].isnull().all()]

        if not valid_features or len(valid_features) < 2: return None # Need at least 2 valid features for correlation

        # Fill NaNs in the valid features before correlation calculation
        # Use ffill/bfill then mean as a robust strategy
        data_copy_valid = data_copy[valid_features].copy()
        data_copy_valid.ffill(inplace=True); data_copy_valid.bfill(inplace=True)
        data_copy_valid.fillna(data_copy_valid.mean(), inplace=True)

        corr_matrix = data_copy_valid.corr()

        if corr_matrix.empty: return None

        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                      colorscale='RdBu', zmin=-1, zmax=1, text=corr_matrix.values.round(2),
                                      texttemplate="%{text}", textfont={"size":10}, hoverongaps=False))
        fig.update_layout(title="Advanced Feature Correlation Matrix", template="plotly_dark", height=600,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)',
                          title_font_color=theme_colors.get('title', AIB_WHITE), title_x=0.5,
                          xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange='reversed',
                          margin=dict(t=50,l=80,r=20,b=80))
        return fig
    except Exception as e:
        # st.warning(f"Advanced heatmap error: {str(e)}") # Avoid clutter
        return None


def plot_stationarity_test(data, theme_colors):
    try:
        if data is None or data.empty or 'CLOSE' not in data.columns or data['CLOSE'].dropna().empty:
            return None
        cleaned_close = data['CLOSE'].dropna()
        if len(cleaned_close) < 20 :
            # st.warning("Not enough data for stationarity test (min 20 non-NA CLOSE points).") # Avoid clutter
            return None

        # Perform ADF test
        adf_result = adfuller(cleaned_close)
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4] # Critical values dictionary

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cleaned_close.index, y=cleaned_close, name='Price', line=dict(color=theme_colors.get('title', AIB_WHITE), width=1.5)))

        # Rolling Mean and Std Dev (ensure sufficient window length)
        window_size = min(20, len(cleaned_close) // 2)
        if window_size >= 1:
             rolling_mean = cleaned_close.rolling(window=window_size, min_periods=1).mean()
             rolling_std = cleaned_close.rolling(window=window_size, min_periods=1).std()

             if not rolling_mean.isnull().all():
                  fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, name=f'Rolling Mean ({window_size}D)', line=dict(color=theme_colors.get('accent', AIB_GOLD), width=1.5)))
             if not rolling_std.isnull().all():
                  fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, name=f'Rolling Std ({window_size}D)', line=dict(color=theme_colors.get('up', AIB_RED), width=1.5)))

        # Add Critical Values as dashed lines
        critical_colors = {'1%': '#FF0000', '5%': '#FFAA00', '10%': '#FFFF00'} # Red, Orange, Yellow
        for key, value in critical_values.items():
             fig.add_hline(y=value, line_dash="dot", line_color=critical_colors.get(key, 'grey'), opacity=0.8, annotation_text=f'Critical Value ({key})', annotation_position="bottom right", annotation_font_size=9)

        # Add ADF Statistic as a dashed line
        fig.add_hline(y=adf_statistic, line_dash="dash", line_color="cyan", opacity=0.9, annotation_text=f'ADF Statistic: {adf_statistic:.2f}', annotation_position="top left", annotation_font_size=9)


        p_value_color = "lightgreen" if p_value <= 0.05 else "salmon"
        title_text = f"Stationarity Test (ADF Statistic: {adf_statistic:.2f}, <span style='color:{p_value_color};'>p-value: {p_value:.3f}</span>)"

        fig.update_layout(title=title_text, template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(28,28,30,0.85)', showlegend=True,
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1, font_size=9),
                          margin=dict(t=60,l=40,r=20,b=40))
        fig.update_yaxes(title_text="Price / Statistic", title_font_size=11, tickfont_size=10)

        return fig
    except Exception as e:
        # st.warning(f"Stationarity test error: {str(e)}") # Avoid clutter
        return None

def display_kmeans_clustering(prediction_data, theme_colors):
    kmeans_data = prediction_data.get('kmeans', {})
    if not kmeans_data or 'cluster_stats' not in kmeans_data or not kmeans_data['cluster_stats']:
        st.warning("K-means clustering results or cluster statistics not available. Ensure enough data points (> number of clusters) were available for training.")
        return

    cluster_stats = pd.DataFrame(kmeans_data['cluster_stats'])
    latest_cluster = kmeans_data.get('latest_cluster', 0) # Default to 0 if missing
    n_clusters = kmeans_data.get('n_clusters', 2) # Default to 2

    if cluster_stats.empty or 'Mean_Value' not in cluster_stats.columns:
        st.warning("Cluster statistics DataFrame is empty or missing 'Mean_Value'. Cannot display cluster analysis.")
        return

    # Determine bullish/bearish clusters based on mean value
    high_cluster = cluster_stats.loc[cluster_stats['Mean_Value'].idxmax()]['Cluster'] if not cluster_stats['Mean_Value'].empty else 0
    low_cluster = cluster_stats.loc[cluster_stats['Mean_Value'].idxmin()]['Cluster'] if not cluster_stats['Mean_Value'].empty else (1 if n_clusters > 1 else 0)

    st.markdown(f""" <div class='hover-card' style='background-color:rgba(0,0,0,0.5); padding:15px; border-radius:10px; border-left:4px solid {theme_colors.get("title", SQ_NEON_PINK)}; margin-bottom:20px;'>
        <h3 style='color:{theme_colors.get("title", SQ_NEON_PINK)}; text-align:center;'>K-means Clustering Analysis</h3>
        <p style='color:white; text-align:center; font-size:0.95em;'>The current data point is most similar to the <span style='color:{theme_colors.get("cluster1", SQ_NEON_GREEN) if latest_cluster == high_cluster else theme_colors.get("cluster2", "#FF3030")}; font-weight:bold;'>Cluster {latest_cluster}</span> data points used for training.</p>
    </div> """, unsafe_allow_html=True)

    # Ensure train_clusters exists and has enough data for time analysis
    train_clusters = kmeans_data.get('train_clusters', [])
    can_plot_time = train_clusters is not None and len(train_clusters) > n_clusters

    col_layout = [1, 1] if can_plot_time else [1] # Adjust layout based on whether time plot is possible

    main_cols = st.columns(col_layout)

    with main_cols[0]:
        st.markdown(f"<h4 style='color:{theme_colors.get('title', SQ_NEON_PINK)};'>Cluster Distribution & Stats</h4>", unsafe_allow_html=True)
        fig = go.Figure()
        for cluster_idx in range(n_clusters):
            color = theme_colors.get('cluster1', SQ_NEON_GREEN) if cluster_idx == high_cluster else theme_colors.get('cluster2', '#FF3030') if cluster_idx == low_cluster else "#888888"
            name = "Bullish Cluster" if cluster_idx == high_cluster else "Bearish Cluster" if cluster_idx == low_cluster else f"Cluster {cluster_idx}"

            current_cluster_stats = cluster_stats[cluster_stats['Cluster'] == cluster_idx]
            if current_cluster_stats.empty: continue

            cluster_mean = current_cluster_stats['Mean_Value'].values[0]
            cluster_std = current_cluster_stats['Std_Value'].values[0]
            cluster_count = current_cluster_stats['Count'].values[0]

            fig.add_trace(go.Bar(x=[f'Cluster {cluster_idx}'], y=[cluster_count], name=name, marker_color=color, opacity=0.9 if cluster_idx == latest_cluster else 0.6,
                width=0.6, marker_line_width=3 if cluster_idx == latest_cluster else 0, marker_line_color='white' if cluster_idx == latest_cluster else None,
                text=[f"Mean: ${cluster_mean:.2f}<br>Count: {cluster_count}<br>Std: ${cluster_std:.2f}"], hoverinfo="text", textposition="auto"))
        fig.update_layout(template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)',
            margin=dict(t=30, b=30, l=40, r=30), title="Number of Data Points per Cluster", title_font_color=theme_colors.get('title', SQ_NEON_PINK),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f""" <div style='background-color:rgba(30,30,32,0.7); padding:15px; border-radius:5px; margin-top:10px;'>
            <h5 style='color:{theme_colors.get("title", SQ_NEON_PINK)};'>Key Insights for Cluster {latest_cluster}</h5>
            <ul style='color:white; margin-left:20px; font-size:0.95em;'>
                <li>Your data point belongs to <span style='color:{theme_colors.get("cluster1", SQ_NEON_GREEN) if latest_cluster == high_cluster else theme_colors.get("cluster2", "#FF3030")}; font-weight:bold;'>Cluster {latest_cluster}</span>.</li>
                <li>This cluster contains <span style='font-weight:bold;'>{kmeans_data.get('latest_cluster_count', 0)}</span> historical data points.</li>
                <li>Average <span style='font-weight:bold;'>5-day future price change</span> in this cluster: <span style='font-weight:bold;'>${kmeans_data.get('latest_cluster_mean', np.nan):.2f}</span> (compared to price 5 days prior).</li>
                <li>This cluster is primarily <span style='font-weight:bold;'>{("Bullish" if latest_cluster == high_cluster else "Bearish") if n_clusters == 2 else "Neutral"}</span>.</li>
            </ul> </div> """, unsafe_allow_html=True)

    if can_plot_time:
        with main_cols[1]:
            st.markdown(f"<h4 style='color:{theme_colors.get('title', SQ_NEON_PINK)};'>Cluster Distribution Over Time</h4>", unsafe_allow_html=True)
            points_over_time = pd.DataFrame({'date_index': range(len(train_clusters)), 'cluster': train_clusters})
            # Use original index for time plot if available
            if st.session_state.get('corr_df') is not None and len(st.session_state.corr_df) >= len(train_clusters):
                time_index_for_clusters = st.session_state.corr_df.index[-len(train_clusters):]
                points_over_time['date'] = time_index_for_clusters
            else: # Fallback to simple date range if index not available/matching
                 last_date = datetime.now().date() # Approximate end date
                 start_date = last_date - timedelta(days=len(train_clusters) - 1)
                 points_over_time['date'] = pd.date_range(start=start_date, periods=len(train_clusters), freq='D')


            # Plot clusters over time directly
            fig_time = go.Figure()
            for cluster_id_time in range(n_clusters):
                 cluster_color_time = theme_colors.get('cluster1', SQ_NEON_GREEN) if cluster_id_time == high_cluster else theme_colors.get('cluster2', '#FF3030') if cluster_id_time == low_cluster else "#888888"
                 cluster_points = points_over_time[points_over_time['cluster'] == cluster_id_time]
                 if not cluster_points.empty:
                    fig_time.add_trace(go.Scatter(x=cluster_points['date'], y=[cluster_id_time] * len(cluster_points), mode='markers', name=f'Cluster {cluster_id_time}',
                                                  marker=dict(color=cluster_color_time, size=8, opacity=0.9 if cluster_id_time == latest_cluster else 0.6,
                                                              line=dict(width=2 if cluster_id_time == latest_cluster else 0, color='white'))))

            fig_time.update_layout(template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)',
                margin=dict(t=30, b=40, l=40, r=30), yaxis_title="Cluster ID", xaxis_title="Date", title="Cluster Assignment Over Time",
                title_font_color=theme_colors.get('title', SQ_NEON_PINK), showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                yaxis=dict(tickvals=list(range(n_clusters)), ticktext=[f'Cluster {i}' for i in range(n_clusters)]))
            st.plotly_chart(fig_time, use_container_width=True)
    elif not can_plot_time and len(train_clusters) > 0:
         st.info(f"Not enough data points ({len(train_clusters)}) compared to number of clusters ({n_clusters}) to meaningfully plot cluster distribution over time.")


    st.markdown(f"<h4 style='color:{theme_colors.get('title', SQ_NEON_PINK)};margin-top:20px;'>Detailed Cluster Metrics</h4>", unsafe_allow_html=True)
    formatted_stats = cluster_stats.copy()
    if not formatted_stats.empty:
        # Rename columns before formatting if necessary
        formatted_stats.rename(columns={'Mean_Value': 'Mean Price Change', 'Std_Value': 'Standard Deviation'}, inplace=True)

        formatted_stats['Mean Price Change'] = formatted_stats['Mean Price Change'].map('${:.2f}'.format)
        formatted_stats['Standard Deviation'] = formatted_stats['Standard Deviation'].map('${:.2f}'.format)
        formatted_stats.columns = ['Cluster', 'Mean Price Change (5D Fwd)', 'Standard Deviation', 'Number of Points']
        formatted_stats.set_index('Cluster', inplace=True)
        def highlight_row(row): return ['background-color: rgba(255,255,255,0.15)' if row.name == latest_cluster else '' for _ in row]
        st.dataframe(formatted_stats.style.apply(highlight_row, axis=1), use_container_width=True, height=formatted_stats.shape[0] * 40 + 50 if not formatted_stats.empty else 150)

    st.markdown(f""" <div style='background-color:rgba(20,20,22,0.85); padding:20px; border-radius:10px; margin-top:20px; border:1px solid {theme_colors.get("title", SQ_NEON_PINK)}40;'>
        <h4 style='color:{theme_colors.get("title", SQ_NEON_PINK)}; text-align:center;'>Trading Signal Interpretation</h4>
        <div style='display:flex; justify-content:space-between; margin:15px 0; gap:20px;'>
            <div style='flex:1; padding:15px; background-color:rgba(0,0,0,0.5); border-radius:8px; border-left:3px solid {theme_colors.get("cluster1", SQ_NEON_GREEN)};'>
                <h5 style='color:{theme_colors.get("cluster1", SQ_NEON_GREEN)}; margin-bottom:8px;'>Bullish Signal Strength</h5>
                <div style='height:25px; background-color:#333; border-radius:12px; overflow:hidden; margin:10px 0; box-shadow: inset 0 0 5px rgba(0,0,0,0.5);'>
                    <div style='height:100%; width:{(90 if latest_cluster == high_cluster else 20)}%; background-color:{theme_colors.get("cluster1", SQ_NEON_GREEN)}; border-radius:12px; transition: width 0.5s ease;'></div>
                </div> <p style='color:white; font-size:0.9em; margin-top:15px;'> {( "Strong bullish signal: Current pattern aligns with historical periods that saw significant price increases." if latest_cluster == high_cluster else "Weak bullish signal: Current pattern is not typical of past bullish market conditions.")} </p>
            </div>
             <div style='flex:1; padding:15px; background-color:rgba(0,0,0,0.5); border-radius:8px; border-left:3px solid {theme_colors.get("cluster2", "#FF3030")};'>
                <h5 style='color:{theme_colors.get("cluster2", "#FF3030")}; margin-bottom:8px;'>Bearish Signal Strength</h5>
                <div style='height:25px; background-color:#333; border-radius:12px; overflow:hidden; margin:10px 0; box-shadow: inset 0 0 5px rgba(0,0,0,0.5);'>
                    <div style='height:100%; width:{(90 if latest_cluster == low_cluster else 20)}%; background-color:{theme_colors.get("cluster2", "#FF3030")}; border-radius:12px; transition: width 0.5s ease;'></div>
                </div> <p style='color:white; font-size:0.9em; margin-top:15px;'> {( "Strong bearish signal: Current pattern aligns with historical periods that saw significant price decreases." if latest_cluster == low_cluster else "Weak bearish signal: Current pattern is not typical of past bearish market conditions.")} </p>
            </div>
        </div>
        <div style='text-align:center; margin-top:20px; padding:15px; background-color:rgba(40,40,45,0.7); border-radius:8px; border:1px solid {theme_colors.get("title", SQ_NEON_PINK)}40;'>
            <span style='color:white; font-weight:bold; font-size:1.1em;'>Overall Indication: </span>
            <span style='color:{(theme_colors.get("cluster1", SQ_NEON_GREEN) if latest_cluster == high_cluster else theme_colors.get("cluster2", "#FF3030") if latest_cluster == low_cluster else "white")}; font-weight:bold; font-size:1.1em;'>
                {( "BULLISH bias indicated by K-means clustering." if latest_cluster == high_cluster else "BEARISH bias indicated by K-means clustering." if latest_cluster == low_cluster else "NEUTRAL bias indicated by K-means clustering.")}
            </span>
             <p style='color:#AAA; font-size:0.8em; margin-top:10px;'>Note: This is an experimental analysis. Use alongside other indicators and predictions.</p>
        </div>
    </div> """, unsafe_allow_html=True)


# --- RENDER WELCOME PAGE ---
def render_welcome_page():
    # Ensure base CSS is applied
    st.markdown(APP_BASE_CSS, unsafe_allow_html=True)

    # Welcome page specific CSS inspired by E-Corp / Dark Army, now with red as primary
    welcome_page_specific_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
        body {{
            background-color: {DA_BLACK} !important;
            color: {DA_RED_ACCENT} !important;
            font-family: 'VT323', monospace, 'Courier New', Courier !important;
            background: linear-gradient(120deg, {DA_BLACK} 80%, {DA_RED_ACCENT}30 100%), url('{BINARY_PATTERN_RED_BG}');
            animation: bg-shift 40s infinite alternate ease-in-out;
        }}
        @keyframes bg-shift {{
            0% {{ background-position: 0% 0%; }}
            100% {{ background-position: 100% 100%; }}
        }}
        [data-testid="stHeader"], footer {{ display: none !important; }}
        div.stApp > section.main > div.block-container {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            min-height: 100vh !important;
            padding: 40px 20px !important;
        }}
        /* Sidebar */
        [data-testid="stSidebar"][aria-expanded="true"] {{
            background: rgba(30,0,0,0.95) !important;
            border-right: 2px solid {DA_RED_ACCENT}A0 !important;
            width: 360px !important;
            box-shadow: 5px 0px 30px {DA_RED_ACCENT}60;
            padding: 30px 25px !important;
            backdrop-filter: blur(14px);
            font-family: 'VT323', monospace !important;
            color: {DA_RED_ACCENT} !important;
        }}
        [data-testid="stSidebar"] h2 {{
            color: {DA_RED_ACCENT} !important;
            font-family: VT323, monospace;
            text-align:center;
            margin-bottom:25px;
            text-shadow: 0 0 18px {DA_RED_ACCENT}B0;
            letter-spacing: 2px;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio > label span {{
            color: {DA_RED_ACCENT}E0 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1em !important;
            transition: all 0.3s;
            letter-spacing: 0.8px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover span {{
            color: {DA_RED_ACCENT} !important;
            text-shadow: 0 0 10px {DA_RED_ACCENT}90;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            background: transparent !important; color: {DA_RED_ACCENT} !important;
            border: 1px solid {DA_RED_ACCENT}A0 !important; padding: 14px !important; font-size: 1.1em !important;
            font-family: 'VT323', monospace !important; border-radius: 8px !important; margin-top: 20px;
            transition: all 0.3s ease-out !important; letter-spacing: 1.5px;
            position: relative; overflow: hidden; z-index: 1;
            text-shadow: 0 0 8px {DA_RED_ACCENT}80;
            box-shadow: 0 0 12px {DA_RED_ACCENT}50;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            color: {DA_BLACK} !important;
            box-shadow: 0 0 30px {DA_RED_ACCENT}C0 !important;
            transform: translateY(-3px) scale(1.05);
            background: {DA_RED_ACCENT} !important;
        }}
        [data-testid="stSidebar"] .stTextInput > div > div > input,
        [data-testid="stSidebar"] .stSelectbox > div > div {{
            background-color: rgba(40,0,0,0.8) !important; color: {DA_RED_ACCENT} !important;
            border: 1px solid {DA_RED_ACCENT}80 !important; font-family: 'VT323', monospace !important;
            border-radius: 8px !important; padding: 10px 15px;
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
            text-shadow: 0 0 3px {DA_RED_ACCENT}50;
        }}
        [data-testid="stSidebar"] .stTextInput > div > div > input:focus,
        [data-testid="stSidebar"] .stSelectbox > div > div:focus-within {{
            border: 1px solid {DA_RED_ACCENT} !important;
            box-shadow: 0 0 18px {DA_RED_ACCENT}A0 !important;
        }}
        /* Main Content Area */
        .welcome-main-content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
            text-align: center;
            background: rgba(30,0,0,0.7);
            border-radius: 24px;
            border: 2.5px solid {DA_RED_ACCENT}A0;
            box-shadow: 0 0 60px {DA_RED_ACCENT}30, 0 0 40px {DA_BLACK}B0;
            backdrop-filter: blur(10px);
            max-width: 950px;
            width: 98%;
            margin-bottom: 40px;
            animation: glass-fade-in 1.2s cubic-bezier(.68,-0.55,.27,1.55);
        }}
        @keyframes glass-fade-in {{
            0% {{ opacity: 0; transform: scale(0.95); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        .welcome-main-content:hover {{
            box-shadow: 0 0 90px {DA_RED_ACCENT}80, 0 0 60px {DA_RED_ACCENT}40;
            border-color: {DA_RED_ACCENT}FF;
            transform: scale(1.01) rotate(-1deg);
        }}
        .welcome-logo-text {{
            font-family: 'VT323', monospace;
            font-size: 6em;
            color: {DA_RED_ACCENT};
            text-shadow: 0 0 30px {DA_RED_ACCENT}, 0 0 60px {DA_RED_ACCENT}80;
            animation: logo-pulse-red 3.5s infinite alternate cubic-bezier(.68,-0.55,.27,1.55);
            margin-bottom: 10px;
            letter-spacing: 8px;
        }}
        @keyframes logo-pulse-red {{
            0%,100% {{ text-shadow: 0 0 30px {DA_RED_ACCENT}, 0 0 60px {DA_RED_ACCENT}80; opacity: 1; }}
            50% {{ text-shadow: 0 0 60px {DA_RED_ACCENT}, 0 0 120px {DA_RED_ACCENT}A0; opacity: 0.85; }}
        }}
        .welcome-main-title {{
            font-size: 2.7em;
            color: #fff;
            text-shadow: 0 0 18px {DA_RED_ACCENT}A0, 0 0 30px {DA_RED_ACCENT}60;
            animation: title-flicker 2.5s infinite alternate;
            margin-bottom: 30px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            letter-spacing: 1.5px;
        }}
        @keyframes title-flicker {{
            0%,100% {{ opacity: 0.95; }}
            45% {{ opacity: 0.7; }}
            50% {{ opacity: 0.3; }}
            55% {{ opacity: 0.7; }}
            60% {{ opacity: 1; }}
        }}
        .welcome-subtitle {{
            font-size: 1.7em;
            color: {DA_RED_ACCENT};
            margin-bottom: 40px;
            letter-spacing: 1.5px;
            opacity: 0.97;
            font-family: 'VT323', monospace, 'Courier New', Courier;
            text-shadow: 0 0 12px {DA_RED_ACCENT}A0;
            animation: subtitle-bounce 2.2s infinite alternate cubic-bezier(.68,-0.55,.27,1.55);
        }}
        @keyframes subtitle-bounce {{
            0% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-8px) scale(1.04); }}
            100% {{ transform: translateY(0); }}
        }}
        /* Funky module cards */
        .welcome-modules-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 40px;
            margin: 40px 0;
            width: 100%;
        }}
        [data-testid="column"] {{
            background: rgba(40,0,0,0.7);
            border: 2.5px solid {DA_RED_ACCENT}A0;
            border-radius: 18px;
            padding: 28px;
            margin: 15px;
            text-align: center;
            transition: all 0.4s cubic-bezier(.68,-0.55,.27,1.55);
            max-width: 270px;
            /* cursor: pointer; /* Button provides cursor indication */
            box-shadow: 0 0 22px {DA_RED_ACCENT}30, 0 0 10px {DA_RED_ACCENT}10;
            position: relative;
            overflow: hidden;
            display: flex; /* Added for flex layout of content */
            flex-direction: column; /* Added for flex layout of content */
            justify-content: space-between; /* Distribute space */
        }}
        [data-testid="column"]:hover {{
            transform: translateY(-12px) scale(1.07) rotate(-2deg);
            box-shadow: 0 0 60px {DA_RED_ACCENT}B0, 0 0 30px {DA_RED_ACCENT}80 inset;
            border-color: {DA_RED_ACCENT}FF;
            background: linear-gradient(120deg, rgba(60,0,0,0.9) 60%, {DA_RED_ACCENT}40 100%);
        }}
        [data-testid="column"]:before {{
            content: '';
            position: absolute;
            top: -40%; left: -40%; width: 180%; height: 180%;
            background: radial-gradient(circle, {DA_RED_ACCENT}30 0%, transparent 80%);
            opacity: 0.7;
            z-index: 0;
            pointer-events: none;
            animation: card-glow 3s infinite alternate;
        }}
        @keyframes card-glow {{
            0% {{ opacity: 0.5; }}
            100% {{ opacity: 0.9; }}
        }}
        .welcome-module-icon-text {{
            font-size: 3.2em;
            margin-bottom: 15px; /* Increased margin for button */
            color: {DA_RED_ACCENT};
            text-shadow: 0 0 18px {DA_RED_ACCENT}A0, 0 0 30px {DA_RED_ACCENT}60;
            animation: icon-spin 4s infinite linear;
        }}
        @keyframes icon-spin {{
            0% {{ transform: rotate(-8deg) scale(1); }}
            50% {{ transform: rotate(8deg) scale(1.12); }}
            100% {{ transform: rotate(-8deg) scale(1); }}
        }}
        /* .welcome-module-title removed */
        .welcome-module-description {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 1em;
            color: #fff;
            margin-top: 15px; /* Increased margin from button */
            line-height: 1.5;
            text-shadow: 0 0 4px {DA_RED_ACCENT}40;
        }}
        .welcome-footer-text {{
            font-size:0.7em; text-align:right; color:{DA_RED_ACCENT}70; margin-top:auto;
            font-family:'VT323', monospace; position:absolute; bottom:20px; right:30px;
            animation: flicker 5s infinite step-end 1s;
        }}
        .welcome-quote {{
            font-size: 1.1em;
            color: {DA_RED_ACCENT}D0;
            max-width: 700px;
            border: 1.5px dashed {DA_RED_ACCENT}A0;
            padding: 25px;
            margin-top: 40px;
            line-height: 1.7;
            background: rgba(60,0,0,0.5);
            position: relative;
            transition: all 0.4s ease;
            font-family: 'VT323', monospace, 'Courier New', Courier;
            letter-spacing: 0.8px;
            box-shadow: 0 0 18px {DA_RED_ACCENT}30;
            border-radius: 14px;
            animation: quote-fade-in 1.5s cubic-bezier(.68,-0.55,.27,1.55);
        }}
        @keyframes quote-fade-in {{
            0% {{ opacity: 0; transform: translateY(30px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .welcome-quote:hover {{
            border-color: {DA_RED_ACCENT}FF;
            box-shadow: 0 0 30px {DA_RED_ACCENT}80;
            transform: scale(1.03);
            background: rgba(80,0,0,0.6);
        }}
        .cursor-blink {{
            display: inline-block;
            width: 12px;
            height: 1.4em;
            background-color: {DA_RED_ACCENT};
            animation: blink-c 1s step-end infinite;
            margin-left: 8px;
            vertical-align: text-bottom;
            border-radius: 2px;
        }}
        @keyframes blink-c {{
            from,to{{background-color:transparent;}}
            50%{{background-color:{DA_RED_ACCENT};}}
        }}
        /* Style for buttons inside welcome module cards if different from global */
        [data-testid="column"] .stButton > button {{
            background: transparent !important;
            color: {DA_RED_ACCENT} !important;
            border: 1px solid {DA_RED_ACCENT}A0 !important;
            font-size: 1.2em !important; /* Make it more prominent like a title */
            font-family: 'VT323', monospace !important;
            border-radius: 6px !important;
            margin-top: 5px; /* Adjust as needed */
            margin-bottom: 5px; /* Adjust as needed */
            padding: 10px !important;
            width: auto; /* Allow button to size to content + padding */
            min-width: 150px; /* Ensure a minimum width */
            align-self: center; /* Center the button within the card column */
            text-shadow: 0 0 8px {DA_RED_ACCENT}80;
            box-shadow: 0 0 10px {DA_RED_ACCENT}40;
        }}
        [data-testid="column"] .stButton > button:hover {{
            color: {DA_BLACK} !important;
            box-shadow: 0 0 20px {DA_RED_ACCENT}B0 !important;
            background: {DA_RED_ACCENT} !important;
            transform: scale(1.08); /* Slightly more pop */
        }}
        [data-testid="column"] .stButton > button::before {{
            display:none; /* Disable general hover effect for these specific buttons */
        }}

    </style>
    """
    st.markdown(welcome_page_specific_css, unsafe_allow_html=True)


    with st.sidebar:
        st.markdown(f"<h2 style='color:{DA_RED_ACCENT} !important; text-shadow: 0 0 10px {DA_RED_ACCENT}90 !important;'>E-CORP UPLINK TERMINAL</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{DA_RED_ACCENT}BF; font-size:0.8em; text-align:center; margin-top:-15px; margin-bottom:25px;'>CONNECT TO MARKET DATA NODE</p>", unsafe_allow_html=True)
        
        data_source = st.radio( "DATA SOURCE SELECTION:", ["Yahoo Finance API", "Upload CSV File"], key="welcome_data_source_radio", horizontal=False)

        if data_source == "Yahoo Finance API":
            ticker_input = st.text_input("TARGET IDENTIFIER (TICKER):", value=st.session_state.get("ticker", "AAPL"), key="welcome_ticker_api_input").upper()
            
            st.markdown(f"<p style='color:{DA_RED_ACCENT}BF; font-size:0.9em; margin-top:5px; margin-bottom:10px;'>Select the time range for historical data analysis:</p>", unsafe_allow_html=True)
            
            period_options = ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max"]
            period_descriptions = {
                "3 Months": "Short-term analysis (90 days)",
                "6 Months": "Medium-term analysis (180 days)",
                "1 Year": "Annual trend analysis (365 days)",
                "2 Years": "Extended trend analysis (730 days)",
                "5 Years": "Long-term analysis (1825 days)",
                "Max": "Maximum available historical data"
            }
            
            with st.container():
                st.markdown(
                    f"""
                    <div style='background:rgba(40,0,0,0.7); border:2px solid {DA_RED_ACCENT}A0; border-radius:12px; padding:18px 18px 10px 18px; margin-bottom:18px;'>
                        <div style='margin-bottom:8px; color:{DA_RED_ACCENT}; font-weight:700; font-size:1.1em;'>HISTORICAL DEPTH:</div>
                    """,
                    unsafe_allow_html=True
                )
                selected_period = st.selectbox(
                    "Select period:", # Changed from "Select period (hidden label):"
                    period_options,
                    index=period_options.index(st.session_state.get("selected_period", "1 Year")),
                    key="welcome_period_selectbox",
                    help="Choose how far back to analyze historical data",
                    label_visibility="visible" # Changed from "collapsed"
                )
                st.markdown(
                    f"<div style='color:#FFD6D6; font-size:0.95em; margin-top:8px; font-weight:600;'>{period_descriptions[selected_period]}</div></div>",
                    unsafe_allow_html=True
                )
            
            st.session_state.ticker_input_welcome = ticker_input
            st.session_state.selected_period_welcome = selected_period
        else:
            uploaded_file = st.file_uploader("SECURE DATA UPLOAD (CSV):", type="csv", key="welcome_csv_uploader_input")
            st.session_state.uploaded_file_welcome = uploaded_file
            if uploaded_file: st.caption(f"FILE: {uploaded_file.name}")

        st.markdown(f"<hr style='border-color: {DA_RED_ACCENT}50;'>", unsafe_allow_html=True) 
        if st.button("ESTABLISH CONNECTION", key="welcome_load_button", use_container_width=True):
            with st.spinner(">>> DATA STREAM ESTABLISHED... VALIDATING PROTOCOL..."):
                error = None; data = None; info = None
                if data_source == "Yahoo Finance API":
                    current_ticker = st.session_state.ticker_input_welcome
                    current_period = st.session_state.selected_period_welcome
                    period_map = {"3 Months":"3mo", "6 Months":"6mo", "1 Year":"1y", "2 Years":"2y", "5 Years":"5y", "Max":"max"}
                    data, info, error = load_stock_data(current_ticker, period=period_map.get(current_period, "1y")) 
                    if not error: st.session_state.ticker = current_ticker
                elif st.session_state.get('uploaded_file_welcome') is not None:
                    up_file = st.session_state.uploaded_file_welcome
                    try:
                        temp_data = pd.read_csv(up_file)
                        column_variations = {
                            'open': ['open', 'opening', 'open_price', 'first', 'first_price', 'start', 'start_price', 'opening_price'],
                            'high': ['high', 'highest', 'high_price', 'max', 'maximum', 'highest_price', 'upper', 'top'],
                            'low': ['low', 'lowest', 'low_price', 'min', 'minimum', 'lowest_price', 'bottom', 'lower'],
                            'close': ['close', 'closing', 'close_price', 'last', 'last_price', 'price', 'final', 'end', 'end_price'],
                            'volume': ['volume', 'vol', 'trade_volume', 'trading_volume', 'quantity', 'amount', 'trades']
                        }
                        date_variations = ['date', 'time', 'datetime', 'timestamp', 'day', 'trading_day', 'date_time', 'period']
                        date_col = None
                        for col_name_csv in temp_data.columns:
                            if any(date_var in col_name_csv.lower() for date_var in date_variations):
                                try:
                                    temp_data[col_name_csv] = pd.to_datetime(temp_data[col_name_csv], errors='coerce')
                                    if temp_data[col_name_csv].notna().any():
                                        date_col = col_name_csv
                                        break 
                                except: continue 

                        if date_col is None and not temp_data.columns.empty:
                             try:
                                 temp_data[temp_data.columns[0]] = pd.to_datetime(temp_data[temp_data.columns[0]], errors='coerce')
                                 if temp_data[temp_data.columns[0]].notna().any():
                                      date_col = temp_data.columns[0]
                             except: pass 

                        if date_col is None:
                             raise ValueError("Could not find a valid date column. Please ensure your CSV has a column containing dates.")

                        temp_data.set_index(date_col, inplace=True)
                        temp_data.sort_index(inplace=True) 

                        column_mapping = {}; missing_cols = []; available_cols = []
                        for std_col, variations in column_variations.items():
                            found = False
                            for col_name_csv_map in temp_data.columns:
                                if any(var in col_name_csv_map.lower() for var in variations):
                                    column_mapping[std_col] = col_name_csv_map
                                    available_cols.append(f"{std_col.upper()}: '{col_name_csv_map}'")
                                    found = True; break
                            if not found: missing_cols.append(std_col)

                        if 'volume' in missing_cols and all(col_map_key_vol in column_mapping for col_map_key_vol in ['high', 'low', 'close']):
                             st.info("VOLUME column not found. Generating synthetic volume data.")
                             temp_data['synthetic_volume'] = ((temp_data[column_mapping['high']] - temp_data[column_mapping['low']]) * temp_data[column_mapping['close']] * 10000).astype(int).clip(lower=1000)
                             column_mapping['volume'] = 'synthetic_volume'
                             available_cols.append("VOLUME: 'synthetic_volume'")
                             missing_cols.remove('volume') 

                        core_price_cols = [mc for mc in missing_cols if mc in ['open', 'high', 'low', 'close']]
                        if core_price_cols:
                             error_msg_csv = f"Missing required price columns: {', '.join(m.upper() for m in core_price_cols)}.\n"
                             error_msg_csv += f"Attempted to map available columns: {', '.join(available_cols)}\n"
                             error_msg_csv += "Your CSV columns: " + ", ".join(temp_data.columns)
                             raise ValueError(error_msg_csv)
                        elif missing_cols: 
                            st.warning(f"Missing non-core columns: {', '.join(m.upper() for m in missing_cols)}. Analysis may be limited.")


                        selected_cols = [column_mapping[c_key_data] for c_key_data in column_variations.keys() if c_key_data in column_mapping]
                        if not selected_cols: raise ValueError("No relevant columns found in CSV.")

                        data = temp_data[selected_cols].copy()
                        data.columns = [col_name_data_std.upper() for col_name_data_std in column_variations.keys() if col_name_data_std in column_mapping]

                        for col_name_final in data.columns:
                             if col_name_final != 'DATE': 
                                data[col_name_final] = pd.to_numeric(data[col_name_final], errors='coerce')
                        data.dropna(subset=['CLOSE'], inplace=True) 
                        data.ffill().bfill(inplace=True) 

                        if 'VOLUME' in data.columns:
                             data['VOLUME'] = data['VOLUME'].fillna(0)
                             data['VOLUME'] = data['VOLUME'].astype(float).fillna(0).astype(np.int64)

                        if data.empty: raise ValueError("Data is empty after processing CSV.")

                        info = {'longName': f'CSV: {up_file.name}', 'symbol': 'CustomCSV'}; st.session_state.ticker = "CSV Data"
                    except Exception as e_csv: error = f"CSV Processing Error: {str(e_csv)}" 
                else: error = "No data source selected or file uploaded."

                if data is not None and not data.empty and error is None:
                    if 'CLOSE' not in data.columns or data['CLOSE'].isnull().all():
                         error = "Loaded data does not contain a valid 'CLOSE' price column."
                         data = None 
                    elif len(data) < 2:
                         error = f"Loaded data has only {len(data)} data point(s). Need at least 2 for basic analysis."
                         data = None 

                if data is not None and not data.empty and error is None:
                    st.session_state.data = data; st.session_state.info = info; st.session_state.error = None
                    st.session_state.data_loaded_from_welcome = True
                    st.session_state.current_stage = "aib_analysis" 
                    st.success("PROTOCOL SYNCED. ARENA INITIALIZED."); time.sleep(0.5); 
                    st.rerun()
                else:
                    st.session_state.data = None 
                    st.session_state.info = None
                    st.session_state.error = error or "Data acquisition failed or resulted in empty/invalid data."
                    st.sidebar.error(f"LINK ERROR: {st.session_state.error}")

        st.markdown(f"""
        <div class='welcome-footer-text'>
            SYSTEM: ONLINE<br>
            PROTOCOL: ACTIVE<br>
            v2.3 // E-Corp-DA Nexus
        </div>
        """, unsafe_allow_html=True)


    # Main welcome page content
    st.markdown(f"""
        <div class="welcome-main-content">
            <div class="welcome-logo-text">SVN</div>
            <h1 class="welcome-main-title">E-Corp StockVerse</h1>
            <p class="welcome-subtitle">Market Simulation Protocol Initialized <span class="cursor-blink"></span></p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='welcome-modules-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1_button_key = "welcome_btn_col1"
    col2_button_key = "welcome_btn_col2"
    col3_button_key = "welcome_btn_col3"

    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="welcome-module-icon-text">{DARK_ARMY_MASK_TEXT}</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("STOCK ARENA", key=col1_button_key, use_container_width=False) # use_container_width=False for custom sizing via CSS
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="welcome-module-description">Navigate market volatility and uncover trends.</div>
            </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="welcome-module-icon-text">{DARK_ARMY_MASK_TEXT}</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("SQUID ML", key=col2_button_key, use_container_width=False)
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="welcome-module-description">Engage AI in predictive market games.</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="welcome-module-icon-text">{DARK_ARMY_MASK_TEXT}</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("HEIST HUB", key=col3_button_key, use_container_width=False)
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="welcome-module-description">Gather intel for the ultimate financial operation.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True) 

    if st.session_state.get(col1_button_key): st.session_state.current_stage = "aib_analysis"; st.rerun()
    if st.session_state.get(col2_button_key): st.session_state.current_stage = "squid_ml"; st.rerun()
    if st.session_state.get(col3_button_key): st.session_state.current_stage = "moneyheist"; st.rerun()

    st.markdown(f"""
        <div class="welcome-quote">
            <q>"OUR DEMOCRACY HAS BEEN HACKED. REAL POWER HAS BEEN CONCENTRATED IN THE HANDS OF A FEW CORPS."</q>
            <br><span style="font-size:0.9em; color:{DA_RED_ACCENT}CC; display:block; margin-top:15px;">> INITIATE DATA UPLINK VIA LEFT PANEL TO BEGIN SIMULATION.</span>
        </div>
    """, unsafe_allow_html=True)


# --- RENDER AIB STOCK ANALYSIS MODULE ---
def render_aib_stock_analysis_module():
    # Apply base CSS first, then page-specific overrides
    st.markdown(APP_BASE_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
        body {{ background-color: {AIB_BLACK} !important; }}
        .stApp > section.main > div.block-container {{ display: block !important; min-height: auto !important; padding: 1.5rem 2rem 1rem 2rem !important; }}
        [data-testid="stSidebar"][aria-expanded="true"] {{
            background-color: #101010 !important; border-right: 3px solid {AIB_RED} !important;
            box-shadow: 3px 0 15px {AIB_RED}50; width: 320px !important;
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; /* Reset sidebar font */
             padding: 20px !important; /* Reset sidebar padding */
        }}
         [data-testid="stSidebar"][aria-expanded="true"] h2 {{ color:{AIB_RED}; text-align:center; font-family:"Arial Black", Gadget, sans-serif; letter-spacing:1px; text-shadow: 1px 1px 2px {AIB_BLACK}; margin-bottom:15px; }}
         [data-testid="stSidebar"][aria-expanded="true"] p {{ color:{AIB_WHITE}BF; font-size:0.8em; text-align:center; margin-top:-10px; margin-bottom:15px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button {{
            background-color: {AIB_BLACK} !important; color: {AIB_WHITE} !important; border: 1px solid {AIB_RED}BF !important;
            box-shadow: 0 2px 4px {AIB_BLACK}80 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; padding: 0.6rem 1.2rem !important;
            font-size: 0.95em !important; letter-spacing: 0.5px; text-transform: uppercase; position: static !important;
             border-radius: 6px; /* Smaller radius than base */
        }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button:hover {{
            background-color: {AIB_RED} !important; color: {AIB_WHITE} !important; border-color: {AIB_RED} !important;
            box-shadow: 0 0 10px {AIB_RED}, 0 0 5px {AIB_RED} inset !important; transform: translateY(-1px) !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button:before {{ display: none !important; }} /* Disable base button hover effect */
        [data-testid="stSidebar"][aria-expanded="true"] .stTextInput > div > div > input,
        [data-testid="stSidebar"][aria-expanded="true"] .stSelectbox > div > div {{
            background-color: {AIB_BLACK} !important; color: {AIB_WHITE} !important;
            border: 1px solid {AIB_RED}70 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
             border-radius: 6px; /* Smaller radius */
        }}
        [data-testid="stSidebar"][aria-expanded="true"] label,
        [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stRadio"] label span {{
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
             color: {AIB_WHITE}CC !important; text-shadow: none !important;
        }}
        .metric-card {{ border-left-color: {AIB_RED} !important; background-color: #1F1F21; border-color: #303032;}}
        .metric-card.up {{ border-left-color: {AIB_GOLD} !important; }} /* Gold for Price Up in AIB */
        .metric-card.down {{ border-left-color: #B00000 !important; }} /* Dark Red for Price Down in AIB */

        .stTabs [data-baseweb="tab"] {{ background-color: {AIB_BLACK}; border-color: {AIB_RED}50; color: {AIB_WHITE}CC;}}
        .stTabs [aria-selected="true"] {{ background-color: {AIB_RED}C0; border-color: {AIB_RED}; color: {AIB_WHITE}; text-shadow: 0 0 5px {AIB_WHITE}40; }}
        .stTabs [role="tablist"] {{ border-bottom: 2px solid {AIB_RED}40 !important; gap: 8px !important; }}
        .stTabs [role="tab"] {{ background-color: rgba(0,0,0,0.3) !important; border: 1px solid {AIB_RED}40 !important;
            border-bottom: none !important; border-radius: 5px 5px 0 0 !important; padding: 0.75rem 1rem !important;
            font-weight: 600 !important; transition: all 0.2s ease; }}
        .stTabs [role="tab"][aria-selected="true"] {{ background-color: {AIB_RED}30 !important;
            border-color: {AIB_RED} !important; box-shadow: 0 0 10px {AIB_RED}30 !important; }}
        .stTabs [role="tab"]:hover {{ background-color: {AIB_RED}15 !important; border-color: {AIB_RED}80 !important; color: {AIB_WHITE} !important; }}

         ::-webkit-scrollbar-thumb {{ background: {AIB_RED}CC; border-radius: 5px;}}
         ::-webkit-scrollbar-thumb:hover {{ background: {AIB_GOLD}; }}
         h1, h2, h3, h4 {{ color: {AIB_WHITE}; text-shadow: 1px 1px 2px {AIB_BLACK}; }}
         hr {{ border-top: 1px solid {AIB_RED}80 !important; }}
         .hover-card {{ background-color: rgba(28,28,30,0.85); }} /* Ensure override */
         .dashboard-section {{ background-color:rgba(24,24,26,0.75); }} /* Ensure override */
         .dashboard-section:hover {{ background-color:rgba(30,30,32,0.85); }} /* Ensure override */
         .stPlotlyChart {{ background-color: rgba(28,28,30,0.7); border-radius: 10px; margin: 15px 0; border: 1px solid #404042; }}


    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<h2 style='color:{AIB_RED}; text-align:center; font-family:\"Arial Black\", Gadget, sans-serif; letter-spacing:1px; text-shadow: 1px 1px 2px {AIB_BLACK};'>BORDERLAND ARENA</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{AIB_WHITE}BF; font-size:0.8em; text-align:center; margin-top:-10px; margin-bottom:15px;'>Difficulty: Market Volatility</p>", unsafe_allow_html=True)
        aib_quotes = {
            "♦️ WITS": 'Chishiya: "The key isn\'t strength, but how you use your head."',
            "♠️ PHYSICAL": 'Arisu: "If you hesitate, you die. It\'s that simple."',
            "♣️ TEAM": 'Usagi: "We survive together, or not at all."',
            "♥️ PSYCHOLOGICAL": 'Mira: "Everyone wears a mask. The game is to see through them."'
        }
        st.markdown(f"<p style='color:{AIB_RED}; font-weight:bold; text-align:center; font-size:0.9em; margin-top:20px;'>GAME MASTER'S HINT:</p>", unsafe_allow_html=True)
        selected_quote_key = st.selectbox("Select Card Suit:", options=list(aib_quotes.keys()), key="aib_quote_select", label_visibility="collapsed")
        st.session_state.sidebar_quote_aib = aib_quotes[selected_quote_key] 
        st.markdown(f"<div class='hover-card' style='background-color:{AIB_BLACK}CC; border-left:3px solid {AIB_RED}; color:{AIB_WHITE}E0; font-size:0.85em; font-style:italic; padding:12px; margin-top:5px;'>{st.session_state.sidebar_quote_aib}</div>", unsafe_allow_html=True) 
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.session_state.get('info'): display_company_info(st.session_state.info, theme_accent_color=AIB_RED, theme_key_color=AIB_GOLD)
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("ADVANCE TO PREDICTION GAME ▶️", key="aib_goto_squid_ml", use_container_width=True):
            st.session_state.current_stage = "squid_ml"; st.rerun()
        if st.button("ADVANCE TO INTELLIGENCE HUB ▶️", key="aib_goto_moneyheist", use_container_width=True):
            st.session_state.current_stage = "moneyheist"; st.rerun()
        if st.button("🏠 RETURN TO HOME", key="aib_goto_home", use_container_width=True):
            st.session_state.current_stage = "welcome"; st.rerun()

    st.markdown(f"""
    <div style='display: flex; align-items: center; margin-bottom: 15px; background-color: rgba(0,0,0,0.6); padding: 15px; border-radius: 10px; border-left: 5px solid {AIB_RED};'>
        <div style='width: 60px; height: 60px; margin-right: 15px;'>{AIB_CARDS_SVG_HEADER}</div>
        <div>
            <h1 style='color: {AIB_RED}; font-family: "Impact", Charcoal, sans-serif; font-size: 2.5em; text-shadow: 1px 1px 3px {AIB_BLACK}, 0 0 10px {AIB_RED}50; letter-spacing: 1px; margin-bottom: 5px;'>
                STOCK ARENA <span style='font-size: 0.6em; color: {AIB_GOLD}; vertical-align: super;'>LIVE</span>
            </h1>
            <p style='color: {AIB_GOLD}; font-size: 1.1em; margin-top: 0; text-shadow: 1px 1px 2px {AIB_BLACK};'>
                {st.session_state.ticker.upper()} COMBAT ANALYSIS </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.data; ticker = st.session_state.ticker
    current_price = data['CLOSE'].iloc[-1] if not data.empty else np.nan 
    with st.expander("📄 View Raw Data Log", expanded=False):
        if data is not None and not data.empty:
            st.dataframe(data.head(1000), height=300, use_container_width=True)
        else:
            st.info("No data available to display.")

    st.markdown(f"#### Current Market Vitals – <span style='color:{AIB_GOLD}'>{ticker}</span>", unsafe_allow_html=True)
    prev_close = data['CLOSE'].iloc[-2] if len(data) > 1 else current_price
    price_chg = current_price - prev_close if pd.notna(current_price) and pd.notna(prev_close) else np.nan
    pct_chg_curr = (price_chg / prev_close * 100) if pd.notna(price_chg) and pd.notna(prev_close) and prev_close != 0 else np.nan

    price_color = AIB_GOLD if price_chg >= 0 else "#FF5050" if pd.notna(price_chg) else "white"
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    last_price_str = f"${current_price:.2f}" if pd.notna(current_price) else "N/A"
    chg_str = f" ({price_chg:+.2f}, {pct_chg_curr:+.2f}%)" if pd.notna(price_chg) and pd.notna(pct_chg_curr) else ""
    st.markdown(f"<div class='metric-card {'up' if price_chg >=0 else 'down'}'><div class='metric-title'>Last Price</div><div class='metric-value' style='color:{price_color};'>{last_price_str} <span style='font-size:0.7em;'>{chg_str}</span></div></div>", unsafe_allow_html=True)
    daily_h = data['HIGH'].iloc[-1] if not data.empty else np.nan; daily_l = data['LOW'].iloc[-1] if not data.empty else np.nan; vol = data['VOLUME'].iloc[-1] if not data.empty else np.nan
    st.markdown(f"<div class='metric-card neutral'><div class='metric-title'>Day High</div><div class='metric-value' style='color:{AIB_WHITE};'>{f'${daily_h:.2f}' if pd.notna(daily_h) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card neutral'><div class='metric-title'>Day Low</div><div class='metric-value' style='color:{AIB_WHITE};'>{f'${daily_l:.2f}' if pd.notna(daily_l) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card neutral'><div class='metric-title'>Volume</div><div class='metric-value' style='color:{AIB_WHITE};'>{f'{vol:,.0f}' if pd.notna(vol) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Game Board Visualization (Price & Indicators)")
    aib_chart_theme = {
        'price_inc': AIB_RED, 'price_dec': "#505050", 'ma20': AIB_WHITE, 'ma50': AIB_GOLD,
        'rsi_line': "#29B6F6", 'rsi_over': AIB_RED, 'rsi_under': AIB_GOLD, 'title': AIB_WHITE,
        'accent': AIB_GOLD, 'up': AIB_GOLD, 'down': '#FF3030'
        }
    fig_price_chart, fig_volume_chart = display_stock_charts_themed(data, ticker, theme_colors=aib_chart_theme)
    if fig_price_chart: st.plotly_chart(fig_price_chart, use_container_width=True)
    if fig_volume_chart: st.plotly_chart(fig_volume_chart, use_container_width=True)
    st.markdown("---"); st.markdown("#### Extended Game Intel")
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        returns_fig = plot_returns_distribution(data, theme_colors=aib_chart_theme)
        if returns_fig: st.plotly_chart(returns_fig, use_container_width=True)
    with col_viz2:
        volatility_fig = plot_rolling_volatility(data, theme_colors=aib_chart_theme)
        if volatility_fig: st.plotly_chart(volatility_fig, use_container_width=True)

    st.markdown("#### Price Correlation Matrix")
    corr_heatmap_fig = plot_feature_correlation_heatmap(data.copy(), theme_colors=aib_chart_theme)
    if corr_heatmap_fig: st.plotly_chart(corr_heatmap_fig, use_container_width=True)
    else: st.info("Not enough data points to calculate correlation matrix.")

    st.markdown("#### Advanced Time Series Analysis"); col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        st.markdown("##### Seasonal Decomposition")
        fig_seasonal = plot_seasonal_decomposition(data, theme_colors=aib_chart_theme)
        if fig_seasonal: st.plotly_chart(fig_seasonal, use_container_width=True)
        else: st.info("Not enough data for Seasonal Decomposition (min ~60 days).")
    with col_ts2:
        st.markdown("##### Autocorrelation Analysis")
        fig_autocorr = plot_autocorrelation(data, theme_colors=aib_chart_theme)
        if fig_autocorr: st.plotly_chart(fig_autocorr, use_container_width=True)
        else: st.info("Not enough data for Autocorrelation Analysis (min 20 days).")

    st.markdown("#### Advanced Feature Analysis")
    fig_heatmap_adv = plot_heatmap_advanced(data, theme_colors=aib_chart_theme)
    if fig_heatmap_adv: st.plotly_chart(fig_heatmap_adv, use_container_width=True)
    else: st.info("Not enough data points to calculate advanced feature correlation matrix.")

    st.markdown("#### Stationarity Analysis")
    fig_stationary = plot_stationarity_test(data, theme_colors=aib_chart_theme)
    if fig_stationary: st.plotly_chart(fig_stationary, use_container_width=True)
    else: st.info("Not enough data for Stationarity Test (min 20 days).")


# --- RENDER SQUID ML MODULE ---
def render_squid_ml_module():
     # Apply base CSS first, then page-specific overrides
    st.markdown(APP_BASE_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
        body {{ background-color: {SQ_DARK_BG} !important; }}
         .stApp > section.main > div.block-container {{
            display: block !important; min-height: auto !important; padding: 1.5rem 2rem 1rem 2rem !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] {{
            background-color: #0A0A0A !important; border-right: 3px solid {SQ_NEON_PINK}50 !important;
            box-shadow: 2px 0 12px {SQ_NEON_PINK}40; width: 320px !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; 
            padding: 20px !important; 
        }}
         [data-testid="stSidebar"][aria-expanded="true"] h2 {{ color:{SQ_NEON_PINK}; text-align:center; font-family:"Courier New", monospace; text-shadow: 0 0 5px {SQ_NEON_PINK}90; margin-bottom:15px; }}
         [data-testid="stSidebar"][aria-expanded="true"] p {{ color:{SQ_NEON_GREEN}BF; font-size:0.8em; text-align:center; margin-top:-10px; margin-bottom:15px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button {{
            background-color: {SQ_DARK_BG} !important; color: {SQ_NEON_PINK} !important; border: 1px solid {SQ_NEON_PINK} !important;
            box-shadow: 0 0 5px {SQ_NEON_PINK}80 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; padding: 0.6rem 1.2rem !important;
            font-size: 0.95em !important; letter-spacing: 0.5px; text-transform: uppercase; position: static !important;
             border-radius: 6px;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button:hover {{
            background-color: {SQ_NEON_PINK} !important; color: {SQ_DARK_BG} !important; border-color: {SQ_NEON_PINK} !important;
            box-shadow: 0 0 12px {SQ_NEON_PINK}, 0 0 8px {SQ_NEON_PINK} inset !important; transform: translateY(-1px) !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button:before {{ display: none !important; }} 
        [data-testid="stSidebar"][aria-expanded="true"] .stTextInput > div > div > input,
        [data-testid="stSidebar"][aria-expanded="true"] .stSelectbox > div > div {{
            background-color: {SQ_DARK_BG} !important; color: {SQ_NEON_PINK} !important;
            border: 1px solid {SQ_NEON_PINK}70 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
             border-radius: 6px;
             text-shadow: 0 0 3px {SQ_NEON_PINK}50;
        }}
         [data-testid="stSidebar"][aria-expanded="true"] .stTextInput > div > div > input:focus,
        [data-testid="stSidebar"][aria-expanded="true"] .stSelectbox > div > div:focus-within {{
            border: 1px solid {SQ_NEON_PINK} !important;
            box-shadow: 0 0 10px {SQ_NEON_PINK}60 !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] label,
        [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stRadio"] label span {{
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
             color: {SQ_NEON_PINK}CC !important; text-shadow: none !important;
        }}

        .metric-card {{ border-left-color: {SQ_NEON_PINK} !important; background-color: #18181A; border-color: {SQ_NEON_PINK}30;}}
        .metric-card.up {{ border-left-color: {SQ_NEON_GREEN} !important; }} 
        .metric-card.down {{ border-left-color: #FF3030 !important; }} 
        .metric-value {{ text-shadow: 0 0 5px currentColor; }} 


        .stTabs [data-baseweb="tab"] {{ background-color: #111; border-color: {SQ_NEON_PINK}50; color: {SQ_NEON_PINK}CC;}}
        .stTabs [aria-selected="true"] {{ background-color: {SQ_NEON_PINK}40; border-color: {SQ_NEON_PINK}; color: white; text-shadow: 0 0 5px white; }}
        .stTabs [role="tablist"] {{ border-bottom: 2px solid {SQ_NEON_PINK}40 !important; gap: 8px !important; }}
        .stTabs [role="tab"] {{ background-color: rgba(0,0,0,0.3) !important; border: 1px solid {SQ_NEON_PINK}40 !important;
            border-bottom: none !important; border-radius: 5px 5px 0 0 !important; padding: 0.75rem 1rem !important;
            font-weight: 600 !important; transition: all 0.2s ease; }}
        .stTabs [role="tab"][aria-selected="true"] {{ background-color: {SQ_NEON_PINK}30 !important;
            border-color: {SQ_NEON_PINK} !important; box-shadow: 0 0 10px {SQ_NEON_PINK}30 !important; }}
        .stTabs [role="tab"]:hover {{ background-color: {SQ_NEON_PINK}15 !important; border-color: {SQ_NEON_PINK}80 !important; color: {SQ_NEON_GREEN} !important; }}


        ::-webkit-scrollbar-thumb {{ background: {SQ_NEON_PINK}; border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {SQ_NEON_GREEN}; }}

        h1, h2, h3, h4 {{ color: {SQ_NEON_PINK}; text-shadow: 0 0 5px {SQ_NEON_PINK}80; }}
        hr {{ border-top: 1px solid {SQ_NEON_PINK}80 !important; }}
        .ml-step-container {{ display:flex; justify-content:space-around; gap:10px; margin: 15px 0; }}
        .ml-step {{ flex:1; background-color: rgba(10,0,20,0.5); border: 1px solid {SQ_NEON_PINK}80; border-radius: 8px; padding: 12px; text-align: center; font-size: 0.8em; color: #E0E0E0; transition: all 0.3s ease; position: relative; overflow: hidden; }}
        .ml-step:hover {{ border-color: {SQ_NEON_GREEN}; box-shadow: 0 0 10px {SQ_NEON_GREEN}80; background-color: rgba(15,0,25,0.6); }}
        .ml-step .step-title {{ font-weight: bold; color: {SQ_NEON_PINK}; margin-bottom: 5px; font-size: 0.9em; text-shadow: 0 0 3px {SQ_NEON_PINK}50; }}
        .ml-step .status-icon {{ font-size: 1.5em; margin: 5px 0; color: {SQ_NEON_PINK}; }}
        .ml-step .status-message {{ font-size: 0.85em; color: #E0E0E0; }}
        .ml-step.success {{ border-color: {SQ_NEON_GREEN}; background-color: rgba(0,25,10,0.5); box-shadow: 0 0 10px {SQ_NEON_GREEN}80; }}
        .ml-step.success .step-title {{ color: {SQ_NEON_GREEN}; text-shadow: 0 0 3px {SQ_NEON_GREEN}50; }}
        .ml-step.success .status-icon {{ color: {SQ_NEON_GREEN}; }}
        .ml-step.success .status-message {{ color: {SQ_NEON_GREEN}; }}
         .hover-card {{ background-color: rgba(28,28,30,0.85); }} 
        .dashboard-section {{ background-color:rgba(24,24,26,0.75); }} 
        .dashboard-section:hover {{ background-color:rgba(30,30,32,0.85); }} 
        .stPlotlyChart {{ background-color: rgba(28,28,30,0.7); border-radius: 10px; margin: 15px 0; border: 1px solid #404042; }}

        @keyframes sg-pulse-glow-h {{{{ 0%,100% {{{{ opacity:0.7; transform: scale(0.98);}}}} 50% {{{{ opacity:1; transform: scale(1.02); filter: drop-shadow(0 0 6px {SQ_NEON_PINK});}}}} }}}}

    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<h2 style='color:{SQ_NEON_PINK}; text-align:center; font-family:\"Courier New\", monospace; text-shadow: 0 0 5px {SQ_NEON_PINK}90;'>SQUIDSTOCK GAME</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{SQ_NEON_GREEN}BF; font-size:0.8em; text-align:center; margin-top:-10px; margin-bottom:15px;'>Prediction Protocol: Active</p>", unsafe_allow_html=True)
        squid_quotes = {"⭕ CIRCLE": 'Player 067: "We\'re teammates in this game called life."', "△ TRIANGLE": 'Sang-woo: "Never trust anyone in debt."', "□ SQUARE": 'Front Man: "All players are equal under the game\'s conditions."', "☂ UMBRELLA": 'Il-nam: "Trust isn\'t about worthiness, but necessity."'}
        st.markdown(f"<p style='color:{SQ_NEON_GREEN}; font-weight:bold; text-align:center;font-size:0.9em; margin-top:20px;'>GAME DIRECTIVES:</p>", unsafe_allow_html=True)
        selected_sq_quote_key = st.selectbox("Select Symbol:", options=list(squid_quotes.keys()), key="sq_quote_select", label_visibility="collapsed")
        st.session_state.sidebar_quote_sq = squid_quotes.get(selected_sq_quote_key, "") 
        st.markdown(f"<div class='hover-card' style='background-color:{SQ_DARK_BG}CC; border-left:3px solid {SQ_NEON_PINK}; color:#E0E0E0; font-size:0.85em; font-style:italic; padding:12px; margin-top:5px;'>{st.session_state.sidebar_quote_sq}</div>", unsafe_allow_html=True) 
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.session_state.get('info'): display_company_info(st.session_state.info, theme_accent_color=SQ_NEON_PINK, theme_key_color=SQ_NEON_GREEN)
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("◀️ RETURN TO STOCK ARENA", key="squid_goto_aib", use_container_width=True):
            st.session_state.current_stage = "aib_analysis"; st.rerun()
        if st.button("ADVANCE TO INTELLIGENCE HUB ▶️", key="squid_goto_moneyheist", use_container_width=True):
            st.session_state.current_stage = "moneyheist"; st.rerun()
        if st.button("🏠 RETURN TO HOME", key="squid_goto_home", use_container_width=True):
            st.session_state.current_stage = "welcome"; st.rerun()

    st.markdown(f"""
    <div style='display: flex; align-items: center; margin-bottom: 15px; background-color: rgba(0,0,0,0.6); padding: 15px; border-radius: 10px; border-left: 5px solid {SQ_NEON_PINK};'>
        <div style='width: 60px; height: 60px; margin-right: 15px;'>{SQUIDGAME_SVG_HEADER}</div>
        <div>
            <h1 style='color: {SQ_NEON_PINK}; font-family: "Courier New", monospace; font-size: 2.4em; text-shadow: 0 0 6px {SQ_NEON_PINK}, 0 0 12px {SQ_NEON_PINK}80; margin-bottom: 5px;'>
                SQUID ML CHALLENGES <span style='font-size: 0.6em; color: {SQ_NEON_GREEN}; vertical-align: super;'>V1.0</span>
            </h1>
            <p style='color: {SQ_NEON_GREEN}; font-size: 1.1em; margin-top: 0; text-shadow: 0 0 2px {SQ_DARK_BG};'>
                {st.session_state.ticker.upper()} PREDICTION ARENA </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.data; ticker = st.session_state.ticker
    current_price = data['CLOSE'].iloc[-1] if not data.empty else np.nan 

    st.markdown(f"<div class='hover-card' style='border-left: 3px solid {SQ_NEON_PINK}; background-color: rgba(18,18,20,0.85);'><h4 style='color:{SQ_NEON_PINK}; margin-bottom: 8px;'>SELECT PREDICTION GAME:</h4></div>", unsafe_allow_html=True)
    selected_ml_game = st.selectbox("Choose your prediction challenge:", ["Price Prediction Challenge (Linear Regression)", "Trend Prediction Gauntlet (Logistic Regression)", "K-means Clustering"], index=0 if st.session_state.get("selected_model_type", "Price") == "Price" else (1 if st.session_state.get("selected_model_type", "Trend") == "Trend" else 2), key="ml_game_selection_dropdown_sq", label_visibility="collapsed")
    st.session_state.selected_model_type = "Price" if "Price Prediction" in selected_ml_game else ("Trend" if "Logistic Regression" in selected_ml_game else "KMeans")
    st.markdown("---")

    st.markdown(f"<div class='dashboard-section'><h3 style='color:{SQ_NEON_PINK}; text-align:center; text-shadow: 0 0 3px {SQ_NEON_PINK};'>ML GAME PIPELINE</h3></div>", unsafe_allow_html=True)

    pipeline_cols_placeholder = st.empty() 
    progress_bar_placeholder = st.empty()
    status_text_area_placeholder = st.empty()

    ml_steps_list = ["Data Feed", "Pre-Game Prep", "Feature Craft", "Team Split", "Training Camp", "Final Score", "Outcome"]
    num_steps = len(ml_steps_list)

    run_ml_pipeline_flag = st.session_state.get('ml_results') is None and st.session_state.get('ml_error') is None
    if st.button("RESTART PREDICTION GAME", key="squid_ml_rerun_button_sidebar", use_container_width=True):
        run_ml_pipeline_flag = True
        st.session_state.ml_results = None
        st.session_state.trained_models = None
        st.session_state.corr_df = None
        st.session_state.ml_error = None
        pipeline_cols_placeholder.empty()
        progress_bar_placeholder.empty()
        status_text_area_placeholder.empty()
        time.sleep(0.1)
        st.rerun() 

    if run_ml_pipeline_flag:
        with st.spinner("RUNNING PREDICTION PROTOCOL..."):
            pipeline_cols = pipeline_cols_placeholder.columns(num_steps) 
            progress_bar = progress_bar_placeholder.progress(0)
            status_text_area = status_text_area_placeholder.empty()

            try:
                if data is None or data.empty: raise ValueError("Stock data missing or empty.")

                with pipeline_cols[0]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[0]}</p><p class='status-icon'>📡</p><p class='status-message'>{data.shape[0]} records</p></div>", unsafe_allow_html=True)
                progress_bar.progress(int(100/num_steps))
                time.sleep(0.05) 

                X_tr,X_te,y_r_tr,y_r_te,y_c_tr,y_c_te,latest_d_s,corr_df_res = prepare_ml_data(data)

                if isinstance(corr_df_res, str): 
                     for i in range(1, num_steps-2): 
                          with pipeline_cols[i]: st.markdown(f"<div class='ml-step'><p class='step-title'>{ml_steps_list[i]}</p><p class='status-icon'>⏳</p><p class='status-message'>Skipped</p></div>", unsafe_allow_html=True)
                     raise ValueError(corr_df_res) 

                st.session_state.corr_df = corr_df_res

                with pipeline_cols[1]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[1]}</p><p class='status-icon'>⚙️</p><p class='status-message'>Processed</p></div>", unsafe_allow_html=True)
                progress_bar.progress(int(200/num_steps))
                time.sleep(0.05)
                with pipeline_cols[2]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[2]}</p><p class='status-icon'>🛠️</p><p class='status-message'>{X_tr.shape[1] if X_tr is not None else 0} factors</p></div>", unsafe_allow_html=True)
                progress_bar.progress(int(300/num_steps))
                time.sleep(0.05)
                with pipeline_cols[3]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[3]}</p><p class='status-icon'>✂️</p><p class='status-message'>Divided</p></div>", unsafe_allow_html=True)
                progress_bar.progress(int(400/num_steps))
                time.sleep(0.05)

                ml_res, trained_m, ml_err = train_and_predict(X_tr,X_te,y_r_tr,y_r_te,y_c_tr,y_c_te,latest_d_s)
                if ml_err: raise ValueError(ml_err) 

                st.session_state.ml_results = ml_res; st.session_state.trained_models = trained_m

                with pipeline_cols[4]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[4]}</p><p class='status-icon'>🧠</p><p class='status-message'>Learned</p></div>", unsafe_allow_html=True)
                progress_bar.progress(int(500/num_steps))
                time.sleep(0.05)

                with pipeline_cols[5]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[5]}</p><p class='status-icon'>📊</p><p class='status-message'>Judged</p></div>", unsafe_allow_html=True)
                progress_bar.progress(int(600/num_steps))
                time.sleep(0.05)

                with pipeline_cols[6]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps_list[6]}</p><p class='status-icon'>💡</p><p class='status-message'>Revealed</p></div>", unsafe_allow_html=True)

                progress_bar.progress(100)
                status_text_area.success("✅ Prediction Protocol Complete!");
                st.session_state.ml_error = None 

            except Exception as e_pipe:
                st.session_state.ml_error = f"PROTOCOL ERROR: {e_pipe}";
                status_text_area.error(f"❌ {st.session_state.ml_error}");
                st.session_state.ml_results=None;st.session_state.trained_models=None;st.session_state.corr_df=None;
                progress_bar.progress(0)
                for i in range(num_steps):
                     if not pipeline_cols[i].container.css_styles or "success" not in pipeline_cols[i].container.css_styles:
                           with pipeline_cols[i]: st.markdown(f"<div class='ml-step'><p class='step-title'>{ml_steps_list[i]}</p><p class='status-icon'>❌</p><p class='status-message'>Failed</p></div>", unsafe_allow_html=True)


    else: 
        pipeline_cols = pipeline_cols_placeholder.columns(num_steps)
        progress_bar = progress_bar_placeholder.progress(0)
        status_text_area = status_text_area_placeholder.empty()

        if st.session_state.get('ml_error'):
            status_text_area.error(f"❌ Previous Protocol Error: {st.session_state.ml_error}")
            for i in range(num_steps):
                 with pipeline_cols[i]: st.markdown(f"<div class='ml-step'><p class='step-title'>{ml_steps_list[i]}</p><p class='status-icon'>❌</p><p class='status-message'>Failed</p></div>", unsafe_allow_html=True)

        elif st.session_state.get('ml_results'):
            status_text_area.success("✅ Previous Protocol Data Loaded.");
            progress_bar.progress(100)
            for i_s, s_name in enumerate(ml_steps_list):
                 with pipeline_cols[i_s]: st.markdown(f"<div class='ml-step success'><p class='step-title'>{s_name}</p><p class='status-icon'>✔️</p><p class='status-message'>Complete</p></div>", unsafe_allow_html=True)
        else:
            status_text_area.info("Awaiting Protocol Execution. Press 'RESTART PREDICTION GAME'.");
            progress_bar.progress(0)
            for i in range(num_steps):
                 with pipeline_cols[i]: st.markdown(f"<div class='ml-step'><p class='step-title'>{ml_steps_list[i]}</p><p class='status-icon'>⏳</p><p class='status-message'>Pending</p></div>", unsafe_allow_html=True)


    st.markdown(f"<hr style='border-color: {SQ_NEON_PINK}50;'>", unsafe_allow_html=True)

    if st.session_state.get('ml_results'):
        verdict_html = generate_prediction_verdict(current_price, st.session_state.ml_results, theme_accent_color=SQ_NEON_PINK, theme_up_color=SQ_NEON_GREEN)
        st.markdown(verdict_html, unsafe_allow_html=True)

        if st.session_state.selected_model_type == "Price":
            st.markdown(f"<div class='dashboard-section'><h3 style='color:{SQ_NEON_PINK};text-align:center; text-shadow: 0 0 3px {SQ_NEON_PINK};'>Price Prediction Game Outcome</h3></div>", unsafe_allow_html=True)
            if st.session_state.ml_results.get('regression'):
                display_price_prediction(current_price, st.session_state.ml_results, ticker,
                    theme_colors={
                        'pred_marker': SQ_NEON_PINK, 'ci': SQ_NEON_GREEN, 'historical_line': '#999999', 
                        'actual_price_marker': SQ_NEON_PINK, 'prediction_path_line': SQ_NEON_GREEN, 
                        'up': SQ_NEON_GREEN, 'down': '#FF3030' },
                    historical_data_df=st.session_state.data)
            else: st.warning("Price Prediction (Linear Regression) results not available. Run the ML pipeline.")
        elif st.session_state.selected_model_type == "Trend":
            st.markdown(f"<div class='dashboard-section'><h3 style='color:{SQ_NEON_PINK};text-align:center; text-shadow: 0 0 3px {SQ_NEON_PINK};'>Trend Prediction Game Outcome</h3></div>", unsafe_allow_html=True)
            if st.session_state.ml_results.get('classification'):
                display_trend_prediction(st.session_state.ml_results,
                                         theme_colors={'title': SQ_NEON_PINK, 'up': SQ_NEON_GREEN, 'down': '#FF3030'})
            else: st.warning("Trend Prediction (Logistic Regression) results not available. Run the ML pipeline.")
        elif st.session_state.selected_model_type == "KMeans":
            st.markdown(f"<div class='dashboard-section'><h3 style='color:{SQ_NEON_PINK};text-align:center; text-shadow: 0 0 3px {SQ_NEON_PINK};'>K-means Clustering Analysis</h3></div>", unsafe_allow_html=True)
            if st.session_state.ml_results.get('kmeans'):
                display_kmeans_clustering(st.session_state.ml_results,
                                          theme_colors={'title': SQ_NEON_PINK, 'cluster1': SQ_NEON_GREEN, 'cluster2': '#FF3030'}) 
            else: st.warning("K-means Clustering results not available. Run the ML pipeline.")


        with st.expander("⚙️ Post-Game Analysis & Data Intel", expanded=False):
             st.markdown(f"<h4 style='color:{SQ_NEON_PINK};'>ML Feature Correlations:</h4>", unsafe_allow_html=True)
             if st.session_state.get('corr_df') is not None and not st.session_state.corr_df.empty:
                 try:
                     ml_features_corr = st.session_state.corr_df.corr()
                     if not ml_features_corr.empty:
                          fig_ml_corr = px.imshow(ml_features_corr, text_auto=".2f", aspect="auto",
                                                  color_continuous_scale='RdBu_r', range_color=[-1,1])
                          fig_ml_corr.update_layout(title="ML Feature Correlation Heatmap", template="plotly_dark", height=500,
                                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(28,28,30,0.85)',
                                                  title_font_color=SQ_NEON_PINK, title_x=0.5)
                          st.plotly_chart(fig_ml_corr, use_container_width=True)
                     else: st.info("Correlation matrix is empty.")
                 except Exception as e:
                      st.warning(f"Error generating ML feature correlation heatmap: {e}")
             else: st.info("ML correlation data not available. Run the ML pipeline.")


    elif st.session_state.get('ml_error'):
        st.error(f"Prediction Protocol Failure: {st.session_state.ml_error}")
    else:
         st.info("No prediction results available yet. Run the ML pipeline using the button above.")



# --- RENDER MONEY HEIST DASHBOARD ---
def render_moneyheist_dashboard():
    st.markdown(APP_BASE_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
        body {{ background-color: {MH_BLACK} !important; }}
         .stApp > section.main > div.block-container {{
            display: block !important; min-height: auto !important; padding: 1.5rem 2rem 1rem 2rem !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] {{
            background-color: #0A0A0A !important; border-right: 3px solid {MH_RED}50 !important;
            box-shadow: 2px 0 12px {MH_RED}40; width: 320px !important;
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; 
             padding: 20px !important; 
        }}
        [data-testid="stSidebar"][aria-expanded="true"] h2 {{ color:{MH_RED}; text-align:center; font-family:"Arial Black", Gadget, sans-serif; text-shadow: 0 0 5px {MH_RED}90; margin-bottom:15px; }}
        [data-testid="stSidebar"][aria-expanded="true"] p {{ color:{MH_GOLD}BF; font-size:0.8em; text-align:center; margin-top:-10px; margin-bottom:15px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button {{
            background-color: {MH_BLACK} !important; color: {MH_RED} !important; border: 1px solid {MH_RED} !important;
            box-shadow: 0 0 5px {MH_RED}80 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; padding: 0.6rem 1.2rem !important;
            font-size: 0.95em !important; letter-spacing: 0.5px; text-transform: uppercase; position: static !important;
            border-radius: 6px;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button:hover {{
            background-color: {MH_RED} !important; color: {MH_BLACK} !important; border-color: {MH_RED} !important;
            box-shadow: 0 0 12px {MH_RED}, 0 0 8px {MH_RED}50 inset !important; transform: translateY(-1px) !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] .stButton > button:before {{ display: none !important; }} 
        [data-testid="stSidebar"][aria-expanded="true"] .stTextInput > div > div > input,
        [data-testid="stSidebar"][aria-expanded="true"] .stSelectbox > div > div {{
            background-color: {MH_BLACK} !important; color: {MH_WHITE} !important;
            border: 1px solid {MH_RED}70 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            border-radius: 6px;
             text-shadow: 0 0 3px {MH_RED}50;
        }}
         [data-testid="stSidebar"][aria-expanded="true"] .stTextInput > div > div > input:focus,
        [data-testid="stSidebar"][aria-expanded="true"] .stSelectbox > div > div:focus-within {{
            border: 1px solid {MH_RED} !important;
            box-shadow: 0 0 10px {MH_RED}60 !important;
        }}
        [data-testid="stSidebar"][aria-expanded="true"] label,
        [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stRadio"] label span {{
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
             color: {MH_WHITE}CC !important; text-shadow: none !important;
        }}
        .metric-card {{ border-left-color: {MH_RED} !important; background-color: #18181A; border-color: {MH_RED}30;}}
        .metric-card.up {{ border-left-color: {MH_GOLD} !important; }} 
        .metric-card.down {{ border-left-color: #B00000 !important; }} 
        .metric-value {{ text-shadow: 0 0 5px currentColor; }} 


        .stTabs [data-baseweb="tab"] {{ background-color: #111; border-color: {MH_RED}50; color: {MH_RED}CC;}}
        .stTabs [aria-selected="true"] {{ background-color: {MH_RED}40; border-color: {MH_RED}; color: white; text-shadow: 0 0 5px white; }}
        .stTabs [role="tablist"] {{ border-bottom: 2px solid {MH_RED}40 !important; gap: 8px !important; }}
        .stTabs [role="tab"] {{ background-color: rgba(0,0,0,0.3) !important; border: 1px solid {MH_RED}40 !important;
            border-bottom: none !important; border-radius: 5px 5px 0 0 !important; padding: 0.75rem 1rem !important;
            font-weight: 600 !important; transition: all 0.2s ease; }}
        .stTabs [role="tab"][aria-selected="true"] {{ background-color: {MH_RED}30 !important;
            border-color: {MH_RED} !important; box-shadow: 0 0 10px {MH_RED}30 !important; }}
        .stTabs [role="tab"]:hover {{ background-color: {MH_RED}15 !important; border-color: {MH_RED}80 !important; color: {MH_GOLD} !important;}}

        ::-webkit-scrollbar-thumb {{ background: {MH_RED}; border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {MH_GOLD}; }}

        h1, h2, h3, h4 {{ color: {MH_RED}; text-shadow: 0 0 5px {MH_RED}80; }}
        hr {{ border-top: 1px solid {MH_RED}80 !important; }}
        .mh-dashboard-card {{ background-color: rgba(24,24,26,0.8); border-radius: 12px; padding: 20px; margin-bottom: 25px;
            border: 1px solid {MH_RED}40; transition: all 0.3s ease; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }}
        .mh-dashboard-card:hover {{ background-color: rgba(30,30,32,0.9); border-color: {MH_GOLD};
            box-shadow: 0 6px 18px rgba(0,0,0,0.5), 0 0 12px {MH_RED}40 inset; transform: translateY(-4px); }}
        .mh-stat-value {{ font-size: 2.4em; font-weight: bold; margin: 8px 0; text-shadow: 0 0 8px currentColor; }}
        .mh-stat-label {{ color: #BBB; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1.2px; }}
        .mh-intel-section {{ background-color: rgba(20,20,22,0.85); border-radius: 12px; padding: 25px;
            margin: 25px 0; border-left: 4px solid {MH_RED}; box-shadow: 0 5px 15px rgba(0,0,0,0.5); }}
        .mh-intel-section:hover {{ background-color: rgba(25,25,28,0.9);
            box-shadow: 0 7px 20px rgba(0,0,0,0.6), 0 0 18px {MH_RED}20 inset; }}
        .mh-intel-header {{ color: {MH_RED}; font-size: 1.6em; margin-bottom: 15px;
            text-shadow: 0 0 8px {MH_RED}60; letter-spacing: 1px; }}
        .mh-prof-note {{ background-color: rgba(0,0,0,0.6); border-left: 3px solid {MH_GOLD};
            padding: 15px; margin: 20px 0; border-radius: 6px; font-style: italic; color: {MH_GOLD}; position: relative; box-shadow: inset 0 0 10px {MH_GOLD}20; }}
        .mh-prof-note::before {{ content: '"'; font-size: 2.5em; position: absolute; left: 8px; top: -8px; opacity: 0.5; color: {MH_GOLD}; }}
        .mh-tag {{ display: inline-block; padding: 4px 10px; background-color: {MH_RED}40;
            color: white; border-radius: 6px; font-size: 0.85em; margin-right: 8px; margin-bottom: 8px; letter-spacing: 0.5px; }}
        .mh-tag.bullish {{ background-color: rgba(0,180,0,0.4); }}
        .mh-tag.bearish {{ background-color: rgba(200,0,0,0.4); }}
        .js-plotly-plot .plotly .modebar {{ background-color: transparent !important; }}
        .mh-grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            grid-gap: 25px; margin-bottom: 30px; }}
        .mh-loading {{ padding: 15px; border-radius: 8px; background-color: rgba(0,0,0,0.7);
            color: {MH_RED}; text-align: center; animation: mh-pulse 1.5s infinite; }}
        @keyframes mh-pulse {{{{ 0%, 100% {{{{ opacity: 0.7; }}}} 50% {{{{ opacity: 1; }}}} }}}}
        .stPlotlyChart {{ background-color: rgba(28,28,30,0.7); border-radius: 10px; margin: 15px 0; border: 1px solid #404042; }}


    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f""" <h2 style='color:{MH_RED}; text-align:center; font-family:\"Arial Black\", Gadget, sans-serif; text-shadow: 0 0 5px {MH_RED}90;'>
            <span style='color:{MH_GOLD}; text-shadow: 0 0 5px {MH_GOLD}90;'>$</span> INTELLIGENCE HUB <span style='color:{MH_GOLD}; text-shadow: 0 0 5px {MH_GOLD}90;'>$</span> </h2> """, unsafe_allow_html=True)
        st.markdown(f"<p style='color:{MH_GOLD}BF; font-size:0.8em; text-align:center; margin-top:-10px; margin-bottom:15px;'>Market Intelligence Network</p>", unsafe_allow_html=True)
        mh_quotes = { "💰 CAPITAL FLOWS": "The true victory isn't capturing the gold, but understanding its patterns.", "🧠 STRATEGIC INSIGHT": "Knowledge in trading is more valuable than capital.",
            "🔒 SECURITY PROTOCOLS": "Security in investments comes from planning, not chance.", "🕒 TIMING ANALYTICS": "Timing isn't everything, but in markets, it's a lot." }
        st.markdown(f"<p style='color:{MH_RED}; font-weight:bold; text-align:center; font-size:0.9em; margin-top:20px;'>PROFESSOR'S NOTES:</p>", unsafe_allow_html=True)
        selected_mh_quote_key = st.selectbox("Select Intel Type:", options=list(mh_quotes.keys()), key="mh_quote_select", label_visibility="collapsed")
        st.session_state.sidebar_quote_mh = mh_quotes.get(selected_mh_quote_key, "") 
        st.markdown(f""" <div class='hover-card' style='background-color:{MH_BLACK}CC; border-left:3px solid {MH_RED}; margin-top:5px;'>
            <div style='color:#E0E0E0; font-size:0.85em; font-style:italic; padding:12px; position:relative;'>
                <span style='position:absolute; font-size:2em; opacity:0.3; top:-5px; left:5px; color:{MH_GOLD};'>"</span>
                <span style='margin-left:15px;'>{st.session_state.sidebar_quote_mh}</span>
                <span style='position:absolute; font-size:2em; opacity:0.3; bottom:-15px; right:5px; color:{MH_GOLD};'>"</span>
            </div> </div> """, unsafe_allow_html=True) 
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.session_state.get('info'): display_company_info(st.session_state.info, theme_accent_color=MH_RED, theme_key_color=MH_GOLD)
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("◀️ RETURN TO SQUID ML", key="mh_goto_squid", use_container_width=True):
            st.session_state.current_stage = "squid_ml"; st.rerun()
        if st.button("◀️ RETURN TO STOCK ARENA", key="mh_goto_aib", use_container_width=True):
            st.session_state.current_stage = "aib_analysis"; st.rerun()
        if st.button("🏠 RETURN TO HOME", key="mh_goto_home", use_container_width=True):
            st.session_state.current_stage = "welcome"; st.rerun()

    st.markdown(f""" <div style='display: flex; align-items: center; margin-bottom: 15px; background-color: rgba(0,0,0,0.6); padding: 15px; border-radius: 10px; border-left: 5px solid {MH_RED};'>
        <div style='width: 60px; height: 60px; margin-right: 15px;'>{MONEYHEIST_SVG_HEADER}</div>
        <div> <h1 style='color: {MH_RED}; font-family: "Impact", Charcoal, sans-serif; font-size: 2.4em; text-shadow: 1px 1px 3px {MH_BLACK}, 0 0 10px {MH_RED}50; letter-spacing: 1px; margin-bottom: 5px;'>
                INTELLIGENCE HUB<span style='font-size: 0.6em; color: {MH_GOLD}; vertical-align: super;'>BETA</span> </h1>
            <p style='color: {MH_GOLD}; font-size: 1.1em; margin-top: 0; text-shadow: 1px 1px 2px {MH_BLACK};'>
                {st.session_state.ticker.upper()} MARKET ANALYSIS </p> </div> </div> """, unsafe_allow_html=True)

    data = st.session_state.data; ticker = st.session_state.ticker
    current_price = data['CLOSE'].iloc[-1] if not data.empty else np.nan 

    st.markdown("<div class='mh-grid-container'>", unsafe_allow_html=True)
    prev_close = data['CLOSE'].iloc[-2] if len(data) > 1 and 'CLOSE' in data.columns and pd.notna(data['CLOSE'].iloc[-2]) else current_price
    price_chg = current_price - prev_close if pd.notna(current_price) and pd.notna(prev_close) else np.nan
    pct_chg = (price_chg / prev_close * 100) if pd.notna(price_chg) and pd.notna(prev_close) and prev_close != 0 else np.nan

    price_color = MH_GOLD if price_chg >= 0 else "#FF5050" if pd.notna(price_chg) else "white"
    chg_icon = "▲" if pd.notna(price_chg) and price_chg > 0 else "▼" if pd.notna(price_chg) and price_chg < 0 else "■"

    current_price_str = f"${current_price:.2f}" if pd.notna(current_price) else "N/A"
    price_chg_str = f" {chg_icon} {price_chg:+.2f}" if pd.notna(price_chg) else ""
    pct_chg_str = f" ({pct_chg:+.2f}%)" if pd.notna(pct_chg) else ""

    st.markdown(f""" <div class="mh-dashboard-card" style="position: relative; overflow: hidden;">
        <div style="position: absolute; top: -15px; right: -15px; font-size: 5em; opacity: 0.1; color: {price_color};">{chg_icon}</div>
        <div class="mh-stat-label">Current Price</div> <div class="mh-stat-value" style="color:{price_color};">{current_price_str}</div>
        <div style="color:{price_color}; font-size:1em; font-weight: 500;"> {price_chg_str} {pct_chg_str} </div>
        <div style="margin-top: 8px;"> <span class="mh-tag" style="background-color: rgba(40,40,40,0.7); border: 1px solid rgba(255,255,255,0.2);">LAST UPDATED</span>
            <span style="color: #CCC; font-size: 0.8em;">{datetime.now().strftime('%H:%M:%S')}</span> </div> </div> """, unsafe_allow_html=True)

    rsi_val = calculate_rsi(data['CLOSE']).iloc[-1] if not data.empty and 'CLOSE' in data.columns and not data['CLOSE'].isnull().all() and len(data['CLOSE'].dropna()) >= 14 else np.nan
    rsi_status = "OVERSOLD" if pd.notna(rsi_val) and rsi_val < 30 else "OVERBOUGHT" if pd.notna(rsi_val) and rsi_val > 70 else "NEUTRAL"
    rsi_color = "#FF5050" if rsi_status == "OVERSOLD" else MH_GOLD if rsi_status == "OVERBOUGHT" else "white" if pd.notna(rsi_val) else "#888"

    gradient_color = "linear-gradient(90deg, #FF5050 0%, #FFAA00 50%, #00AAFF 100%)"; indicator_position = min(max(rsi_val if pd.notna(rsi_val) else 50, 0), 100) 

    st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">RSI (14-day)</div>
        <div class="mh-stat-value" style="color:{rsi_color};">{f'{rsi_val:.1f}' if pd.notna(rsi_val) else 'N/A'}</div>
        <div style="margin: 10px 0; height: 8px; border-radius: 4px; background: {gradient_color}; position: relative;">
             {'<div style="position: absolute; left: ' + str(indicator_position) + '%; top: -6px; width: 5px; height: 20px; background-color: white; transform: translateX(-50%); border-radius: 2px;"></div>' if pd.notna(rsi_val) else ''}
        </div> <div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #AAA; margin-bottom: 8px;">
            <div>OVERSOLD</div> <div>NEUTRAL</div> <div>OVERBOUGHT</div> </div> <div style="margin-top: 5px;">
            <span class="mh-tag" style="background-color: {MH_RED if rsi_status == 'OVERSOLD' else MH_GOLD if rsi_status == 'OVERBOUGHT' else '#666'}80; border: 1px solid rgba(255,255,255,0.2);"> {rsi_status} </span>
        </div> </div> """, unsafe_allow_html=True)

    vol_20d_series = data['CLOSE'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100 if 'CLOSE' in data.columns and len(data['CLOSE']) >= 20 else pd.Series([], dtype=float)
    current_vol_val = vol_20d_series.iloc[-1] if not vol_20d_series.empty else np.nan
    avg_vol_val = vol_20d_series.mean() if not vol_20d_series.empty else np.nan

    vol_status = "HIGH" if pd.notna(current_vol_val) and pd.notna(avg_vol_val) and avg_vol_val != 0 and current_vol_val > avg_vol_val * 1.2 else ("LOW" if pd.notna(current_vol_val) and pd.notna(avg_vol_val) and avg_vol_val != 0 and current_vol_val < avg_vol_val * 0.8 else "AVERAGE" if pd.notna(current_vol_val) else "N/A")
    vol_color = "#FF5050" if vol_status == "HIGH" else MH_GOLD if vol_status == "LOW" else "white" if vol_status == "AVERAGE" else "#888"
    vol_percent = min(100, max(0, (current_vol_val / (avg_vol_val * 2) * 100) if pd.notna(current_vol_val) and pd.notna(avg_vol_val) and avg_vol_val != 0 else 50)) if pd.notna(current_vol_val) else 0 

    st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">Volatility (Annualized)</div>
        <div class="mh-stat-value" style="color:{vol_color};">{f'{current_vol_val:.1f}%' if pd.notna(current_vol_val) else 'N/A'}</div>
        <div style="margin: 10px 0; height: 8px; border-radius: 4px; background: linear-gradient(90deg, {MH_GOLD} 0%, #FF5050 100%); position: relative;">
            {'<div style="position: absolute; left: ' + str(vol_percent) + '%; top: -6px; width: 5px; height: 20px; background-color: white; transform: translateX(-50%); border-radius: 2px;"></div>' if pd.notna(current_vol_val) else ''}
        </div> <div style="color: #AAA; font-size: 0.85em; margin-top: 8px;">
            <span style="color: {vol_color}; font-weight: bold;">{vol_status}</span> volatility compared to avg of {f'{avg_vol_val:.1f}%' if pd.notna(avg_vol_val) else 'N/A'}
        </div> </div> """, unsafe_allow_html=True)

    avg_vol_data = data['VOLUME'].rolling(window=20).mean().iloc[-1] if 'VOLUME' in data.columns and len(data) >= 20 and not data['VOLUME'].isnull().all() else (data['VOLUME'].mean() if 'VOLUME' in data.columns and not data['VOLUME'].isnull().all() else np.nan)
    current_vol_data = data['VOLUME'].iloc[-1] if 'VOLUME' in data.columns and not data.empty else np.nan

    vol_ratio = (current_vol_data / avg_vol_data) if pd.notna(current_vol_data) and pd.notna(avg_vol_data) and avg_vol_data != 0 else (1 if pd.notna(current_vol_data) else np.nan) 

    vol_status_act = "HIGH" if pd.notna(vol_ratio) and vol_ratio > 1.5 else ("LOW" if pd.notna(vol_ratio) and vol_ratio < 0.7 else "AVERAGE" if pd.notna(vol_ratio) else "N/A")
    vol_color_act = MH_GOLD if vol_status_act == "HIGH" else "#FF5050" if vol_status_act == "LOW" else "white" if vol_status_act == "AVERAGE" else "#888"

    def format_volume(v):
        if pd.isna(v): return "N/A"
        v = int(v) 
        if v >= 1e9: return f"{v/1e9:.2f}B"
        elif v >= 1e6: return f"{v/1e6:.2f}M"
        elif v >= 1e3: return f"{v/1e3:.1f}K"
        else: return f"{v:.0f}"

    formatted_vol = format_volume(current_vol_data); formatted_avg = format_volume(avg_vol_data)
    vol_ratio_percent = min(100, max(0, (vol_ratio * 100) if pd.notna(vol_ratio) else 0)) 

    st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">Volume</div>
        <div class="mh-stat-value" style="color:{vol_color_act};">{formatted_vol}</div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="flex-grow: 1; height: 5px; background-color: #444; border-radius: 3px; margin-right: 10px; position: relative;">
                 {'<div style="position: absolute; width: ' + str(vol_ratio_percent) + '%; height: 100%; background-color: ' + vol_color_act + '; border-radius: 3px; transition: width 0.3s ease;"></div>' if pd.notna(vol_ratio_percent) else ''}
            </div> <div style="color: #AAA; font-size: 0.9em;">{f'{vol_ratio:.2f}x' if pd.notna(vol_ratio) else 'N/A'}</div>
        </div> <div style="color: #999; font-size: 0.85em;"> 20-day avg: <span style="color: white;">{formatted_avg}</span> </div>
        <div style="margin-top: 5px;"> <span class="mh-tag" style="background-color: {MH_GOLD if vol_status_act == 'HIGH' else '#FF5050' if vol_status_act == 'LOW' else '#666'}80; border: 1px solid rgba(255,255,255,0.2);">
                {vol_status_act} ACTIVITY </span> </div> </div> """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True) 

    st.markdown(f""" <div class="mh-intel-section"> <h3 class="mh-intel-header">📊 COMPREHENSIVE MARKET INTELLIGENCE</h3>
        <p style="color: #AAA; margin-bottom: 20px;"> In-depth analysis of <span style="color: {MH_GOLD}; font-weight: bold;">{ticker.upper()}</span> market behavior and performance metrics </p>
        <div class="mh-prof-note"> <q>"Look beyond the numbers. The patterns will reveal themselves when you analyze from multiple angles."</q> </div>
    </div> """, unsafe_allow_html=True)

    tabs = st.tabs(["📈 Price Action", "🔬 Technical Analysis", "📊 Statistical Insights"])

    mh_theme_colors = {
        'price_inc': MH_GOLD, 'price_dec': "#B00000", 'ma20': MH_WHITE, 'ma50': MH_GOLD,
        'rsi_line': "#29B6F6", 'rsi_over': "#B00000", 'rsi_under': MH_GOLD, 'title': MH_RED,
        'accent': MH_GOLD, 'up': MH_GOLD, 'down': '#B00000'
        }


    with tabs[0]:
        st.markdown(f"<h4 style='color:{MH_RED}; text-shadow: 0 0 3px {MH_RED};'>Price Action Dashboard</h4>", unsafe_allow_html=True)
        fig_price, fig_vol = display_stock_charts_themed(data, ticker, theme_colors=mh_theme_colors)
        if fig_price: st.plotly_chart(fig_price, use_container_width=True)
        if fig_vol: st.plotly_chart(fig_vol, use_container_width=True)
        if fig_price is None and fig_vol is None: st.info("Not enough valid data to display Price Action Charts.")

    with tabs[1]:
        st.markdown(f"<h4 style='color:{MH_RED}; text-shadow: 0 0 3px {MH_RED};'>Technical Analysis Dashboard</h4>", unsafe_allow_html=True)
        col_ta1, col_ta2 = st.columns(2)
        with col_ta1:
            ma_short = data['CLOSE'].rolling(window=20).mean() if 'CLOSE' in data.columns and len(data) >= 20 else pd.Series([], dtype=float)
            ma_long = data['CLOSE'].rolling(window=50).mean() if 'CLOSE' in data.columns and len(data) >= 50 else pd.Series([], dtype=float)
            signal_msg = "Insufficient Data"; signal_color_ma = "#888888"
            if not ma_short.empty and not ma_long.empty and len(ma_short) >= 2 and len(ma_long) >= 2:
                if ma_short.iloc[-2] <= ma_long.iloc[-2] and ma_short.iloc[-1] > ma_long.iloc[-1]: signal_msg, signal_color_ma = "Golden Cross (Bullish)", MH_GOLD
                elif ma_short.iloc[-2] >= ma_long.iloc[-2] and ma_short.iloc[-1] < ma_long.iloc[-1]: signal_msg, signal_color_ma = "Death Cross (Bearish)", "#FF5050"
                else: signal_msg, signal_color_ma = "No Recent Crossover", "white"
            st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">MA Crossover Signal (20 vs 50)</div>
                <div style="font-size:1.5em; color:{signal_color_ma}; font-weight:bold; margin:10px 0;">{signal_msg}</div>
                <div style="color:#AAA; font-size:0.8em;">Analysis based on 20-day and 50-day moving averages</div> </div> """, unsafe_allow_html=True)

            bb_window = 20
            if 'CLOSE' in data.columns and len(data) >= bb_window:
                 rolling_mean_bb = data['CLOSE'].rolling(window=bb_window).mean()
                 rolling_std_bb = data['CLOSE'].rolling(window=bb_window).std()
                 upper_band = rolling_mean_bb + (rolling_std_bb * 2)
                 lower_band = rolling_mean_bb - (rolling_std_bb * 2)
                 bb_signal, bb_color = "Insufficient Data", "#888888"
                 if not upper_band.empty and not lower_band.empty and not data.empty:
                      last_price_bb = data['CLOSE'].iloc[-1]
                      if pd.notna(last_price_bb) and pd.notna(upper_band.iloc[-1]) and last_price_bb > upper_band.iloc[-1]: bb_signal, bb_color = "Overbought (Above Upper Band)", "#FF5050"
                      elif pd.notna(last_price_bb) and pd.notna(lower_band.iloc[-1]) and last_price_bb < lower_band.iloc[-1]: bb_signal, bb_color = "Oversold (Below Lower Band)", MH_GOLD
                      else: bb_signal, bb_color = "Within Bands (Neutral)", "white"
            else:
                bb_signal, bb_color = "Insufficient Data", "#888888"

            st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">Bollinger Bands Signal</div>
                <div style="font-size:1.5em; color:{bb_color}; font-weight:bold; margin:10px 0;">{bb_signal}</div>
                <div style="color:#AAA; font-size:0.8em;">Price relative to 20-day Bollinger Bands</div> </div> """, unsafe_allow_html=True)
        with col_ta2:
            rsi_mh = calculate_rsi(data['CLOSE']) if 'CLOSE' in data.columns else pd.Series([], dtype=float)
            rsi_trend, rsi_trend_color = "Insufficient Data", "#888888"
            rsi_current_val_str = "N/A"; rsi_past_val_str = "N/A"
            if len(rsi_mh) >= 5:
                rsi_5d = rsi_mh.iloc[-5:]
                rsi_current_val = rsi_5d.iloc[-1]
                rsi_past_val = rsi_5d.iloc[0]
                if pd.notna(rsi_current_val) and pd.notna(rsi_past_val):
                     rsi_current_val_str = f"{rsi_current_val:.1f}"
                     rsi_past_val_str = f"{rsi_past_val:.1f}"
                     if rsi_current_val > rsi_past_val: rsi_trend, rsi_trend_color = "Increasing", MH_GOLD
                     elif rsi_current_val < rsi_past_val: rsi_trend, rsi_trend_color = "Decreasing", "#FF5050"
                     else: rsi_trend, rsi_trend_color = "Flat", "white"
            elif not rsi_mh.empty:
                rsi_current_val_str = f"{rsi_mh.iloc[-1]:.1f}"
                rsi_trend, rsi_trend_color = "Single Point", "white"


            st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">RSI Trend (5 Days)</div>
                <div style="font-size:1.5em; color:{rsi_trend_color}; font-weight:bold; margin:10px 0;">{rsi_trend}</div>
                <div style="color:#AAA; font-size:0.8em;">Current: {rsi_current_val_str} | 5 days ago: {rsi_past_val_str}</div>
            </div> """, unsafe_allow_html=True)

            vol_trend_mh, vol_trend_color_mh = "Insufficient Data", "#888888"
            if 'VOLUME' in data.columns and not data['VOLUME'].isnull().all():
                 if len(data['VOLUME']) >= 20: 
                      recent_vol_avg = data['VOLUME'].iloc[-5:].mean()
                      past_vol_avg = data['VOLUME'].iloc[-20:-5].mean()
                      if pd.notna(recent_vol_avg) and pd.notna(past_vol_avg) and past_vol_avg > 0:
                           if recent_vol_avg > past_vol_avg: vol_trend_mh, vol_trend_color_mh = "Increasing", MH_GOLD
                           else: vol_trend_mh, vol_trend_color_mh = "Decreasing", "#FF5050"
                      elif pd.notna(recent_vol_avg): 
                           vol_trend_mh, vol_trend_color_mh = "Recent Activity", "white"
                 elif len(data['VOLUME']) >= 2: 
                      if data['VOLUME'].iloc[-1] > data['VOLUME'].iloc[-2]: vol_trend_mh, vol_trend_color_mh = "Increasing (Recent)", MH_GOLD
                      elif data['VOLUME'].iloc[-1] < data['VOLUME'].iloc[-2]: vol_trend_mh, vol_trend_color_mh = "Decreasing (Recent)", "#FF5050"
                      else: vol_trend_mh, vol_trend_color_mh = "Flat (Recent)", "white"
                 else:
                      vol_trend_mh, vol_trend_color_mh = "Not Enough Data", "#888888"
            st.markdown(f""" <div class="mh-dashboard-card"> <div class="mh-stat-label">Volume Trend</div>
                <div style="font-size:1.5em; color:{vol_trend_color_mh}; font-weight:bold; margin:10px 0;">{vol_trend_mh}</div>
                <div style="color:#AAA; font-size:0.8em;">Recent 5-day vs previous 15-day average (min 20 days needed)</div> </div> """, unsafe_allow_html=True)

        st.markdown(f"<h4 style='color:{MH_RED}; margin-top:20px; text-shadow: 0 0 3px {MH_RED};'>Advanced Technical Indicators</h4>", unsafe_allow_html=True)
        macd_fig = go.Figure()
        if 'CLOSE' in data.columns and not data['CLOSE'].isnull().all() and len(data['CLOSE'].dropna()) >= 34: 
            cleaned_close = data['CLOSE'].dropna()
            exp12 = cleaned_close.ewm(span=12, adjust=False).mean()
            exp26 = cleaned_close.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal_macd = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal_macd

            valid_indices = macd.dropna().index
            if not valid_indices.empty:
                macd_fig.add_trace(go.Scatter(x=valid_indices, y=macd.loc[valid_indices], line=dict(color=MH_WHITE, width=2), name='MACD Line'))
                signal_valid_indices = signal_macd.dropna().index
                if not signal_valid_indices.empty:
                     macd_fig.add_trace(go.Scatter(x=signal_valid_indices, y=signal_macd.loc[signal_valid_indices], line=dict(color=MH_GOLD, width=1.5), name='Signal Line'))
                hist_valid_indices = histogram.dropna().index
                if not hist_valid_indices.empty:
                     macd_fig.add_trace(go.Bar(x=hist_valid_indices, y=histogram.loc[hist_valid_indices], marker_color=np.where(histogram.loc[hist_valid_indices] >= 0, MH_GOLD, '#FF5050'), name='Histogram'))

                macd_fig.update_layout(title='MACD (12,26,9)', template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(28,28,30,0.85)', margin=dict(t=50,b=30,l=30,r=20), legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font_size=10))
                st.plotly_chart(macd_fig, use_container_width=True)
            else: st.warning("Insufficient data points for MACD plot after dropping NaNs.")
        else: st.info("Insufficient data to calculate MACD (need at least 34 data points for CLOSE).")

    with tabs[2]:
        st.markdown(f"<h4 style='color:{MH_RED}; text-shadow: 0 0 3px {MH_RED};'>Statistical Insights</h4>", unsafe_allow_html=True)
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            ret_fig = plot_returns_distribution(data, theme_colors=mh_theme_colors)
            if ret_fig: st.plotly_chart(ret_fig, use_container_width=True)
            else: st.info("Not enough data for Returns Distribution plot.")
        with col_stats2:
            vol_fig = plot_rolling_volatility(data, theme_colors=mh_theme_colors)
            if vol_fig: st.plotly_chart(vol_fig, use_container_width=True)
            else: st.info("Not enough data for Rolling Volatility plot (min 21 days).")

        st.markdown(f"<h4 style='color:{MH_RED}; margin-top:20px; text-shadow: 0 0 3px {MH_RED};'>Price Correlation Matrix</h4>", unsafe_allow_html=True)
        corr_fig = plot_feature_correlation_heatmap(data.copy(), theme_colors=mh_theme_colors)
        if corr_fig: st.plotly_chart(corr_fig, use_container_width=True)
        else: st.info("Not enough data points to calculate correlation matrix.")


        st.markdown(f"<h4 style='color:{MH_RED}; margin-top:20px; text-shadow: 0 0 3px {MH_RED};'>Seasonal Patterns Analysis</h4>", unsafe_allow_html=True)
        fig_seasonal = plot_seasonal_decomposition(data, theme_colors=mh_theme_colors)
        if fig_seasonal: st.plotly_chart(fig_seasonal, use_container_width=True)
        else: st.info("Not enough data for Seasonal Decomposition (min ~60 days).")

        st.markdown(f"<h4 style='color:{MH_RED}; margin-top:20px; text-shadow: 0 0 3px {MH_RED};'>Stationarity Analysis (ADF Test)</h4>", unsafe_allow_html=True)
        fig_stationary = plot_stationarity_test(data, theme_colors=mh_theme_colors)
        if fig_stationary: st.plotly_chart(fig_stationary, use_container_width=True)
        else: st.info("Not enough data for Stationarity Test (min 20 days).")

        st.markdown(f"<h4 style='color:{MH_RED}; margin-top:20px; text-shadow: 0 0 3px {MH_RED};'>Autocorrelation Analysis</h4>", unsafe_allow_html=True)
        fig_autocorr = plot_autocorrelation(data, theme_colors=mh_theme_colors)
        if fig_autocorr: st.plotly_chart(fig_autocorr, use_container_width=True)
        else: st.info("Not enough data for Autocorrelation Analysis (min 20 days).")


# --- MAIN APP ROUTER ---
def main():
    session_keys = [
        "current_stage", "data_loaded_from_welcome", "ticker", "selected_period",
        "ml_results", "trained_models", "corr_df", "ml_error", "selected_model_type",
        "sidebar_quote_aib", "sidebar_quote_sq", "sidebar_quote_mh",
        "ticker_input_welcome", "selected_period_welcome", "uploaded_file_welcome",
        "data", "info", "error" 
        ]
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None if key in ["ml_results", "trained_models", "corr_df", "ml_error", "uploaded_file_welcome", "data", "info", "error"] else \
                                     False if key == "data_loaded_from_welcome" else \
                                     "AAPL" if key == "ticker" else \
                                     "1 Year" if key == "selected_period" else \
                                     "Price" if key == "selected_model_type" else \
                                     "" if key in ["sidebar_quote_aib", "sidebar_quote_sq", "sidebar_quote_mh", "ticker_input_welcome", "selected_period_welcome"] else \
                                     "welcome" 

    if st.session_state.current_stage == "welcome":
         render_welcome_page()
    elif st.session_state.get('data') is not None and not st.session_state.data.empty and st.session_state.get('error') is None:
        if st.session_state.current_stage == "aib_analysis":
            render_aib_stock_analysis_module()
        elif st.session_state.current_stage == "squid_ml":
            render_squid_ml_module()
        elif st.session_state.current_stage == "moneyheist":
            render_moneyheist_dashboard()
        else: 
             st.session_state.current_stage = "aib_analysis"
             st.rerun()
    else:
        if st.session_state.current_stage != "welcome":
            st.warning("No valid data loaded. Returning to home page.")
        st.session_state.current_stage = "welcome"
        st.session_state.ml_results = None
        st.session_state.trained_models = None
        st.session_state.corr_df = None
        st.session_state.ml_error = None
        st.session_state.data_loaded_from_welcome = False
        st.session_state.data = None
        st.session_state.info = None
        st.rerun() 

if __name__ == "__main__":
    main()

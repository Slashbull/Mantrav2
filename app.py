# app.py — M.A.N.T.R.A. FINAL VERSION
# The most powerful, simplest stock edge dashboard.

import streamlit as st
import pandas as pd
from datetime import datetime

from config import CONFIG
from data_loader import load_and_process
from engine import run_signal_engine

st.set_page_config(
    page_title="M.A.N.T.R.A.",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
html, body, .main { background: #f7faff !important; }
.stButton > button, .stDownloadButton > button { background: #1769aa; color: #fff; font-weight:600; border-radius:7px;}
.stButton > button:hover { background: #104c7d;}
.card { background: #fff; border-radius: 14px; box-shadow: 0 2px 14px #e3eaf7; padding: 1.2rem 1.5rem; margin-bottom: 1rem;}
.metric-label, h1, h2, h3 { color: #14213d !important; }
::-webkit-scrollbar {width:7px; background:#f7faff;} ::-webkit-scrollbar-thumb {background:#e3eaf7;}
</style>
""", unsafe_allow_html=True)

# ---- Data Load & Analysis ----
@st.cache_data(ttl=600, show_spinner="Loading & analyzing data…")
def get_signals():
    stocks, sector, summary = load_and_process()
    if stocks is None or stocks.empty:
        st.error("No stock data loaded. Check Google Sheet.")
        st.stop()
    df = run_signal_engine(stocks, sector)
    return df

# ---- Main App ----
st.title("M.A.N.T.R.A. — Personal Stock Edge Engine")
st.caption("Decisions, Not Guesses. Built For My Edge.")

# -- Filter Bar --
col1, col2, col3, col4 = st.columns([2,2,2,2])
with col1:
    tags = ["All", "STRONG_BUY", "BUY", "HOLD", "SELL"]
    tag = st.selectbox("Signal", tags, index=0)
with col2:
    conf_min = st.slider("Min Confidence %", 0, 100, 70, 1)
with col3:
    search = st.text_input("Search (Ticker/Name)", "")
with col4:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# -- Load Data --
signals = get_signals()
show_df = signals.copy()
if tag != "All" and "signal" in show_df.columns:
    show_df = show_df[show_df.signal == tag]
if search:
    mask = pd.Series([False]*len(show_df))
    if "ticker" in show_df.columns:
        mask |= show_df.ticker.str.contains(search, case=False, na=False)
    if "company_name" in show_df.columns:
        mask |= show_df.company_name.str.contains(search, case=False, na=False)
    show_df = show_df[mask]
if "confidence" in show_df.columns:
    show_df = show_df[show_df.confidence >= conf_min]
show_df = show_df.sort_values(["confidence", "signal"], ascending=[False, True])

# -- KPIs --
k1, k2, k3, k4 = st.columns(4)
k1.metric("Stocks Scanned", len(signals))
k2.metric("Strong Buy", int((signals.signal == "STRONG_BUY").sum()) if "signal" in signals else 0)
k3.metric("Buy", int((signals.signal == "BUY").sum()) if "signal" in signals else 0)
k4.metric("Updated", datetime.now().strftime("%d %b, %I:%M %p"))

st.divider()

# -- Opportunities Cards --
if show_df.empty:
    st.warning("No stocks found. Adjust filter or search.")
else:
    st.markdown("#### 🏆 Top Opportunities")
    topn = show_df.head(10)
    for i, row in topn.iterrows():
        st.markdown(f"""<div class='card'>
        <span style='font-size:1.25rem; font-weight:700;'>{row.get('ticker','')}</span>
        <span style='font-size:1rem; color:#444;'> — {row.get('company_name','')[:40]}</span>
        <span style='float:right; font-weight:700; color:#1769aa;'>{row.get('signal','')}</span>
        <br>
        <b>₹{row.get('price',0):,.0f}</b> ({row.get('sector','')}) | Conf: <b>{row.get('confidence',0):.0f}%</b>
        <span style='margin-left:1.2rem;'>30D: <b>{row.get('ret_30d',0):+.1f}%</b></span>
        <span style='margin-left:1.2rem;'>Vol: <b>{row.get('rvol',0):.1f}x</b></span>
        <span style='margin-left:1.2rem;'>P/E: <b>{row.get('pe','N/A') if row.get('pe',0)>0 else 'N/A'}</b></span>
        </div>""", unsafe_allow_html=True)
    st.divider()

    with st.expander("Show All (Table)", expanded=False):
        cols = ['ticker','company_name','signal','confidence','price','ret_1d','ret_30d','pe','eps_change_pct','rvol','sector']
        dfshow = show_df[cols] if all(c in show_df.columns for c in cols) else show_df
        st.dataframe(dfshow, use_container_width=True, hide_index=True)

# -- Download Button --
st.download_button(
    label="Download Filtered CSV",
    data=show_df.to_csv(index=False).encode(),
    file_name=f"mantra_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=True
)

# -- Help/Legend --
with st.expander("ℹ️ Legend & Quick Help", expanded=False):
    st.markdown("""
**Signal meanings:**  
- 🟢 **STRONG_BUY** = highest conviction  
- 🟢 **BUY** = very good setup  
- 🟡 **HOLD** = neutral/wait  
- 🔴 **SELL** = avoid or exit  
**Confidence** = overall score (0-100)  
**You can filter, search, download, or refresh any time.**  
_No signals? Try "All" or reduce confidence filter._  
---
**Philosophy:** _All edge, no noise. Only what matters._
""")

# END OF FILE

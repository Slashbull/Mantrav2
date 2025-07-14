# app.py - M.A.N.T.R.A. ULTRA SIMPLE, FINAL VERSION (2025)
# "Decisions, not guesses. My Edge, My Logic, My Market."

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config import CONFIG
from data_loader import DataLoader
from professional_signal_engine import ProfessionalSignalEngine

st.set_page_config(
    page_title="M.A.N.T.R.A.",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
body, .main, .block-container { background: #fcfcfc; }
.metric-label, h1, h2, h3, .st-emotion-cache-1wmy9hl, .stMarkdown { color: #222 !important; }
.stButton > button { background: #2962ff; color: #fff; font-weight: 500; border-radius: 6px; }
.stButton > button:hover { background: #1941a9;}
.card { background: #fff; border-radius: 12px; box-shadow: 0 1px 6px #d3d3d3; padding: 1.2rem 1.5rem; margin-bottom: 1rem;}
.top-bar { background: #f7faff; border-radius: 10px; padding: 0.7rem 1rem 0.7rem 1.2rem; margin-bottom: 1.5rem; display: flex; align-items: center;}
.signal-STRONG_BUY {color: #00c853; font-weight: 600;}
.signal-BUY {color: #1976d2; font-weight: 600;}
.signal-HOLD {color: #fb8c00;}
.signal-SELL {color: #d32f2f;}
::-webkit-scrollbar {width:6px; background:#f3f3f3;} ::-webkit-scrollbar-thumb {background:#e0e0e0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600, show_spinner="Loading & analyzing data…")
def get_data():
    loader = DataLoader()
    ok, msg = loader.load_and_process()
    if not ok: st.stop()
    stocks = loader.get_stocks_data()
    sector = loader.get_sector_data()
    signals = ProfessionalSignalEngine().analyze(stocks, sector)
    return signals

def filter_data(df, tag, search, conf_min):
    if tag != "All": df = df[df.signal == tag]
    if search: df = df[df.ticker.str.contains(search, case=False) | df.company_name.str.contains(search, case=False)]
    df = df[df.confidence >= conf_min]
    return df

# === Top Bar Controls ===
with st.container():
    st.markdown("<div class='top-bar'>"
                "<span style='font-size:1.4rem; font-weight:700; letter-spacing:1px;'>M.A.N.T.R.A. &mdash; Personal Stock Intelligence</span>"
                "</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([2,2,2,3,3])

tags = ["All", "STRONG_BUY", "BUY", "HOLD", "SELL"]
with col1: tag = st.selectbox("Signal", tags, index=1, key="sigfilter")
with col2: conf_min = st.slider("Min Confidence %", 0, 100, 70, 1)
with col3: search = st.text_input("Search (Ticker/Name)", "")
with col4:
    if st.button("🔄 Refresh", use_container_width=True): st.cache_data.clear()
with col5:
    st.info("Signals: 🟢 Strong Buy / Buy, 🟡 Hold, 🔴 Sell", icon="📈")

# === Data Load & Filter ===
signals = get_data()
signals = signals.copy()
show_df = filter_data(signals, tag, search, conf_min)
show_df = show_df.sort_values(["confidence", "signal"], ascending=[False, True])

# === Key KPIs ===
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Stocks Scanned", len(signals))
with k2:
    st.metric("Strong Buy", int((signals.signal == "STRONG_BUY").sum()))
with k3:
    st.metric("Buy", int((signals.signal == "BUY").sum()))
with k4:
    st.metric("Updated", datetime.now().strftime("%d %b, %I:%M %p"))

st.divider()

# === Main Results Area ===
if len(show_df) == 0:
    st.warning("No stocks found. Adjust filter or search.")
else:
    st.markdown("#### 🏆 Top Opportunities")
    topn = show_df.head(12)
    for i, row in topn.iterrows():
        st.markdown(f"""<div class='card'>
        <span style='font-size:1.3rem; font-weight:700;'>{row.ticker}</span>
        <span style='font-size:1rem; color:#444;'> — {row.company_name[:40]}</span>
        <span class='signal-{row.signal}' style='float:right;font-weight:700;font-size:1.2rem;'>{row.signal}</span>
        <br>
        <b>₹{row.price:,.0f}</b> ({row.sector}) | Conf: <b>{row.confidence:.0f}%</b>
        <span style='margin-left:1rem;'>30D: <b>{row.ret_30d:+.1f}%</b></span>
        <span style='margin-left:1rem;'>Vol: <b>{row.rvol:.1f}x</b></span>
        <span style='margin-left:1rem;'>P/E: <b>{row.pe:.1f if row.pe>0 else 'N/A'}</b></span>
        <br>
        <span style='font-size:0.98rem;color:#888;'>{row.key_insights if 'key_insights' in row and row.key_insights else ''}</span>
        </div>""", unsafe_allow_html=True)
    st.divider()
    with st.expander("Show All (Table)", expanded=False):
        cols = ['ticker','company_name','signal','confidence','price','ret_1d','ret_30d','pe','eps_change_pct','rvol','sector']
        dfshow = show_df[cols] if all(c in show_df.columns for c in cols) else show_df
        st.dataframe(dfshow, use_container_width=True, hide_index=True)

# === Download Button ===
st.download_button(
    label="Download Filtered CSV",
    data=show_df.to_csv(index=False).encode(),
    file_name=f"mantra_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=True
)

# === Help / Legend ===
with st.expander("ℹ️ Legend & Quick Help", expanded=False):
    st.markdown("""
- **Signal meanings**:  
    🟢 **STRONG_BUY** = highest conviction,  
    🟢 **BUY** = very good setup,  
    🟡 **HOLD** = neutral/wait,  
    🔴 **SELL** = avoid or exit.
- **Confidence** = overall combined score (0-100).
- **P/E, rvol, 30D return** = standard metrics.
- You can filter, search, download, or refresh any time.
- No signals = adjust confidence filter or try "All".
---
**Philosophy:** _All edge, no noise. No tips, no confusion. Only what matters._
""")

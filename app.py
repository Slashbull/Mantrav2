"""
app.py - M.A.N.T.R.A. Version 3 FINAL
=====================================
Ultimate simple UI with best UX
No bugs, ultra-reliable, locked forever
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

from config import CONFIG
from data_loader import DataLoader
from signal_engine import SignalEngine

# Page configuration
st.set_page_config(
    page_title=f"{CONFIG.APP_ICON} {CONFIG.APP_NAME} {CONFIG.APP_VERSION}",
    page_icon=CONFIG.APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .app-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Stock card styling */
    .stock-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stock-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Signal badge */
    .signal-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Metric styling */
    .metric-container {
        text-align: center;
        padding: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin: 0;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.8rem;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border: none;
        color: white;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #20c997 0%, #28a745 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'signal_engine' not in st.session_state:
    st.session_state.signal_engine = SignalEngine()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def render_header():
    """Render application header"""
    st.markdown(f"""
    <div class="app-header">
        <h1>{CONFIG.APP_ICON} {CONFIG.APP_NAME} {CONFIG.APP_VERSION}</h1>
        <p>{CONFIG.APP_SUBTITLE}</p>
        <p style="font-size: 1rem; margin-top: 1rem;">Ultra-high confidence signals with crystal-clear reasoning</p>
    </div>
    """, unsafe_allow_html=True)

def format_number(value, decimals=0, prefix="", suffix=""):
    """Format number for display"""
    if pd.isna(value):
        return "N/A"
    if decimals == 0:
        return f"{prefix}{value:,.0f}{suffix}"
    else:
        return f"{prefix}{value:,.{decimals}f}{suffix}"

def format_change(value):
    """Format percentage change with color"""
    if pd.isna(value):
        return "N/A"
    color = "green" if value > 0 else "red" if value < 0 else "gray"
    return f'<span style="color: {color}; font-weight: bold;">{value:+.1f}%</span>'

def render_stock_card(stock):
    """Render a single stock opportunity card"""
    
    # Get signal color
    signal_color = CONFIG.SIGNAL_COLORS.get(stock['signal'], '#868e96')
    
    # Create card HTML
    card_html = f"""
    <div class="stock-card" style="border: 2px solid {signal_color};">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h3 style="margin: 0; color: #333;">{stock.get('ticker', 'N/A')}</h3>
                <p style="margin: 0.3rem 0; color: #666; font-size: 0.9rem;">
                    {stock.get('company_name', 'Unknown Company')}
                </p>
                <p style="margin: 0.3rem 0; color: #888; font-size: 0.8rem;">
                    {stock.get('sector', 'Unknown Sector')} | {stock.get('category', 'Unknown Category')}
                </p>
            </div>
            <div style="text-align: right;">
                <div class="signal-badge" style="background: {signal_color}; color: white;">
                    {stock['signal']}
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">
                    ₹{format_number(stock.get('price', 0))}
                </p>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">
                    Confidence: {stock.get('confidence', 0):.0f}%
                </p>
            </div>
        </div>
        
        <hr style="margin: 1rem 0; opacity: 0.3;">
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div class="metric-container">
                <p class="metric-label">1D Return</p>
                <p class="metric-value">{format_change(stock.get('ret_1d', 0))}</p>
            </div>
            <div class="metric-container">
                <p class="metric-label">30D Return</p>
                <p class="metric-value">{format_change(stock.get('ret_30d', 0))}</p>
            </div>
            <div class="metric-container">
                <p class="metric-label">P/E Ratio</p>
                <p class="metric-value">{format_number(stock.get('pe', 0), 1)}</p>
            </div>
            <div class="metric-container">
                <p class="metric-label">Volume</p>
                <p class="metric-value">{format_number(stock.get('rvol', 1), 1)}x</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
            <div class="metric-container">
                <p class="metric-label">EPS Growth</p>
                <p class="metric-value">{format_change(stock.get('eps_change_pct', 0))}</p>
            </div>
            <div class="metric-container">
                <p class="metric-label">52W Position</p>
                <p class="metric-value">{format_number(stock.get('position_52w', 50), 0)}%</p>
            </div>
            <div class="metric-container">
                <p class="metric-label">Momentum</p>
                <p class="metric-value">{format_number(stock.get('momentum_score', 0), 0)}/100</p>
            </div>
            <div class="metric-container">
                <p class="metric-label">Value</p>
                <p class="metric-value">{format_number(stock.get('value_score', 0), 0)}/100</p>
            </div>
        </div>
        
        <hr style="margin: 1rem 0; opacity: 0.3;">
        
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
            <p style="margin: 0; font-weight: bold; color: #333;">📊 Signal Analysis</p>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                {stock.get('explanation', 'Analysis based on 8-factor scoring system')}
            </p>
            <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.8rem;">
                Composite Score: {stock.get('composite_score', 0):.1f}/100
            </p>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def render_market_summary(signals_df):
    """Render market summary metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = len(signals_df)
        st.metric("📊 Total Stocks", f"{total_stocks:,}")
    
    with col2:
        strong_signals = len(signals_df[signals_df['signal'].isin(['STRONG_BUY', 'BUY'])])
        st.metric("🎯 Strong Signals", strong_signals)
    
    with col3:
        if 'ret_1d' in signals_df.columns:
            positive = (signals_df['ret_1d'] > 0).sum()
            breadth = (positive / len(signals_df) * 100) if len(signals_df) > 0 else 0
            st.metric("📈 Market Breadth", f"{breadth:.1f}%")
        else:
            st.metric("📈 Market Breadth", "N/A")
    
    with col4:
        if st.session_state.last_update:
            time_diff = (datetime.now() - st.session_state.last_update).total_seconds() / 60
            st.metric("🕒 Last Update", f"{time_diff:.0f}m ago")
        else:
            st.metric("🕒 Last Update", "Never")

def load_and_analyze_data():
    """Load data and run analysis"""
    
    with st.spinner("🔄 Loading market data..."):
        # Load data
        success, message = st.session_state.data_loader.load_and_process(use_cache=True)
        
        if not success:
            st.error(f"❌ {message}")
            return None
        
        stocks_df = st.session_state.data_loader.get_stocks_data()
        sector_df = st.session_state.data_loader.get_sector_data()
        
        if stocks_df.empty:
            st.error("❌ No data loaded")
            return None
    
    with st.spinner(f"🧠 Analyzing {len(stocks_df)} stocks with 8-factor system..."):
        # Run signal analysis
        signals_df = st.session_state.signal_engine.analyze(stocks_df, sector_df)
        
        # Update last update time
        st.session_state.last_update = datetime.now()
        
        # Get signal summary
        signal_summary = st.session_state.signal_engine.get_signal_summary()
        
        # Success message
        st.success(f"""
        ✅ **Analysis Complete!**
        - Analyzed {len(signals_df)} stocks in {st.session_state.data_loader.health['processing_time_s']:.1f} seconds
        - Found {signal_summary.get('STRONG_BUY', 0)} STRONG_BUY signals
        - Found {signal_summary.get('BUY', 0)} BUY signals
        - Data quality score: {st.session_state.data_loader.health['quality_analysis']['quality_score']:.0f}/100
        """)
        
        return signals_df

def main():
    """Main application"""
    
    # Render header
    render_header()
    
    # Load data button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Load Market Intelligence", type="primary", use_container_width=True):
            signals_df = load_and_analyze_data()
            if signals_df is not None:
                st.session_state.signals_df = signals_df
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.data_loader.clear_cache()
            st.success("✅ Cache cleared")
    
    with col3:
        if st.button("📊 Health Check", use_container_width=True):
            health = st.session_state.data_loader.get_health()
            if health:
                st.json(health['quality_analysis'])
    
    # Check if we have data
    if 'signals_df' not in st.session_state:
        # Welcome message
        st.info("""
        👋 **Welcome to M.A.N.T.R.A. Version 3 Final**
        
        Click **"Load Market Intelligence"** to analyze 2200+ stocks and find the best opportunities with:
        - 🎯 Ultra-conservative signals (92+ for STRONG_BUY)
        - 📊 8-factor precision analysis
        - 🧠 Clear reasoning for every signal
        - ⚡ 1-3 second performance
        
        **No bugs. No crashes. Just pure market intelligence.**
        """)
        return
    
    signals_df = st.session_state.signals_df
    
    # Market summary
    st.markdown("---")
    render_market_summary(signals_df)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Top Opportunities", 
        "📊 All Stocks", 
        "📈 Signal Distribution",
        "📋 Export Data"
    ])
    
    with tab1:
        # Top opportunities
        st.markdown("### 🚀 Today's Top Opportunities")
        
        # Get top opportunities
        top_opps = st.session_state.signal_engine.get_top_opportunities(limit=CONFIG.MAX_OPPORTUNITIES)
        
        if top_opps.empty:
            st.warning("📭 No high-confidence opportunities found at this time.")
        else:
            # Render each opportunity as a card
            for _, stock in top_opps.iterrows():
                render_stock_card(stock)
    
    with tab2:
        # All stocks view
        st.markdown("### 📊 All Stocks Analysis")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sectors = ['All'] + sorted(signals_df['sector'].dropna().unique().tolist())
            selected_sector = st.selectbox("🏢 Sector", sectors, key="sector_filter")
        
        with col2:
            categories = ['All'] + sorted(signals_df['category'].dropna().unique().tolist()) if 'category' in signals_df.columns else ['All']
            selected_category = st.selectbox("📁 Category", categories, key="category_filter")
        
        with col3:
            signals = ['All'] + ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH']
            selected_signal = st.selectbox("🎯 Signal", signals, key="signal_filter")
        
        with col4:
            min_confidence = st.slider("🎚️ Min Confidence", 0, 100, 60, key="confidence_filter")
        
        # Apply filters
        filtered_df = signals_df.copy()
        
        if selected_sector != 'All':
            filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
        
        if selected_category != 'All' and 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if selected_signal != 'All':
            filtered_df = filtered_df[filtered_df['signal'] == selected_signal]
        
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display count
        st.info(f"📊 Showing {len(filtered_df)} stocks (filtered from {len(signals_df)} total)")
        
        # Display table
        if not filtered_df.empty:
            # Select columns to display
            display_cols = [
                'ticker', 'company_name', 'signal', 'confidence', 'composite_score',
                'price', 'ret_1d', 'ret_30d', 'pe', 'eps_change_pct', 'rvol', 
                'momentum_score', 'value_score', 'sector', 'category'
            ]
            
            # Keep only available columns
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Format the dataframe for display
            display_df = filtered_df[display_cols].copy()
            
            # Apply conditional formatting
            st.dataframe(
                display_df.head(CONFIG.MAX_TABLE_ROWS),
                use_container_width=True,
                height=600,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "company_name": st.column_config.TextColumn("Company", width="medium"),
                    "signal": st.column_config.TextColumn("Signal", width="small"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100),
                    "composite_score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "price": st.column_config.NumberColumn("Price", format="₹%.0f"),
                    "ret_1d": st.column_config.NumberColumn("1D%", format="%.1f%%"),
                    "ret_30d": st.column_config.NumberColumn("30D%", format="%.1f%%"),
                    "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                    "eps_change_pct": st.column_config.NumberColumn("EPS Δ%", format="%.1f%%"),
                    "rvol": st.column_config.NumberColumn("RVol", format="%.1fx"),
                    "momentum_score": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100),
                    "value_score": st.column_config.ProgressColumn("Value", min_value=0, max_value=100),
                }
            )
        else:
            st.warning("No stocks match the selected filters")
    
    with tab3:
        # Signal distribution
        st.markdown("### 📈 Signal Distribution Analysis")
        
        # Get signal counts
        signal_counts = signals_df['signal'].value_counts()
        
        # Create columns for signal boxes
        cols = st.columns(len(signal_counts))
        
        for i, (signal, count) in enumerate(signal_counts.items()):
            with cols[i]:
                color = CONFIG.SIGNAL_COLORS.get(signal, '#868e96')
                percentage = (count / len(signals_df) * 100)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {color}20; 
                            border: 2px solid {color}; border-radius: 10px;">
                    <h3 style="margin: 0; color: {color};">{signal}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;">{count}</p>
                    <p style="margin: 0; color: #666;">{percentage:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Sector performance
        if 'sector' in signals_df.columns:
            st.markdown("### 🏢 Sector Analysis")
            
            sector_summary = signals_df.groupby('sector').agg({
                'signal': 'count',
                'ret_30d': 'mean',
                'composite_score': 'mean'
            }).round(1)
            
            sector_summary.columns = ['Stock Count', 'Avg 30D Return %', 'Avg Score']
            sector_summary = sector_summary.sort_values('Avg Score', ascending=False)
            
            st.dataframe(
                sector_summary,
                use_container_width=True,
                column_config={
                    "Stock Count": st.column_config.NumberColumn("Stocks", format="%d"),
                    "Avg 30D Return %": st.column_config.NumberColumn("Avg 30D%", format="%.1f%%"),
                    "Avg Score": st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=100)
                }
            )
    
    with tab4:
        # Export functionality
        st.markdown("### 📋 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare CSV
            csv = signals_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Analysis (CSV)",
                data=csv,
                file_name=f"mantra_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Prepare top opportunities CSV
            top_opps = st.session_state.signal_engine.get_top_opportunities(limit=50)
            if not top_opps.empty:
                top_csv = top_opps.to_csv(index=False)
                st.download_button(
                    label="🎯 Download Top 50 Opportunities (CSV)",
                    data=top_csv,
                    file_name=f"mantra_top50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()

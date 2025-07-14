"""
app.py - M.A.N.T.R.A. Version 3 FINAL (Simplified)
==================================================
Ultimate simple UI with native Streamlit components
No complex HTML, just pure functionality
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

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'signal_engine' not in st.session_state:
    st.session_state.signal_engine = SignalEngine()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown(f"# {CONFIG.APP_ICON} {CONFIG.APP_NAME} {CONFIG.APP_VERSION}")
        st.markdown(f"**{CONFIG.APP_SUBTITLE}**")
        st.caption("Ultra-high confidence signals with crystal-clear reasoning")
    
    st.markdown("---")

def render_stock_card(stock):
    """Render a single stock opportunity card using native Streamlit components"""
    
    # Get signal color
    signal_color = CONFIG.SIGNAL_COLORS.get(stock['signal'], '#868e96')
    
    # Create an expander for each stock
    with st.expander(f"**{stock.get('ticker', 'N/A')} - {stock.get('company_name', 'Unknown')}**", expanded=True):
        
        # Header information
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Sector:** {stock.get('sector', 'Unknown')}")
            st.markdown(f"**Category:** {stock.get('category', 'Unknown')}")
        
        with col2:
            # Signal badge
            if stock['signal'] == 'STRONG_BUY':
                st.success(f"🚀 {stock['signal']}")
            elif stock['signal'] == 'BUY':
                st.success(f"📈 {stock['signal']}")
            elif stock['signal'] == 'ACCUMULATE':
                st.info(f"📊 {stock['signal']}")
            else:
                st.warning(f"👀 {stock['signal']}")
        
        with col3:
            st.metric("Price", f"₹{stock.get('price', 0):,.0f}")
            st.caption(f"Confidence: {stock.get('confidence', 0):.0f}%")
        
        # Key metrics
        st.markdown("### 📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ret_1d = stock.get('ret_1d', 0)
            st.metric(
                "1D Return", 
                f"{ret_1d:+.1f}%",
                delta=f"{ret_1d:+.1f}%",
                delta_color="normal"
            )
        
        with col2:
            ret_30d = stock.get('ret_30d', 0)
            st.metric(
                "30D Return", 
                f"{ret_30d:+.1f}%",
                delta=f"{ret_30d:+.1f}%",
                delta_color="normal"
            )
        
        with col3:
            pe = stock.get('pe', 0)
            st.metric("P/E Ratio", f"{pe:.1f}" if pe > 0 else "N/A")
        
        with col4:
            rvol = stock.get('rvol', 1)
            st.metric("Volume", f"{rvol:.1f}x")
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            eps_growth = stock.get('eps_change_pct', 0)
            if pd.notna(eps_growth):
                st.metric(
                    "EPS Growth",
                    f"{eps_growth:+.1f}%",
                    delta=f"{eps_growth:+.1f}%",
                    delta_color="normal"
                )
            else:
                st.metric("EPS Growth", "N/A")
        
        with col2:
            position = stock.get('position_52w', 50)
            st.metric("52W Position", f"{position:.0f}%")
        
        with col3:
            momentum = stock.get('momentum_score', 0)
            st.metric("Momentum", f"{momentum:.0f}/100")
        
        with col4:
            value = stock.get('value_score', 0)
            st.metric("Value", f"{value:.0f}/100")
        
        # Signal Analysis
        st.markdown("### 🎯 Signal Analysis")
        st.info(f"""
        **Reasoning:** {stock.get('explanation', 'Analysis based on 8-factor scoring system')}
        
        **Composite Score:** {stock.get('composite_score', 0):.1f}/100
        """)
        
        # Factor scores breakdown
        st.markdown("### 📈 Factor Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Momentum:** {stock.get('momentum_score', 0):.0f}/100")
            st.markdown(f"**Value:** {stock.get('value_score', 0):.0f}/100")
            st.markdown(f"**Growth:** {stock.get('growth_score', 0):.0f}/100")
            st.markdown(f"**Volume:** {stock.get('volume_score', 0):.0f}/100")
        
        with col2:
            st.markdown(f"**Technical:** {stock.get('technical_score', 0):.0f}/100")
            st.markdown(f"**Sector:** {stock.get('sector_score', 0):.0f}/100")
            st.markdown(f"**Risk:** {stock.get('risk_score', 0):.0f}/100")
            st.markdown(f"**Quality:** {stock.get('quality_score', 0):.0f}/100")

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
    
    # Control buttons
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
                with st.expander("System Health", expanded=True):
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
        
        **Simple. Powerful. Reliable.**
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
        st.markdown("## 🚀 Today's Top Opportunities")
        
        # Get top opportunities
        top_opps = st.session_state.signal_engine.get_top_opportunities(limit=CONFIG.MAX_OPPORTUNITIES)
        
        if top_opps.empty:
            st.warning("📭 No high-confidence opportunities found at this time.")
        else:
            # Show signal summary
            signal_counts = top_opps['signal'].value_counts()
            summary_cols = st.columns(len(signal_counts))
            
            for i, (signal, count) in enumerate(signal_counts.items()):
                with summary_cols[i]:
                    if signal == 'STRONG_BUY':
                        st.success(f"🚀 {signal}: {count}")
                    elif signal == 'BUY':
                        st.success(f"📈 {signal}: {count}")
                    else:
                        st.info(f"📊 {signal}: {count}")
            
            st.markdown("---")
            
            # Render each opportunity
            for idx, stock in top_opps.iterrows():
                render_stock_card(stock)
                st.markdown("---")
    
    with tab2:
        # All stocks view
        st.markdown("## 📊 All Stocks Analysis")
        
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
                'sector', 'category'
            ]
            
            # Keep only available columns
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Display the dataframe
            st.dataframe(
                filtered_df[display_cols].head(CONFIG.MAX_TABLE_ROWS),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No stocks match the selected filters")
    
    with tab3:
        # Signal distribution
        st.markdown("## 📈 Signal Distribution Analysis")
        
        # Get signal counts
        signal_counts = signals_df['signal'].value_counts()
        
        # Display signal distribution
        st.markdown("### Signal Count Overview")
        cols = st.columns(len(signal_counts))
        
        for i, (signal, count) in enumerate(signal_counts.items()):
            with cols[i]:
                percentage = (count / len(signals_df) * 100)
                st.metric(
                    label=signal,
                    value=count,
                    delta=f"{percentage:.1f}%"
                )
        
        # Sector performance
        if 'sector' in signals_df.columns:
            st.markdown("### 🏢 Top Sectors by Average Score")
            
            sector_summary = signals_df.groupby('sector').agg({
                'signal': 'count',
                'ret_30d': 'mean',
                'composite_score': 'mean'
            }).round(1)
            
            sector_summary.columns = ['Stock Count', 'Avg 30D Return %', 'Avg Score']
            sector_summary = sector_summary.sort_values('Avg Score', ascending=False).head(10)
            
            st.dataframe(
                sector_summary,
                use_container_width=True
            )
    
    with tab4:
        # Export functionality
        st.markdown("## 📋 Export Data")
        
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

"""
app.py - M.A.N.T.R.A. PROFESSIONAL PRODUCTION VERSION
=====================================================
Clean, Simple, Beautiful, Bug-Free
Designed with proper wireframing and professional UX principles
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import traceback

from config import CONFIG
from data_loader import DataLoader
from professional_signal_engine import ProfessionalSignalEngine

# ============================================================================
# PROFESSIONAL PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. - Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL DESIGN SYSTEM
# ============================================================================
st.markdown("""
<style>
    /* Professional, Clean Design */
    .main {
        padding: 0rem 1rem;
    }
    
    .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    /* Card Design */
    .stock-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stock-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2962ff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1e40af;
        box-shadow: 0 2px 8px rgba(41, 98, 255, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        background-color: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #fafafa;
        padding-top: 2rem;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
        border: 1px solid #e0e0e0;
    }
    
    /* Professional Signal Colors */
    .signal-strong-buy { color: #00c853; font-weight: 600; }
    .signal-buy { color: #43a047; font-weight: 600; }
    .signal-hold { color: #fb8c00; font-weight: 500; }
    .signal-sell { color: #e53935; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT (ROBUST)
# ============================================================================
class SessionManager:
    """Manage session state professionally"""
    
    @staticmethod
    def initialize():
        """Initialize all session states with defaults"""
        defaults = {
            'data_loader': DataLoader(),
            'signal_engine': ProfessionalSignalEngine(),
            'data_loaded': False,
            'signals_df': pd.DataFrame(),
            'filtered_df': pd.DataFrame(),
            'last_update': None,
            'selected_filter': 'top_opportunities',
            'error_message': None,
            'loading': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def get(key, default=None):
        """Safely get session state value"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key, value):
        """Safely set session state value"""
        st.session_state[key] = value

# Initialize session
SessionManager.initialize()

# ============================================================================
# PROFESSIONAL FILTERS
# ============================================================================
SMART_FILTERS = {
    "top_opportunities": {
        "name": "🎯 Top Opportunities",
        "description": "Best overall trading opportunities",
        "filter": lambda df: df[df['signal'].isin(['STRONG_BUY', 'BUY'])].nlargest(20, 'confidence')
    },
    "momentum_leaders": {
        "name": "🚀 Momentum Leaders",
        "description": "Stocks with strongest momentum",
        "filter": lambda df: df[(df['momentum_score'] > 75) & (df['ret_30d'] > 10)].nlargest(20, 'momentum_score')
    },
    "value_picks": {
        "name": "💎 Value Picks",
        "description": "Undervalued opportunities",
        "filter": lambda df: df[(df['pe'] > 0) & (df['pe'] < 20) & (df['value_score'] > 70)].nlargest(20, 'value_score')
    },
    "volume_surge": {
        "name": "📊 Volume Surge",
        "description": "Unusual volume activity",
        "filter": lambda df: df[df['rvol'] > 3].nlargest(20, 'rvol')
    },
    "breakout_watch": {
        "name": "📈 Breakout Watch",
        "description": "Near 52-week highs",
        "filter": lambda df: df[df['position_52w'] > 80].nlargest(20, 'technical_score')
    }
}

# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================
def handle_errors(func):
    """Decorator for robust error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            st.error(error_msg)
            SessionManager.set('error_message', error_msg)
            return None
    return wrapper

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@handle_errors
def load_data():
    """Load and analyze data with proper error handling"""
    SessionManager.set('loading', True)
    
    # Progress tracking
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Step 1: Load data
        status.text("📥 Loading market data...")
        progress.progress(25)
        
        data_loader = SessionManager.get('data_loader')
        success, message = data_loader.load_and_process(use_cache=True)
        
        if not success:
            st.error(f"Failed to load data: {message}")
            return False
        
        # Step 2: Get dataframes
        progress.progress(50)
        status.text("📊 Processing data...")
        
        stocks_df = data_loader.get_stocks_data()
        sector_df = data_loader.get_sector_data()
        
        if stocks_df.empty:
            st.error("No stock data available")
            return False
        
        # Step 3: Run analysis
        progress.progress(75)
        status.text("🧠 Running intelligent analysis...")
        
        signal_engine = SessionManager.get('signal_engine')
        signals_df = signal_engine.analyze(stocks_df, sector_df)
        
        # Step 4: Save results
        progress.progress(100)
        status.text("✅ Analysis complete!")
        
        SessionManager.set('signals_df', signals_df)
        SessionManager.set('filtered_df', signals_df)  # Initialize with full data
        SessionManager.set('data_loaded', True)
        SessionManager.set('last_update', datetime.now())
        
        # Clear progress indicators
        progress.empty()
        status.empty()
        
        # Show success metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success(f"✅ Analyzed {len(signals_df)} stocks")
        with col2:
            buy_signals = len(signals_df[signals_df['signal'].isin(['STRONG_BUY', 'BUY'])])
            st.success(f"🎯 {buy_signals} buy signals")
        with col3:
            avg_confidence = signals_df['confidence'].mean()
            st.success(f"📊 {avg_confidence:.0f}% avg confidence")
        with col4:
            quality = data_loader.get_health()['quality_analysis']['quality_score']
            st.success(f"✨ {quality:.0f}% data quality")
        
        return True
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        SessionManager.set('loading', False)

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar():
    """Professional sidebar with all controls"""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style='text-align: center; padding-bottom: 2rem;'>
            <h1 style='font-size: 2rem; margin: 0;'>📊 M.A.N.T.R.A.</h1>
            <p style='color: #666; margin: 0;'>Market Analysis & Trading Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Management Section
        st.markdown("### 📁 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Data", use_container_width=True, type="primary"):
                load_data()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                SessionManager.get('data_loader').clear_cache()
                load_data()
        
        # Show data status
        if SessionManager.get('data_loaded'):
            last_update = SessionManager.get('last_update')
            if last_update:
                mins_ago = (datetime.now() - last_update).seconds // 60
                st.success(f"✅ Data loaded {mins_ago}m ago")
        else:
            st.info("💤 Data not loaded")
        
        st.markdown("---")
        
        # Filters Section
        if SessionManager.get('data_loaded'):
            st.markdown("### 🎯 Smart Filters")
            
            # Filter buttons
            for key, filter_info in SMART_FILTERS.items():
                if st.button(
                    filter_info['name'],
                    key=f"filter_{key}",
                    use_container_width=True,
                    type="primary" if SessionManager.get('selected_filter') == key else "secondary"
                ):
                    SessionManager.set('selected_filter', key)
                    apply_current_filter()
            
            # Show active filter description
            current_filter = SMART_FILTERS[SessionManager.get('selected_filter')]
            st.info(f"📌 {current_filter['description']}")
            
            st.markdown("---")
            
            # Advanced Filters
            st.markdown("### ⚙️ Advanced Filters")
            
            signals_df = SessionManager.get('signals_df', pd.DataFrame())
            
            if not signals_df.empty:
                # Sector filter
                sectors = ['All'] + sorted(signals_df['sector'].dropna().unique().tolist())
                selected_sector = st.selectbox("📂 Sector", sectors)
                
                # Signal filter
                signals = ['All'] + sorted(signals_df['signal'].unique().tolist())
                selected_signal = st.selectbox("🎯 Signal Type", signals)
                
                # Confidence range
                confidence_range = st.slider(
                    "📊 Confidence Range",
                    0, 100, (60, 100),
                    help="Filter by confidence score"
                )
                
                # Apply advanced filters button
                if st.button("🔍 Apply Filters", use_container_width=True):
                    apply_advanced_filters(selected_sector, selected_signal, confidence_range)
        
        # Market Stats Section
        st.markdown("---")
        st.markdown("### 📈 Market Stats")
        
        if SessionManager.get('data_loaded'):
            signals_df = SessionManager.get('signals_df', pd.DataFrame())
            
            if not signals_df.empty:
                # Calculate stats
                total_stocks = len(signals_df)
                bullish = len(signals_df[signals_df['ret_1d'] > 0]) if 'ret_1d' in signals_df else 0
                bearish = total_stocks - bullish
                breadth = (bullish / total_stocks * 100) if total_stocks > 0 else 50
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Total Stocks", f"{total_stocks:,}")
                    st.metric("🟢 Bullish", f"{bullish} ({bullish/total_stocks*100:.0f}%)")
                
                with col2:
                    st.metric("🌍 Market Breadth", f"{breadth:.0f}%")
                    st.metric("🔴 Bearish", f"{bearish} ({bearish/total_stocks*100:.0f}%)")
        
        # Help Section
        st.markdown("---")
        with st.expander("❓ Help"):
            st.markdown("""
            **Quick Start:**
            1. Click 'Load Data' to begin
            2. Use Smart Filters for quick analysis
            3. Apply Advanced Filters for specific criteria
            4. Review opportunities in the main panel
            
            **Signals:**
            - 🟢 STRONG_BUY: Highest conviction
            - 🟢 BUY: High conviction
            - 🟡 HOLD: Monitor position
            - 🔴 SELL: Consider exit
            """)

def apply_current_filter():
    """Apply the currently selected smart filter"""
    signals_df = SessionManager.get('signals_df', pd.DataFrame())
    if signals_df.empty:
        return
    
    selected_filter = SessionManager.get('selected_filter')
    filter_func = SMART_FILTERS[selected_filter]['filter']
    
    try:
        filtered_df = filter_func(signals_df)
        SessionManager.set('filtered_df', filtered_df)
    except Exception as e:
        st.error(f"Filter error: {str(e)}")
        SessionManager.set('filtered_df', pd.DataFrame())

def apply_advanced_filters(sector, signal, confidence_range):
    """Apply advanced filters"""
    signals_df = SessionManager.get('signals_df', pd.DataFrame())
    if signals_df.empty:
        return
    
    filtered = signals_df.copy()
    
    # Apply filters
    if sector != 'All':
        filtered = filtered[filtered['sector'] == sector]
    
    if signal != 'All':
        filtered = filtered[filtered['signal'] == signal]
    
    # Confidence filter
    filtered = filtered[
        (filtered['confidence'] >= confidence_range[0]) & 
        (filtered['confidence'] <= confidence_range[1])
    ]
    
    SessionManager.set('filtered_df', filtered)

def render_header():
    """Render professional header"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("# 📊 Market Intelligence Dashboard")
        st.markdown("*Professional trading insights powered by 8-factor analysis*")
    
    with col2:
        if SessionManager.get('data_loaded'):
            if st.button("📥 Export", use_container_width=True):
                export_data()
    
    with col3:
        if st.button("🔄 Refresh View", use_container_width=True):
            st.rerun()

def export_data():
    """Export filtered data"""
    filtered_df = SessionManager.get('filtered_df', pd.DataFrame())
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"mantra_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

def render_stock_card(stock):
    """Render a clean, professional stock card"""
    # Determine signal color
    signal_colors = {
        'STRONG_BUY': '#00c853',
        'BUY': '#43a047',
        'HOLD': '#fb8c00',
        'SELL': '#e53935'
    }
    color = signal_colors.get(stock.get('signal', 'HOLD'), '#757575')
    
    # Create card layout
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"### {stock.get('ticker', 'N/A')} - {stock.get('company_name', 'Unknown')[:40]}")
            st.caption(f"{stock.get('sector', 'Unknown')} | {stock.get('category', 'Unknown')}")
        
        with col2:
            signal = stock.get('signal', 'HOLD')
            st.markdown(f"<h4 style='color: {color}; text-align: center;'>{signal}</h4>", unsafe_allow_html=True)
            confidence = stock.get('confidence', 0)
            st.progress(confidence / 100)
            st.caption(f"{confidence:.0f}% confidence")
        
        with col3:
            price = stock.get('price', 0)
            ret_1d = stock.get('ret_1d', 0)
            st.metric("Price", f"₹{price:,.0f}", f"{ret_1d:+.1f}%")
        
        with col4:
            ret_30d = stock.get('ret_30d', 0)
            st.metric("30D Return", f"{ret_30d:+.1f}%")
        
        # Key metrics row
        st.markdown("---")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            pe = stock.get('pe', 0)
            st.metric("P/E", f"{pe:.1f}" if pe > 0 else "N/A")
        
        with col2:
            eps_growth = stock.get('eps_change_pct', 0)
            st.metric("EPS Growth", f"{eps_growth:+.0f}%")
        
        with col3:
            rvol = stock.get('rvol', 1)
            st.metric("Rel Volume", f"{rvol:.1f}x")
        
        with col4:
            position = stock.get('position_52w', 50)
            st.metric("52W Position", f"{position:.0f}%")
        
        with col5:
            momentum = stock.get('momentum_score', 50)
            st.metric("Momentum", f"{momentum:.0f}")
        
        with col6:
            value = stock.get('value_score', 50)
            st.metric("Value", f"{value:.0f}")
        
        # Show key insights if available
        if 'key_insights' in stock and stock['key_insights']:
            st.info(f"💡 {stock['key_insights']}")
        
        st.markdown("---")

def render_main_content():
    """Render main content area"""
    if not SessionManager.get('data_loaded'):
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h1 style='color: #1a1a1a; font-size: 3rem;'>Welcome to M.A.N.T.R.A.</h1>
            <p style='color: #666; font-size: 1.2rem; margin: 2rem 0;'>
                Your professional market analysis and trading assistant
            </p>
            <div style='display: flex; justify-content: center; gap: 3rem; margin: 3rem 0;'>
                <div style='text-align: center;'>
                    <h2 style='color: #2962ff;'>2,200+</h2>
                    <p>Stocks Analyzed</p>
                </div>
                <div style='text-align: center;'>
                    <h2 style='color: #2962ff;'>8-Factor</h2>
                    <p>Analysis Engine</p>
                </div>
                <div style='text-align: center;'>
                    <h2 style='color: #2962ff;'>Real-time</h2>
                    <p>Market Data</p>
                </div>
            </div>
            <p style='font-size: 1.1rem; color: #444;'>
                👈 Click <strong>"Load Data"</strong> in the sidebar to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Opportunities",
        "📊 Market Overview",
        "🏢 Sector Analysis",
        "📈 Analytics"
    ])
    
    with tab1:
        render_opportunities_tab()
    
    with tab2:
        render_market_overview_tab()
    
    with tab3:
        render_sector_analysis_tab()
    
    with tab4:
        render_analytics_tab()

def render_opportunities_tab():
    """Render opportunities tab"""
    filtered_df = SessionManager.get('filtered_df', pd.DataFrame())
    
    if filtered_df.empty:
        st.warning("No stocks match the current filters. Try adjusting your criteria.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Stocks", len(filtered_df))
    
    with col2:
        strong_buy = len(filtered_df[filtered_df['signal'] == 'STRONG_BUY'])
        st.metric("Strong Buy", strong_buy)
    
    with col3:
        avg_return = filtered_df['ret_30d'].mean() if 'ret_30d' in filtered_df else 0
        st.metric("Avg 30D Return", f"{avg_return:.1f}%")
    
    with col4:
        avg_confidence = filtered_df['confidence'].mean() if 'confidence' in filtered_df else 0
        st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
    
    st.markdown("---")
    
    # Display mode selector
    display_mode = st.radio(
        "Display Mode",
        ["Cards", "Table"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if display_mode == "Cards":
        # Show top 10 as cards
        st.markdown("### 🏆 Top Opportunities")
        for _, stock in filtered_df.head(10).iterrows():
            render_stock_card(stock)
    else:
        # Show as table
        st.markdown("### 📊 Opportunities Table")
        
        # Select columns to display
        display_columns = [
            'ticker', 'company_name', 'signal', 'confidence',
            'price', 'ret_1d', 'ret_30d', 'pe', 'eps_change_pct',
            'rvol', 'sector', 'momentum_score', 'value_score'
        ]
        
        # Filter available columns
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # Display dataframe
        st.dataframe(
            filtered_df[available_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "company_name": st.column_config.TextColumn("Company"),
                "signal": st.column_config.TextColumn("Signal", width="small"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100),
                "price": st.column_config.NumberColumn("Price", format="₹%.0f"),
                "ret_1d": st.column_config.NumberColumn("1D%", format="%.1f%%"),
                "ret_30d": st.column_config.NumberColumn("30D%", format="%.1f%%"),
                "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                "eps_change_pct": st.column_config.NumberColumn("EPS Δ%", format="%.0f%%"),
                "rvol": st.column_config.NumberColumn("Vol", format="%.1fx")
            }
        )

def render_market_overview_tab():
    """Render market overview tab"""
    signals_df = SessionManager.get('signals_df', pd.DataFrame())
    
    if signals_df.empty:
        st.warning("No data available")
        return
    
    # Market metrics
    st.markdown("### 📊 Market Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = len(signals_df)
        st.metric("Total Stocks", f"{total_stocks:,}")
    
    with col2:
        if 'ret_1d' in signals_df:
            avg_return = signals_df['ret_1d'].mean()
            st.metric("Avg Daily Return", f"{avg_return:.2f}%")
    
    with col3:
        buy_signals = len(signals_df[signals_df['signal'].isin(['STRONG_BUY', 'BUY'])])
        st.metric("Buy Signals", buy_signals)
    
    with col4:
        if 'rvol' in signals_df:
            high_volume = len(signals_df[signals_df['rvol'] > 2])
            st.metric("High Volume", high_volume)
    
    # Signal distribution
    st.markdown("### 📈 Signal Distribution")
    
    signal_counts = signals_df['signal'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=signal_counts.index,
            y=signal_counts.values,
            marker_color=['#00c853', '#43a047', '#fb8c00', '#e53935', '#757575'][:len(signal_counts)]
        )
    ])
    
    fig.update_layout(
        title="Distribution of Trading Signals",
        xaxis_title="Signal Type",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market breadth chart
    if 'ret_1d' in signals_df:
        st.markdown("### 🌍 Market Breadth")
        
        returns = signals_df['ret_1d']
        positive = len(returns[returns > 0])
        negative = len(returns[returns < 0])
        neutral = len(returns[returns == 0])
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[positive, negative, neutral],
                marker_colors=['#00c853', '#e53935', '#757575'],
                hole=0.4
            )
        ])
        
        fig.update_layout(
            title="Market Breadth (1-Day Returns)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_sector_analysis_tab():
    """Render sector analysis tab"""
    signals_df = SessionManager.get('signals_df', pd.DataFrame())
    
    if signals_df.empty or 'sector' not in signals_df.columns:
        st.warning("No sector data available")
        return
    
    # Sector performance
    st.markdown("### 🏢 Sector Performance")
    
    sector_stats = signals_df.groupby('sector').agg({
        'ticker': 'count',
        'ret_30d': 'mean',
        'signal': lambda x: (x.isin(['STRONG_BUY', 'BUY'])).sum()
    }).round(2)
    
    sector_stats.columns = ['Count', 'Avg 30D Return', 'Buy Signals']
    sector_stats = sector_stats.sort_values('Avg 30D Return', ascending=False)
    
    # Sector performance chart
    fig = go.Figure(data=[
        go.Bar(
            x=sector_stats.index,
            y=sector_stats['Avg 30D Return'],
            marker_color=np.where(sector_stats['Avg 30D Return'] >= 0, '#00c853', '#e53935'),
            text=sector_stats['Avg 30D Return'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Average 30-Day Returns by Sector",
        xaxis_title="Sector",
        yaxis_title="Average Return (%)",
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector table
    st.markdown("### 📊 Sector Statistics")
    st.dataframe(
        sector_stats,
        use_container_width=True,
        column_config={
            "Count": st.column_config.NumberColumn("Stocks"),
            "Avg 30D Return": st.column_config.NumberColumn("Avg Return %", format="%.2f%%"),
            "Buy Signals": st.column_config.NumberColumn("Buy Signals")
        }
    )

def render_analytics_tab():
    """Render analytics tab"""
    signals_df = SessionManager.get('signals_df', pd.DataFrame())
    
    if signals_df.empty:
        st.warning("No data available for analytics")
        return
    
    # Factor correlations
    st.markdown("### 🔍 Factor Analysis")
    
    factor_cols = ['momentum_score', 'value_score', 'growth_score', 'volume_score', 'technical_score']
    available_factors = [col for col in factor_cols if col in signals_df.columns]
    
    if len(available_factors) >= 2:
        corr_matrix = signals_df[available_factors].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Factor Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Return distribution
    if 'ret_30d' in signals_df:
        st.markdown("### 📊 Return Distribution")
        
        fig = go.Figure(data=[
            go.Histogram(
                x=signals_df['ret_30d'],
                nbinsx=50,
                marker_color='#2962ff',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="30-Day Return Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    try:
        # Render sidebar
        render_sidebar()
        
        # Render header
        render_header()
        
        # Render main content
        render_main_content()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again.")
        if st.button("🔄 Reset Application"):
            for key in st.session_state:
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

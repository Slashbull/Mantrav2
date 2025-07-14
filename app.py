"""
app.py - M.A.N.T.R.A. ULTIMATE FINAL VERSION
============================================
The definitive implementation - Clean, Smart, Powerful
No further changes after this - Everything optimized for perfection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

from config import CONFIG
from data_loader import DataLoader
from ultimate_signal_engine import UltimateSignalEngine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🔱 M.A.N.T.R.A. | Market Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - CLEAN & PROFESSIONAL
# ============================================================================
st.markdown("""
<style>
    /* Clean, modern design system */
    :root {
        --primary: #1e293b;
        --secondary: #334155;
        --accent: #3b82f6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --background: #ffffff;
        --surface: #f8fafc;
        --text: #1e293b;
        --text-secondary: #64748b;
        --border: #e2e8f0;
        --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    /* Remove default padding for clean layout */
    .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: none;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    
    /* Metrics with clean design */
    div[data-testid="metric-container"] {
        background-color: var(--background);
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: var(--shadow);
        transition: all 0.2s;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: var(--surface);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background-color: transparent;
        border-radius: 0.375rem;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--background);
        color: var(--accent);
        box-shadow: var(--shadow);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        font-weight: 500;
    }
    
    /* DataFrames */
    .dataframe {
        font-size: 0.875rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        color: var(--primary);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
        border: 1px solid;
        font-size: 0.875rem;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--accent);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'signal_engine' not in st.session_state:
    st.session_state.signal_engine = UltimateSignalEngine()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'selected_filter' not in st.session_state:
    st.session_state.selected_filter = 'high_conviction'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ============================================================================
# SMART FILTERS CONFIGURATION
# ============================================================================
SMART_FILTERS = {
    "high_conviction": {
        "name": "🎯 High Conviction",
        "icon": "🎯",
        "description": "Ultra-high confidence opportunities",
        "filter": lambda df: df[df['composite_score'] >= 85].head(20),
        "color": "#10b981"
    },
    "momentum_surge": {
        "name": "🚀 Momentum Surge",
        "icon": "🚀",
        "description": "Accelerating momentum with volume",
        "filter": lambda df: df[(df['momentum_score'] > 80) & (df['volume_score'] > 70) & (df['ret_7d'] > 5)],
        "color": "#3b82f6"
    },
    "value_gems": {
        "name": "💎 Value Gems",
        "icon": "💎",
        "description": "Undervalued with growth catalysts",
        "filter": lambda df: df[(df['pe'] < 20) & (df['pe'] > 0) & (df['eps_change_pct'] > 30) & (df['value_score'] > 75)],
        "color": "#8b5cf6"
    },
    "breakout_alerts": {
        "name": "⚡ Breakout Alerts",
        "icon": "⚡",
        "description": "Near 52W high with strong technicals",
        "filter": lambda df: df[(df['position_52w'] > 80) & (df['rvol'] > 2) & (df['technical_score'] > 75)],
        "color": "#f59e0b"
    },
    "smart_money": {
        "name": "🏦 Smart Money",
        "icon": "🏦",
        "description": "Institutional accumulation patterns",
        "filter": lambda df: df[(df['volume_30d'] > df['volume_3m'] * 1.2) & (df['ret_30d'].between(-5, 15)) & (df['pe'].between(10, 30))],
        "color": "#14b8a6"
    },
    "earnings_stars": {
        "name": "⭐ Earnings Stars",
        "icon": "⭐",
        "description": "Exceptional earnings growth",
        "filter": lambda df: df[(df['eps_change_pct'] > 50) & (df['growth_score'] > 80) & (df['eps_current'] > 0)],
        "color": "#ec4899"
    },
    "turnaround_plays": {
        "name": "🔄 Turnaround Plays",
        "icon": "🔄",
        "description": "Reversing from oversold conditions",
        "filter": lambda df: df[(df['position_52w'] < 30) & (df['ret_7d'] > 5) & (df['momentum_quality'] == 'REVERSAL') & (df['pe'].between(5, 25))],
        "color": "#06b6d4"
    }
}

# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================
def render_sidebar():
    """Render sidebar with controls and filters"""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='margin: 0; font-size: 2rem;'>🔱 M.A.N.T.R.A.</h1>
            <p style='margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;'>
                Market Analysis Neural Trading Research Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Controls
        st.markdown("### 📊 Data Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Data", type="primary", use_container_width=True):
                with st.spinner("Loading..."):
                    success, message = st.session_state.data_loader.load_and_process(use_cache=True)
                    if success:
                        stocks_df = st.session_state.data_loader.get_stocks_data()
                        sector_df = st.session_state.data_loader.get_sector_data()
                        st.session_state.signals_df = st.session_state.signal_engine.analyze(stocks_df, sector_df)
                        st.session_state.last_update = datetime.now()
                        st.session_state.data_loaded = True
                        st.success("✅ Data loaded!")
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.data_loader.clear_cache()
                st.rerun()
        
        # Smart Filters
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 🎯 Smart Filters")
            
            # Display filter buttons
            for key, filter_config in SMART_FILTERS.items():
                if st.button(
                    f"{filter_config['icon']} {filter_config['name']}", 
                    key=f"filter_{key}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_filter == key else "secondary"
                ):
                    st.session_state.selected_filter = key
                    st.rerun()
            
            # Current filter info
            current_filter = SMART_FILTERS[st.session_state.selected_filter]
            st.info(f"**Active:** {current_filter['description']}")
            
            # Advanced Filters
            st.markdown("---")
            st.markdown("### ⚙️ Advanced Filters")
            
            # Sector filter
            all_sectors = ['All'] + sorted(st.session_state.signals_df['sector'].dropna().unique().tolist())
            selected_sector = st.selectbox("Sector", all_sectors, key="sector_filter")
            
            # Signal filter
            all_signals = ['All'] + sorted(st.session_state.signals_df['signal'].unique().tolist())
            selected_signal = st.selectbox("Signal Type", all_signals, key="signal_filter")
            
            # Confidence slider
            min_confidence = st.slider("Min Confidence", 0, 100, 60, key="confidence_filter")
            
            # PE Range
            pe_range = st.slider("P/E Range", 0, 100, (0, 50), key="pe_filter")
            
            # Market Cap
            market_cap_options = ['All', 'Large Cap', 'Mid Cap', 'Small Cap']
            selected_market_cap = st.selectbox("Market Cap", market_cap_options, key="mcap_filter")
        
        # Market Stats
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📈 Market Stats")
            
            signals_df = st.session_state.signals_df
            
            # Calculate stats
            total_stocks = len(signals_df)
            buy_signals = len(signals_df[signals_df['signal'].isin(['STRONG_BUY', 'BUY'])])
            avg_confidence = signals_df['confidence'].mean()
            
            # Display stats
            st.metric("Total Stocks", f"{total_stocks:,}")
            st.metric("Buy Signals", buy_signals, delta=f"{(buy_signals/total_stocks*100):.1f}%")
            st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
            
            # Market breadth gauge
            if 'ret_1d' in signals_df.columns:
                positive = (signals_df['ret_1d'] > 0).sum()
                breadth = (positive / total_stocks * 100) if total_stocks > 0 else 50
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=breadth,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Market Breadth"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#10b981" if breadth > 60 else "#ef4444" if breadth < 40 else "#f59e0b"},
                        'steps': [
                            {'range': [0, 40], 'color': "#fee2e2"},
                            {'range': [40, 60], 'color': "#fef3c7"},
                            {'range': [60, 100], 'color': "#d1fae5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        # About
        st.markdown("---")
        with st.expander("ℹ️ About M.A.N.T.R.A."):
            st.markdown("""
            **Version:** 3.0 ULTIMATE FINAL
            
            **Features:**
            - 🧠 8-Factor Analysis Engine
            - 🎯 Pattern Recognition
            - 📊 Volume Intelligence
            - 🚀 Smart Entry Zones
            - 📈 Sector Rotation
            
            **Data Source:** Live Google Sheets
            **Update Frequency:** Real-time
            """)

# ============================================================================
# MAIN CONTENT COMPONENTS
# ============================================================================
def render_header():
    """Render clean header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("# 📊 Market Intelligence Dashboard")
        if st.session_state.last_update:
            time_diff = (datetime.now() - st.session_state.last_update).total_seconds() / 60
            st.caption(f"Last updated {time_diff:.0f} minutes ago")
    
    with col2:
        if st.session_state.data_loaded:
            current_filter = SMART_FILTERS[st.session_state.selected_filter]
            st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem; background-color: {current_filter['color']}20; 
                        border: 1px solid {current_filter['color']}; border-radius: 0.5rem;'>
                <span style='color: {current_filter['color']}; font-weight: 600;'>
                    {current_filter['icon']} {current_filter['name']}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.data_loaded:
            st.download_button(
                "📥 Export Data",
                st.session_state.signals_df.to_csv(index=False),
                f"mantra_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

def render_market_overview(signals_df):
    """Render market overview section"""
    st.markdown("## 🌍 Market Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_analyzed = len(signals_df)
        st.metric("Stocks Analyzed", f"{total_analyzed:,}")
    
    with col2:
        strong_buy = len(signals_df[signals_df['signal'] == 'STRONG_BUY'])
        st.metric("Strong Buy", strong_buy, delta="Top picks")
    
    with col3:
        # Average return
        avg_return = signals_df['ret_30d'].mean()
        st.metric("Avg 30D Return", f"{avg_return:.1f}%", delta_color="normal")
    
    with col4:
        # Hot sector
        if 'sector_performance' in signals_df.columns:
            hot_sector = signals_df.groupby('sector')['sector_performance'].first().idxmax()
            hot_sector_perf = signals_df.groupby('sector')['sector_performance'].first().max()
            st.metric("Hot Sector", hot_sector[:15], delta=f"+{hot_sector_perf:.1f}%")
    
    with col5:
        # Volume surge count
        volume_surge = len(signals_df[signals_df['rvol'] > 3])
        st.metric("Volume Surges", volume_surge, delta="High activity")

def render_opportunity_card(stock):
    """Render clean opportunity card"""
    # Determine signal color
    signal_colors = CONFIG.SIGNAL_COLORS
    signal_color = signal_colors.get(stock['signal'], '#868e96')
    
    # Create card
    with st.container():
        # Card header
        col1, col2, col3 = st.columns([3, 1.5, 1.5])
        
        with col1:
            st.markdown(f"### {stock['ticker']} - {stock.get('company_name', 'N/A')[:40]}")
            st.caption(f"{stock.get('sector', 'Unknown')} | {stock.get('category', 'Unknown')}")
        
        with col2:
            # Signal badge
            st.markdown(f"""
            <div style='text-align: center; background-color: {signal_color}20; 
                        border: 2px solid {signal_color}; border-radius: 0.5rem; 
                        padding: 0.5rem; margin-top: 0.5rem;'>
                <div style='color: {signal_color}; font-weight: 700; font-size: 1rem;'>
                    {stock['signal']}
                </div>
                <div style='color: {signal_color}; font-size: 0.75rem;'>
                    {stock.get('confidence', 0):.0f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            price = stock.get('price', 0)
            ret_1d = stock.get('ret_1d', 0)
            st.metric("Price", f"₹{price:,.0f}", delta=f"{ret_1d:+.1f}%", delta_color="normal")
        
        # Insights section
        if 'smart_insights' in stock and stock['smart_insights']:
            insights = stock['smart_insights']
            for insight_type, insight_data in insights.items():
                if insight_data['detected']:
                    st.info(f"{insight_data['icon']} **{insight_type}:** {insight_data['message']}")
        
        # Metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("30D Return", f"{stock.get('ret_30d', 0):+.1f}%", delta_color="normal")
            st.metric("P/E Ratio", f"{stock.get('pe', 0):.1f}" if stock.get('pe', 0) > 0 else "N/A")
        
        with col2:
            st.metric("EPS Growth", f"{stock.get('eps_change_pct', 0):+.1f}%", delta_color="normal")
            st.metric("Volume", f"{stock.get('rvol', 1):.1f}x avg")
        
        with col3:
            st.metric("52W Position", f"{stock.get('position_52w', 50):.0f}%")
            st.metric("Momentum", f"{stock.get('momentum_score', 50):.0f}/100")
        
        with col4:
            # Entry zone
            if 'entry_zone' in stock:
                zone = stock['entry_zone']
                zone_colors = {
                    'HOT_ZONE': '#ef4444',
                    'ACCUMULATION_ZONE': '#10b981',
                    'VALUE_ZONE': '#8b5cf6',
                    'WATCH_ZONE': '#f59e0b'
                }
                zone_color = zone_colors.get(zone['type'], '#64748b')
                st.markdown(f"""
                <div style='text-align: center; margin-top: 1.5rem;'>
                    <div style='color: {zone_color}; font-weight: 600;'>
                        {zone['label']}
                    </div>
                    <div style='font-size: 0.75rem; color: #64748b;'>
                        {zone['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Factor analysis
        with st.expander("📊 Factor Analysis"):
            factors = ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']
            scores = [stock.get(f'{f}_score', 50) for f in factors]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores,
                y=[f.capitalize() for f in factors],
                orientation='h',
                marker_color=['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899'],
                text=[f"{s:.0f}" for s in scores],
                textposition='inside'
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 100], showgrid=False),
                yaxis=dict(showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

def render_sector_analysis(signals_df):
    """Render sector analysis"""
    st.markdown("## 🏢 Sector Analysis")
    
    if 'sector' not in signals_df.columns:
        st.warning("Sector data not available")
        return
    
    # Sector performance
    sector_stats = signals_df.groupby('sector').agg({
        'ticker': 'count',
        'ret_30d': 'mean',
        'signal': lambda x: (x.isin(['STRONG_BUY', 'BUY'])).sum(),
        'confidence': 'mean'
    }).round(1)
    
    sector_stats.columns = ['Count', 'Avg Return', 'Buy Signals', 'Avg Confidence']
    sector_stats = sector_stats.sort_values('Avg Return', ascending=False)
    
    # Create visualizations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sector returns heatmap
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sector_stats['Avg Return'],
            y=sector_stats.index,
            orientation='h',
            marker=dict(
                color=sector_stats['Avg Return'],
                colorscale='RdYlGn',
                cmin=-20,
                cmax=20,
                showscale=True,
                colorbar=dict(title="Return %")
            ),
            text=sector_stats['Avg Return'].apply(lambda x: f"{x:.1f}%"),
            textposition='inside'
        ))
        fig.update_layout(
            title="Average 30-Day Returns by Sector",
            height=400,
            xaxis_title="Return %",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector opportunities
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sector_stats['Avg Confidence'],
            y=sector_stats['Buy Signals'] / sector_stats['Count'] * 100,
            mode='markers+text',
            marker=dict(
                size=sector_stats['Count'] * 2,
                color=sector_stats['Avg Return'],
                colorscale='RdYlGn',
                showscale=False,
                line=dict(width=1, color='white')
            ),
            text=sector_stats.index,
            textposition="top center"
        ))
        fig.update_layout(
            title="Sector Opportunity Map",
            height=400,
            xaxis_title="Average Confidence",
            yaxis_title="% Buy Signals",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top stocks by sector
    st.markdown("### 🌟 Sector Leaders")
    
    top_sectors = sector_stats.head(3).index
    cols = st.columns(3)
    
    for idx, sector in enumerate(top_sectors):
        with cols[idx]:
            st.markdown(f"**{sector}**")
            sector_stocks = signals_df[
                (signals_df['sector'] == sector) & 
                (signals_df['signal'].isin(['STRONG_BUY', 'BUY']))
            ].sort_values('confidence', ascending=False).head(3)
            
            for _, stock in sector_stocks.iterrows():
                st.caption(f"• {stock['ticker']} ({stock['confidence']:.0f}%)")

def render_pattern_insights(signals_df):
    """Render pattern insights"""
    st.markdown("## 🎯 Pattern Recognition")
    
    # Count patterns
    pattern_summary = {}
    stocks_with_patterns = 0
    
    for _, stock in signals_df.iterrows():
        if 'patterns' in stock and stock['patterns']:
            stocks_with_patterns += 1
            for pattern in stock['patterns']:
                if pattern not in pattern_summary:
                    pattern_summary[pattern] = 0
                pattern_summary[pattern] += 1
    
    if not pattern_summary:
        st.info("No significant patterns detected in current filter")
        return
    
    # Pattern statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Patterns", len(pattern_summary))
    with col2:
        st.metric("Stocks with Patterns", stocks_with_patterns)
    with col3:
        st.metric("Avg Patterns/Stock", f"{sum(pattern_summary.values())/stocks_with_patterns:.1f}")
    
    # Pattern distribution
    pattern_df = pd.DataFrame(list(pattern_summary.items()), columns=['Pattern', 'Count'])
    pattern_df = pattern_df.sort_values('Count', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pattern_df['Count'],
        y=pattern_df['Pattern'],
        orientation='h',
        marker_color='#3b82f6',
        text=pattern_df['Count'],
        textposition='inside'
    ))
    fig.update_layout(
        title="Pattern Distribution",
        height=300,
        xaxis_title="Number of Stocks",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def render_analytics_dashboard(signals_df):
    """Render analytics dashboard"""
    st.markdown("## 📈 Analytics")
    
    # Create tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Returns", "🔍 Correlations"])
    
    with tab1:
        # Signal distribution
        col1, col2 = st.columns(2)
        
        with col1:
            signal_counts = signals_df['signal'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                hole=0.4,
                marker_colors=[CONFIG.SIGNAL_COLORS.get(s, '#868e96') for s in signal_counts.index]
            )])
            fig.update_layout(title="Signal Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=signals_df['confidence'],
                nbinsx=20,
                marker_color='#3b82f6',
                name='Confidence'
            ))
            fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Returns analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Return distribution
            returns_df = signals_df[['ret_1d', 'ret_7d', 'ret_30d']].melt()
            fig = go.Figure()
            for period in ['ret_1d', 'ret_7d', 'ret_30d']:
                fig.add_trace(go.Box(
                    y=signals_df[period],
                    name=period.replace('ret_', '').replace('d', 'D'),
                    boxpoints='outliers'
                ))
            fig.update_layout(
                title="Return Distribution by Period",
                yaxis_title="Return %",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return vs Confidence scatter
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=signals_df['confidence'],
                y=signals_df['ret_30d'],
                mode='markers',
                marker=dict(
                    color=signals_df['confidence'],
                    colorscale='Viridis',
                    size=5,
                    showscale=True,
                    colorbar=dict(title="Confidence")
                ),
                text=signals_df['ticker'],
                hovertemplate='%{text}<br>Confidence: %{x}<br>30D Return: %{y}%'
            ))
            fig.update_layout(
                title="30D Return vs Confidence",
                xaxis_title="Confidence Score",
                yaxis_title="30D Return %",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation matrix
        numeric_cols = ['momentum_score', 'value_score', 'growth_score', 'volume_score', 
                       'technical_score', 'ret_30d', 'pe', 'eps_change_pct']
        
        available_cols = [col for col in numeric_cols if col in signals_df.columns]
        if len(available_cols) > 2:
            corr_matrix = signals_df[available_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title="Factor Correlation Matrix",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def apply_filters(df, sector, signal, confidence, pe_range, market_cap):
    """Apply advanced filters to dataframe"""
    filtered = df.copy()
    
    # Sector filter
    if sector != 'All':
        filtered = filtered[filtered['sector'] == sector]
    
    # Signal filter
    if signal != 'All':
        filtered = filtered[filtered['signal'] == signal]
    
    # Confidence filter
    filtered = filtered[filtered['confidence'] >= confidence]
    
    # PE filter
    if 'pe' in filtered.columns:
        filtered = filtered[(filtered['pe'] >= pe_range[0]) & (filtered['pe'] <= pe_range[1])]
    
    # Market cap filter
    if market_cap != 'All' and 'market_cap' in filtered.columns:
        if market_cap == 'Large Cap':
            filtered = filtered[filtered['market_cap'] > 20000]
        elif market_cap == 'Mid Cap':
            filtered = filtered[(filtered['market_cap'] >= 5000) & (filtered['market_cap'] <= 20000)]
        elif market_cap == 'Small Cap':
            filtered = filtered[filtered['market_cap'] < 5000]
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 5rem 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🔱 Welcome to M.A.N.T.R.A.</h1>
            <p style='font-size: 1.25rem; color: #64748b; margin-bottom: 2rem;'>
                Your intelligent companion for market analysis and trading decisions
            </p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                        gap: 2rem; max-width: 800px; margin: 0 auto;'>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem;'>🧠</div>
                    <h3>8-Factor Analysis</h3>
                    <p style='color: #64748b;'>Advanced multi-factor scoring system</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem;'>🎯</div>
                    <h3>Smart Filters</h3>
                    <p style='color: #64748b;'>Pre-configured opportunity scanners</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem;'>📊</div>
                    <h3>Pattern Detection</h3>
                    <p style='color: #64748b;'>Identify high-probability setups</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem;'>🚀</div>
                    <h3>Real-time Analysis</h3>
                    <p style='color: #64748b;'>Live data from Google Sheets</p>
                </div>
            </div>
            <p style='margin-top: 3rem; font-size: 1.125rem;'>
                👈 Click <strong>"Load Data"</strong> in the sidebar to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get data
    signals_df = st.session_state.signals_df
    
    # Apply smart filter
    current_filter = SMART_FILTERS[st.session_state.selected_filter]
    filtered_df = current_filter['filter'](signals_df)
    
    # Apply advanced filters
    filtered_df = apply_filters(
        filtered_df,
        st.session_state.get('sector_filter', 'All'),
        st.session_state.get('signal_filter', 'All'),
        st.session_state.get('confidence_filter', 60),
        st.session_state.get('pe_filter', (0, 50)),
        st.session_state.get('mcap_filter', 'All')
    )
    
    # Render header
    render_header()
    
    # Render content based on tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Opportunities",
        "🏢 Sectors",
        "🔍 Patterns",
        "📊 All Stocks",
        "📈 Analytics"
    ])
    
    with tab1:
        render_market_overview(signals_df)
        
        st.markdown("---")
        st.markdown(f"## {current_filter['icon']} {current_filter['name']}")
        st.caption(f"Showing {len(filtered_df)} opportunities")
        
        if filtered_df.empty:
            st.warning("No stocks match the current filters. Try adjusting your criteria.")
        else:
            # Display top opportunities
            for _, stock in filtered_df.head(10).iterrows():
                render_opportunity_card(stock)
    
    with tab2:
        render_sector_analysis(signals_df)
    
    with tab3:
        render_pattern_insights(filtered_df)
    
    with tab4:
        st.markdown("## 📊 All Stocks")
        
        # Additional table filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by", ['Confidence', '30D Return', 'Volume', 'P/E'])
        with col2:
            show_only = st.multiselect("Show only", ['STRONG_BUY', 'BUY', 'ACCUMULATE'])
        with col3:
            search = st.text_input("Search ticker/company", "")
        
        # Apply table filters
        display_df = filtered_df.copy()
        if show_only:
            display_df = display_df[display_df['signal'].isin(show_only)]
        if search:
            mask = (display_df['ticker'].str.contains(search.upper(), na=False) | 
                   display_df['company_name'].str.contains(search, case=False, na=False))
            display_df = display_df[mask]
        
        # Sort
        sort_map = {
            'Confidence': 'confidence',
            '30D Return': 'ret_30d',
            'Volume': 'rvol',
            'P/E': 'pe'
        }
        display_df = display_df.sort_values(
            sort_map[sort_by], 
            ascending=(sort_by == 'P/E')
        )
        
        # Display count
        st.info(f"Displaying {len(display_df)} stocks")
        
        # Show dataframe
        if not display_df.empty:
            # Select display columns
            display_cols = [
                'ticker', 'company_name', 'signal', 'confidence',
                'price', 'ret_1d', 'ret_30d', 'pe', 'eps_change_pct',
                'rvol', 'sector', 'momentum_score', 'value_score'
            ]
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            # Configure column display
            column_config = {
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "company_name": st.column_config.TextColumn("Company", width="medium"),
                "signal": st.column_config.TextColumn("Signal", width="small"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d"),
                "price": st.column_config.NumberColumn("Price", format="₹%.0f"),
                "ret_1d": st.column_config.NumberColumn("1D%", format="%.1f%%"),
                "ret_30d": st.column_config.NumberColumn("30D%", format="%.1f%%"),
                "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                "eps_change_pct": st.column_config.NumberColumn("EPS Δ%", format="%.1f%%"),
                "rvol": st.column_config.NumberColumn("RVol", format="%.1fx"),
                "sector": st.column_config.TextColumn("Sector", width="small"),
                "momentum_score": st.column_config.NumberColumn("Mom", format="%.0f"),
                "value_score": st.column_config.NumberColumn("Val", format="%.0f")
            }
            
            st.dataframe(
                display_df[display_cols].head(100),
                column_config=column_config,
                use_container_width=True,
                height=600,
                hide_index=True
            )
    
    with tab5:
        render_analytics_dashboard(signals_df)

if __name__ == "__main__":
    main()

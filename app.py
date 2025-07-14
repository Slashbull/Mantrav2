"""
app.py - M.A.N.T.R.A. Ultimate Version
======================================
The all-time best UI/UX with enhanced intelligence
Simple, beautiful, and incredibly powerful
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
from enhanced_signal_engine import EnhancedSignalEngine

# Page configuration
st.set_page_config(
    page_title=f"{CONFIG.APP_ICON} {CONFIG.APP_NAME} Ultimate",
    page_icon=CONFIG.APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Beautiful metrics */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Hover effects */
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'signal_engine' not in st.session_state:
    st.session_state.signal_engine = EnhancedSignalEngine()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'selected_filter' not in st.session_state:
    st.session_state.selected_filter = 'all'

# Smart Filters Configuration
SMART_FILTERS = {
    "all": {
        "name": "🎯 All Opportunities",
        "filter": lambda df: df[df['signal'].isin(['STRONG_BUY', 'BUY', 'ACCUMULATE'])],
        "description": "All stocks with positive signals"
    },
    "volume_explosion": {
        "name": "🔥 Volume Explosions",
        "filter": lambda df: df[(df['rvol'] > 3) & (df['signal'].isin(['STRONG_BUY', 'BUY']))],
        "description": "Unusual volume with strong signals"
    },
    "hidden_value": {
        "name": "💎 Hidden Value",
        "filter": lambda df: df[(df['pe'] < 15) & (df['eps_change_pct'] > 30) & (df['from_low_pct'] < 40)],
        "description": "Undervalued with growing earnings"
    },
    "breakout_ready": {
        "name": "🚀 Breakout Ready",
        "filter": lambda df: df[(df['from_high_pct'] > -10) & (df['ret_7d'] > 5) & (df['rvol'] > 2)],
        "description": "Near 52W high with momentum"
    },
    "sector_leaders": {
        "name": "👑 Sector Leaders",
        "filter": lambda df: df[df.apply(lambda x: x['ret_30d'] > x.get('sector_performance', 0) * 1.5, axis=1)],
        "description": "Outperforming their sectors"
    },
    "smart_money": {
        "name": "🏦 Smart Money",
        "filter": lambda df: df[(df['volume_30d'] > df['volume_3m'] * 1.2) & (df['ret_30d'] > 5)],
        "description": "Institutional accumulation detected"
    }
}

def render_header():
    """Render beautiful header"""
    # Create gradient header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
        <h1 style='margin: 0; font-size: 3rem;'>🔱 M.A.N.T.R.A. Ultimate</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.95;'>
            Market Analysis Neural Trading Research Assistant
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>
            Enhanced Intelligence • Pattern Recognition • Smart Signals
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

def render_market_pulse(signals_df):
    """Render market pulse dashboard"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_stocks = len(signals_df)
        st.metric(
            "📊 Stocks Analyzed", 
            f"{total_stocks:,}",
            delta="Live data"
        )
    
    with col2:
        strong_signals = len(signals_df[signals_df['signal'].isin(['STRONG_BUY', 'BUY'])])
        st.metric(
            "🎯 Strong Signals", 
            strong_signals,
            delta=f"{(strong_signals/total_stocks*100):.1f}%" if total_stocks > 0 else "0%"
        )
    
    with col3:
        if 'ret_1d' in signals_df.columns:
            positive = (signals_df['ret_1d'] > 0).sum()
            breadth = (positive / len(signals_df) * 100) if len(signals_df) > 0 else 0
            market_status = "🟢 Bullish" if breadth > 60 else "🔴 Bearish" if breadth < 40 else "🟡 Neutral"
            st.metric("Market Breadth", f"{breadth:.0f}%", delta=market_status)
        else:
            st.metric("Market Breadth", "N/A")
    
    with col4:
        # Top performing sector
        if 'sector_performance' in signals_df.columns:
            top_sector_data = signals_df.groupby('sector')['sector_performance'].first().sort_values(ascending=False)
            if not top_sector_data.empty:
                top_sector = top_sector_data.index[0]
                top_sector_perf = top_sector_data.iloc[0]
                st.metric("🔥 Hot Sector", top_sector[:12], delta=f"+{top_sector_perf:.1f}%")
            else:
                st.metric("🔥 Hot Sector", "N/A")
        else:
            st.metric("🔥 Hot Sector", "N/A")
    
    with col5:
        if st.session_state.last_update:
            time_diff = (datetime.now() - st.session_state.last_update).total_seconds() / 60
            st.metric("Last Update", f"{time_diff:.0f}m ago", delta="Auto-refresh")
        else:
            st.metric("Last Update", "Never")

def render_smart_filters():
    """Render smart filter buttons"""
    st.markdown("### 🎯 Smart Filters")
    
    # Create columns for filter buttons
    cols = st.columns(len(SMART_FILTERS))
    
    for idx, (key, filter_config) in enumerate(SMART_FILTERS.items()):
        with cols[idx]:
            if st.button(
                filter_config['name'], 
                key=f"filter_{key}",
                use_container_width=True,
                type="primary" if st.session_state.selected_filter == key else "secondary"
            ):
                st.session_state.selected_filter = key
                st.rerun()
    
    # Show filter description
    current_filter = SMART_FILTERS[st.session_state.selected_filter]
    st.caption(f"**Active Filter:** {current_filter['description']}")

def render_signal_strength_visual(stock):
    """Render visual signal strength indicator"""
    factors = {
        'Momentum': stock.get('momentum_score', 0),
        'Value': stock.get('value_score', 0),
        'Growth': stock.get('growth_score', 0),
        'Volume': stock.get('volume_score', 0),
        'Technical': stock.get('technical_score', 0),
        'Sector': stock.get('sector_score', 0)
    }
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD']
    
    for idx, (factor, score) in enumerate(factors.items()):
        fig.add_trace(go.Bar(
            x=[score],
            y=[factor],
            orientation='h',
            name=factor,
            marker_color=colors[idx],
            text=f"{score:.0f}",
            textposition='inside',
            showlegend=False
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay'
    )
    
    return fig

def render_enhanced_stock_card(stock):
    """Render beautiful enhanced stock card"""
    
    # Determine card color based on signal
    signal_colors = {
        'STRONG_BUY': '#28a745',
        'BUY': '#40c057',
        'ACCUMULATE': '#74c0fc',
        'WATCH': '#ffd43b',
        'NEUTRAL': '#868e96'
    }
    
    signal_color = signal_colors.get(stock['signal'], '#868e96')
    
    # Create container with colored border
    with st.container():
        # Custom styling for this card
        st.markdown(f"""
        <div style='border: 3px solid {signal_color}; border-radius: 15px; 
                    padding: 20px; margin: 10px 0; background: white;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        
        # Header row
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"## {stock.get('ticker', 'N/A')}")
            st.markdown(f"**{stock.get('company_name', 'Unknown')}**")
            st.caption(f"{stock.get('sector', 'Unknown')} | {stock.get('category', 'Unknown')}")
        
        with col2:
            # Signal badge with custom styling
            signal_emoji = {
                'STRONG_BUY': '🚀',
                'BUY': '📈',
                'ACCUMULATE': '📊',
                'WATCH': '👀',
                'NEUTRAL': '➖'
            }
            st.markdown(f"""
            <div style='text-align: center; background: {signal_color}; color: white; 
                        padding: 10px; border-radius: 50px; font-weight: bold; font-size: 16px;'>
                {signal_emoji.get(stock['signal'], '❓')} {stock['signal']}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.metric("💰 Price", f"₹{stock.get('price', 0):,.0f}")
            confidence = stock.get('confidence', 0)
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence:.0f}%")
        
        # Smart insights
        if 'smart_explanation' in stock and stock['smart_explanation']:
            st.info(f"💡 **Insight:** {stock['smart_explanation']}")
        
        # Pattern alerts
        if 'patterns' in stock and stock['patterns']:
            with st.expander("🎯 Detected Patterns", expanded=True):
                for pattern in stock['patterns'][:3]:  # Top 3 patterns
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{pattern.pattern_name}**")
                        st.caption(f"{pattern.description}")
                    with col2:
                        st.metric("Success Rate", pattern.success_rate)
                        st.caption(f"Action: {pattern.action}")
        
        # Key metrics in a beautiful grid
        st.markdown("---")
        st.markdown("### 📊 Key Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            ret_1d = stock.get('ret_1d', 0)
            st.metric("1D Return", f"{ret_1d:+.1f}%", delta_color="normal")
            
            ret_30d = stock.get('ret_30d', 0)
            st.metric("30D Return", f"{ret_30d:+.1f}%", delta_color="normal")
        
        with metric_col2:
            pe = stock.get('pe', 0)
            st.metric("P/E Ratio", f"{pe:.1f}" if pe > 0 else "N/A")
            
            eps_growth = stock.get('eps_change_pct', 0)
            if pd.notna(eps_growth):
                st.metric("EPS Growth", f"{eps_growth:+.1f}%", delta_color="normal")
            else:
                st.metric("EPS Growth", "N/A")
        
        with metric_col3:
            rvol = stock.get('rvol', 1)
            vol_emoji = "🔥" if rvol > 3 else "📊"
            st.metric(f"{vol_emoji} Volume", f"{rvol:.1f}x")
            
            position = stock.get('position_52w', 50)
            pos_emoji = "🎯" if position > 80 else "📍"
            st.metric(f"{pos_emoji} 52W Pos", f"{position:.0f}%")
        
        with metric_col4:
            # Entry zone
            if 'entry_zone' in stock:
                zone = stock['entry_zone']
                zone_emoji = {
                    'GOLDEN_ENTRY': '🎯',
                    'BREAKOUT_ZONE': '🚀',
                    'VALUE_ZONE': '💎',
                    'MOMENTUM_ZONE': '📈'
                }.get(zone['zone'], '📍')
                st.metric(f"{zone_emoji} Entry", zone['zone'].replace('_', ' '))
                st.caption(zone['action'])
        
        # Signal strength visualization
        st.markdown("---")
        st.markdown("### 💪 Signal Strength Breakdown")
        fig = render_signal_strength_visual(stock)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Close the card div
        st.markdown("</div>", unsafe_allow_html=True)

def render_sector_intelligence(signals_df):
    """Render sector intelligence dashboard"""
    st.markdown("## 🏢 Sector Intelligence")
    
    if 'sector_performance' not in signals_df.columns:
        st.warning("Sector data not available")
        return
    
    # Get sector summary
    sector_summary = signals_df.groupby('sector').agg({
        'ticker': 'count',
        'ret_30d': 'mean',
        'sector_performance': 'first',
        'signal': lambda x: (x.isin(['STRONG_BUY', 'BUY'])).sum()
    }).round(1)
    
    sector_summary.columns = ['Total Stocks', 'Avg 30D Return', 'Sector Performance', 'Buy Signals']
    sector_summary = sector_summary.sort_values('Sector Performance', ascending=False)
    
    # Create two columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Sector performance heatmap
        st.markdown("### 🔥 Sector Heatmap")
        
        fig = go.Figure(data=go.Bar(
            x=sector_summary['Sector Performance'],
            y=sector_summary.index,
            orientation='h',
            marker=dict(
                color=sector_summary['Sector Performance'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Performance %")
            ),
            text=sector_summary['Sector Performance'].apply(lambda x: f"{x:.1f}%"),
            textposition='inside'
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="30D Performance %",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top stocks in hot sectors
        st.markdown("### 🚀 Leaders in Hot Sectors")
        
        # Get top 3 sectors
        top_sectors = sector_summary.head(3).index
        
        for sector in top_sectors:
            sector_stocks = signals_df[
                (signals_df['sector'] == sector) & 
                (signals_df['signal'].isin(['STRONG_BUY', 'BUY']))
            ].sort_values('composite_score', ascending=False).head(3)
            
            if not sector_stocks.empty:
                st.markdown(f"**{sector}** ({sector_summary.loc[sector, 'Sector Performance']:.1f}% sector return)")
                
                for _, stock in sector_stocks.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"• **{stock['ticker']}** - {stock['company_name'][:30]}")
                    with col2:
                        st.caption(f"Return: {stock['ret_30d']:.1f}%")
                    with col3:
                        st.caption(f"Signal: {stock['signal']}")
                
                st.markdown("---")

def render_pattern_scanner(signals_df):
    """Render pattern detection results"""
    st.markdown("## 🎯 Pattern Scanner")
    
    # Count patterns
    pattern_counts = {}
    for _, stock in signals_df.iterrows():
        if 'patterns' in stock and stock['patterns']:
            for pattern in stock['patterns']:
                pattern_counts[pattern.pattern_name] = pattern_counts.get(pattern.pattern_name, 0) + 1
    
    if not pattern_counts:
        st.info("No patterns detected in current filter")
        return
    
    # Display pattern summary
    cols = st.columns(len(pattern_counts))
    for idx, (pattern_name, count) in enumerate(pattern_counts.items()):
        with cols[idx]:
            st.metric(pattern_name.replace('_', ' '), count)
    
    # Show stocks for each pattern
    selected_pattern = st.selectbox(
        "Select pattern to view stocks:",
        options=list(pattern_counts.keys()),
        format_func=lambda x: x.replace('_', ' ')
    )
    
    # Get stocks with selected pattern
    pattern_stocks = signals_df[
        signals_df['patterns'].apply(
            lambda p: any(pat.pattern_name == selected_pattern for pat in p) if p else False
        )
    ].sort_values('composite_score', ascending=False)
    
    if not pattern_stocks.empty:
        st.markdown(f"### Stocks with {selected_pattern.replace('_', ' ')} Pattern")
        
        for _, stock in pattern_stocks.head(5).iterrows():
            with st.expander(f"{stock['ticker']} - {stock['company_name']}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Signal", stock['signal'])
                with col2:
                    st.metric("30D Return", f"{stock['ret_30d']:.1f}%")
                with col3:
                    st.metric("Volume", f"{stock['rvol']:.1f}x")
                with col4:
                    st.metric("Confidence", f"{stock['confidence']:.0f}%")
                
                # Pattern details
                pattern_obj = next(p for p in stock['patterns'] if p.pattern_name == selected_pattern)
                st.info(f"**Pattern:** {pattern_obj.description}")
                st.success(f"**Action:** {pattern_obj.action}")
                st.caption(f"**Success Rate:** {pattern_obj.success_rate}")

def load_and_analyze_data():
    """Load data and run enhanced analysis"""
    
    with st.spinner("🔄 Loading market data..."):
        success, message = st.session_state.data_loader.load_and_process(use_cache=True)
        
        if not success:
            st.error(f"❌ {message}")
            return None
        
        stocks_df = st.session_state.data_loader.get_stocks_data()
        sector_df = st.session_state.data_loader.get_sector_data()
        
        if stocks_df.empty:
            st.error("❌ No data loaded")
            return None
    
    with st.spinner(f"🧠 Running enhanced analysis on {len(stocks_df)} stocks..."):
        signals_df = st.session_state.signal_engine.analyze(stocks_df, sector_df)
        st.session_state.last_update = datetime.now()
        
        # Show success message
        pattern_count = sum(1 for _, s in signals_df.iterrows() if s.get('patterns'))
        
        st.success(f"""
        ✅ **Enhanced Analysis Complete!**
        - Analyzed {len(signals_df)} stocks with advanced intelligence
        - Detected {pattern_count} stocks with high-probability patterns
        - Found {len(signals_df[signals_df['signal'] == 'STRONG_BUY'])} STRONG_BUY signals
        - Data quality: {st.session_state.data_loader.health['quality_analysis']['quality_score']:.0f}/100
        """)
        
        return signals_df

def main():
    """Main application"""
    
    # Render header
    render_header()
    
    # Control center
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🚀 Load Market Intelligence", type="primary", use_container_width=True):
            signals_df = load_and_analyze_data()
            if signals_df is not None:
                st.session_state.signals_df = signals_df
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.data_loader.clear_cache()
            if 'signals_df' in st.session_state:
                signals_df = load_and_analyze_data()
                if signals_df is not None:
                    st.session_state.signals_df = signals_df
    
    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.data_loader.clear_cache()
            st.success("✅ Cache cleared")
    
    with col4:
        if st.button("ℹ️ Help", use_container_width=True):
            with st.expander("How to use M.A.N.T.R.A. Ultimate", expanded=True):
                st.markdown("""
                **🚀 Quick Start:**
                1. Click "Load Market Intelligence" to analyze stocks
                2. Use Smart Filters to find specific opportunities
                3. Review detailed cards for each opportunity
                4. Check Sector Intelligence for rotation plays
                5. Use Pattern Scanner for high-probability setups
                
                **🎯 Signals:**
                - **STRONG_BUY** (92+): Ultra-high confidence, top 2-3%
                - **BUY** (82+): High confidence, top 8-10%
                - **ACCUMULATE** (72+): Good opportunities
                
                **💡 Tips:**
                - Focus on stocks with multiple pattern detections
                - Check volume explosions for immediate opportunities
                - Use sector intelligence for thematic investing
                """)
    
    # Check if data exists
    if 'signals_df' not in st.session_state:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px; background: #f8f9fa; 
                    border-radius: 15px; margin: 20px 0;'>
            <h2>👋 Welcome to M.A.N.T.R.A. Ultimate</h2>
            <p style='font-size: 18px; margin: 20px 0;'>
                The most advanced stock analysis system with:
            </p>
            <div style='display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;'>
                <div style='text-align: center;'>
                    <h3>🧠</h3>
                    <p>Enhanced AI</p>
                </div>
                <div style='text-align: center;'>
                    <h3>🎯</h3>
                    <p>Pattern Detection</p>
                </div>
                <div style='text-align: center;'>
                    <h3>📊</h3>
                    <p>Smart Filters</p>
                </div>
                <div style='text-align: center;'>
                    <h3>🚀</h3>
                    <p>Volume Intelligence</p>
                </div>
            </div>
            <p style='margin-top: 30px;'>
                <strong>Click "Load Market Intelligence" to begin!</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    signals_df = st.session_state.signals_df
    
    # Market Pulse
    st.markdown("---")
    render_market_pulse(signals_df)
    
    # Smart Filters
    st.markdown("---")
    render_smart_filters()
    
    # Apply selected filter
    current_filter = SMART_FILTERS[st.session_state.selected_filter]
    filtered_df = current_filter['filter'](signals_df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Top Opportunities",
        "🏢 Sector Intelligence", 
        "🔍 Pattern Scanner",
        "📊 All Stocks",
        "📈 Analytics"
    ])
    
    with tab1:
        st.markdown("## 🚀 Top Opportunities")
        
        if filtered_df.empty:
            st.warning(f"No stocks match the filter: {current_filter['name']}")
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filtered Stocks", len(filtered_df))
            with col2:
                strong_buy = len(filtered_df[filtered_df['signal'] == 'STRONG_BUY'])
                st.metric("STRONG_BUY Signals", strong_buy)
            with col3:
                avg_confidence = filtered_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
            
            st.markdown("---")
            
            # Display top opportunities
            top_stocks = filtered_df.head(10)
            for _, stock in top_stocks.iterrows():
                render_enhanced_stock_card(stock)
    
    with tab2:
        render_sector_intelligence(signals_df)
    
    with tab3:
        render_pattern_scanner(filtered_df)
    
    with tab4:
        st.markdown("## 📊 All Stocks Analysis")
        
        # Additional filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sectors = ['All'] + sorted(signals_df['sector'].dropna().unique().tolist())
            selected_sector = st.selectbox("Sector", sectors)
        
        with col2:
            signals = ['All'] + sorted(signals_df['signal'].unique().tolist())
            selected_signal = st.selectbox("Signal", signals)
        
        with col3:
            min_confidence = st.slider("Min Confidence", 0, 100, 60)
        
        with col4:
            sort_by = st.selectbox("Sort By", ['Confidence', '30D Return', 'Volume', 'PE'])
            sort_column = {
                'Confidence': 'confidence',
                '30D Return': 'ret_30d',
                'Volume': 'rvol',
                'PE': 'pe'
            }[sort_by]
        
        # Apply filters
        display_df = filtered_df.copy()
        
        if selected_sector != 'All':
            display_df = display_df[display_df['sector'] == selected_sector]
        
        if selected_signal != 'All':
            display_df = display_df[display_df['signal'] == selected_signal]
        
        display_df = display_df[display_df['confidence'] >= min_confidence]
        
        # Sort
        display_df = display_df.sort_values(sort_column, ascending=(sort_by == 'PE'))
        
        # Display
        st.info(f"Showing {len(display_df)} stocks")
        
        if not display_df.empty:
            # Select columns
            display_cols = [
                'ticker', 'company_name', 'signal', 'confidence', 
                'smart_explanation', 'price', 'ret_1d', 'ret_30d', 
                'pe', 'eps_change_pct', 'rvol', 'sector'
            ]
            
            # Keep available columns
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            # Show dataframe with custom formatting
            st.dataframe(
                display_df[display_cols].head(100),
                use_container_width=True,
                height=600,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "company_name": st.column_config.TextColumn("Company"),
                    "signal": st.column_config.TextColumn("Signal", width="small"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100),
                    "smart_explanation": st.column_config.TextColumn("Intelligence", width="large"),
                    "price": st.column_config.NumberColumn("Price", format="₹%.0f"),
                    "ret_1d": st.column_config.NumberColumn("1D%", format="%.1f%%"),
                    "ret_30d": st.column_config.NumberColumn("30D%", format="%.1f%%"),
                    "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                    "eps_change_pct": st.column_config.NumberColumn("EPS Δ%", format="%.1f%%"),
                    "rvol": st.column_config.NumberColumn("Vol", format="%.1fx"),
                }
            )
    
    with tab5:
        st.markdown("## 📈 Market Analytics")
        
        # Signal distribution pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Signal Distribution")
            signal_counts = signals_df['signal'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                hole=0.4,
                marker_colors=[signal_colors.get(s, '#868e96') for s in signal_counts.index]
            )])
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Top Patterns Detected")
            pattern_summary = {}
            for _, stock in signals_df.iterrows():
                if 'patterns' in stock and stock['patterns']:
                    for pattern in stock['patterns']:
                        pattern_summary[pattern.pattern_name] = pattern_summary.get(pattern.pattern_name, 0) + 1
            
            if pattern_summary:
                pattern_df = pd.DataFrame(
                    list(pattern_summary.items()), 
                    columns=['Pattern', 'Count']
                ).sort_values('Count', ascending=True)
                
                fig = go.Figure(data=[go.Bar(
                    x=pattern_df['Count'],
                    y=pattern_df['Pattern'],
                    orientation='h',
                    marker_color='#667eea'
                )])
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Number of Stocks"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Market breadth over time
        st.markdown("### Market Momentum Distribution")
        
        momentum_ranges = pd.cut(
            signals_df['ret_30d'], 
            bins=[-100, -20, -10, 0, 10, 20, 50, 100],
            labels=['< -20%', '-20 to -10%', '-10 to 0%', '0 to 10%', '10 to 20%', '20 to 50%', '> 50%']
        )
        
        momentum_dist = momentum_ranges.value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=momentum_dist.index,
            y=momentum_dist.values,
            marker_color=['#e74c3c' if '0%' in str(x) and '-' in str(x) else '#2ecc71' for x in momentum_dist.index]
        )])
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="30-Day Return Range",
            yaxis_title="Number of Stocks"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export section
    st.markdown("---")
    with st.expander("📥 Export Data"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📊 Download Filtered Data",
                csv,
                f"mantra_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            top_50 = signals_df.head(50).to_csv(index=False)
            st.download_button(
                "🎯 Download Top 50",
                top_50,
                f"mantra_top50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            # Pattern stocks
            pattern_stocks = signals_df[
                signals_df['patterns'].apply(lambda p: len(p) > 0 if p else False)
            ]
            if not pattern_stocks.empty:
                pattern_csv = pattern_stocks.to_csv(index=False)
                st.download_button(
                    "🎯 Download Pattern Stocks",
                    pattern_csv,
                    f"mantra_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()

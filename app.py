"""
M.A.N.T.R.A. ULTIMATE - Market Analysis Neural Trading Research Assistant
==========================================================================
FINAL ULTIMATE EDITION - Maximum signal accuracy with enterprise performance

Built for:
- 2200+ stocks with lightning speed
- 7-factor advanced signal system  
- Maximum information density
- Professional trading decisions
- Ultimate signal accuracy

"All signal, no noise. Every element is intentional."
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging

# Import ultimate modules
from config import configure_ultimate_page, APP_CONFIG, DISPLAY_CONFIG, SIGNALS, TIER_CLASSIFICATIONS
from core import UltimateMANTRAEngine
from components import UltimateUIComponents

# =============================================================================
# ULTIMATE APPLICATION SETUP
# =============================================================================

# Configure ultimate page (must be first)
configure_ultimate_page()

# Initialize ultimate UI
UltimateUIComponents.load_ultimate_css()

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ULTIMATE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ULTIMATE SESSION STATE MANAGEMENT
# =============================================================================

def initialize_ultimate_session_state():
    """Initialize ultimate session state for maximum performance"""
    defaults = {
        'ultimate_engine': UltimateMANTRAEngine(),
        'data_loaded': False,
        'last_refresh': None,
        'processing_time': 0,
        'show_advanced_analytics': False,
        'auto_refresh_enabled': False,
        'current_view': 'overview',
        'filter_presets': {
            'growth_stars': {'signals': ['STRONG_BUY', 'BUY'], 'min_score': 80, 'risk': ['Low', 'Medium']},
            'value_picks': {'min_score': 70, 'categories': ['Large Cap', 'Mid Cap']},
            'momentum_plays': {'signals': ['STRONG_BUY', 'BUY'], 'min_score': 75}
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize ultimate session state
initialize_ultimate_session_state()

# =============================================================================
# ULTIMATE DATA LOADING WITH CACHING
# =============================================================================

@st.cache_data(ttl=DISPLAY_CONFIG['cache_ttl'], show_spinner=False)
def load_ultimate_data():
    """Load ultimate market data with enterprise caching"""
    start_time = time.time()
    
    engine = UltimateMANTRAEngine()
    success, message = engine.load_and_process()
    
    processing_time = time.time() - start_time
    
    if success:
        return {
            'success': True,
            'master_df': engine.master_df,
            'sectors_df': engine.sectors_df,
            'summary': engine.get_ultimate_summary(),
            'top_opportunities': engine.get_ultimate_opportunities(DISPLAY_CONFIG['top_opportunities']),
            'message': message,
            'processing_time': processing_time,
            'engine': engine
        }
    else:
        return {
            'success': False,
            'master_df': pd.DataFrame(),
            'sectors_df': pd.DataFrame(),
            'summary': {},
            'top_opportunities': pd.DataFrame(),
            'message': message,
            'processing_time': processing_time,
            'engine': None
        }

# =============================================================================
# ULTIMATE HEADER WITH CONTROLS
# =============================================================================

def render_ultimate_header():
    """Render ultimate header with advanced controls"""
    
    # Main header
    UltimateUIComponents.render_ultimate_header()
    
    # Control bar
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        # Data quality and status
        if st.session_state.data_loaded and 'summary' in st.session_state:
            quality = st.session_state.summary.get('data_quality', {})
            if quality:
                UltimateUIComponents.quality_indicator_ultimate(quality)
    
    with col2:
        # Refresh button
        if st.button("🔄 Refresh", 
                    help="Reload market data", 
                    use_container_width=True,
                    key="refresh_btn"):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()
    
    with col3:
        # View toggle
        view_options = ["📊 Overview", "📈 Detailed", "🎯 Opportunities", "🏭 Sectors"]
        current_view = st.selectbox(
            "View",
            options=view_options,
            index=0,
            key="view_selector",
            label_visibility="collapsed"
        )
        st.session_state.current_view = current_view.split(" ")[1].lower()
    
    with col4:
        # Advanced analytics toggle
        if st.button("📈 Analytics" if not st.session_state.show_advanced_analytics else "📊 Simple",
                    help="Toggle advanced analytics",
                    use_container_width=True):
            st.session_state.show_advanced_analytics = not st.session_state.show_advanced_analytics
    
    with col5:
        # Quick export
        if st.session_state.data_loaded and 'master_df' in st.session_state:
            export_data = st.session_state.master_df.head(500)  # Limit for performance
            csv_data = export_data.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            st.download_button(
                label="📥 Export",
                data=csv_data,
                file_name=f"mantra_ultimate_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download top 500 results"
            )

# =============================================================================
# ULTIMATE MARKET OVERVIEW
# =============================================================================

def render_ultimate_overview(summary: dict):
    """Render ultimate market overview with maximum insights"""
    
    UltimateUIComponents.section_header_ultimate(
        "🎯 Market Intelligence", 
        "🎯", 
        f"Real-time analysis of {summary.get('total_stocks', 0):,} stocks"
    )
    
    # Primary metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = summary.get('total_stocks', 0)
        UltimateUIComponents.metric_card_ultimate(
            "Total Stocks", 
            f"{total_stocks:,}",
            icon="📊"
        )
    
    with col2:
        buy_signals = summary.get('buy_signals', 0)
        accumulate_signals = summary.get('accumulate_signals', 0)
        total_signals = buy_signals + accumulate_signals
        signal_pct = (total_signals / total_stocks * 100) if total_stocks > 0 else 0
        
        UltimateUIComponents.metric_card_ultimate(
            "Buy Signals", 
            str(total_signals),
            delta=f"{signal_pct:.1f}%",
            delta_color="green" if signal_pct > 15 else "neutral",
            icon="🎯"
        )
    
    with col3:
        avg_score = summary.get('avg_composite_score', 0)
        UltimateUIComponents.metric_card_ultimate(
            "Avg Score", 
            f"{avg_score:.0f}",
            delta_color="green" if avg_score > 60 else "red",
            icon="⭐"
        )
    
    with col4:
        avg_confidence = summary.get('avg_confidence', 0)
        UltimateUIComponents.metric_card_ultimate(
            "Avg Confidence", 
            f"{avg_confidence:.0f}%",
            delta_color="green" if avg_confidence > 70 else "neutral",
            icon="🔥"
        )
    
    with col5:
        market_breadth = summary.get('market_breadth', 50)
        UltimateUIComponents.metric_card_ultimate(
            "Market Breadth", 
            f"{market_breadth:.0f}%",
            delta_color="green" if market_breadth > 50 else "red",
            icon="🌊"
        )
    
    with col6:
        high_conf_signals = summary.get('high_confidence_signals', 0)
        UltimateUIComponents.metric_card_ultimate(
            "High Conf Signals", 
            str(high_conf_signals),
            icon="🚀"
        )
    
    # Secondary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strong_momentum = summary.get('strong_momentum', 0)
        st.metric("🚀 Strong Momentum", strong_momentum)
    
    with col2:
        volume_spikes = summary.get('volume_spikes', 0)
        st.metric("📊 Volume Spikes", volume_spikes)
    
    with col3:
        # Risk distribution
        risk_dist = summary.get('risk_distribution', {})
        low_risk_pct = risk_dist.get('Low', 0) / total_stocks * 100 if total_stocks > 0 else 0
        st.metric("✅ Low Risk %", f"{low_risk_pct:.0f}%")
    
    with col4:
        # Processing performance
        processing_time = summary.get('processing_stats', {}).get('processing_time', 0)
        st.metric("⚡ Process Time", f"{processing_time:.1f}s")

# =============================================================================
# ULTIMATE FILTERING SYSTEM
# =============================================================================

def render_ultimate_filters(master_df: pd.DataFrame) -> pd.DataFrame:
    """Ultimate filtering system for maximum flexibility"""
    
    if master_df.empty:
        return master_df
    
    # Main filters
    selected_sectors, selected_categories, min_score, selected_signals, selected_risks = \
        UltimateUIComponents.render_smart_filters(master_df)
    
    # Apply filters efficiently
    filtered_df = master_df.copy()
    
    # Sector filter
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
    
    # Category filter  
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Score filter
    if 'composite_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['composite_score'] >= min_score]
    
    # Signal filter
    if selected_signals:
        filtered_df = filtered_df[filtered_df['signal'].isin(selected_signals)]
    
    # Risk filter
    if selected_risks:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(selected_risks)]
    
    # Quick stats for filtered results
    UltimateUIComponents.create_quick_stats_panel(filtered_df)
    
    return filtered_df

# =============================================================================
# ULTIMATE OPPORTUNITIES DISPLAY
# =============================================================================

def render_ultimate_opportunities(opportunities_df: pd.DataFrame):
    """Render ultimate opportunities with maximum information density"""
    
    opportunity_count = len(opportunities_df)
    UltimateUIComponents.section_header_ultimate(
        f"🎯 Ultimate Opportunities", 
        "🎯", 
        f"Top {opportunity_count} high-confidence signals"
    )
    
    if opportunities_df.empty:
        st.info("🔍 No opportunities match your current filters. Consider adjusting criteria for more results.")
        return
    
    # Render opportunities in responsive grid
    num_cols = DISPLAY_CONFIG['cards_per_row']
    
    for i in range(0, len(opportunities_df), num_cols):
        cols = st.columns(num_cols)
        batch = opportunities_df.iloc[i:i+num_cols]
        
        for j, (_, stock) in enumerate(batch.iterrows()):
            if j < len(cols):
                with cols[j]:
                    UltimateUIComponents.stock_card_ultimate(stock)

# =============================================================================
# ULTIMATE DATA TABLE
# =============================================================================

def render_ultimate_table(filtered_df: pd.DataFrame):
    """Render ultimate data table with maximum information density"""
    
    if filtered_df.empty:
        st.info("No stocks match your current filters.")
        return
    
    # Table controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        UltimateUIComponents.section_header_ultimate(
            f"📊 Ultimate Analysis", 
            "📊", 
            f"{len(filtered_df):,} stocks | Showing top {min(len(filtered_df), DISPLAY_CONFIG['table_rows_max'])}"
        )
    
    with col2:
        # Rows to display
        rows_to_show = st.selectbox(
            "Rows",
            options=[50, 100, 200, 500],
            index=1,
            key="rows_selector"
        )
    
    with col3:
        # Sort options
        sort_options = ['composite_score', 'confidence', 'momentum_score', 'value_score', 'ret_30d']
        available_sort_options = [opt for opt in sort_options if opt in filtered_df.columns]
        
        sort_by = st.selectbox(
            "Sort by",
            options=available_sort_options,
            index=0,
            key="sort_selector"
        )
    
    # Prepare display data
    display_df = filtered_df.nlargest(rows_to_show, sort_by)
    formatted_df = UltimateUIComponents.format_ultimate_dataframe(display_df)
    
    # Enhanced column configuration
    column_config = {
        "ticker": st.column_config.TextColumn("Symbol", width="small"),
        "name": st.column_config.TextColumn("Company", width="medium"),
        "Signal": st.column_config.TextColumn("Signal", width="small"),
        "Score": st.column_config.NumberColumn("Score", width="small"),
        "Conf%": st.column_config.NumberColumn("Conf%", width="small"),
        "price": st.column_config.TextColumn("Price", width="small"),
        "sector": st.column_config.TextColumn("Sector", width="medium"),
        "category": st.column_config.TextColumn("Category", width="small")
    }
    
    # Display ultimate table
    st.dataframe(
        formatted_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=600
    )

# =============================================================================
# ULTIMATE SECTOR ANALYSIS
# =============================================================================

def render_ultimate_sector_analysis(sectors_df: pd.DataFrame, master_df: pd.DataFrame):
    """Ultimate sector analysis with comprehensive insights"""
    
    if sectors_df.empty:
        return
    
    UltimateUIComponents.section_header_ultimate("🏭 Sector Intelligence", "🏭")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced sector heatmap
        fig = UltimateUIComponents.create_ultimate_sector_heatmap(sectors_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector signal distribution
        if not master_df.empty and 'sector' in master_df.columns:
            st.subheader("🎯 Sector Opportunities")
            
            sector_signals = master_df[
                master_df['signal'].isin(['STRONG_BUY', 'BUY'])
            ]['sector'].value_counts().head(8)
            
            for sector, count in sector_signals.items():
                total_in_sector = len(master_df[master_df['sector'] == sector])
                percentage = (count / total_in_sector * 100) if total_in_sector > 0 else 0
                st.metric(
                    sector, 
                    f"{count} signals",
                    f"{percentage:.0f}% of sector"
                )

# =============================================================================
# ULTIMATE ANALYTICS DASHBOARD
# =============================================================================

def render_ultimate_analytics(master_df: pd.DataFrame):
    """Ultimate analytics dashboard for deep insights"""
    
    if master_df.empty or not st.session_state.show_advanced_analytics:
        return
    
    UltimateUIComponents.section_header_ultimate("📈 Advanced Analytics", "📈")
    
    # Performance dashboard
    fig = UltimateUIComponents.create_ultimate_performance_dashboard(master_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional analytics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏆 Top Performers")
        if 'ret_30d' in master_df.columns:
            top_performers = master_df.nlargest(5, 'ret_30d')[['ticker', 'ret_30d', 'composite_score']]
            st.dataframe(top_performers, hide_index=True)
    
    with col2:
        st.subheader("💎 Value Opportunities")
        if 'value_score' in master_df.columns:
            value_picks = master_df.nlargest(5, 'value_score')[['ticker', 'pe', 'value_score']]
            st.dataframe(value_picks, hide_index=True)
    
    with col3:
        st.subheader("🚀 Momentum Leaders")
        if 'momentum_score' in master_df.columns:
            momentum_leaders = master_df.nlargest(5, 'momentum_score')[['ticker', 'ret_30d', 'momentum_score']]
            st.dataframe(momentum_leaders, hide_index=True)

# =============================================================================
# ULTIMATE FOOTER
# =============================================================================

def render_ultimate_footer():
    """Ultimate footer with system information"""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Performance stats
        if st.session_state.data_loaded:
            last_update = st.session_state.last_refresh
            processing_time = st.session_state.processing_time
            
            update_time = last_update.strftime('%H:%M:%S') if last_update else 'Never'
            
            st.markdown(f"""
            <div style="text-align: center; color: #888888; padding: 20px;">
                <p style="font-size: 1.2rem; margin-bottom: 10px; color: #00d4ff;">
                    🔱 <strong>M.A.N.T.R.A. {APP_CONFIG['version']}</strong>
                </p>
                <p style="font-size: 0.9rem; margin-bottom: 8px;">
                    {APP_CONFIG['subtitle']}
                </p>
                <p style="font-size: 0.75rem; color: #666666;">
                    ⚡ Last update: {update_time} | ⏱️ Processing: {processing_time:.2f}s | 🎯 Ultimate signal accuracy<br>
                    📊 Built for 2200+ stocks | 🏭 7-factor analysis | 🔍 Maximum information density
                </p>
                <p style="font-size: 0.7rem; color: #444444; margin-top: 10px;">
                    📋 Educational purposes only. Always conduct independent research before trading.
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# ULTIMATE MAIN APPLICATION
# =============================================================================

def main():
    """Ultimate main application with enterprise-grade performance"""
    
    try:
        # Render header with controls
        render_ultimate_header()
        
        # Load data if needed
        if not st.session_state.data_loaded:
            
            UltimateUIComponents.render_loading_ultimate("Loading ultimate market intelligence...")
            
            # Load ultimate data
            data = load_ultimate_data()
            
            if data['success']:
                # Store in session state
                st.session_state.update({
                    'master_df': data['master_df'],
                    'sectors_df': data['sectors_df'],
                    'summary': data['summary'],
                    'top_opportunities': data['top_opportunities'],
                    'data_loaded': True,
                    'last_refresh': datetime.now(),
                    'processing_time': data['processing_time'],
                    'ultimate_engine': data['engine']
                })
                
                UltimateUIComponents.status_message(data['message'], "success")
                time.sleep(1)  # Brief pause to show success
                st.rerun()
                
            else:
                UltimateUIComponents.status_message(data['message'], "error")
                st.error("⚠️ Unable to load market data. Please check your connection and try refreshing.")
                st.stop()
        
        # Get data from session state
        master_df = st.session_state.master_df
        sectors_df = st.session_state.sectors_df
        summary = st.session_state.summary
        top_opportunities = st.session_state.top_opportunities
        
        # Render market overview
        render_ultimate_overview(summary)
        
        # Apply ultimate filters
        filtered_data = render_ultimate_filters(master_df)
        
        # Render content based on current view
        current_view = st.session_state.current_view
        
        if current_view == "overview":
            # Overview: Top opportunities + summary table
            render_ultimate_opportunities(filtered_data.head(DISPLAY_CONFIG['top_opportunities']))
            render_ultimate_table(filtered_data.head(100))
            
        elif current_view == "detailed":
            # Detailed: Full table view
            render_ultimate_table(filtered_data)
            
        elif current_view == "opportunities":
            # Opportunities: Focus on top picks
            render_ultimate_opportunities(filtered_data.head(20))
            
        elif current_view == "sectors":
            # Sectors: Sector analysis focus
            render_ultimate_sector_analysis(sectors_df, master_df)
        
        # Advanced analytics (if enabled)
        render_ultimate_analytics(master_df)
        
        # Ultimate footer
        render_ultimate_footer()
        
        # Sidebar information
        with st.sidebar:
            st.markdown("### 🎯 Quick Stats")
            
            if not filtered_data.empty:
                st.metric("Filtered Stocks", len(filtered_data))
                
                if 'signal' in filtered_data.columns:
                    strong_signals = len(filtered_data[filtered_data['signal'].isin(['STRONG_BUY', 'BUY'])])
                    st.metric("Strong Signals", strong_signals)
                
                if 'confidence' in filtered_data.columns:
                    avg_confidence = filtered_data['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
            
            st.markdown("---")
            st.markdown("### ⚙️ System Status")
            
            if st.session_state.last_refresh:
                refresh_time = st.session_state.last_refresh.strftime('%H:%M:%S')
                st.text(f"Last refresh: {refresh_time}")
            
            processing_time = st.session_state.processing_time
            st.text(f"Processing: {processing_time:.2f}s")
            
            # Data quality indicator
            quality = summary.get('data_quality', {})
            if quality:
                st.text(f"Data quality: {quality.get('status', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Ultimate application error: {e}", exc_info=True)
        UltimateUIComponents.status_message(f"⚠️ System error: {str(e)}", "error")
        st.error("🔧 Please refresh the page. If the problem persists, check the data source.")

# =============================================================================
# ULTIMATE APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting M.A.N.T.R.A. Ultimate Edition")
    main()

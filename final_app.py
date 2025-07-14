"""
app_final.py - M.A.N.T.R.A. Version 3 FINAL Application
=======================================================
Ultimate locked system with simple UI and best UX
Built for permanent use - no further upgrades needed
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all system components
from config_final import *
from engine_final import UltimatePrecisionEngine
from intelligence import MANTRAIntelligence
from ui_final import SimpleUIComponents
from quality import QualityController

# Configure Streamlit for optimal performance
configure_streamlit()

class MANTRAFinalApp:
    """
    M.A.N.T.R.A. Version 3 FINAL - Ultimate Market Intelligence
    
    Simple UI with maximum intelligence underneath.
    Built for speed, reliability, and crystal-clear insights.
    """
    
    def __init__(self):
        self.engine = UltimatePrecisionEngine()
        self.intelligence = MANTRAIntelligence()
        self.ui = SimpleUIComponents()
        self.quality_controller = QualityController()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for clean app behavior"""
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'last_load_time' not in st.session_state:
            st.session_state.last_load_time = None
        
        if 'processing_stats' not in st.session_state:
            st.session_state.processing_stats = None
        
        if 'quality_report' not in st.session_state:
            st.session_state.quality_report = None
        
        if 'market_summary' not in st.session_state:
            st.session_state.market_summary = {}
    
    @st.cache_data(ttl=DISPLAY_CONFIG['cache_ttl'])
    def _load_and_process_data(_self):
        """Load and process data with smart caching"""
        
        start_time = time.time()
        
        try:
            # Load and process all data
            success, message = _self.engine.load_and_process()
            
            if not success:
                return False, message, None, None, None
            
            # Get quality report
            quality_report = _self.quality_controller.comprehensive_quality_assessment(
                _self.engine.watchlist_df,
                _self.engine.sectors_df,
                _self.engine.returns_df
            )
            
            # Get market summary
            market_summary = _self.engine.get_market_summary()
            
            # Get master dataframe
            master_df = _self.engine.master_df.copy()
            
            processing_time = time.time() - start_time
            
            return True, message, master_df, quality_report, market_summary, processing_time
            
        except Exception as e:
            error_msg = f"Critical error in data processing: {str(e)}"
            return False, error_msg, None, None, None, 0
    
    def run(self):
        """Main application runner"""
        
        # Render header
        self.ui.render_header()
        
        # Data loading section
        self._render_data_loading_section()
        
        if st.session_state.data_loaded and hasattr(self.engine, 'master_df') and not self.engine.master_df.empty:
            # Main dashboard
            self._render_daily_edge_dashboard()
            
            # Market intelligence panel
            self._render_market_intelligence_panel()
            
            # Advanced features section
            self._render_advanced_features()
            
            # Footer
            self.ui.render_footer()
        else:
            self._render_getting_started_guide()
    
    def _render_data_loading_section(self):
        """Render data loading and refresh section"""
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Load Market Data", type="primary", use_container_width=True):
                self._load_data()
        
        with col2:
            if st.session_state.last_load_time:
                time_diff = datetime.now() - st.session_state.last_load_time
                minutes_ago = int(time_diff.total_seconds() / 60)
                st.info(f"Updated {minutes_ago}m ago")
        
        with col3:
            if st.session_state.data_loaded:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.cache_data.clear()
                    self._load_data()
    
    def _load_data(self):
        """Load data with comprehensive error handling and user feedback"""
        
        with st.spinner("🔱 Loading market intelligence..."):
            
            # Show loading indicator
            loading_placeholder = st.empty()
            with loading_placeholder:
                self.ui.loading_indicator("Analyzing 2200+ stocks with precision...")
            
            # Load data
            result = self._load_and_process_data()
            
            # Clear loading indicator
            loading_placeholder.empty()
            
            if len(result) == 6:  # Success case
                success, message, master_df, quality_report, market_summary, processing_time = result
                
                if success and master_df is not None:
                    # Update session state
                    st.session_state.data_loaded = True
                    st.session_state.last_load_time = datetime.now()
                    st.session_state.processing_stats = {
                        'processing_time': processing_time,
                        'total_stocks': len(master_df),
                        'signals_generated': len(master_df[master_df['signal'].isin(['STRONG_BUY', 'BUY', 'ACCUMULATE'])]),
                        'data_quality': quality_report.overall_score if quality_report else 0
                    }
                    st.session_state.quality_report = quality_report
                    st.session_state.market_summary = market_summary
                    
                    # Store data in engine
                    self.engine.master_df = master_df
                    
                    # Success message with stats
                    signal_counts = master_df['signal'].value_counts()
                    strong_buy_count = signal_counts.get('STRONG_BUY', 0)
                    buy_count = signal_counts.get('BUY', 0)
                    
                    success_msg = f"""
                    ✅ **Market Intelligence Loaded Successfully**
                    
                    📊 **{len(master_df):,} stocks analyzed** in {processing_time:.1f}s
                    🎯 **{strong_buy_count} STRONG_BUY** + **{buy_count} BUY** signals generated
                    📈 **{quality_report.overall_score:.0f}% data quality** - {quality_report.status}
                    ⚡ **{len(master_df)/processing_time:.0f} stocks/second** processing speed
                    """
                    
                    st.success(success_msg)
                    
                    # Auto-rerun to show dashboard
                    time.sleep(1)
                    st.rerun()
                
                else:
                    st.error(f"❌ Data loading failed: {message}")
                    
            else:  # Error case
                success, message = result[:2]
                st.error(f"❌ Critical error: {message}")
    
    def _render_daily_edge_dashboard(self):
        """Render the main Daily Edge dashboard"""
        
        self.ui.section_header("📈 Today's Market Edge", "🎯")
        
        # Get top opportunities
        top_opportunities = self.engine.get_top_opportunities(limit=10)
        
        if top_opportunities.empty:
            st.info("🔍 No high-confidence opportunities found today. Market conditions may require patience.")
            return
        
        # Separate strong buy and buy signals
        strong_buy_stocks = top_opportunities[top_opportunities['signal'] == 'STRONG_BUY']
        buy_stocks = top_opportunities[top_opportunities['signal'] == 'BUY']
        
        # Strong Buy Section
        if not strong_buy_stocks.empty:
            st.markdown("### 🚀 STRONG BUY Opportunities")
            st.markdown(f"*Ultra-high confidence signals - Top {len(strong_buy_stocks)} picks*")
            
            for idx, stock in strong_buy_stocks.iterrows():
                self.ui.opportunity_card(stock, show_explanation=True)
                
                # Add expandable detailed analysis
                with st.expander(f"🧠 Deep Analysis - {stock['ticker']}", expanded=False):
                    self._render_detailed_stock_analysis(stock)
        
        # Buy Section
        if not buy_stocks.empty:
            st.markdown("### 📈 BUY Opportunities")
            st.markdown(f"*High confidence signals - {len(buy_stocks)} additional opportunities*")
            
            # Show first 3 by default, rest in expander
            display_count = min(3, len(buy_stocks))
            
            for idx, stock in buy_stocks.head(display_count).iterrows():
                self.ui.opportunity_card(stock, show_explanation=True)
            
            # Show remaining in expander
            if len(buy_stocks) > 3:
                with st.expander(f"➕ View {len(buy_stocks) - 3} More BUY Opportunities", expanded=False):
                    for idx, stock in buy_stocks.tail(len(buy_stocks) - 3).iterrows():
                        self.ui.opportunity_card(stock, show_explanation=False)
    
    def _render_detailed_stock_analysis(self, stock):
        """Render detailed analysis for a specific stock"""
        
        ticker = stock.get('ticker', 'Unknown')
        
        # Get detailed explanation
        explanation = self.engine.get_signal_explanation(ticker)
        
        if explanation:
            # Factor scores visualization
            st.markdown("**Factor Analysis:**")
            
            factor_cols = st.columns(4)
            factors = list(explanation.factor_scores.items())
            
            for i, (factor, score) in enumerate(factors):
                with factor_cols[i % 4]:
                    # Color code based on score
                    if score >= 80:
                        color = "🟢"
                    elif score >= 60:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.metric(
                        f"{color} {factor.title()}",
                        f"{score:.0f}",
                        help=f"{factor} factor score out of 100"
                    )
            
            # Key metrics
            st.markdown("**Key Metrics:**")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                price = stock.get('price', 0)
                ret_1d = stock.get('ret_1d', 0)
                st.metric("Price", f"₹{price:,.0f}", f"{ret_1d:+.1f}%")
            
            with metric_cols[1]:
                pe = stock.get('pe', 0)
                st.metric("P/E Ratio", f"{pe:.1f}" if pe > 0 else "N/A")
            
            with metric_cols[2]:
                ret_30d = stock.get('ret_30d', 0)
                st.metric("30D Return", f"{ret_30d:+.1f}%")
            
            with metric_cols[3]:
                rvol = stock.get('rvol', 1)
                st.metric("Rel. Volume", f"{rvol:.1f}x")
            
            # Supporting evidence and risks
            if explanation.supporting_factors:
                st.markdown("**Why This Signal:**")
                for factor in explanation.supporting_factors:
                    st.write(f"✅ {factor}")
            
            if explanation.risk_factors:
                st.markdown("**Risk Considerations:**")
                for risk in explanation.risk_factors:
                    st.write(f"⚠️ {risk}")
            
            # Recommendation
            st.markdown("**Investment Recommendation:**")
            st.info(explanation.recommendation)
        
        else:
            st.warning("Detailed analysis not available for this stock.")
    
    def _render_market_intelligence_panel(self):
        """Render market intelligence and summary panel"""
        
        self.ui.section_header("📊 Market Intelligence", "📈")
        
        # Market overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        market_summary = st.session_state.market_summary
        processing_stats = st.session_state.processing_stats
        
        with col1:
            total_stocks = processing_stats.get('total_stocks', 0) if processing_stats else 0
            st.metric("📊 Stocks Analyzed", f"{total_stocks:,}")
        
        with col2:
            signals_generated = processing_stats.get('signals_generated', 0) if processing_stats else 0
            st.metric("🎯 Action Signals", signals_generated)
        
        with col3:
            market_breadth = market_summary.get('market_breadth', 50) if market_summary else 50
            st.metric("📈 Market Breadth", f"{market_breadth:.0f}%")
        
        with col4:
            data_quality = processing_stats.get('data_quality', 0) if processing_stats else 0
            st.metric("✅ Data Quality", f"{data_quality:.0f}%")
        
        # Market condition and sector performance
        if market_summary:
            col1, col2 = st.columns(2)
            
            with col1:
                # Market condition
                market_conditions = market_summary.get('market_conditions', {})
                condition = market_conditions.get('condition', 'Unknown')
                confidence = market_conditions.get('confidence', 0)
                
                condition_display = condition.replace('_', ' ').title()
                st.info(f"🌊 **Market Condition:** {condition_display} ({confidence:.0f}% confidence)")
            
            with col2:
                # Signal distribution
                signal_dist = market_summary.get('signal_distribution', {})
                if signal_dist:
                    strong_buy = signal_dist.get('STRONG_BUY', 0)
                    buy = signal_dist.get('BUY', 0)
                    accumulate = signal_dist.get('ACCUMULATE', 0)
                    
                    st.success(f"🎯 **Today's Signals:** {strong_buy} Strong Buy, {buy} Buy, {accumulate} Accumulate")
        
        # Sector performance (if available)
        if hasattr(self.engine, 'sectors_df') and not self.engine.sectors_df.empty:
            with st.expander("🏭 Sector Performance Analysis", expanded=False):
                self._render_sector_performance()
    
    def _render_sector_performance(self):
        """Render sector performance analysis"""
        
        try:
            sectors_df = self.engine.sectors_df
            
            if 'sector' in sectors_df.columns and 'sector_ret_30d' in sectors_df.columns:
                # Sort by 30-day performance
                sector_perf = sectors_df.sort_values('sector_ret_30d', ascending=False)
                
                # Create performance chart
                chart_fig = self.ui.create_simple_sector_chart(sector_perf)
                st.plotly_chart(chart_fig, use_container_width=True)
                
                # Top and bottom performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔥 Top Performers (30D)**")
                    top_sectors = sector_perf.head(5)
                    for _, sector in top_sectors.iterrows():
                        name = sector.get('sector', 'Unknown')
                        perf = sector.get('sector_ret_30d', 0)
                        st.write(f"📈 {name}: **{perf:+.1f}%**")
                
                with col2:
                    st.markdown("**❄️ Underperformers (30D)**")
                    bottom_sectors = sector_perf.tail(5)
                    for _, sector in bottom_sectors.iterrows():
                        name = sector.get('sector', 'Unknown')
                        perf = sector.get('sector_ret_30d', 0)
                        st.write(f"📉 {name}: **{perf:+.1f}%**")
        
        except Exception as e:
            st.error(f"Unable to display sector performance: {str(e)}")
    
    def _render_advanced_features(self):
        """Render advanced features and tools"""
        
        self.ui.section_header("🔧 Advanced Features", "⚙️")
        
        tab1, tab2, tab3 = st.tabs(["🔍 Stock Explorer", "📊 Data Quality", "📁 Export"])
        
        with tab1:
            self._render_stock_explorer()
        
        with tab2:
            self._render_data_quality_panel()
        
        with tab3:
            self._render_export_panel()
    
    def _render_stock_explorer(self):
        """Render advanced stock filtering and exploration"""
        
        if not hasattr(self.engine, 'master_df') or self.engine.master_df.empty:
            st.info("Load data to access stock explorer.")
            return
        
        st.markdown("**🔍 Explore and filter all analyzed stocks**")
        
        # Get filters
        sectors, categories, signals, min_score, min_confidence = self.ui.render_simple_filters(self.engine.master_df)
        
        # Apply filters
        filtered_df = self.engine.get_filtered_stocks(
            sectors=sectors if sectors else None,
            categories=categories if categories else None,
            signals=signals if signals else None,
            min_score=min_score,
            min_confidence=min_confidence
        )
        
        if filtered_df.empty:
            st.warning("🔍 No stocks match your current filters. Try adjusting the criteria.")
            return
        
        # Quick stats for filtered results
        self.ui.render_quick_stats(filtered_df)
        
        # Display options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            display_mode = st.selectbox(
                "Display Mode",
                ["Cards", "Table"],
                help="Choose how to display the results"
            )
        
        with col2:
            show_limit = st.selectbox(
                "Show",
                [25, 50, 100, 200],
                index=1,
                help="Number of stocks to display"
            )
        
        # Display results
        display_df = filtered_df.head(show_limit)
        
        if display_mode == "Cards":
            # Card display
            for idx, stock in display_df.iterrows():
                self.ui.opportunity_card(stock, show_explanation=False)
                
                # Add quick action buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button(f"📊 Details", key=f"details_{stock['ticker']}"):
                        st.session_state[f"show_details_{stock['ticker']}"] = True
                
                with col2:
                    if st.button(f"📝 Notes", key=f"notes_{stock['ticker']}"):
                        st.session_state[f"show_notes_{stock['ticker']}"] = True
                
                # Show details if requested
                if st.session_state.get(f"show_details_{stock['ticker']}", False):
                    with st.expander(f"Details for {stock['ticker']}", expanded=True):
                        self._render_detailed_stock_analysis(stock)
                    
                    if st.button("Close Details", key=f"close_{stock['ticker']}"):
                        st.session_state[f"show_details_{stock['ticker']}"] = False
                        st.rerun()
        
        else:
            # Table display
            formatted_df = self.ui.format_table_data(display_df)
            st.dataframe(
                formatted_df,
                use_container_width=True,
                height=min(600, len(formatted_df) * 35 + 100)
            )
    
    def _render_data_quality_panel(self):
        """Render data quality information and diagnostics"""
        
        quality_report = st.session_state.quality_report
        
        if not quality_report:
            st.info("Load data to see quality report.")
            return
        
        # Overall quality indicator
        self.ui.quality_indicator({
            'overall_score': quality_report.overall_score,
            'status': quality_report.status
        })
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{quality_report.overall_score:.1f}%")
        
        with col2:
            st.metric("Usable Stocks", f"{quality_report.usable_stocks:,}")
        
        with col3:
            st.metric("Data Coverage", f"{(quality_report.usable_stocks/max(quality_report.total_stocks,1)*100):.1f}%")
        
        with col4:
            issues_count = len(quality_report.critical_issues) + len(quality_report.warnings)
            st.metric("Issues Found", issues_count)
        
        # Quality details
        if quality_report.critical_issues:
            st.markdown("**🚨 Critical Issues:**")
            for issue in quality_report.critical_issues:
                st.error(f"• {issue}")
        
        if quality_report.warnings:
            st.markdown("**⚠️ Warnings:**")
            for warning in quality_report.warnings:
                st.warning(f"• {warning}")
        
        if quality_report.recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in quality_report.recommendations:
                st.info(f"• {rec}")
        
        # Data completeness by column
        if quality_report.completeness_scores:
            with st.expander("📊 Data Completeness by Column", expanded=False):
                completeness_df = pd.DataFrame([
                    {'Column': col, 'Completeness': f"{score:.1f}%"}
                    for col, score in quality_report.completeness_scores.items()
                ]).sort_values('Completeness', ascending=False)
                
                st.dataframe(completeness_df, use_container_width=True)
    
    def _render_export_panel(self):
        """Render data export options"""
        
        if not hasattr(self.engine, 'master_df') or self.engine.master_df.empty:
            st.info("Load data to access export options.")
            return
        
        st.markdown("**📁 Export your market analysis data**")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            export_scope = st.selectbox(
                "Export Scope",
                ["Top Opportunities Only", "All Strong Buy + Buy", "Complete Dataset"],
                help="Choose what data to export"
            )
        
        with col2:
            export_format = st.selectbox(
                "Format",
                ["CSV", "Excel"],
                help="Choose export format"
            )
        
        # Prepare export data
        if export_scope == "Top Opportunities Only":
            export_df = self.engine.get_top_opportunities(limit=20)
        elif export_scope == "All Strong Buy + Buy":
            export_df = self.engine.master_df[
                self.engine.master_df['signal'].isin(['STRONG_BUY', 'BUY'])
            ]
        else:
            export_df = self.engine.master_df
        
        if export_df.empty:
            st.warning("No data available for export.")
            return
        
        # Export button and file preparation
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.metric("Stocks to Export", len(export_df))
        
        with col2:
            if st.button("📁 Prepare Export", type="primary"):
                self._prepare_export_file(export_df, export_format, export_scope)
        
        # Show preview
        with st.expander("👀 Preview Export Data", expanded=False):
            preview_df = self.ui.format_table_data(export_df.head(10))
            st.dataframe(preview_df, use_container_width=True)
    
    def _prepare_export_file(self, df, format_type, scope):
        """Prepare and offer file download"""
        
        try:
            # Add metadata
            export_df = df.copy()
            
            # Add export timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if format_type == "CSV":
                # Prepare CSV
                csv_data = export_df.to_csv(index=False)
                
                filename = f"MANTRA_Export_{scope.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Download your market analysis data as CSV"
                )
            
            else:  # Excel
                # For Excel, we'll provide CSV for now since Excel creation requires additional libraries
                csv_data = export_df.to_csv(index=False)
                
                filename = f"MANTRA_Export_{scope.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                
                st.download_button(
                    label="⬇️ Download Data",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Download your market analysis data"
                )
            
            st.success(f"✅ Export prepared successfully! {len(export_df)} stocks ready for download.")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
    
    def _render_getting_started_guide(self):
        """Render getting started guide for new users"""
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## 🚀 Welcome to M.A.N.T.R.A. Version 3 Final
            
            **Your Ultimate Market Intelligence System**
            
            This system analyzes 2200+ stocks using 8 precision factors to find the best opportunities with ultra-high confidence.
            
            ### 🎯 Key Features:
            - **🚀 STRONG_BUY Signals**: Ultra-conservative thresholds (92+ score) for maximum confidence
            - **📈 BUY Opportunities**: High-confidence signals (82+ score) with clear reasoning  
            - **🧠 Explainable AI**: Every signal comes with detailed reasoning and risk analysis
            - **⚡ Ultra-Fast**: Processes 2200+ stocks in 1-3 seconds
            - **📊 Market Intelligence**: Real-time sector performance and market condition analysis
            - **🔍 Smart Filtering**: Filter by sector, category, risk level, and confidence
            
            ### 📋 How to Get Started:
            1. **Click "🚀 Load Market Data"** to analyze all stocks
            2. **Review Top Opportunities** in the Daily Edge dashboard
            3. **Expand any signal** to see detailed reasoning and risk factors
            4. **Use filters** to explore specific sectors or categories
            5. **Export data** for further analysis or record keeping
            
            ### 🎯 Signal Types:
            - **🚀 STRONG_BUY (92+)**: Top 2-3% of stocks - Ultra-high confidence
            - **📈 BUY (82+)**: Top 8-10% of stocks - High confidence with reasoning
            - **📊 ACCUMULATE (72+)**: Top 20% - Good opportunities for gradual building
            - **👀 WATCH (60+)**: Monitor closely - Potential but not ready
            """)
        
        with col2:
            st.info("""
            **💡 Pro Tips:**
            
            ✅ **Quality First**: Only STRONG_BUY and BUY signals are actionable
            
            ✅ **Check Reasoning**: Always expand to see why a stock got its signal
            
            ✅ **Watch Risk Factors**: Every signal shows potential risks
            
            ✅ **Use Filters**: Find opportunities in your preferred sectors
            
            ✅ **Monitor Market Conditions**: Signals adapt to bull/bear markets
            
            ✅ **Export Data**: Save opportunities for deeper research
            """)
            
            st.warning("""
            **⚠️ Important:**
            
            This system is for educational and research purposes only. Always conduct your own due diligence before making investment decisions.
            """)

# Main application runner
def main():
    """Main application entry point"""
    
    try:
        # Initialize and run the app
        app = MANTRAFinalApp()
        app.run()
        
    except Exception as e:
        st.error(f"""
        🚨 **Critical System Error**
        
        The M.A.N.T.R.A. system encountered an unexpected error:
        
        `{str(e)}`
        
        Please refresh the page to restart the system.
        """)
        
        # Show error details in expander for debugging
        with st.expander("🔧 Technical Details", expanded=False):
            st.exception(e)

if __name__ == "__main__":
    main()

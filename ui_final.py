"""
ui_final.py - M.A.N.T.R.A. Version 3 FINAL UI Components
========================================================
Simple UI with the best UX - Clean, fast, intuitive
Every element serves a purpose, nothing unnecessary
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from config_final import *
from intelligence import MANTRAIntelligence, DetailedExplanation

class SimpleUIComponents:
    """Simple but best UI components - clean, fast, intuitive"""
    
    def __init__(self):
        self.intelligence = MANTRAIntelligence()
        self.load_simple_css()
    
    def load_simple_css(self):
        """Load clean, simple CSS with best UX principles"""
        st.markdown("""
        <style>
        /* Clean, professional font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global clean design */
        .stApp {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #ffffff;
            color: #212529;
        }
        
        /* Hide Streamlit elements for clean look */
        #MainMenu, footer, header, .stDeployButton {
            visibility: hidden;
        }
        .stApp > div:first-child {
            margin-top: -80px;
        }
        
        /* Clean container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Simple header */
        .app-header {
            text-align: center;
            padding: 40px 20px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .app-title {
            font-size: 3rem;
            font-weight: 700;
            color: #212529;
            margin: 0;
            letter-spacing: -0.02em;
        }
        
        .app-subtitle {
            font-size: 1.1rem;
            color: #6c757d;
            margin-top: 10px;
            font-weight: 400;
        }
        
        .app-philosophy {
            font-size: 0.9rem;
            color: #007bff;
            margin-top: 5px;
            font-style: italic;
        }
        
        /* Simple metric cards */
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: all 0.2s ease;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .metric-card:hover {
            border-color: #007bff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #212529;
            margin: 5px 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-delta {
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 3px;
        }
        
        /* Clean opportunity cards */
        .opportunity-card {
            background: #ffffff;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .opportunity-card:hover {
            border-color: #007bff;
            box-shadow: 0 6px 20px rgba(0, 123, 255, 0.1);
            transform: translateY(-3px);
        }
        
        .stock-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        
        .stock-ticker {
            font-size: 1.2rem;
            font-weight: 700;
            color: #212529;
        }
        
        .stock-name {
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 3px;
        }
        
        .stock-price {
            font-size: 1.6rem;
            font-weight: 700;
            color: #212529;
            margin: 10px 0;
        }
        
        .stock-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
            font-size: 0.85rem;
            color: #6c757d;
        }
        
        .stock-explanation {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 12px;
            margin-top: 12px;
            font-size: 0.9rem;
            color: #495057;
            border-left: 4px solid #007bff;
        }
        
        /* Simple signal badges */
        .signal-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            border: none;
        }
        
        .signal-strong-buy {
            background-color: #28a745;
            color: white;
        }
        
        .signal-buy {
            background-color: #40c057;
            color: white;
        }
        
        .signal-accumulate {
            background-color: #74c0fc;
            color: #212529;
        }
        
        .signal-watch {
            background-color: #ffd43b;
            color: #212529;
        }
        
        .signal-neutral {
            background-color: #868e96;
            color: white;
        }
        
        .signal-avoid {
            background-color: #fa5252;
            color: white;
        }
        
        .signal-strong-avoid {
            background-color: #e03131;
            color: white;
        }
        
        /* Clean confidence indicator */
        .confidence-bar {
            width: 100%;
            height: 6px;
            background-color: #e9ecef;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .confidence-fill {
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 3px;
        }
        
        .confidence-high {
            background: linear-gradient(90deg, #28a745, #40c057);
        }
        
        .confidence-medium {
            background: linear-gradient(90deg, #ffd43b, #fd7e14);
        }
        
        .confidence-low {
            background: linear-gradient(90deg, #fa5252, #e03131);
        }
        
        /* Simple section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #212529;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Clean filter panel */
        .filter-panel {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Simple quality indicator */
        .quality-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .quality-excellent {
            border-color: #28a745;
            color: #28a745;
            background: rgba(40, 167, 69, 0.1);
        }
        
        .quality-good {
            border-color: #007bff;
            color: #007bff;
            background: rgba(0, 123, 255, 0.1);
        }
        
        .quality-acceptable {
            border-color: #ffc107;
            color: #856404;
            background: rgba(255, 193, 7, 0.1);
        }
        
        /* Clean table styling */
        .dataframe {
            border: 1px solid #e9ecef !important;
            border-radius: 8px !important;
        }
        
        .dataframe thead th {
            background: #f8f9fa !important;
            color: #495057 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #dee2e6 !important;
        }
        
        .dataframe tbody td {
            border-bottom: 1px solid #f1f3f4 !important;
        }
        
        /* Simple status messages */
        .success-message {
            background: rgba(40, 167, 69, 0.1);
            border: 1px solid #28a745;
            color: #155724;
            padding: 12px 20px;
            border-radius: 6px;
            margin: 15px 0;
        }
        
        .error-message {
            background: rgba(220, 53, 69, 0.1);
            border: 1px solid #dc3545;
            color: #721c24;
            padding: 12px 20px;
            border-radius: 6px;
            margin: 15px 0;
        }
        
        .info-message {
            background: rgba(0, 123, 255, 0.1);
            border: 1px solid #007bff;
            color: #004085;
            padding: 12px 20px;
            border-radius: 6px;
            margin: 15px 0;
        }
        
        /* Loading indicator */
        .loading-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: #6c757d;
        }
        
        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #e9ecef;
            border-left-color: #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .app-title {
                font-size: 2rem;
            }
            
            .opportunity-card {
                margin: 10px 0;
                padding: 15px;
            }
            
            .stock-metrics {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render clean, simple header"""
        st.markdown(f"""
        <div class="app-header">
            <h1 class="app-title">{APP_CONFIG['icon']} {APP_CONFIG['title']}</h1>
            <p class="app-subtitle">{APP_CONFIG['subtitle']}</p>
            <p class="app-philosophy">{APP_CONFIG['philosophy']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def metric_card(self, label: str, value: str, delta: str = "", delta_color: str = ""):
        """Simple, clean metric card"""
        
        delta_html = ""
        if delta:
            color = COLORS['success'] if delta_color == "green" else COLORS['danger'] if delta_color == "red" else COLORS['text_secondary']
            delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def opportunity_card(self, stock: pd.Series, show_explanation: bool = True):
        """Clean opportunity card with optional explanation"""
        
        # Extract stock data
        ticker = stock.get('ticker', 'N/A')
        name = str(stock.get('name', ticker))[:40] + ('...' if len(str(stock.get('name', ticker))) > 40 else '')
        price = stock.get('price', 0)
        ret_1d = stock.get('ret_1d', 0)
        ret_30d = stock.get('ret_30d', 0)
        signal = stock.get('signal', 'NEUTRAL')
        confidence = stock.get('confidence', 50)
        pe = stock.get('pe', 0)
        rvol = stock.get('rvol', 1)
        sector = stock.get('sector', 'Unknown')
        
        # Signal styling
        signal_class = f"signal-{signal.lower().replace('_', '-')}"
        
        # Price change color
        price_color = COLORS['success'] if ret_1d >= 0 else COLORS['danger']
        
        # Confidence bar
        if confidence >= 80:
            conf_class = "confidence-high"
        elif confidence >= 60:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        # Format price
        price_display = f"₹{price:,.2f}" if price < 1000 else f"₹{price:,.0f}"
        
        # Generate simple explanation if requested
        explanation_html = ""
        if show_explanation:
            try:
                simple_explanation = self.intelligence.generate_simple_explanation(stock)
                # Clean explanation for display
                clean_explanation = simple_explanation.split(' - ', 1)[-1] if ' - ' in simple_explanation else simple_explanation
                explanation_html = f'<div class="stock-explanation">{clean_explanation}</div>'
            except:
                explanation_html = ""
        
        st.markdown(f"""
        <div class="opportunity-card">
            <div class="stock-header">
                <div>
                    <div class="stock-ticker">{ticker}</div>
                    <div class="stock-name">{name}</div>
                </div>
                <span class="signal-badge {signal_class}">{signal}</span>
            </div>
            
            <div class="stock-price">
                {price_display}
                <span style="font-size: 1rem; color: {price_color}; margin-left: 10px; font-weight: 500;">
                    {'+' if ret_1d >= 0 else ''}{ret_1d:.1f}%
                </span>
            </div>
            
            <div style="margin: 10px 0;">
                <span style="font-size: 0.85rem; color: #6c757d;">Confidence: {confidence:.0f}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill {conf_class}" style="width: {confidence}%"></div>
                </div>
            </div>
            
            <div class="stock-metrics">
                <div><strong>Sector:</strong> {sector}</div>
                <div><strong>P/E:</strong> {pe:.1f if pe > 0 else 'N/A'}</div>
                <div><strong>30D:</strong> <span style="color: {'#28a745' if ret_30d >= 0 else '#dc3545'}">{ret_30d:+.1f}%</span></div>
            </div>
            
            {explanation_html}
        </div>
        """, unsafe_allow_html=True)
    
    def quality_indicator(self, quality_info: Dict):
        """Simple data quality indicator"""
        
        score = quality_info.get('overall_score', 0)
        status = quality_info.get('status', 'Unknown')
        
        if score >= 95:
            badge_class = "quality-excellent"
            icon = "✅"
        elif score >= 85:
            badge_class = "quality-good" 
            icon = "✅"
        else:
            badge_class = "quality-acceptable"
            icon = "⚠️"
        
        st.markdown(f"""
        <div class="quality-indicator {badge_class}">
            <span>{icon}</span>
            <span>Data: {status} ({score:.0f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    def section_header(self, title: str, icon: str = ""):
        """Simple section header"""
        icon_html = f'<span style="font-size: 1.3rem; margin-right: 8px;">{icon}</span>' if icon else ''
        st.markdown(f'<div class="section-header">{icon_html}{title}</div>', unsafe_allow_html=True)
    
    def success_message(self, message: str):
        """Simple success message"""
        st.markdown(f'<div class="success-message">✅ {message}</div>', unsafe_allow_html=True)
    
    def error_message(self, message: str):
        """Simple error message"""
        st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)
    
    def info_message(self, message: str):
        """Simple info message"""
        st.markdown(f'<div class="info-message">ℹ️ {message}</div>', unsafe_allow_html=True)
    
    def loading_indicator(self, message: str = "Processing market data..."):
        """Simple loading indicator"""
        st.markdown(f"""
        <div class="loading-indicator">
            <div class="loading-spinner"></div>
            <span>{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def render_simple_filters(self, stocks_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], float, float]:
        """Simple but effective filtering interface"""
        
        if stocks_df.empty:
            return [], [], [], 0, 0
        
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown("**🎯 Quick Filters**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Sector filter (main requirement)
            available_sectors = sorted(stocks_df['sector'].dropna().unique()) if 'sector' in stocks_df.columns else []
            selected_sectors = st.multiselect(
                "Sectors",
                options=available_sectors,
                default=[],
                help="Filter by industry sectors"
            )
        
        with col2:
            # Category filter (main requirement)
            available_categories = sorted(stocks_df['category'].dropna().unique()) if 'category' in stocks_df.columns else []
            selected_categories = st.multiselect(
                "Categories", 
                options=available_categories,
                default=[],
                help="Filter by market cap categories"
            )
        
        with col3:
            # Signal filter
            signal_options = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH']
            selected_signals = st.multiselect(
                "Signals",
                options=signal_options,
                default=['STRONG_BUY', 'BUY'],
                help="Select signal types to display"
            )
        
        with col4:
            # Score and confidence sliders
            min_score = st.slider("Min Score", 0, 100, 75, 5, help="Minimum composite score")
            min_confidence = st.slider("Min Confidence", 0, 100, 60, 5, help="Minimum signal confidence")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return selected_sectors, selected_categories, selected_signals, min_score, min_confidence
    
    def create_simple_sector_chart(self, sectors_df: pd.DataFrame) -> go.Figure:
        """Simple, clean sector performance chart"""
        
        if sectors_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=COLORS['text_secondary'])
            )
            fig.update_layout(
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
        
        # Prepare data - focus on key timeframes
        sectors = sectors_df['sector'].tolist()
        periods = ['1D', '7D', '30D']
        columns = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d']
        
        # Build data matrix
        z_values = []
        text_values = []
        
        for _, row in sectors_df.iterrows():
            row_data = []
            row_text = []
            
            for col in columns:
                if col in sectors_df.columns and pd.notna(row[col]):
                    val = float(row[col])
                    row_data.append(val)
                    row_text.append(f'{val:+.1f}%')
                else:
                    row_data.append(0)
                    row_text.append('N/A')
            
            z_values.append(row_data)
            text_values.append(row_text)
        
        # Create clean heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=periods,
            y=sectors,
            text=text_values,
            texttemplate='%{text}',
            textfont=dict(size=11, color='white'),
            colorscale=[
                [0, COLORS['danger']],
                [0.5, '#f8f9fa'],
                [1, COLORS['success']]
            ],
            zmid=0,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
            showscale=True,
            colorbar=dict(
                title="Return %",
                titlefont=dict(color=COLORS['text_primary']),
                tickfont=dict(color=COLORS['text_primary'])
            )
        ))
        
        # Clean layout
        fig.update_layout(
            height=max(350, len(sectors) * 25),
            font=dict(color=COLORS['text_primary']),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=100, r=40, t=20, b=40),
            xaxis=dict(
                side='top',
                tickfont=dict(size=12),
                showgrid=False
            ),
            yaxis=dict(
                tickfont=dict(size=11),
                showgrid=False
            )
        )
        
        return fig
    
    def create_simple_summary_chart(self, summary_data: Dict) -> go.Figure:
        """Simple market summary visualization"""
        
        # Create subplot for key metrics
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Signal Distribution', 'Market Breadth'],
            specs=[[{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # Signal distribution pie chart
        signal_dist = summary_data.get('signal_distribution', {})
        if signal_dist:
            labels = list(signal_dist.keys())
            values = list(signal_dist.values())
            colors = [COLORS['success'], COLORS['primary'], COLORS['info'], COLORS['warning'], COLORS['text_secondary']]
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors[:len(labels)]),
                    textinfo='label+percent',
                    textfont=dict(size=10)
                ),
                row=1, col=1
            )
        
        # Market breadth gauge
        market_breadth = summary_data.get('market_breadth', 50)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=market_breadth,
                title={'text': "Market Breadth %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': COLORS['primary']},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=COLORS['text_primary'])
        )
        
        return fig
    
    def format_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataframe for clean table display"""
        
        if df.empty:
            return df
        
        # Create display copy
        display_df = df.copy()
        
        # Select and order columns for display
        display_columns = [
            'ticker', 'name', 'signal', 'composite_score', 'confidence',
            'price', 'ret_1d', 'ret_30d', 'pe', 'rvol', 'sector'
        ]
        
        # Keep only available columns
        available_columns = [col for col in display_columns if col in display_df.columns]
        display_df = display_df[available_columns]
        
        # Format for better readability
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(
                lambda x: f'₹{x:,.2f}' if pd.notna(x) else ''
            )
        
        if 'ret_1d' in display_df.columns:
            display_df['ret_1d'] = display_df['ret_1d'].apply(
                lambda x: f'{x:+.1f}%' if pd.notna(x) else ''
            )
        
        if 'ret_30d' in display_df.columns:
            display_df['ret_30d'] = display_df['ret_30d'].apply(
                lambda x: f'{x:+.1f}%' if pd.notna(x) else ''
            )
        
        if 'pe' in display_df.columns:
            display_df['pe'] = display_df['pe'].apply(
                lambda x: f'{x:.1f}' if pd.notna(x) and x > 0 else 'N/A'
            )
        
        if 'rvol' in display_df.columns:
            display_df['rvol'] = display_df['rvol'].apply(
                lambda x: f'{x:.1f}x' if pd.notna(x) else ''
            )
        
        if 'composite_score' in display_df.columns:
            display_df['composite_score'] = display_df['composite_score'].apply(
                lambda x: f'{x:.0f}' if pd.notna(x) else ''
            )
        
        if 'confidence' in display_df.columns:
            display_df['confidence'] = display_df['confidence'].apply(
                lambda x: f'{x:.0f}%' if pd.notna(x) else ''
            )
        
        # Rename columns for better display
        column_renames = {
            'composite_score': 'Score',
            'confidence': 'Conf%',
            'ret_1d': '1D%',
            'ret_30d': '30D%',
            'pe': 'P/E',
            'rvol': 'RVol'
        }
        
        for old_name, new_name in column_renames.items():
            if old_name in display_df.columns:
                display_df = display_df.rename(columns={old_name: new_name})
        
        return display_df
    
    def render_quick_stats(self, filtered_df: pd.DataFrame):
        """Render quick statistics for filtered data"""
        
        if filtered_df.empty:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Filtered Stocks", len(filtered_df))
        
        with col2:
            if 'signal' in filtered_df.columns:
                strong_signals = len(filtered_df[filtered_df['signal'].isin(['STRONG_BUY', 'BUY'])])
                st.metric("🎯 Strong Signals", strong_signals)
        
        with col3:
            if 'composite_score' in filtered_df.columns:
                avg_score = filtered_df['composite_score'].mean()
                st.metric("⭐ Avg Score", f"{avg_score:.0f}")
        
        with col4:
            if 'confidence' in filtered_df.columns:
                avg_confidence = filtered_df['confidence'].mean()
                st.metric("🔥 Avg Confidence", f"{avg_confidence:.0f}%")
    
    def render_explanation_panel(self, stock: pd.Series):
        """Render detailed explanation panel for a stock"""
        
        try:
            explanation = self.intelligence.generate_comprehensive_explanation(stock)
            
            with st.expander(f"🧠 Detailed Analysis - {stock.get('ticker', 'Stock')}", expanded=False):
                
                # Header
                st.markdown(f"**{explanation.headline}**")
                
                # Main thesis
                st.markdown("**Investment Thesis:**")
                st.write(explanation.primary_thesis)
                
                # Supporting evidence
                if explanation.supporting_evidence:
                    st.markdown("**Supporting Evidence:**")
                    for evidence in explanation.supporting_evidence:
                        st.write(f"• {evidence}")
                
                # Risk considerations
                if explanation.risk_considerations:
                    st.markdown("**Risk Considerations:**")
                    for risk in explanation.risk_considerations:
                        st.write(f"• {risk}")
                
                # Recommendation
                st.markdown("**Recommendation:**")
                st.write(f"• {explanation.recommendation}")
                st.write(f"• {explanation.target_action}")
                st.write(f"• {explanation.risk_management}")
                
                # Factor breakdown
                st.markdown("**Factor Analysis:**")
                factor_cols = st.columns(3)
                
                for i, (factor, analysis) in enumerate(explanation.factor_analysis.items()):
                    with factor_cols[i % 3]:
                        score = analysis['score']
                        rating = analysis['rating']
                        icon = analysis['icon']
                        st.metric(
                            f"{icon} {factor.title()}",
                            f"{score:.0f}",
                            f"{rating}"
                        )
        
        except Exception as e:
            st.error(f"Unable to generate detailed explanation: {str(e)}")
    
    def create_performance_summary(self, processing_stats: Dict) -> Dict[str, str]:
        """Create simple performance summary"""
        
        if not processing_stats:
            return {}
        
        processing_time = processing_stats.get('processing_time', 0)
        total_stocks = processing_stats.get('total_stocks', 0)
        strong_buy_count = processing_stats.get('strong_buy_count', 0)
        buy_count = processing_stats.get('buy_count', 0)
        
        return {
            'speed': f"⚡ Processed {total_stocks:,} stocks in {processing_time:.1f}s",
            'signals': f"🎯 Generated {strong_buy_count} Strong Buy + {buy_count} Buy signals",
            'rate': f"📊 Processing rate: {total_stocks/max(processing_time, 0.1):.0f} stocks/second" if processing_time > 0 else ""
        }
    
    def render_footer(self):
        """Simple, clean footer"""
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; padding: 20px;">
            <p style="font-size: 1rem; margin-bottom: 8px;">
                <strong>{APP_CONFIG['icon']} {APP_CONFIG['title']} {APP_CONFIG['version']}</strong>
            </p>
            <p style="font-size: 0.9rem; margin-bottom: 12px;">
                {APP_CONFIG['subtitle']}
            </p>
            <p style="font-size: 0.8rem; color: #868e96;">
                🎯 Built for ultra-high confidence trading signals | 📊 Simple UI, powerful intelligence underneath<br>
                📋 For educational purposes only. Always conduct your own research before trading.
            </p>
        </div>
        """, unsafe_allow_html=True)
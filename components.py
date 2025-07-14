"""
components.py - M.A.N.T.R.A. ULTIMATE UI Components
===================================================
Maximum information density with enterprise-grade aesthetics
Optimized for 2200+ stocks with lightning-fast rendering
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import COLORS, APP_CONFIG, DISPLAY_CONFIG, ESSENTIAL_DISPLAY_COLUMNS

class UltimateUIComponents:
    """Ultimate UI components for maximum information density and speed"""
    
    @staticmethod
    def load_ultimate_css():
        """Ultra-modern CSS with maximum information density"""
        st.markdown("""
        <style>
        /* Import professional fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global high-density theme */
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #0a0a0a 50%, #111111 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 13px;
            line-height: 1.4;
        }
        
        /* Hide Streamlit branding */
        #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
        .stApp > div:first-child {margin-top: -90px;}
        
        /* Compact spacing */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }
        
        /* Ultra-modern header */
        .ultimate-header {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 107, 53, 0.1) 100%);
            backdrop-filter: blur(30px);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 16px;
            padding: 20px 30px;
            margin-bottom: 25px;
            position: relative;
            overflow: hidden;
        }
        
        .ultimate-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00d4ff 0%, #ff6b35 100%);
        }
        
        .ultimate-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 50%, #ff6b35 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            letter-spacing: -0.03em;
            text-align: center;
        }
        
        .ultimate-subtitle {
            font-size: 1rem;
            color: #888888;
            margin-top: 8px;
            text-align: center;
            font-weight: 500;
        }
        
        /* High-density metric cards */
        .metric-card-ultimate {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            transition: all 0.2s ease;
            height: 110px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            cursor: pointer;
        }
        
        .metric-card-ultimate:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 212, 255, 0.3);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
        }
        
        .metric-icon-ultimate {
            font-size: 1.8rem;
            margin-bottom: 6px;
            filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.3));
        }
        
        .metric-value-ultimate {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin: 4px 0;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .metric-label-ultimate {
            font-size: 0.75rem;
            color: #888888;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-delta-ultimate {
            font-size: 0.7rem;
            font-weight: 600;
            margin-top: 2px;
        }
        
        /* Ultra-compact stock cards */
        .stock-card-ultimate {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 14px;
            margin: 8px 0;
            transition: all 0.2s ease;
            position: relative;
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .stock-card-ultimate:hover {
            transform: translateY(-1px);
            border-color: rgba(0, 212, 255, 0.2);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        .stock-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        
        .stock-ticker-ultimate {
            font-size: 0.95rem;
            font-weight: 700;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .stock-name-ultimate {
            font-size: 0.7rem;
            color: #888888;
            margin-top: 2px;
            line-height: 1.2;
        }
        
        .stock-price-ultimate {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            margin: 6px 0;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .stock-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            font-size: 0.65rem;
            color: #888888;
        }
        
        .stock-metric {
            padding: 2px 4px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 4px;
        }
        
        /* Advanced signal badges */
        .signal-badge-ultimate {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .signal-strong-buy {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.25), rgba(0, 255, 136, 0.4));
            color: #00ff88;
            border-color: rgba(0, 255, 136, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
        }
        
        .signal-buy {
            background: linear-gradient(135deg, rgba(102, 255, 153, 0.2), rgba(102, 255, 153, 0.35));
            color: #66ff99;
            border-color: rgba(102, 255, 153, 0.3);
        }
        
        .signal-accumulate {
            background: linear-gradient(135deg, rgba(153, 221, 255, 0.2), rgba(153, 221, 255, 0.35));
            color: #99ddff;
            border-color: rgba(153, 221, 255, 0.3);
        }
        
        .signal-watch {
            background: linear-gradient(135deg, rgba(255, 183, 0, 0.2), rgba(255, 183, 0, 0.35));
            color: #ffb700;
            border-color: rgba(255, 183, 0, 0.3);
        }
        
        .signal-neutral {
            background: linear-gradient(135deg, rgba(136, 136, 136, 0.2), rgba(136, 136, 136, 0.35));
            color: #888888;
            border-color: rgba(136, 136, 136, 0.3);
        }
        
        .signal-avoid {
            background: linear-gradient(135deg, rgba(255, 102, 102, 0.2), rgba(255, 102, 102, 0.35));
            color: #ff6666;
            border-color: rgba(255, 102, 102, 0.3);
        }
        
        .signal-strong-avoid {
            background: linear-gradient(135deg, rgba(255, 51, 102, 0.25), rgba(255, 51, 102, 0.4));
            color: #ff3366;
            border-color: rgba(255, 51, 102, 0.4);
        }
        
        /* High-density data quality indicator */
        .quality-indicator-ultimate {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .quality-excellent {
            border-color: rgba(0, 255, 136, 0.3);
            color: #00ff88;
        }
        
        .quality-good {
            border-color: rgba(0, 212, 255, 0.3);
            color: #00d4ff;
        }
        
        .quality-acceptable {
            border-color: rgba(255, 183, 0, 0.3);
            color: #ffb700;
        }
        
        /* Section headers with information density */
        .section-header-ultimate {
            font-size: 1.2rem;
            font-weight: 700;
            color: #ffffff;
            margin: 20px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .section-icon-ultimate {
            font-size: 1.1rem;
            filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.3));
        }
        
        /* Advanced filter panel */
        .filter-panel-ultimate {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
        
        /* High-density table styling */
        .dataframe {
            font-size: 0.75rem !important;
            line-height: 1.3 !important;
        }
        
        .dataframe thead th {
            background: rgba(0, 212, 255, 0.1) !important;
            color: #00d4ff !important;
            font-weight: 600 !important;
            font-size: 0.7rem !important;
            padding: 6px 8px !important;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3) !important;
        }
        
        .dataframe tbody td {
            padding: 4px 8px !important;
            font-size: 0.7rem !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Performance indicators */
        .performance-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 0.6rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .perf-excellent {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }
        
        .perf-good {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }
        
        .perf-average {
            background: rgba(255, 183, 0, 0.2);
            color: #ffb700;
        }
        
        .perf-poor {
            background: rgba(255, 102, 102, 0.2);
            color: #ff6666;
        }
        
        /* Responsive design for maximum density */
        @media (max-width: 1200px) {
            .ultimate-title {
                font-size: 2.2rem;
            }
            .stock-card-ultimate {
                height: 130px;
            }
        }
        
        /* Loading optimizations */
        .loading-ultimate {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: #00d4ff;
        }
        
        .loading-spinner {
            width: 24px;
            height: 24px;
            border: 2px solid rgba(0, 212, 255, 0.2);
            border-left-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 12px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Success/Error states */
        .status-message {
            padding: 12px 20px;
            border-radius: 10px;
            margin: 12px 0;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .status-success {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            color: #00ff88;
        }
        
        .status-error {
            background: rgba(255, 51, 102, 0.1);
            border: 1px solid rgba(255, 51, 102, 0.3);
            color: #ff3366;
        }
        
        /* Confidence indicators */
        .confidence-bar {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 4px;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        
        .confidence-high {
            background: linear-gradient(90deg, #00ff88, #00d4ff);
        }
        
        .confidence-medium {
            background: linear-gradient(90deg, #ffb700, #ff6b35);
        }
        
        .confidence-low {
            background: linear-gradient(90deg, #ff6666, #ff3366);
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_ultimate_header():
        """Render ultimate header with maximum impact"""
        st.markdown(f"""
        <div class="ultimate-header">
            <h1 class="ultimate-title">{APP_CONFIG['icon']} {APP_CONFIG['title']}</h1>
            <p class="ultimate-subtitle">{APP_CONFIG['subtitle']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card_ultimate(label: str, value: str, delta: str = "", 
                            delta_color: str = "", icon: str = "📊"):
        """Ultra-compact metric card for maximum information density"""
        
        delta_html = ""
        if delta:
            if delta_color == "green":
                color = COLORS['success']
            elif delta_color == "red":
                color = COLORS['danger']
            else:
                color = COLORS['text_muted']
            delta_html = f'<div class="metric-delta-ultimate" style="color: {color};">{delta}</div>'
        
        st.markdown(f"""
        <div class="metric-card-ultimate">
            <div class="metric-icon-ultimate">{icon}</div>
            <div class="metric-value-ultimate">{value}</div>
            <div class="metric-label-ultimate">{label}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def stock_card_ultimate(stock: pd.Series):
        """Ultra-compact stock card with maximum information density"""
        
        # Extract data with intelligent defaults
        ticker = stock.get('ticker', 'N/A')
        name = str(stock.get('name', ticker))[:28] + ('...' if len(str(stock.get('name', ticker))) > 28 else '')
        price = stock.get('price', 0)
        change_1d = stock.get('ret_1d', 0)
        signal = stock.get('signal', 'NEUTRAL')
        score = stock.get('composite_score', 50)
        confidence = stock.get('confidence', 50)
        
        # Additional metrics for information density
        pe = stock.get('pe', 0)
        eps_change = stock.get('eps_change_pct', 0)
        rvol = stock.get('rvol', 1)
        from_low = stock.get('from_low_pct', 0)
        momentum_score = stock.get('momentum_score', 50)
        value_score = stock.get('value_score', 50)
        risk_level = stock.get('risk_level', 'Medium')
        
        # Signal styling
        signal_class = f"signal-{signal.lower().replace('_', '-')}"
        signal_icons = {
            'STRONG_BUY': '🚀', 'BUY': '📈', 'ACCUMULATE': '📊',
            'WATCH': '👀', 'NEUTRAL': '➖', 'AVOID': '⚠️', 'STRONG_AVOID': '🚫'
        }
        signal_icon = signal_icons.get(signal, '➖')
        
        # Price change color
        price_color = COLORS['success'] if change_1d >= 0 else COLORS['danger']
        
        # Format price intelligently
        if price >= 1000:
            price_display = f"₹{price:,.0f}"
        else:
            price_display = f"₹{price:,.2f}"
        
        # Confidence bar
        confidence_class = 'confidence-high' if confidence >= 75 else 'confidence-medium' if confidence >= 50 else 'confidence-low'
        
        # Risk color
        risk_colors = {'Low': '#00ff88', 'Medium': '#ffb700', 'High': '#ff6666', 'Extreme': '#ff3366'}
        risk_color = risk_colors.get(risk_level, '#888888')
        
        st.markdown(f"""
        <div class="stock-card-ultimate">
            <div class="stock-header">
                <div>
                    <div class="stock-ticker-ultimate">{ticker}</div>
                    <div class="stock-name-ultimate">{name}</div>
                </div>
                <span class="signal-badge-ultimate {signal_class}">
                    {signal_icon} {signal}
                </span>
            </div>
            
            <div class="stock-price-ultimate">
                {price_display}
                <span style="font-size: 0.7rem; color: {price_color}; margin-left: 6px; font-weight: 500;">
                    {'+' if change_1d >= 0 else ''}{change_1d:.1f}%
                </span>
            </div>
            
            <div style="margin: 6px 0;">
                <span style="font-size: 0.65rem; color: #888888;">Score: </span>
                <span style="font-size: 0.75rem; color: #ffffff; font-weight: 600;">{int(score)}</span>
                <span style="font-size: 0.65rem; color: #888888; margin-left: 8px;">Conf: </span>
                <span style="font-size: 0.75rem; color: #ffffff; font-weight: 600;">{int(confidence)}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill {confidence_class}" style="width: {confidence}%"></div>
                </div>
            </div>
            
            <div class="stock-metrics">
                <div class="stock-metric">P/E: <strong>{pe:.1f if pe > 0 else 'N/A'}</strong></div>
                <div class="stock-metric">EPS: <strong>{eps_change:+.0f}%</strong></div>
                <div class="stock-metric">RVol: <strong>{rvol:.1f}x</strong></div>
                <div class="stock-metric">52W: <strong>{from_low:.0f}%</strong></div>
                <div class="stock-metric">Mom: <strong>{int(momentum_score)}</strong></div>
                <div class="stock-metric" style="color: {risk_color};">Risk: <strong>{risk_level}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def quality_indicator_ultimate(quality: Dict):
        """Ultra-compact data quality indicator"""
        
        score = quality.get('score', 0)
        status = quality.get('status', 'Unknown')
        
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
        <div class="quality-indicator-ultimate {badge_class}">
            <span>{icon}</span>
            <span>Data: {status} ({score:.0f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def section_header_ultimate(title: str, icon: str = "📊", subtitle: str = ""):
        """Compact section header with optional subtitle"""
        subtitle_html = f'<span style="font-size: 0.8rem; color: #888888; margin-left: 12px; font-weight: 400;">{subtitle}</span>' if subtitle else ''
        
        st.markdown(f"""
        <div class="section-header-ultimate">
            <span class="section-icon-ultimate">{icon}</span>
            <span>{title}</span>
            {subtitle_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def status_message(message: str, status_type: str = "success"):
        """Compact status message"""
        status_class = f"status-{status_type}"
        st.markdown(f'<div class="status-message {status_class}">{message}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_smart_filters(stocks_df: pd.DataFrame) -> Tuple[List[str], List[str], float, List[str], List[str]]:
        """Advanced filtering panel optimized for 2200+ stocks"""
        
        if stocks_df.empty:
            return [], [], 0, [], []
        
        st.markdown('<div class="filter-panel-ultimate">', unsafe_allow_html=True)
        
        # Create compact filter layout
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 2, 1.5])
        
        with col1:
            # Sector filter (main request)
            available_sectors = sorted(stocks_df['sector'].dropna().unique()) if 'sector' in stocks_df.columns else []
            selected_sectors = st.multiselect(
                "🏭 Sectors",
                options=available_sectors,
                default=[],
                help="Filter by industry sectors",
                key="sector_filter"
            )
        
        with col2:
            # Category filter (main request)
            available_categories = sorted(stocks_df['category'].dropna().unique()) if 'category' in stocks_df.columns else []
            selected_categories = st.multiselect(
                "📊 Categories",
                options=available_categories,
                default=[],
                help="Filter by market cap categories",
                key="category_filter"
            )
        
        with col3:
            # Score threshold
            min_score = st.slider(
                "Min Score",
                min_value=0,
                max_value=100,
                value=70,
                step=5,
                help="Minimum composite score",
                key="score_filter"
            )
        
        with col4:
            # Signal types
            signal_options = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH', 'NEUTRAL', 'AVOID', 'STRONG_AVOID']
            selected_signals = st.multiselect(
                "🎯 Signals",
                options=signal_options,
                default=['STRONG_BUY', 'BUY'],
                help="Trading signal types",
                key="signal_filter"
            )
        
        with col5:
            # Risk levels
            risk_options = ['Low', 'Medium', 'High', 'Extreme']
            selected_risks = st.multiselect(
                "⚠️ Risk",
                options=risk_options,
                default=['Low', 'Medium'],
                help="Acceptable risk levels",
                key="risk_filter"
            )
        
        # Advanced filters in expandable section
        with st.expander("🔧 Advanced Filters", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Confidence filter
                min_confidence = st.slider("Min Confidence %", 0, 100, 60, 5, key="conf_filter")
            
            with col2:
                # Volume filter
                min_volume = st.number_input("Min Volume (K)", 0, 10000, 100, 50, key="vol_filter") * 1000
            
            with col3:
                # PE ratio filter
                max_pe = st.slider("Max P/E Ratio", 0, 100, 50, 5, key="pe_filter")
            
            with col4:
                # Position filter
                min_52w_position = st.slider("Min 52W Position %", 0, 100, 20, 10, key="pos_filter")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return selected_sectors, selected_categories, min_score, selected_signals, selected_risks
    
    @staticmethod
    def create_ultimate_sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
        """Advanced sector heatmap with multiple timeframes"""
        
        if sector_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS['text_muted'])
            )
            fig.update_layout(
                height=250,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Prepare comprehensive data
        sectors = sector_df['sector'].tolist()
        periods = ['1D', '7D', '30D', '3M', '1Y']
        columns = ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 'sector_ret_3m', 'sector_ret_1y']
        
        # Build enhanced data matrix
        z_values = []
        text_values = []
        hover_values = []
        
        for _, row in sector_df.iterrows():
            row_data = []
            row_text = []
            row_hover = []
            
            for col in columns:
                if col in sector_df.columns and pd.notna(row[col]):
                    val = float(row[col])
                    row_data.append(val)
                    row_text.append(f'{val:+.1f}%')
                    row_hover.append(f'{val:+.2f}%')
                else:
                    row_data.append(0)
                    row_text.append('N/A')
                    row_hover.append('No data')
            
            z_values.append(row_data)
            text_values.append(row_text)
            hover_values.append(row_hover)
        
        # Create advanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=periods,
            y=sectors,
            text=text_values,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}%<br><extra></extra>',
            texttemplate='%{text}',
            textfont=dict(size=10, color='white', family='JetBrains Mono'),
            colorscale=[
                [0, COLORS['danger']],
                [0.3, '#ff6666'],
                [0.45, '#333333'],
                [0.55, '#333333'],
                [0.7, '#66ff99'],
                [1, COLORS['success']]
            ],
            zmid=0,
            showscale=True,
            colorbar=dict(
                title="Return %",
                titlefont=dict(color=COLORS['text'], size=12),
                tickfont=dict(color=COLORS['text'], size=10),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor='rgba(0, 212, 255, 0.2)',
                borderwidth=1,
                thickness=15,
                len=0.8
            )
        ))
        
        # Enhanced layout
        fig.update_layout(
            height=max(400, len(sectors) * 28),
            font=dict(family='Inter', color=COLORS['text']),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=140, r=60, t=30, b=30),
            xaxis=dict(
                side='top',
                tickfont=dict(size=11, color=COLORS['text']),
                linecolor='rgba(0, 212, 255, 0.2)',
                showgrid=False
            ),
            yaxis=dict(
                tickfont=dict(size=10, color=COLORS['text']),
                linecolor='rgba(0, 212, 255, 0.2)',
                showgrid=False
            )
        )
        
        return fig
    
    @staticmethod
    def create_ultimate_performance_dashboard(df: pd.DataFrame) -> go.Figure:
        """Comprehensive performance dashboard with multiple charts"""
        
        if df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Score Distribution', 'Signal Breakdown', 'Risk vs Return', 'Confidence Analysis'],
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # 1. Score distribution
        fig.add_trace(
            go.Histogram(
                x=df['composite_score'] if 'composite_score' in df.columns else [],
                nbinsx=20,
                name='Score Distribution',
                marker=dict(color=COLORS['primary'], opacity=0.7),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Signal breakdown
        if 'signal' in df.columns:
            signal_counts = df['signal'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=signal_counts.index,
                    values=signal_counts.values,
                    name='Signals',
                    marker=dict(colors=[COLORS['success'], COLORS['primary'], COLORS['info'], 
                                      COLORS['warning'], COLORS['text_muted'], COLORS['danger']]),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Risk vs Return analysis
        if all(col in df.columns for col in ['ret_30d', 'risk_level']):
            risk_colors = {'Low': COLORS['success'], 'Medium': COLORS['warning'], 
                          'High': COLORS['danger'], 'Extreme': COLORS['danger']}
            
            for risk in df['risk_level'].unique():
                if pd.notna(risk):
                    risk_data = df[df['risk_level'] == risk]
                    fig.add_trace(
                        go.Scatter(
                            x=risk_data['composite_score'] if 'composite_score' in risk_data.columns else [],
                            y=risk_data['ret_30d'],
                            mode='markers',
                            name=f'{risk} Risk',
                            marker=dict(color=risk_colors.get(risk, COLORS['text_muted']), size=4),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        # 4. Confidence analysis
        if 'confidence' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['confidence'],
                    nbinsx=15,
                    name='Confidence Distribution',
                    marker=dict(color=COLORS['info'], opacity=0.7),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'], family='Inter'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    @staticmethod
    def format_ultimate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format dataframe for ultra-high information density display"""
        
        if df.empty:
            return df
        
        display_df = df.copy()
        
        # Intelligent column selection for maximum information density
        essential_cols = [col for col in ESSENTIAL_DISPLAY_COLUMNS if col in display_df.columns]
        display_df = display_df[essential_cols]
        
        # Format functions with enhanced precision
        def format_price(val):
            if pd.isna(val) or val == 0:
                return ''
            return f'₹{val:,.0f}' if val >= 100 else f'₹{val:.2f}'
        
        def format_percentage(val):
            if pd.isna(val):
                return ''
            color = '#00ff88' if val >= 0 else '#ff3366'
            return f'<span style="color: {color}; font-weight: 600;">{val:+.1f}%</span>'
        
        def format_score(val):
            if pd.isna(val):
                return ''
            if val >= 80:
                color = '#00ff88'
            elif val >= 60:
                color = '#00d4ff'
            elif val >= 40:
                color = '#ffb700'
            else:
                color = '#ff6666'
            return f'<span style="color: {color}; font-weight: 600;">{val:.0f}</span>'
        
        def format_signal(val):
            if pd.isna(val):
                return ''
            
            colors = {
                'STRONG_BUY': 'background: linear-gradient(135deg, #00ff88, #00d4ff); color: #000; font-weight: 700;',
                'BUY': 'background: #66ff99; color: #000; font-weight: 600;',
                'ACCUMULATE': 'background: #99ddff; color: #000; font-weight: 600;',
                'WATCH': 'background: #ffb700; color: #000; font-weight: 600;',
                'NEUTRAL': 'background: #888888; color: #fff;',
                'AVOID': 'background: #ff6666; color: #fff;',
                'STRONG_AVOID': 'background: #ff3366; color: #fff; font-weight: 700;'
            }
            
            style = colors.get(val, 'background: #888888; color: #fff;')
            return f'<div style="{style} padding: 2px 6px; border-radius: 6px; text-align: center; font-size: 0.7rem;">{val}</div>'
        
        def format_risk(val):
            if pd.isna(val):
                return ''
            colors = {'Low': '#00ff88', 'Medium': '#ffb700', 'High': '#ff6666', 'Extreme': '#ff3366'}
            color = colors.get(val, '#888888')
            return f'<span style="color: {color}; font-weight: 600;">{val}</span>'
        
        # Apply formatting
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(format_price)
        
        # Format percentage columns
        for col in ['ret_1d', 'ret_30d', 'ret_3m', 'from_low_pct']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_percentage)
        
        # Format score columns
        for col in ['composite_score', 'confidence', 'momentum_score', 'value_score', 'growth_score', 'volume_score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_score)
        
        # Format special columns
        if 'signal' in display_df.columns:
            display_df['signal'] = display_df['signal'].apply(format_signal)
        
        if 'risk_level' in display_df.columns:
            display_df['risk_level'] = display_df['risk_level'].apply(format_risk)
        
        # Format other columns
        if 'pe' in display_df.columns:
            display_df['pe'] = display_df['pe'].apply(lambda x: f'{x:.1f}' if pd.notna(x) and x > 0 else 'N/A')
        
        if 'rvol' in display_df.columns:
            display_df['rvol'] = display_df['rvol'].apply(lambda x: f'{x:.1f}x' if pd.notna(x) else '')
        
        # Rename columns for better display
        column_renames = {
            'composite_score': 'Score',
            'confidence': 'Conf%',
            'ret_1d': '1D%',
            'ret_30d': '30D%',
            'ret_3m': '3M%',
            'from_low_pct': '52W%',
            'momentum_score': 'Mom',
            'value_score': 'Val',
            'growth_score': 'Grw',
            'volume_score': 'Vol',
            'risk_level': 'Risk',
            'position_strength': 'Pos'
        }
        
        for old_name, new_name in column_renames.items():
            if old_name in display_df.columns:
                display_df = display_df.rename(columns={old_name: new_name})
        
        return display_df
    
    @staticmethod
    def render_loading_ultimate(message: str = "Processing ultimate signals..."):
        """Enhanced loading indicator"""
        st.markdown(f"""
        <div class="loading-ultimate">
            <div class="loading-spinner"></div>
            <span style="font-size: 1rem; font-weight: 500;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_quick_stats_panel(df: pd.DataFrame) -> None:
        """Quick statistics panel for immediate insights"""
        
        if df.empty:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            strong_signals = len(df[df['signal'].isin(['STRONG_BUY', 'BUY'])]) if 'signal' in df.columns else 0
            st.metric("🎯 Strong Signals", strong_signals, f"{strong_signals/len(df)*100:.1f}%")
        
        with col2:
            avg_score = df['composite_score'].mean() if 'composite_score' in df.columns else 0
            st.metric("📊 Avg Score", f"{avg_score:.0f}", f"±{df['composite_score'].std():.0f}" if 'composite_score' in df.columns else "")
        
        with col3:
            high_conf = len(df[df['confidence'] >= 80]) if 'confidence' in df.columns else 0
            st.metric("🔥 High Confidence", high_conf, f"{high_conf/len(df)*100:.1f}%")
        
        with col4:
            low_risk = len(df[df['risk_level'] == 'Low']) if 'risk_level' in df.columns else 0
            st.metric("✅ Low Risk", low_risk, f"{low_risk/len(df)*100:.1f}%")

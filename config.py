"""
config.py - M.A.N.T.R.A. ULTIMATE Configuration
===============================================
Complete data utilization for maximum signal accuracy
Built for 2200+ stocks with enterprise-grade performance
"""

import streamlit as st

# =============================================================================
# GOOGLE SHEETS CONFIGURATION
# =============================================================================

SHEET_CONFIG = {
    "id": "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk",
    "sheets": {
        "watchlist": "2026492216",      # Main comprehensive watchlist
        "returns": "1234567890",        # Stock return analysis sheet  
        "sectors": "140104095"          # Sector performance analysis
    }
}

# Build direct CSV export URLs for maximum speed
DATA_URLS = {
    "watchlist": f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['id']}/export?format=csv&gid={SHEET_CONFIG['sheets']['watchlist']}",
    "returns": f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['id']}/export?format=csv&gid={SHEET_CONFIG['sheets']['returns']}",
    "sectors": f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['id']}/export?format=csv&gid={SHEET_CONFIG['sheets']['sectors']}"
}

# =============================================================================
# ULTIMATE SIGNAL SYSTEM - 7-Factor Advanced Scoring
# =============================================================================

# Signal thresholds (0-100 scale) - Tuned for maximum accuracy
SIGNALS = {
    "STRONG_BUY": 88,    # Top 5% opportunities - highest accuracy
    "BUY": 78,           # Top 15% opportunities - strong signals
    "ACCUMULATE": 68,    # Top 30% - gradual position building
    "WATCH": 55,         # Monitor for entry opportunities  
    "NEUTRAL": 40,       # No clear signal
    "AVOID": 25,         # Poor prospects
    "STRONG_AVOID": 0    # High risk situations
}

# Advanced 7-factor scoring weights (optimized for Indian markets)
WEIGHTS = {
    "momentum": 0.25,        # Multi-timeframe momentum analysis
    "value": 0.20,          # Comprehensive valuation metrics
    "growth": 0.18,         # EPS growth and earnings quality
    "volume": 0.15,         # Advanced volume pattern analysis
    "technical": 0.12,      # Technical position and trend strength
    "sector": 0.06,         # Sector rotation and relative strength
    "quality": 0.04         # Financial quality and stability
}

# =============================================================================
# COMPREHENSIVE COLUMN MAPPING - ALL DATA UTILIZATION
# =============================================================================

# Complete watchlist columns mapping
WATCHLIST_COLUMNS = {
    # Core identifiers
    "ticker": "ticker",
    "exchange": "exchange", 
    "company_name": "name",
    "sector": "sector",
    "category": "category",        # Market cap category
    
    # Price and market data
    "price": "price",
    "prev_close": "prev_close",
    "market_cap": "mcap",
    "year": "year_founded",
    
    # Complete returns data (1d to 5y)
    "ret_1d": "ret_1d",
    "ret_3d": "ret_3d", 
    "ret_7d": "ret_7d",
    "ret_30d": "ret_30d",
    "ret_3m": "ret_3m",
    "ret_6m": "ret_6m",
    "ret_1y": "ret_1y",
    "ret_3y": "ret_3y",
    "ret_5y": "ret_5y",
    
    # 52-week analysis
    "low_52w": "low_52w",
    "high_52w": "high_52w",
    "from_low_pct": "from_low_pct",
    "from_high_pct": "from_high_pct",
    
    # Moving averages
    "sma_20d": "sma20",
    "sma_50d": "sma50", 
    "sma_200d": "sma200",
    "trading_under": "trading_under",
    
    # Advanced volume analysis
    "volume_1d": "vol_1d",
    "volume_7d": "vol_7d",
    "volume_30d": "vol_30d", 
    "volume_3m": "vol_3m",
    "vol_ratio_1d_90d": "vol_ratio_1d_90d",
    "vol_ratio_7d_90d": "vol_ratio_7d_90d",
    "vol_ratio_30d_90d": "vol_ratio_30d_90d",
    "rvol": "rvol",
    
    # Comprehensive EPS data
    "pe": "pe",
    "eps_current": "eps_current",
    "eps_last_qtr": "eps_last_qtr",
    "eps_duplicate": "eps_duplicate",
    "eps_change_pct": "eps_change_pct",
    
    # Tier classifications
    "eps_tier": "eps_tier",
    "price_tier": "price_tier"
}

# Returns analysis sheet columns
RETURNS_COLUMNS = {
    "ticker": "ticker",
    "company_name": "name",
    "returns_ret_1d": "returns_1d",
    "returns_ret_3d": "returns_3d", 
    "returns_ret_7d": "returns_7d",
    "returns_ret_30d": "returns_30d",
    "returns_ret_3m": "returns_3m",
    "returns_ret_6m": "returns_6m",
    "returns_ret_1y": "returns_1y",
    "returns_ret_3y": "returns_3y",
    "returns_ret_5y": "returns_5y",
    "avg_ret_30d": "avg_ret_30d",
    "avg_ret_3m": "avg_ret_3m",
    "avg_ret_6m": "avg_ret_6m", 
    "avg_ret_1y": "avg_ret_1y",
    "avg_ret_3y": "avg_ret_3y",
    "avg_ret_5y": "avg_ret_5y"
}

# Comprehensive sector analysis
SECTOR_COLUMNS = {
    "sector": "sector",
    "sector_ret_1d": "sector_ret_1d",
    "sector_ret_3d": "sector_ret_3d",
    "sector_ret_7d": "sector_ret_7d", 
    "sector_ret_30d": "sector_ret_30d",
    "sector_ret_3m": "sector_ret_3m",
    "sector_ret_6m": "sector_ret_6m",
    "sector_ret_1y": "sector_ret_1y",
    "sector_ret_3y": "sector_ret_3y",
    "sector_ret_5y": "sector_ret_5y",
    "sector_avg_30d": "sector_avg_30d",
    "sector_avg_3m": "sector_avg_3m",
    "sector_avg_6m": "sector_avg_6m",
    "sector_avg_1y": "sector_avg_1y", 
    "sector_avg_3y": "sector_avg_3y",
    "sector_avg_5y": "sector_avg_5y",
    "sector_count": "sector_count"
}

# =============================================================================
# ADVANCED BENCHMARKS FOR MAXIMUM ACCURACY
# =============================================================================

# Multi-timeframe momentum thresholds
MOMENTUM_THRESHOLDS = {
    "excellent": {"1d": 2, "7d": 5, "30d": 15, "3m": 25},    # Strong momentum
    "good": {"1d": 1, "7d": 3, "30d": 8, "3m": 12},          # Decent momentum  
    "neutral": {"1d": 0, "7d": 0, "30d": 0, "3m": 0},        # Baseline
    "poor": {"1d": -1, "7d": -3, "30d": -8, "3m": -12},      # Weak momentum
    "terrible": {"1d": -2, "7d": -5, "30d": -15, "3m": -25}  # Strong decline
}

# Value assessment benchmarks
VALUE_BENCHMARKS = {
    "deep_value": 12,     # PE < 12 = deep value
    "fair_value": 18,     # PE 12-18 = fair value
    "growth_premium": 25, # PE 18-25 = growth premium
    "expensive": 35,      # PE 25-35 = expensive
    "overvalued": 50      # PE > 50 = significantly overvalued
}

# EPS growth quality thresholds
GROWTH_BENCHMARKS = {
    "excellent": 25,      # >25% EPS growth = excellent
    "strong": 15,         # 15-25% = strong growth
    "decent": 8,          # 8-15% = decent growth
    "weak": 0,           # 0-8% = weak growth
    "declining": -5      # <-5% = declining earnings
}

# Advanced volume analysis thresholds
VOLUME_BENCHMARKS = {
    "extreme_spike": 4.0,     # 4x+ volume = extreme interest
    "strong_spike": 2.5,      # 2.5-4x = strong spike
    "elevated": 1.5,          # 1.5-2.5x = elevated activity
    "normal": 0.8,            # 0.8-1.5x = normal range
    "weak": 0.5              # <0.5x = weak interest
}

# 52-week position strength
POSITION_BENCHMARKS = {
    "near_highs": 85,        # >85% of 52w range = near highs
    "upper_range": 65,       # 65-85% = upper range
    "middle_range": 35,      # 35-65% = middle range  
    "lower_range": 15,       # 15-35% = lower range
    "near_lows": 0          # <15% = near lows
}

# =============================================================================
# UI CONFIGURATION FOR MAXIMUM INFORMATION DENSITY
# =============================================================================

APP_CONFIG = {
    "title": "M.A.N.T.R.A.",
    "subtitle": "Market Analysis Neural Trading Research Assistant - Ultimate Edition",
    "icon": "🔱", 
    "layout": "wide",
    "version": "Ultimate 1.0",
    "max_stocks": 2200
}

# Display configuration for 2200+ stocks
DISPLAY_CONFIG = {
    "top_opportunities": 16,      # Top opportunities to highlight
    "table_rows_default": 100,   # Default table rows
    "table_rows_max": 500,       # Maximum table rows for performance
    "cache_ttl": 180,            # 3-minute cache for data freshness
    "cards_per_row": 4,          # Opportunity cards per row
    "compact_mode": True         # Dense information display
}

# Essential columns for maximum information density
ESSENTIAL_DISPLAY_COLUMNS = [
    'ticker', 'name', 'signal', 'composite_score', 'confidence',
    'price', 'ret_1d', 'ret_30d', 'ret_3m', 'from_low_pct',
    'pe', 'eps_change_pct', 'rvol', 'vol_ratio_1d_90d',
    'momentum_score', 'value_score', 'growth_score', 'volume_score',
    'sector', 'category', 'risk_level', 'position_strength'
]

# Tier classifications for filtering
TIER_CLASSIFICATIONS = {
    "eps_tiers": ["5↓", "5↑", "15↑", "35↑", "55↑", "75↑", "95↑"],
    "price_tiers": ["100↓", "100↑", "200↑", "500↑", "1K↑", "2K↑", "5K↑"],
    "categories": ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
}

# =============================================================================
# ADVANCED COLOR SCHEME FOR INFORMATION DENSITY
# =============================================================================

COLORS = {
    # Core theme
    "primary": "#00d4ff",         # Bright cyan for key elements
    "secondary": "#ff6b35",       # Orange for attention
    "success": "#00ff88",         # Bright green for positive
    "warning": "#ffb700",         # Amber for caution
    "danger": "#ff3366",          # Red for negative
    "info": "#3d7dff",           # Blue for information
    
    # Signal colors (more nuanced)
    "strong_buy": "#00ff88",      # Bright green
    "buy": "#66ff99",            # Light green
    "accumulate": "#99ddff",      # Light blue
    "watch": "#ffb700",          # Amber
    "neutral": "#888888",         # Gray
    "avoid": "#ff6666",          # Light red
    "strong_avoid": "#ff3366",    # Bright red
    
    # Background and surfaces
    "background": "#0a0a0a",      # Deep black
    "surface": "#1a1a1a",        # Dark surface
    "surface_light": "#2a2a2a",  # Lighter surface
    "border": "#333333",          # Border color
    
    # Text colors
    "text": "#ffffff",            # Primary text
    "text_secondary": "#cccccc", # Secondary text
    "text_muted": "#888888",     # Muted text
    "text_accent": "#00d4ff"     # Accent text
}

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================

PERFORMANCE_CONFIG = {
    "parallel_workers": 3,        # Parallel data loading workers
    "chunk_size": 500,           # Processing chunk size for large datasets
    "memory_limit_mb": 1000,     # Memory usage limit
    "calculation_timeout": 30,    # Calculation timeout in seconds
    "enable_profiling": False,    # Performance profiling
    "optimize_for_speed": True    # Speed vs accuracy balance
}

# =============================================================================
# QUALITY CONTROL THRESHOLDS
# =============================================================================

QUALITY_THRESHOLDS = {
    "excellent": 95,      # >95% data completeness = excellent
    "good": 85,          # 85-95% = good  
    "acceptable": 70,     # 70-85% = acceptable
    "poor": 50,          # 50-70% = poor
    "critical": 0        # <50% = critical issues
}

# Data validation rules
VALIDATION_RULES = {
    "min_price": 1,              # Minimum valid stock price
    "max_pe": 1000,             # Maximum reasonable PE ratio
    "min_volume": 1000,          # Minimum daily volume
    "max_returns": 1000,         # Maximum single-day return (%)
    "required_fields": ["ticker", "price", "sector"]  # Must-have fields
}

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

def configure_ultimate_page():
    """Configure Streamlit for ultimate performance and UX"""
    st.set_page_config(
        page_title=f"{APP_CONFIG['icon']} {APP_CONFIG['title']} Ultimate",
        page_icon=APP_CONFIG['icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state="collapsed",
        menu_items={
            'About': f"**{APP_CONFIG['title']}** {APP_CONFIG['version']}\n\n"
                    f"{APP_CONFIG['subtitle']}\n\n"
                    "🎯 Built for maximum signal accuracy with comprehensive data analysis\n"
                    "⚡ Optimized for 2200+ stocks with enterprise-grade performance\n"
                    "🔍 Advanced 7-factor scoring system for precise trading signals"
        }
    )

# =============================================================================
# ADVANCED SIGNAL CONFIDENCE CALCULATION
# =============================================================================

def calculate_signal_confidence(momentum_score, value_score, growth_score, 
                               volume_score, technical_score, sector_score, 
                               quality_score):
    """Calculate signal confidence based on factor alignment"""
    scores = [momentum_score, value_score, growth_score, volume_score, 
              technical_score, sector_score, quality_score]
    
    # Remove None values
    valid_scores = [s for s in scores if s is not None]
    
    if not valid_scores:
        return 0
    
    # Calculate standard deviation (lower = more aligned = higher confidence)
    import numpy as np
    std_dev = np.std(valid_scores)
    mean_score = np.mean(valid_scores)
    
    # Confidence calculation: high mean + low deviation = high confidence
    base_confidence = mean_score
    alignment_bonus = max(0, 20 - std_dev)  # Bonus for factor alignment
    
    confidence = min(100, base_confidence + alignment_bonus)
    return round(confidence, 1)

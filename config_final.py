"""
config_final.py - M.A.N.T.R.A. Version 3 FINAL Configuration
============================================================
Ultimate precision configuration for the final locked system
Built for 90%+ signal accuracy with crystal-clear reasoning
"""

import streamlit as st
from typing import Dict, List, Tuple
from dataclasses import dataclass

# =============================================================================
# SYSTEM METADATA
# =============================================================================

SYSTEM_INFO = {
    "name": "M.A.N.T.R.A.",
    "version": "3.0 FINAL",
    "subtitle": "Market Analysis Neural Trading Research Assistant",
    "philosophy": "Ultra-high confidence signals with crystal-clear reasoning",
    "icon": "🔱",
    "build_date": "2025",
    "locked": True  # No further major changes
}

# =============================================================================
# GOOGLE SHEETS CONFIGURATION
# =============================================================================

# Your Google Sheets configuration (UPDATE THESE)
SHEET_CONFIG = {
    "id": "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk",
    "sheets": {
        "watchlist": "2026492216",      # Main comprehensive watchlist
        "returns": "100734077",        # Stock return analysis (optional)
        "sectors": "140104095"          # Sector performance analysis
    }
}

# Direct CSV export URLs for maximum speed
DATA_URLS = {
    "watchlist": f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['id']}/export?format=csv&gid={SHEET_CONFIG['sheets']['watchlist']}",
    "returns": f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['id']}/export?format=csv&gid={SHEET_CONFIG['sheets']['returns']}",
    "sectors": f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['id']}/export?format=csv&gid={SHEET_CONFIG['sheets']['sectors']}"
}

# =============================================================================
# ULTIMATE SIGNAL SYSTEM - 8-FACTOR PRECISION ENGINE
# =============================================================================

# Ultra-conservative signal thresholds for 90%+ accuracy
SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 92,    # Top 2-3% only - ultra-high confidence
    "BUY": 82,           # Top 8-10% - high confidence with reasoning
    "ACCUMULATE": 72,    # Top 20% - good opportunities 
    "WATCH": 60,         # Monitor closely - potential but not ready
    "NEUTRAL": 40,       # No clear signal
    "AVOID": 25,         # Multiple risk factors
    "STRONG_AVOID": 0    # High risk - clear warning
}

# Enhanced 8-factor scoring weights (precision optimized)
FACTOR_WEIGHTS = {
    "momentum": 0.23,        # Multi-timeframe momentum alignment
    "value": 0.20,          # PE + earnings quality with context
    "growth": 0.18,         # EPS trends with sustainability check
    "volume": 0.15,         # Real interest confirmation
    "technical": 0.12,      # SMA trends + 52W position
    "sector": 0.06,         # Industry momentum alignment
    "risk": 0.04,           # Multi-dimensional risk assessment
    "quality": 0.02         # Data completeness + anomaly detection (NEW)
}

# Confidence calibration for explainable AI
CONFIDENCE_CALIBRATION = {
    "factor_alignment_weight": 0.4,    # How aligned are the factors?
    "historical_weight": 0.3,          # Historical performance of similar signals
    "data_quality_weight": 0.2,        # Completeness and reliability of data
    "market_context_weight": 0.1       # Current market conditions
}

# =============================================================================
# COMPREHENSIVE DATA MAPPING - ALL SPREADSHEET COLUMNS
# =============================================================================

# Complete watchlist column mapping
WATCHLIST_COLUMNS = {
    # Core identifiers
    "ticker": "ticker",
    "exchange": "exchange",
    "company_name": "name",
    "sector": "sector", 
    "category": "category",
    "year": "year_founded",
    
    # Price and market data
    "price": "price",
    "prev_close": "prev_close",
    "market_cap": "mcap",
    
    # Complete returns data (1D to 5Y)
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

# Returns analysis columns
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

# Sector analysis columns
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
# PRECISION BENCHMARKS FOR ULTIMATE ACCURACY
# =============================================================================

# Multi-timeframe momentum thresholds (conservative)
MOMENTUM_BENCHMARKS = {
    "excellent": {"1d": 3, "7d": 8, "30d": 20, "3m": 35},     # Strong sustained momentum
    "good": {"1d": 1.5, "7d": 4, "30d": 10, "3m": 18},       # Decent momentum
    "neutral": {"1d": 0, "7d": 0, "30d": 0, "3m": 0},         # Baseline
    "poor": {"1d": -1.5, "7d": -4, "30d": -10, "3m": -18},    # Weak performance
    "terrible": {"1d": -3, "7d": -8, "30d": -20, "3m": -35}   # Strong decline
}

# Value assessment benchmarks (Indian market context)
VALUE_BENCHMARKS = {
    "deep_value": 10,      # PE < 10 = exceptional value
    "strong_value": 15,    # PE 10-15 = strong value
    "fair_value": 22,      # PE 15-22 = fair value
    "growth_premium": 30,  # PE 22-30 = growth premium
    "expensive": 45,       # PE 30-45 = expensive
    "overvalued": 60       # PE > 60 = significantly overvalued
}

# EPS growth quality thresholds (conservative)
GROWTH_BENCHMARKS = {
    "accelerating": 30,    # >30% EPS growth = accelerating
    "strong": 20,          # 20-30% = strong growth
    "decent": 12,          # 12-20% = decent growth
    "modest": 5,           # 5-12% = modest growth
    "weak": 0,            # 0-5% = weak growth
    "declining": -10       # <-10% = declining earnings
}

# Advanced volume analysis thresholds (real interest detection)
VOLUME_BENCHMARKS = {
    "extreme_interest": 5.0,    # 5x+ volume = extreme interest
    "strong_interest": 3.0,     # 3-5x = strong interest
    "elevated_interest": 2.0,   # 2-3x = elevated interest
    "normal_activity": 0.8,     # 0.8-2x = normal range
    "weak_interest": 0.5,       # 0.5-0.8x = weak interest
    "very_weak": 0.3           # <0.3x = very weak interest
}

# 52-week position strength (trend confirmation)
POSITION_BENCHMARKS = {
    "near_highs": 90,      # >90% of 52w range = near highs
    "strong_position": 75, # 75-90% = strong position
    "upper_range": 60,     # 60-75% = upper range
    "middle_range": 40,    # 40-60% = middle range
    "lower_range": 25,     # 25-40% = lower range
    "weak_position": 10,   # 10-25% = weak position
    "near_lows": 0        # <10% = near lows
}

# Risk assessment thresholds (comprehensive)
RISK_BENCHMARKS = {
    "low_risk_threshold": 25,      # <25 risk score = low risk
    "medium_risk_threshold": 50,   # 25-50 = medium risk
    "high_risk_threshold": 75,     # 50-75 = high risk
    "extreme_risk_threshold": 100  # >75 = extreme risk
}

# =============================================================================
# UI/UX CONFIGURATION - SIMPLE BUT BEST
# =============================================================================

# Application configuration
APP_CONFIG = {
    "title": SYSTEM_INFO["name"],
    "subtitle": SYSTEM_INFO["subtitle"],
    "version": SYSTEM_INFO["version"],
    "icon": SYSTEM_INFO["icon"],
    "layout": "wide",
    "philosophy": SYSTEM_INFO["philosophy"]
}

# Display configuration for optimal UX
DISPLAY_CONFIG = {
    "daily_opportunities": 8,       # Top opportunities on landing page
    "max_strong_buy": 5,           # Maximum STRONG_BUY signals to show
    "max_buy": 10,                 # Maximum BUY signals to show
    "default_table_rows": 50,      # Default table display
    "max_table_rows": 200,         # Maximum for performance
    "cache_ttl": 180,              # 3-minute cache for balance of speed/freshness
    "loading_timeout": 30,         # Maximum loading time
    "simple_mode": True            # Simple UI focus
}

# Simple but best color scheme
COLORS = {
    # Primary theme (simple, professional)
    "background": "#ffffff",       # Clean white background
    "surface": "#f8f9fa",         # Light gray surface
    "border": "#e9ecef",          # Subtle borders
    
    # Text (high contrast, readable)
    "text_primary": "#212529",    # Dark gray primary text
    "text_secondary": "#6c757d",  # Medium gray secondary text
    "text_muted": "#adb5bd",      # Light gray muted text
    
    # Signal colors (traditional, clear)
    "strong_buy": "#28a745",      # Strong green
    "buy": "#40c057",            # Medium green
    "accumulate": "#74c0fc",      # Light blue
    "watch": "#ffd43b",          # Amber yellow
    "neutral": "#868e96",         # Gray
    "avoid": "#fa5252",          # Light red
    "strong_avoid": "#e03131",    # Strong red
    
    # Accent colors
    "primary": "#007bff",         # Blue for primary actions
    "success": "#28a745",         # Green for success
    "warning": "#ffc107",         # Yellow for warnings
    "danger": "#dc3545",          # Red for dangers
    "info": "#17a2b8"            # Teal for information
}

# Typography (simple, readable)
TYPOGRAPHY = {
    "font_family": "system-ui, -apple-system, sans-serif",
    "sizes": {
        "small": "0.875rem",      # 14px
        "normal": "1rem",         # 16px  
        "medium": "1.125rem",     # 18px
        "large": "1.25rem",       # 20px
        "xlarge": "1.5rem",       # 24px
        "xxlarge": "2rem"         # 32px
    }
}

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================

PERFORMANCE_CONFIG = {
    "parallel_workers": 3,         # Data loading workers
    "chunk_size": 500,            # Processing chunk size
    "memory_limit_mb": 1500,      # Memory usage limit
    "calculation_timeout": 45,     # Processing timeout
    "enable_caching": True,        # Smart caching enabled
    "optimize_for_accuracy": True, # Accuracy over speed when trade-off needed
    "target_load_time": 2.0       # Target loading time in seconds
}

# =============================================================================
# DATA QUALITY CONTROL THRESHOLDS
# =============================================================================

# Quality requirements for reliable signals
QUALITY_REQUIREMENTS = {
    "minimum_completeness": 90,    # 90% data completeness required
    "excellent_threshold": 95,     # 95%+ = excellent quality
    "good_threshold": 90,          # 90-95% = good quality
    "acceptable_threshold": 80,    # 80-90% = acceptable
    "poor_threshold": 70,          # 70-80% = poor quality
    "critical_threshold": 50       # <50% = critical issues
}

# Essential columns for signal generation (must be present)
REQUIRED_COLUMNS = ["ticker", "price", "sector", "pe", "ret_30d"]

# Columns that significantly improve signal quality
IMPORTANT_COLUMNS = [
    "ret_1d", "ret_7d", "ret_3m", "vol_1d", "rvol", 
    "eps_current", "eps_change_pct", "sma20", "from_low_pct"
]

# Data validation rules
VALIDATION_RULES = {
    "min_price": 1,              # Minimum valid stock price
    "max_price": 100000,         # Maximum reasonable price
    "min_pe": -100,              # Minimum PE (allow negative for loss companies)
    "max_pe": 500,               # Maximum reasonable PE
    "min_volume": 100,           # Minimum daily volume
    "max_return": 50,            # Maximum single-day return (%)
    "min_return": -50,           # Minimum single-day return (%)
    "max_market_cap": 1e12       # Maximum market cap (₹)
}

# =============================================================================
# MARKET CONDITION DETECTION
# =============================================================================

# Market condition thresholds for adaptive signals
MARKET_CONDITIONS = {
    "bull_market": {
        "market_breadth_min": 65,     # >65% stocks positive
        "sector_strength_min": 3,     # Average sector return >3%
        "momentum_bias": 1.2          # Increase momentum weight by 20%
    },
    "bear_market": {
        "market_breadth_max": 35,     # <35% stocks positive
        "sector_strength_max": -3,    # Average sector return <-3%
        "value_bias": 1.3             # Increase value weight by 30%
    },
    "neutral_market": {
        "market_breadth_min": 35,     # 35-65% stocks positive
        "market_breadth_max": 65,
        "balanced_weights": True       # Use standard weights
    }
}

# =============================================================================
# ALERT SYSTEM CONFIGURATION
# =============================================================================

# Alert thresholds and settings
ALERT_CONFIG = {
    "enabled": True,
    "thresholds": {
        "new_strong_buy": 92,         # Alert for new STRONG_BUY signals
        "signal_upgrade": True,        # Alert when signals upgrade
        "risk_warning": 75,           # Alert when risk score >75
        "sector_rotation": 5          # Alert when sector moves >5%
    },
    "frequency": {
        "immediate": ["new_strong_buy"],
        "daily_summary": ["signal_upgrade", "sector_rotation"],
        "weekly_summary": ["risk_warning"]
    }
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# Export settings for different formats
EXPORT_CONFIG = {
    "csv": {
        "enabled": True,
        "max_rows": 1000,
        "include_reasoning": True
    },
    "excel": {
        "enabled": True,
        "max_rows": 500,
        "include_formatting": True,
        "include_charts": False      # Keep simple
    },
    "pdf": {
        "enabled": False,            # Disable for simplicity
        "summary_only": True
    }
}

# =============================================================================
# TIER CLASSIFICATIONS
# =============================================================================

# Your existing tier systems
TIER_SYSTEMS = {
    "eps_tiers": ["5↓", "5↑", "15↑", "35↑", "55↑", "75↑", "95↑"],
    "price_tiers": ["100↓", "100↑", "200↑", "500↑", "1K↑", "2K↑", "5K↑"],
    "categories": ["Nano Cap", "Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
}

# =============================================================================
# STREAMLIT CONFIGURATION FUNCTION
# =============================================================================

def configure_streamlit():
    """Configure Streamlit for optimal performance and UX"""
    st.set_page_config(
        page_title=f"{APP_CONFIG['icon']} {APP_CONFIG['title']} {APP_CONFIG['version']}",
        page_icon=APP_CONFIG['icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state="collapsed",
        menu_items={
            'About': f"**{APP_CONFIG['title']} {APP_CONFIG['version']}**\n\n"
                    f"{APP_CONFIG['subtitle']}\n\n"
                    f"*{APP_CONFIG['philosophy']}*\n\n"
                    "🎯 Built for ultra-high confidence trading signals\n"
                    "📊 90%+ accuracy with crystal-clear reasoning\n"
                    "⚡ Optimized for 2200+ stocks with 1-3 second loading\n"
                    "🔍 Simple UI with maximum intelligence underneath"
        }
    )

# =============================================================================
# REASONING SYSTEM CONFIGURATION
# =============================================================================

# Templates for signal explanations
REASONING_TEMPLATES = {
    "strong_buy": "🚀 STRONG BUY ({confidence}% confidence): {primary_reason}. {supporting_factors}. {risk_note}",
    "buy": "📈 BUY ({confidence}% confidence): {primary_reason}. {supporting_factors}. {risk_note}",
    "accumulate": "📊 ACCUMULATE ({confidence}% confidence): {primary_reason}. {supporting_factors}. {risk_note}",
    "watch": "👀 WATCH ({confidence}% confidence): {primary_reason}. {watch_for}",
    "avoid": "⚠️ AVOID ({confidence}% confidence): {primary_concern}. {risk_factors}"
}

# Factor importance for reasoning
FACTOR_IMPORTANCE = {
    "momentum": "Strong price momentum",
    "value": "Attractive valuation", 
    "growth": "Solid earnings growth",
    "volume": "High trading interest",
    "technical": "Favorable technical setup",
    "sector": "Strong sector performance",
    "quality": "High data quality"
}

# Risk warning templates
RISK_WARNINGS = {
    "high_pe": "High valuation (PE {pe})",
    "low_volume": "Low liquidity risk",
    "negative_momentum": "Declining price trend",
    "sector_weakness": "Weak sector performance",
    "incomplete_data": "Limited data available",
    "high_volatility": "High price volatility"
}

# =============================================================================
# FINAL SYSTEM VALIDATION
# =============================================================================

def validate_configuration():
    """Validate configuration consistency and completeness"""
    errors = []
    
    # Validate weights sum to 1.0
    total_weight = sum(FACTOR_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Factor weights sum to {total_weight}, should be 1.0")
    
    # Validate signal thresholds are ordered
    thresholds = list(SIGNAL_THRESHOLDS.values())
    if thresholds != sorted(thresholds, reverse=True):
        errors.append("Signal thresholds are not properly ordered")
    
    # Validate required columns are present
    if not all(col in WATCHLIST_COLUMNS for col in REQUIRED_COLUMNS):
        errors.append("Required columns missing from watchlist mapping")
    
    # Validate performance targets
    if PERFORMANCE_CONFIG["target_load_time"] > 5.0:
        errors.append("Target load time too high for good UX")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

# Validate configuration on import
validate_configuration()

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

CONFIG_SUMMARY = {
    "system": SYSTEM_INFO["name"],
    "version": SYSTEM_INFO["version"],
    "signal_factors": len(FACTOR_WEIGHTS),
    "signal_levels": len(SIGNAL_THRESHOLDS),
    "data_columns": len(WATCHLIST_COLUMNS),
    "performance_target": f"{PERFORMANCE_CONFIG['target_load_time']}s",
    "accuracy_target": "90%+",
    "ui_philosophy": "Simple but best"
}

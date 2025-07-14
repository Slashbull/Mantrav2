"""
config_ultimate.py - M.A.N.T.R.A. Version 3 FINAL Configuration
===============================================================
Ultimate configuration for the locked forever trading system
Built for your exact Google Sheets structure
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """System configuration"""
    APP_NAME: str = "M.A.N.T.R.A."
    APP_VERSION: str = "3.0 FINAL"
    APP_SUBTITLE: str = "Market Analysis Neural Trading Research Assistant"
    APP_ICON: str = "🔱"

# =============================================================================
# DATA SOURCE CONFIGURATION - YOUR ACTUAL SHEETS
# =============================================================================

@dataclass
class DataSourceConfig:
    """Google Sheets data source configuration"""
    # Your spreadsheet ID from the URL
    SHEET_ID: str = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    
    # Sheet GIDs for each tab
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "0",          # Main watchlist sheet
        "returns": "100734077",    # Stock Return Analysis sheet  
        "sectors": "140104095"     # Sector Analysis sheet
    })
    
    # Cache settings
    CACHE_TTL: int = 180  # 3 minutes
    
    def get_sheet_url(self, sheet_name: str) -> str:
        """Get CSV export URL for a sheet"""
        gid = self.SHEET_GIDS.get(sheet_name, "0")
        return f"https://docs.google.com/spreadsheets/d/{self.SHEET_ID}/export?format=csv&gid={gid}"

# =============================================================================
# SIGNAL CONFIGURATION
# =============================================================================

@dataclass
class SignalConfig:
    """Signal generation configuration"""
    
    # Ultra-conservative thresholds
    SIGNAL_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BUY": 92,    # Top 2-3% only
        "BUY": 82,           # Top 8-10%
        "ACCUMULATE": 72,    # Top 20%
        "WATCH": 60,         # Monitor
        "NEUTRAL": 40,       # No signal
        "AVOID": 25,         # Risk present
        "STRONG_AVOID": 0    # High risk
    })
    
    # 8-Factor weights
    FACTOR_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.25,
        "value": 0.20,
        "growth": 0.18,
        "volume": 0.15,
        "technical": 0.12,
        "sector": 0.06,
        "risk": 0.03,
        "quality": 0.01
    })

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

@dataclass
class DisplayConfig:
    """Display and UI configuration"""
    
    # Display limits
    MAX_OPPORTUNITIES: int = 10
    MAX_TABLE_ROWS: int = 100
    
    # Colors (traditional)
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        "strong_buy": "#28a745",   # Dark green
        "buy": "#40c057",          # Green
        "accumulate": "#74c0fc",   # Light blue
        "watch": "#ffd43b",        # Yellow
        "neutral": "#868e96",      # Gray
        "avoid": "#fa5252",        # Light red
        "strong_avoid": "#e03131"  # Dark red
    })

# =============================================================================
# COLUMN MAPPINGS - YOUR ACTUAL COLUMNS
# =============================================================================

@dataclass
class ColumnConfig:
    """Column name mappings for your sheets"""
    
    # Watchlist columns (from your spreadsheet)
    WATCHLIST_COLUMNS: List[str] = field(default_factory=lambda: [
        'ticker', 'exchange', 'company_name', 'year', 'market_cap', 
        'category', 'sector', 'eps_tier', 'price', 'ret_1d', 
        'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d', 'trading_under',
        'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'volume_1d', 'volume_7d', 'volume_30d', 'volume_3m',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'rvol',
        'price_tier', 'prev_close', 'pe', 'eps_current', 'eps_last_qtr', 
        'eps_duplicate', 'eps_change_pct'
    ])
    
    # Returns columns
    RETURNS_COLUMNS: List[str] = field(default_factory=lambda: [
        'ticker', 'company_name',
        'returns_ret_1d', 'returns_ret_3d', 'returns_ret_7d', 'returns_ret_30d',
        'returns_ret_3m', 'returns_ret_6m', 'returns_ret_1y', 'returns_ret_3y', 'returns_ret_5y',
        'avg_ret_30d', 'avg_ret_3m', 'avg_ret_6m', 'avg_ret_1y', 'avg_ret_3y', 'avg_ret_5y'
    ])
    
    # Sector columns
    SECTOR_COLUMNS: List[str] = field(default_factory=lambda: [
        'sector',
        'sector_ret_1d', 'sector_ret_3d', 'sector_ret_7d', 'sector_ret_30d',
        'sector_ret_3m', 'sector_ret_6m', 'sector_ret_1y', 'sector_ret_3y', 'sector_ret_5y',
        'sector_avg_30d', 'sector_avg_3m', 'sector_avg_6m', 
        'sector_avg_1y', 'sector_avg_3y', 'sector_avg_5y',
        'sector_count'
    ])
    
    # Column mappings for easier access
    COLUMN_MAPPINGS: Dict[str, str] = field(default_factory=lambda: {
        'name': 'company_name',
        'sma20': 'sma_20d',
        'sma50': 'sma_50d',
        'sma200': 'sma_200d',
        'vol_1d': 'volume_1d',
        'vol_7d': 'volume_7d',
        'vol_30d': 'volume_30d'
    })

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

@dataclass
class PerformanceConfig:
    """Performance settings"""
    TARGET_LOAD_TIME: float = 2.0  # seconds
    MAX_STOCKS: int = 5000
    BATCH_SIZE: int = 500
    ENABLE_CACHING: bool = True

# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """Main configuration class"""
    system: SystemConfig = field(default_factory=SystemConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

# Create global config instance
CONFIG = Config()

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

def configure_streamlit():
    """Configure Streamlit settings"""
    st.set_page_config(
        page_title=f"{CONFIG.system.APP_ICON} {CONFIG.system.APP_NAME} {CONFIG.system.APP_VERSION}",
        page_icon=CONFIG.system.APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# =============================================================================
# EXPORT
# =============================================================================

__all__ = ['CONFIG', 'configure_streamlit']

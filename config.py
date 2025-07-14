"""
config.py - M.A.N.T.R.A. Version 3 FINAL Configuration
======================================================
Ultimate configuration with proven sheet GIDs
Built from working production code
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple
import warnings

@dataclass
class Config:
    """Configuration for M.A.N.T.R.A. - proven in production"""
    
    # =========================================================================
    # DATA SOURCES (From working code - DO NOT CHANGE)
    # =========================================================================
    BASE_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "2026492216",  # Main watchlist sheet
        "returns": "100734077",      # Stock Return Analysis sheet
        "sector": "140104095"        # Sector Analysis sheet
    })
    
    # =========================================================================
    # REQUIRED COLUMNS (Exact from your sheets)
    # =========================================================================
    REQUIRED_WATCHLIST: Set[str] = field(default_factory=lambda: {
        "ticker", "exchange", "company_name", "year", "market_cap", "category", "sector",
        "eps_tier", "price", "prev_close", "ret_1d", "low_52w", "high_52w",
        "from_low_pct", "from_high_pct", "sma_20d", "sma_50d", "sma_200d",
        "trading_under", "ret_3d", "ret_7d", "ret_30d", "ret_3m", "ret_6m",
        "ret_1y", "ret_3y", "ret_5y", "volume_1d", "volume_7d", "volume_30d",
        "volume_3m", "vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d",
        "rvol", "price_tier", "pe", "eps_current", "eps_last_qtr", "eps_duplicate", 
        "eps_change_pct"
    })
    
    REQUIRED_RETURNS: Set[str] = field(default_factory=lambda: {
        "ticker", "company_name",
        "avg_ret_30d", "avg_ret_3m", "avg_ret_6m", "avg_ret_1y", "avg_ret_3y", "avg_ret_5y"
    })
    
    REQUIRED_SECTOR: Set[str] = field(default_factory=lambda: {
        "sector", "sector_ret_1d", "sector_ret_3d", "sector_ret_7d", "sector_ret_30d",
        "sector_ret_3m", "sector_ret_6m", "sector_ret_1y", "sector_ret_3y", "sector_ret_5y",
        "sector_avg_30d", "sector_avg_3m", "sector_avg_6m", "sector_avg_1y",
        "sector_avg_3y", "sector_avg_5y", "sector_count"
    })
    
    # =========================================================================
    # SIGNAL CONFIGURATION (Ultra-conservative as requested)
    # =========================================================================
    SIGNAL_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BUY": 92,    # Top 2-3% only
        "BUY": 82,           # Top 8-10%
        "ACCUMULATE": 72,    # Top 20%
        "WATCH": 60,         # Monitor
        "NEUTRAL": 40,       # No signal
        "AVOID": 25,         # Risk present
        "STRONG_AVOID": 0    # High risk
    })
    
    # 8-Factor weights (your specification)
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
    
    # =========================================================================
    # PERFORMANCE SETTINGS
    # =========================================================================
    CACHE_TTL: int = 300  # 5 minutes cache
    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    TARGET_LOAD_TIME: float = 2.0  # seconds
    
    # =========================================================================
    # DATA QUALITY THRESHOLDS
    # =========================================================================
    MIN_DATA_QUALITY_SCORE: float = 70.0
    MAX_NULL_PERCENTAGE: float = 20.0
    OUTLIER_THRESHOLD: float = 3.0  # z-score
    
    # Critical fields that must have data
    CRITICAL_FIELDS: Tuple[str, ...] = ("ticker", "price", "eps_current", "sector")
    
    # =========================================================================
    # DISPLAY SETTINGS
    # =========================================================================
    MAX_OPPORTUNITIES: int = 10  # Top opportunities to show
    MAX_TABLE_ROWS: int = 100   # Max rows in tables
    
    # Traditional colors as requested
    SIGNAL_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "STRONG_BUY": "#28a745",   # Dark green
        "BUY": "#40c057",          # Green
        "ACCUMULATE": "#74c0fc",   # Light blue
        "WATCH": "#ffd43b",        # Yellow
        "NEUTRAL": "#868e96",      # Gray
        "AVOID": "#fa5252",        # Light red
        "STRONG_AVOID": "#e03131"  # Dark red
    })
    
    # =========================================================================
    # SYSTEM SETTINGS
    # =========================================================================
    APP_NAME: str = "M.A.N.T.R.A."
    APP_VERSION: str = "3.0 FINAL"
    APP_SUBTITLE: str = "Market Analysis Neural Trading Research Assistant"
    APP_ICON: str = "🔱"
    SCHEMA_VERSION: str = "2025.01.14"
    
    def get_sheet_url(self, name: str) -> str:
        """Get Google Sheets export URL for a sheet"""
        if name not in self.SHEET_GIDS:
            raise ValueError(f"Unknown sheet: {name}")
        return f"{self.BASE_URL}/export?format=csv&gid={self.SHEET_GIDS[name]}"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.CACHE_TTL < 60:
            warnings.warn(f"Cache TTL {self.CACHE_TTL}s is very low, consider increasing")
        
        if self.MIN_DATA_QUALITY_SCORE < 50:
            raise ValueError("MIN_DATA_QUALITY_SCORE must be >= 50")
        
        # Ensure factor weights sum to 1
        weight_sum = sum(self.FACTOR_WEIGHTS.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Factor weights must sum to 1.0, got {weight_sum}")

# Global configuration instance
CONFIG = Config()

# Export
__all__ = ['Config', 'CONFIG']

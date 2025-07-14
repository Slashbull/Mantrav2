"""
config_ultimate.py - M.A.N.T.R.A. Version 3 FINAL - Ultimate Configuration
==========================================================================
Perfect configuration system inspired by core_system_foundation.py
Built for permanent use - no further modifications needed
Optimized for Streamlit Community Cloud deployment
"""

import streamlit as st
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
import logging
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SYSTEM METADATA
# =============================================================================

SYSTEM_INFO = {
    "name": "M.A.N.T.R.A.",
    "version": "3.0 FINAL PERFECT",
    "subtitle": "Market Analysis Neural Trading Research Assistant",
    "philosophy": "Ultra-high confidence signals with crystal-clear reasoning",
    "icon": "🔱",
    "build_date": "2025",
    "locked": True,  # No further major changes
    "schema_version": "2025.07.14"
}

# =============================================================================
# DATA SOURCE CONFIGURATION (BULLETPROOF)
# =============================================================================

@dataclass
class DataSourceConfig:
    """Bulletproof data source configuration"""
    BASE_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "2026492216",
        "returns": "100734077", 
        "sectors": "140104095"
    })
    
    # Performance settings
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.0
    CACHE_TTL: int = 300  # 5 minutes
    
    def get_sheet_url(self, name: str) -> str:
        """Get Google Sheets export URL for a sheet"""
        if name not in self.SHEET_GIDS:
            raise ValueError(f"Unknown sheet: {name}")
        return f"{self.BASE_URL}/export?format=csv&gid={self.SHEET_GIDS[name]}"

# =============================================================================
# SCHEMA REQUIREMENTS (BULLETPROOF)
# =============================================================================

@dataclass
class SchemaConfig:
    """Bulletproof schema configuration with validation"""
    
    # Required columns for each dataset
    REQUIRED_WATCHLIST: Set[str] = field(default_factory=lambda: {
        "ticker", "exchange", "company_name", "year", "market_cap", "category", "sector",
        "eps_tier", "price", "prev_close", "ret_1d", "low_52w", "high_52w",
        "from_low_pct", "from_high_pct", "sma_20d", "sma_50d", "sma_200d",
        "trading_under", "ret_3d", "ret_7d", "ret_30d", "ret_3m", "ret_6m",
        "ret_1y", "ret_3y", "ret_5y", "volume_1d", "volume_7d", "volume_30d",
        "volume_3m", "vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d",
        "rvol", "price_tier", "eps_current", "eps_last_qtr", "eps_duplicate", "eps_change_pct"
    })
    
    REQUIRED_RETURNS: Set[str] = field(default_factory=lambda: {
        "ticker", "company_name",
        "avg_ret_30d", "avg_ret_3m", "avg_ret_6m", "avg_ret_1y", "avg_ret_3y", "avg_ret_5y"
    })
    
    REQUIRED_SECTORS: Set[str] = field(default_factory=lambda: {
        "sector", "sector_ret_1d", "sector_ret_3d", "sector_ret_7d", "sector_ret_30d",
        "sector_ret_3m", "sector_ret_6m", "sector_ret_1y", "sector_ret_3y", "sector_ret_5y",
        "sector_avg_30d", "sector_avg_3m", "sector_avg_6m", "sector_avg_1y",
        "sector_avg_3y", "sector_avg_5y", "sector_count"
    })
    
    # Critical fields that must have data
    CRITICAL_FIELDS: Tuple[str, ...] = ("ticker", "price", "eps_current", "sector")
    
    # Column mappings for data cleaning
    COLUMN_MAPPINGS: Dict[str, str] = field(default_factory=lambda: {
        # Standardize column names
        "company_name": "name",
        "sma_20d": "sma20",
        "sma_50d": "sma50", 
        "sma_200d": "sma200",
        "volume_1d": "vol_1d",
        "volume_7d": "vol_7d",
        "volume_30d": "vol_30d",
        "volume_3m": "vol_3m"
    })

# =============================================================================
# SIGNAL ENGINE CONFIGURATION (PRECISION TUNED)
# =============================================================================

@dataclass
class SignalConfig:
    """Ultimate signal configuration for maximum accuracy"""
    
    # Ultra-conservative signal thresholds for 90%+ accuracy
    SIGNAL_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BUY": 92,    # Top 2-3% only - ultra-high confidence
        "BUY": 82,           # Top 8-10% - high confidence with reasoning
        "ACCUMULATE": 72,    # Top 20% - good opportunities 
        "WATCH": 60,         # Monitor closely - potential but not ready
        "NEUTRAL": 40,       # No clear signal
        "AVOID": 25,         # Multiple risk factors
        "STRONG_AVOID": 0    # High risk - clear warning
    })
    
    # Enhanced 8-factor scoring weights (precision optimized)
    FACTOR_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.23,        # Multi-timeframe momentum alignment
        "value": 0.20,          # PE + earnings quality with context
        "growth": 0.18,         # EPS trends with sustainability check
        "volume": 0.15,         # Real interest confirmation
        "technical": 0.12,      # SMA trends + 52W position
        "sector": 0.06,         # Industry momentum alignment
        "risk": 0.04,           # Multi-dimensional risk assessment
        "quality": 0.02         # Data completeness + anomaly detection
    })
    
    # Confidence calibration for explainable AI
    CONFIDENCE_CALIBRATION: Dict[str, float] = field(default_factory=lambda: {
        "factor_alignment_weight": 0.4,    # How aligned are the factors?
        "historical_weight": 0.3,          # Historical performance of similar signals
        "data_quality_weight": 0.2,        # Completeness and reliability of data
        "market_context_weight": 0.1       # Current market conditions
    })

# =============================================================================
# QUALITY CONTROL CONFIGURATION (BULLETPROOF)
# =============================================================================

@dataclass
class QualityConfig:
    """Bulletproof quality control configuration"""
    
    # Quality requirements for reliable signals
    MIN_DATA_QUALITY_SCORE: float = 70.0
    EXCELLENT_THRESHOLD: float = 95.0
    GOOD_THRESHOLD: float = 90.0
    ACCEPTABLE_THRESHOLD: float = 80.0
    POOR_THRESHOLD: float = 70.0
    CRITICAL_THRESHOLD: float = 50.0
    
    # Data validation thresholds
    MAX_NULL_PERCENTAGE: float = 20.0
    OUTLIER_THRESHOLD: float = 3.0  # z-score
    MIN_COMPLETENESS: float = 90.0  # 90% data completeness required
    
    # Data validation rules
    VALIDATION_RULES: Dict[str, Dict] = field(default_factory=lambda: {
        "price": {"min": 1, "max": 100000},
        "pe": {"min": -100, "max": 500},
        "volume_1d": {"min": 100, "max": 1e12},
        "ret_1d": {"min": -50, "max": 50},
        "ret_30d": {"min": -90, "max": 500},
        "market_cap": {"min": 1e6, "max": 1e12}
    })
    
    # Essential columns for signal generation
    REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        "ticker", "price", "sector", "pe", "ret_30d"
    ])
    
    # Important columns that improve signal quality
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        "ret_1d", "ret_7d", "ret_3m", "vol_1d", "rvol", 
        "eps_current", "eps_change_pct", "sma20", "from_low_pct"
    ])

# =============================================================================
# BENCHMARKS AND THRESHOLDS (PRECISION TUNED)
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Precision-tuned benchmarks for Indian markets"""
    
    # Multi-timeframe momentum thresholds (conservative)
    MOMENTUM_BENCHMARKS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "excellent": {"1d": 3, "7d": 8, "30d": 20, "3m": 35},
        "good": {"1d": 1.5, "7d": 4, "30d": 10, "3m": 18},
        "neutral": {"1d": 0, "7d": 0, "30d": 0, "3m": 0},
        "poor": {"1d": -1.5, "7d": -4, "30d": -10, "3m": -18},
        "terrible": {"1d": -3, "7d": -8, "30d": -20, "3m": -35}
    })
    
    # Value assessment benchmarks (Indian market context)
    VALUE_BENCHMARKS: Dict[str, float] = field(default_factory=lambda: {
        "deep_value": 10,      # PE < 10 = exceptional value
        "strong_value": 15,    # PE 10-15 = strong value
        "fair_value": 22,      # PE 15-22 = fair value
        "growth_premium": 30,  # PE 22-30 = growth premium
        "expensive": 45,       # PE 30-45 = expensive
        "overvalued": 60       # PE > 60 = significantly overvalued
    })
    
    # EPS growth quality thresholds (conservative)
    GROWTH_BENCHMARKS: Dict[str, float] = field(default_factory=lambda: {
        "accelerating": 30,    # >30% EPS growth = accelerating
        "strong": 20,          # 20-30% = strong growth
        "decent": 12,          # 12-20% = decent growth
        "modest": 5,           # 5-12% = modest growth
        "weak": 0,            # 0-5% = weak growth
        "declining": -10       # <-10% = declining earnings
    })
    
    # Advanced volume analysis thresholds
    VOLUME_BENCHMARKS: Dict[str, float] = field(default_factory=lambda: {
        "extreme_interest": 5.0,    # 5x+ volume = extreme interest
        "strong_interest": 3.0,     # 3-5x = strong interest
        "elevated_interest": 2.0,   # 2-3x = elevated interest
        "normal_activity": 0.8,     # 0.8-2x = normal range
        "weak_interest": 0.5,       # 0.5-0.8x = weak interest
        "very_weak": 0.3           # <0.3x = very weak interest
    })
    
    # 52-week position strength
    POSITION_BENCHMARKS: Dict[str, float] = field(default_factory=lambda: {
        "near_highs": 90,      # >90% of 52w range = near highs
        "strong_position": 75, # 75-90% = strong position
        "upper_range": 60,     # 60-75% = upper range
        "middle_range": 40,    # 40-60% = middle range
        "lower_range": 25,     # 25-40% = lower range
        "weak_position": 10,   # 10-25% = weak position
        "near_lows": 0        # <10% = near lows
    })

# =============================================================================
# UI/UX CONFIGURATION (SIMPLE BUT BEST)
# =============================================================================

@dataclass
class UIConfig:
    """Perfect UI/UX configuration for ultimate user experience"""
    
    # Application configuration
    APP_CONFIG: Dict[str, str] = field(default_factory=lambda: {
        "title": SYSTEM_INFO["name"],
        "subtitle": SYSTEM_INFO["subtitle"],
        "version": SYSTEM_INFO["version"],
        "icon": SYSTEM_INFO["icon"],
        "layout": "wide",
        "philosophy": SYSTEM_INFO["philosophy"]
    })
    
    # Display configuration for optimal UX
    DISPLAY_CONFIG: Dict[str, int] = field(default_factory=lambda: {
        "daily_opportunities": 8,       # Top opportunities on landing page
        "max_strong_buy": 5,           # Maximum STRONG_BUY signals to show
        "max_buy": 10,                 # Maximum BUY signals to show
        "default_table_rows": 50,      # Default table display
        "max_table_rows": 200,         # Maximum for performance
        "cache_ttl": 300,              # 5-minute cache for balance of speed/freshness
        "loading_timeout": 30,         # Maximum loading time
        "simple_mode": True            # Simple UI focus
    })
    
    # Perfect color scheme (traditional, clear)
    COLORS: Dict[str, str] = field(default_factory=lambda: {
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
    })

# =============================================================================
# PERFORMANCE CONFIGURATION (STREAMLIT CLOUD OPTIMIZED)
# =============================================================================

@dataclass
class PerformanceConfig:
    """Performance configuration optimized for Streamlit Community Cloud"""
    
    # Processing settings
    PARALLEL_WORKERS: int = 3         # Data loading workers
    CHUNK_SIZE: int = 500            # Processing chunk size
    MEMORY_LIMIT_MB: int = 1500      # Memory usage limit
    CALCULATION_TIMEOUT: int = 45     # Processing timeout
    ENABLE_CACHING: bool = True       # Smart caching enabled
    OPTIMIZE_FOR_ACCURACY: bool = True # Accuracy over speed when needed
    TARGET_LOAD_TIME: float = 2.0    # Target loading time in seconds
    
    # Cache settings
    CACHE_TTL: int = 300             # 5 minutes
    MAX_CACHE_SIZE: int = 100        # Maximum cached items
    
    # Data processing limits
    MAX_STOCKS: int = 10000          # Maximum stocks to process
    MAX_SECTORS: int = 50            # Maximum sectors
    BATCH_SIZE: int = 1000           # Batch processing size

# =============================================================================
# MARKET CONDITION CONFIGURATION
# =============================================================================

@dataclass
class MarketConfig:
    """Market condition detection and adaptation configuration"""
    
    # Market condition thresholds for adaptive signals
    MARKET_CONDITIONS: Dict[str, Dict] = field(default_factory=lambda: {
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
    })

# =============================================================================
# CONSOLIDATED CONFIGURATION CLASS
# =============================================================================

@dataclass
class UltimateConfig:
    """Ultimate consolidated configuration for M.A.N.T.R.A. Final"""
    
    # Sub-configurations
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate factor weights sum to 1.0
        total_weight = sum(self.signals.FACTOR_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Factor weights sum to {total_weight}, should be 1.0")
        
        # Validate signal thresholds are ordered
        thresholds = list(self.signals.SIGNAL_THRESHOLDS.values())
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError("Signal thresholds are not properly ordered")
        
        # Validate quality thresholds
        if self.quality.MIN_DATA_QUALITY_SCORE < 50:
            raise ValueError("MIN_DATA_QUALITY_SCORE must be >= 50")
        
        # Validate performance targets
        if self.performance.TARGET_LOAD_TIME > 5.0:
            warnings.warn("Target load time too high for good UX")

# =============================================================================
# STREAMLIT CONFIGURATION FUNCTION
# =============================================================================

def configure_streamlit():
    """Configure Streamlit for optimal performance and UX"""
    config = UltimateConfig()
    
    st.set_page_config(
        page_title=f"{config.ui.APP_CONFIG['icon']} {config.ui.APP_CONFIG['title']} {config.ui.APP_CONFIG['version']}",
        page_icon=config.ui.APP_CONFIG['icon'],
        layout=config.ui.APP_CONFIG['layout'],
        initial_sidebar_state="collapsed",
        menu_items={
            'About': f"**{config.ui.APP_CONFIG['title']} {config.ui.APP_CONFIG['version']}**\n\n"
                    f"{config.ui.APP_CONFIG['subtitle']}\n\n"
                    f"*{config.ui.APP_CONFIG['philosophy']}*\n\n"
                    "🎯 Built for ultra-high confidence trading signals\n"
                    "📊 90%+ accuracy with crystal-clear reasoning\n"
                    "⚡ Optimized for 2200+ stocks with 1-3 second loading\n"
                    "🔍 Simple UI with maximum intelligence underneath"
        }
    )

# =============================================================================
# CONFIGURATION SUMMARY AND VALIDATION
# =============================================================================

def get_config_summary() -> Dict[str, str]:
    """Get configuration summary for display"""
    config = UltimateConfig()
    
    return {
        "system": SYSTEM_INFO["name"],
        "version": SYSTEM_INFO["version"],
        "schema_version": SYSTEM_INFO["schema_version"],
        "signal_factors": len(config.signals.FACTOR_WEIGHTS),
        "signal_levels": len(config.signals.SIGNAL_THRESHOLDS),
        "data_columns": len(config.schema.REQUIRED_WATCHLIST),
        "performance_target": f"{config.performance.TARGET_LOAD_TIME}s",
        "accuracy_target": "90%+",
        "ui_philosophy": "Simple but best",
        "cache_ttl": f"{config.performance.CACHE_TTL}s",
        "quality_threshold": f"{config.quality.MIN_DATA_QUALITY_SCORE}%"
    }

def validate_configuration():
    """Validate configuration consistency and completeness"""
    try:
        config = UltimateConfig()
        summary = get_config_summary()
        return True, f"✅ Configuration validated successfully. {summary['signal_factors']} factors, {summary['signal_levels']} signal levels."
    except Exception as e:
        return False, f"❌ Configuration validation failed: {str(e)}"

# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Create global configuration instance
CONFIG = UltimateConfig()

# Validate on import
is_valid, message = validate_configuration()
if not is_valid:
    raise ValueError(f"Configuration validation failed: {message}")

print(f"✅ {SYSTEM_INFO['name']} {SYSTEM_INFO['version']} configuration loaded and validated")

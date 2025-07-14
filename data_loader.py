"""
data_loader.py - M.A.N.T.R.A. Version 3 FINAL Data Loader
=========================================================
Bulletproof data loading based on proven production code
Handles Indian market data with all edge cases
"""

import io
import re
import time
import logging
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

from config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [DATA] %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HTTP SESSION WITH RETRY
# ============================================================================

def create_session() -> requests.Session:
    """Create HTTP session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=CONFIG.MAX_RETRIES,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session

# Global session for efficiency
_session = None

def get_session() -> requests.Session:
    """Get or create global session"""
    global _session
    if _session is None:
        _session = create_session()
    return _session

# ============================================================================
# SIMPLE CACHE
# ============================================================================

class SimpleCache:
    """Simple TTL-based in-memory cache"""
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp"""
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()

# Global cache instance
_cache = SimpleCache()

# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize dataframe columns"""
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    
    # Clean column names - convert to lowercase and replace spaces with underscores
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    
    # Remove hidden characters
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace("\u00A0", " ", regex=False)
    
    return df

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """Clean and convert a numeric series - handles Indian currency and percentages"""
    # Convert to string and clean
    s = series.astype(str)
    
    # Remove currency symbols and units (including Indian symbols)
    for symbol in ['₹', '$', '€', '£', 'Cr', 'L', 'K', 'M', 'B', '%', ',', '↑', '↓']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Remove non-ASCII characters
    s = s.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
    
    # Handle empty strings
    s = s.replace('', 'NaN')
    
    # Convert to numeric
    numeric_series = pd.to_numeric(s, errors='coerce')
    
    # Handle percentage columns
    if col_name.endswith('_pct') or '%' in series.astype(str).str.cat():
        return numeric_series
    
    # Auto-detect if values need scaling
    non_null = numeric_series.dropna()
    if len(non_null) > 0 and non_null.max() < 1 and non_null.min() >= 0:
        # Likely percentages stored as decimals
        return numeric_series * 100
    
    return numeric_series

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all numeric columns in dataframe"""
    numeric_pattern = re.compile(
        r"^(price|prev_close|ret_|avg_ret|volume|vol_ratio|"
        r"low_52w|high_52w|from_low_pct|from_high_pct|pe|eps|rvol|"
        r"market_cap|sma_|dma_|sector_ret_|sector_avg_)"
    )
    
    for col in df.columns:
        if numeric_pattern.match(col):
            df[col] = clean_numeric_series(df[col], col)
    
    return df

# ============================================================================
# DATA LOADING
# ============================================================================

def load_sheet(name: str, use_cache: bool = True) -> pd.DataFrame:
    """Load a single sheet from Google Sheets"""
    url = CONFIG.get_sheet_url(name)
    cache_key = f"sheet_{name}_{CONFIG.SCHEMA_VERSION}"
    
    # Check cache
    if use_cache:
        cached = _cache.get(cache_key, CONFIG.CACHE_TTL)
        if cached is not None:
            logger.info(f"✅ Cache hit for sheet '{name}'")
            return cached
    
    # Fetch from remote
    logger.info(f"📥 Loading sheet '{name}' from Google Sheets...")
    try:
        session = get_session()
        response = session.get(url, timeout=CONFIG.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean dataframe
        df = clean_dataframe(df)
        
        # Cache result
        if use_cache:
            _cache.set(cache_key, df)
        
        logger.info(f"✅ Loaded {len(df)} rows from '{name}'")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to load sheet '{name}': {e}")
        raise Exception(f"Cannot load sheet '{name}' from Google Sheets")

# ============================================================================
# DATA PROCESSING
# ============================================================================

def validate_schema(df: pd.DataFrame, required_cols: set, sheet_name: str) -> None:
    """Validate dataframe has required columns"""
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        # Log warning but don't fail - be flexible
        logger.warning(f"⚠️ Missing columns in {sheet_name}: {missing_cols}")

def merge_datasets(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Merge watchlist and returns data"""
    # Drop duplicate columns
    returns_df = returns_df.drop(columns=['company_name'], errors='ignore')
    
    # Merge on ticker
    merged = watchlist_df.merge(
        returns_df,
        on='ticker',
        how='left',
        validate='one_to_one'
    )
    
    # Handle duplicates
    if merged['ticker'].duplicated().any():
        dup_count = merged['ticker'].duplicated().sum()
        logger.warning(f"⚠️ Found {dup_count} duplicate tickers, keeping last")
        merged = merged.drop_duplicates(subset='ticker', keep='last')
    
    # Normalize ticker
    merged['ticker'] = merged['ticker'].astype(str).str.upper().str.strip()
    
    return merged

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for signal generation"""
    
    # 52-week position
    if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
        price = df['price']
        low_52w = df['low_52w']
        high_52w = df['high_52w']
        
        range_52w = high_52w - low_52w
        df['position_52w'] = np.where(range_52w > 0, 
                                      (price - low_52w) / range_52w * 100, 
                                      50).round(1)
    
    # Moving average positions
    for ma in ['20d', '50d', '200d']:
        ma_col = f'sma_{ma}'
        if ma_col in df.columns and 'price' in df.columns:
            df[f'above_sma_{ma}'] = (df['price'] > df[ma_col]).astype(int)
            df[f'pct_from_sma_{ma}'] = ((df['price'] - df[ma_col]) / df[ma_col] * 100).round(1)
    
    # Momentum consistency
    momentum_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m']
    available_momentum = [col for col in momentum_cols if col in df.columns]
    
    if len(available_momentum) >= 2:
        positive_counts = pd.Series(0, index=df.index)
        for col in available_momentum:
            positive_counts += (df[col] > 0).astype(int)
        df['momentum_breadth'] = (positive_counts / len(available_momentum) * 100).round(0)
    
    # Volume spike indicator
    if 'rvol' in df.columns:
        df['volume_spike'] = (df['rvol'] > 2).astype(int)
    
    # EPS growth quality
    if 'eps_change_pct' in df.columns:
        df['eps_growing'] = (df['eps_change_pct'] > 0).astype(int)
    
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage"""
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert low-cardinality object columns to category
    for col in ['sector', 'category', 'exchange', 'price_tier', 'eps_tier']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    return df

# ============================================================================
# DATA QUALITY
# ============================================================================

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data quality and generate metrics"""
    
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Null analysis
    null_counts = df.isnull().sum()
    null_percentage = (null_counts.sum() / (len(df) * len(df.columns)) * 100).round(2)
    
    analysis['null_percentage'] = float(null_percentage)
    analysis['columns_with_nulls'] = int((null_counts > 0).sum())
    
    # Duplicate analysis
    analysis['duplicate_tickers'] = int(df['ticker'].duplicated().sum()) if 'ticker' in df else 0
    
    # Calculate overall quality score
    quality_score = 100.0
    quality_score -= min(50, null_percentage * 2)
    quality_score -= min(20, analysis['duplicate_tickers'] * 0.5)
    
    analysis['quality_score'] = max(0, quality_score)
    analysis['quality_grade'] = (
        'A' if quality_score >= 90 else
        'B' if quality_score >= 80 else
        'C' if quality_score >= 70 else
        'D' if quality_score >= 60 else
        'F'
    )
    
    return analysis

# ============================================================================
# MAIN LOADER
# ============================================================================

class DataLoader:
    """Main data loader class for M.A.N.T.R.A."""
    
    def __init__(self):
        self.config = CONFIG
        self.stocks_df = None
        self.sector_df = None
        self.health = None
        
    def load_and_process(self, use_cache: bool = True) -> Tuple[bool, str]:
        """
        Load and process all data
        
        Returns:
            Tuple of (success, message)
        """
        start_time = time.time()
        
        try:
            # Load sheets
            logger.info("🚀 Starting data load...")
            watchlist_df = load_sheet('watchlist', use_cache)
            returns_df = load_sheet('returns', use_cache)
            self.sector_df = load_sheet('sector', use_cache)
            
            # Validate schemas (but don't fail)
            validate_schema(watchlist_df, self.config.REQUIRED_WATCHLIST, 'watchlist')
            validate_schema(returns_df, self.config.REQUIRED_RETURNS, 'returns')
            validate_schema(self.sector_df, self.config.REQUIRED_SECTOR, 'sector')
            
            # Merge datasets
            logger.info("🔄 Merging datasets...")
            self.stocks_df = merge_datasets(watchlist_df, returns_df)
            
            # Clean numeric columns
            logger.info("🧹 Cleaning data...")
            self.stocks_df = clean_numeric_columns(self.stocks_df)
            self.sector_df = clean_numeric_columns(self.sector_df)
            
            # Add derived features
            logger.info("🔧 Adding features...")
            self.stocks_df = add_derived_features(self.stocks_df)
            
            # Optimize memory
            self.stocks_df = optimize_dtypes(self.stocks_df)
            self.sector_df = optimize_dtypes(self.sector_df)
            
            # Analyze quality
            quality_analysis = analyze_data_quality(self.stocks_df)
            
            # Build health report
            self.health = {
                'processing_time_s': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat(),
                'quality_analysis': quality_analysis,
                'total_stocks': len(self.stocks_df),
                'total_sectors': len(self.sector_df),
                'cache_used': use_cache
            }
            
            logger.info(f"✅ Data loaded successfully in {self.health['processing_time_s']:.2f}s")
            logger.info(f"📊 {len(self.stocks_df)} stocks, quality score: {quality_analysis['quality_score']:.1f}")
            
            return True, f"Loaded {len(self.stocks_df)} stocks successfully"
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return False, f"Data loading failed: {str(e)}"
    
    def get_stocks_data(self) -> pd.DataFrame:
        """Get stocks dataframe"""
        return self.stocks_df if self.stocks_df is not None else pd.DataFrame()
    
    def get_sector_data(self) -> pd.DataFrame:
        """Get sector dataframe"""
        return self.sector_df if self.sector_df is not None else pd.DataFrame()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health report"""
        return self.health if self.health is not None else {}
    
    def clear_cache(self):
        """Clear all cached data"""
        _cache.clear()
        logger.info("🗑️ Cache cleared")

# Export
__all__ = ['DataLoader']

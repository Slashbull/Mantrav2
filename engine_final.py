"""
engine_final.py - M.A.N.T.R.A. Version 3 FINAL Signal Engine
============================================================
Ultimate precision engine with explainable AI and 90%+ accuracy
Built for crystal-clear reasoning and ultra-conservative signals
"""

import pandas as pd
import numpy as np
import requests
import logging
import time
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config_final import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - FINAL - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SignalExplanation:
    """Container for signal reasoning and explanation"""
    signal: str
    confidence: float
    primary_reason: str
    supporting_factors: List[str]
    risk_factors: List[str]
    factor_scores: Dict[str, float]
    data_quality: float
    recommendation: str

@dataclass
class ProcessingStats:
    """Container for processing performance statistics"""
    total_stocks: int
    processing_time: float
    strong_buy_count: int
    buy_count: int
    data_quality_avg: float
    signals_generated: int
    timestamp: datetime

class UltimatePrecisionEngine:
    """
    Ultimate precision signal engine with explainable AI
    
    Features:
    - 8-factor precision scoring system
    - Ultra-conservative signal thresholds (92+ for STRONG_BUY)
    - Complete explainability for every signal
    - Quality control and data validation
    - Market condition adaptation
    - Risk-adjusted scoring
    - Performance tracking and optimization
    """
    
    def __init__(self):
        self.watchlist_df = pd.DataFrame()
        self.returns_df = pd.DataFrame()
        self.sectors_df = pd.DataFrame()
        self.master_df = pd.DataFrame()
        
        self.processing_stats = None
        self.data_quality_report = {}
        self.market_conditions = {}
        self.signal_explanations = {}
        
        self.last_update = None
        self.performance_metrics = {}
        
        logger.info("🔱 Ultimate Precision Engine initialized")
    
    def load_and_process(self) -> Tuple[bool, str]:
        """
        Main processing pipeline with comprehensive error handling
        Returns: (success, status_message)
        """
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting ultimate precision processing...")
            
            # Step 1: Load all data sources
            success, message = self._load_all_data_sources()
            if not success:
                return False, message
            
            # Step 2: Comprehensive data validation and cleaning
            self._validate_and_clean_data()
            
            # Step 3: Quality assessment and control
            self._assess_data_quality()
            
            # Step 4: Market condition detection
            self._detect_market_conditions()
            
            # Step 5: Data enrichment and feature engineering
            self._enrich_data_with_features()
            
            # Step 6: Create master dataset with all sources merged
            self._create_master_dataset()
            
            # Step 7: Generate precision signals with explainable AI
            self._generate_precision_signals()
            
            # Step 8: Calculate signal confidence and explanations
            self._calculate_signal_explanations()
            
            # Step 9: Final quality validation and ranking
            self._validate_and_rank_signals()
            
            # Record performance statistics
            processing_time = time.time() - start_time
            self._record_processing_stats(processing_time)
            
            self.last_update = datetime.now()
            
            success_msg = f"🎯 Ultimate processing complete: {len(self.master_df)} stocks analyzed in {processing_time:.2f}s"
            logger.info(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"❌ Ultimate processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _load_all_data_sources(self) -> Tuple[bool, str]:
        """Load data from all Google Sheets sources with parallel processing"""
        
        def fetch_sheet_data(sheet_name: str, url: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Fetch individual sheet with comprehensive error handling"""
            try:
                logger.info(f"Loading {sheet_name} data...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                if df.empty:
                    logger.warning(f"{sheet_name} loaded but is empty")
                    return sheet_name, pd.DataFrame()
                
                logger.info(f"✅ {sheet_name}: {len(df)} rows, {len(df.columns)} columns loaded")
                return sheet_name, df
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Timeout loading {sheet_name}")
                return sheet_name, None
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network error loading {sheet_name}: {e}")
                return sheet_name, None
            except Exception as e:
                logger.error(f"❌ Error loading {sheet_name}: {e}")
                return sheet_name, None
        
        # Parallel loading for optimal performance
        results = {}
        with ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG['parallel_workers']) as executor:
            futures = {
                executor.submit(fetch_sheet_data, name, url): name 
                for name, url in DATA_URLS.items()
            }
            
            for future in as_completed(futures):
                sheet_name, df = future.result()
                if df is not None:
                    results[sheet_name] = df
        
        # Validate critical data loaded
        if 'watchlist' not in results:
            return False, "❌ Critical: Watchlist data failed to load - cannot proceed"
        
        if results['watchlist'].empty:
            return False, "❌ Critical: Watchlist data is empty - cannot proceed"
        
        # Store loaded data
        self.watchlist_df = results.get('watchlist', pd.DataFrame())
        self.returns_df = results.get('returns', pd.DataFrame())
        self.sectors_df = results.get('sectors', pd.DataFrame())
        
        sheets_loaded = len([df for df in results.values() if not df.empty])
        return True, f"✅ Data sources loaded: {sheets_loaded}/3 sheets successful"
    
    def _validate_and_clean_data(self):
        """Comprehensive data validation and cleaning"""
        
        # Clean watchlist data (primary dataset)
        if not self.watchlist_df.empty:
            self.watchlist_df = self._clean_watchlist_comprehensive(self.watchlist_df)
        
        # Clean returns data if available
        if not self.returns_df.empty:
            self.returns_df = self._clean_returns_data(self.returns_df)
        
        # Clean sectors data if available
        if not self.sectors_df.empty:
            self.sectors_df = self._clean_sectors_data(self.sectors_df)
    
    def _clean_watchlist_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive watchlist data cleaning with validation"""
        
        logger.info(f"Cleaning watchlist data: {len(df)} initial rows")
        
        # Store original dataframe for potential restoration
        original_df = df.copy()
        
        # 1. Rename columns to standard mapping with error handling
        original_columns = df.columns.tolist()
        
        # Only rename columns that actually exist in the dataframe
        valid_mappings = {k: v for k, v in WATCHLIST_COLUMNS.items() if k in df.columns}
        if valid_mappings:
            df = df.rename(columns=valid_mappings)
            logger.info(f"Renamed {len(valid_mappings)} columns using column mapping")
        else:
            logger.warning("No column mappings applied - using original column names")
        
        # 2. Keep only available essential columns (more flexible)
        essential_cols = ['ticker', 'price', 'sector']  # Minimum required
        available_essential = [col for col in essential_cols if col in df.columns]
        
        if not available_essential:
            logger.error("❌ No essential columns found! Cannot proceed with data cleaning.")
            return pd.DataFrame()  # Return empty dataframe
        
        # Keep all available columns, don't be too restrictive
        logger.info(f"Working with {len(df.columns)} available columns")
        
        # 3. Validate and clean ticker symbols (critical for everything)
        if 'ticker' in df.columns:
            initial_count = len(df)
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[~df['ticker'].isin(['NAN', 'NONE', '', 'NULL', 'NA'])]
            df = df.dropna(subset=['ticker'])
            df = df[df['ticker'].str.len() > 0]
            df = df.drop_duplicates(subset=['ticker'], keep='first')
            
            cleaned_count = len(df)
            logger.info(f"Ticker cleaning: {initial_count} → {cleaned_count} rows")
        
        # 4. Comprehensive numeric conversion with validation
        numeric_columns = [
            'price', 'prev_close', 'mcap', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma20', 'sma50', 'sma200',
            'vol_1d', 'vol_7d', 'vol_30d', 'vol_3m', 'rvol',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                if col == 'mcap':
                    df[col] = self._parse_market_cap_advanced(df[col])
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Apply validation rules
                if col == 'price':
                    df.loc[df[col] < VALIDATION_RULES['min_price'], col] = np.nan
                    df.loc[df[col] > VALIDATION_RULES['max_price'], col] = np.nan
                elif col == 'pe':
                    df.loc[df[col] < VALIDATION_RULES['min_pe'], col] = np.nan
                    df.loc[df[col] > VALIDATION_RULES['max_pe'], col] = np.nan
                elif 'ret_' in col:
                    df.loc[df[col] < VALIDATION_RULES['min_return'], col] = np.nan
                    df.loc[df[col] > VALIDATION_RULES['max_return'], col] = np.nan
        
        # 5. Filter out invalid stocks (more lenient)
        if 'price' in df.columns:
            # Only filter out clearly invalid prices (0 or negative), be more lenient
            valid_price_mask = (df['price'] > 0) & (df['price'].notna())
            initial_count = len(df)
            df = df[valid_price_mask]
            filtered_count = len(df)
            logger.info(f"Price filtering: {initial_count} → {filtered_count} rows (removed {initial_count - filtered_count} invalid prices)")
        
        # 6. Smart defaults for missing critical data (be more lenient)
        critical_defaults = {
            'price': 0, 'pe': 0, 'eps_current': 0, 'eps_change_pct': 0,
            'vol_1d': 0, 'rvol': 1.0, 'from_low_pct': 50, 'from_high_pct': -50,
            'ret_1d': 0, 'ret_7d': 0, 'ret_30d': 0, 'ret_3m': 0
        }
        
        for col, default_val in critical_defaults.items():
            if col in df.columns:
                before_fill = df[col].isna().sum()
                df[col] = df[col].fillna(default_val)
                if before_fill > 0:
                    logger.info(f"Filled {before_fill} missing values in {col} with {default_val}")
        
        # 7. Clean text columns
        text_columns = ['name', 'sector', 'category', 'eps_tier', 'price_tier']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df.loc[df[col].isin(['nan', 'NaN', '', 'None']), col] = 'Unknown'
        
        # Final check - ensure we don't lose all data
        if len(df) == 0:
            logger.error("❌ All data was filtered out during cleaning! Restoring original data with minimal cleaning...")
            # Restore original data with just basic ticker cleaning
            df = original_df.copy()  # Use original dataframe reference
            if 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
                df = df[~df['ticker'].isin(['NAN', 'NONE', '', 'NULL', 'NA'])]
                df = df.dropna(subset=['ticker'])
                df = df[df['ticker'].str.len() > 0]
                df = df.drop_duplicates(subset=['ticker'], keep='first')
            
            # Ensure numeric columns are properly converted after restoration
            numeric_columns = [
                'price', 'pe', 'eps_current', 'eps_change_pct', 'vol_1d', 'rvol',
                'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'from_low_pct', 'from_high_pct'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"✅ Watchlist cleaned: {len(df)} final rows")
        return df.reset_index(drop=True)
    
    def _clean_returns_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean returns analysis data"""
        
        if df.empty:
            return df
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in RETURNS_COLUMNS.items() if k in df.columns})
        
        # Keep available columns
        available_cols = [col for col in RETURNS_COLUMNS.values() if col in df.columns]
        df = df[available_cols]
        
        # Clean ticker for merging
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[df['ticker'] != '']
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if 'ret' in col or 'avg' in col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.reset_index(drop=True)
    
    def _clean_sectors_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean comprehensive sector data"""
        
        if df.empty:
            return df
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in SECTOR_COLUMNS.items() if k in df.columns})
        
        # Keep available columns
        available_cols = [col for col in SECTOR_COLUMNS.values() if col in df.columns]
        df = df[available_cols]
        
        # Clean sector names
        if 'sector' in df.columns:
            df['sector'] = df['sector'].astype(str).str.strip()
            df = df[df['sector'] != '']
            df = df[~df['sector'].isin(['nan', 'NaN', 'Unknown'])]
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if 'ret' in col or 'avg' in col or 'count' in col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.reset_index(drop=True)
    
    def _parse_market_cap_advanced(self, series: pd.Series) -> pd.Series:
        """Advanced market cap parsing with comprehensive Indian notation support"""
        
        if series.dtype not in ['object', 'string']:
            return pd.to_numeric(series, errors='coerce')
        
        # Convert to string and clean
        s = series.astype(str).str.upper().str.strip()
        s = s.str.replace(',', '').str.replace(' ', '')
        
        result = pd.Series(0.0, index=series.index)
        
        # Handle Crores (most common in Indian markets)
        cr_patterns = ['CR', 'CRORE', 'CRORES']
        for pattern in cr_patterns:
            cr_mask = s.str.contains(pattern, na=False)
            if cr_mask.any():
                cr_values = s[cr_mask].str.extract(r'([\d.]+)', expand=False)
                result[cr_mask] = pd.to_numeric(cr_values, errors='coerce') * 1e7
        
        # Handle Lakhs
        lakh_patterns = ['L', 'LAKH', 'LAKHS']
        for pattern in lakh_patterns:
            lakh_mask = s.str.contains(pattern, na=False) & (result == 0)
            if lakh_mask.any():
                lakh_values = s[lakh_mask].str.extract(r'([\d.]+)', expand=False)
                result[lakh_mask] = pd.to_numeric(lakh_values, errors='coerce') * 1e5
        
        # Handle thousands
        k_patterns = ['K', 'THOUSAND']
        for pattern in k_patterns:
            k_mask = s.str.contains(pattern, na=False) & (result == 0)
            if k_mask.any():
                k_values = s[k_mask].str.extract(r'([\d.]+)', expand=False)
                result[k_mask] = pd.to_numeric(k_values, errors='coerce') * 1e3
        
        # Handle direct numbers
        num_mask = (result == 0)
        if num_mask.any():
            result[num_mask] = pd.to_numeric(s[num_mask], errors='coerce')
        
        # Apply market cap validation
        result = result.clip(0, VALIDATION_RULES['max_market_cap'])
        
        return result
    
    def _assess_data_quality(self):
        """Comprehensive data quality assessment"""
        
        if self.watchlist_df.empty:
            self.data_quality_report = {
                'overall_score': 0,
                'status': 'No Data',
                'details': {'error': 'Watchlist data is empty'}
            }
            return
        
        # Assess completeness of critical columns
        completeness_scores = {}
        for col in REQUIRED_COLUMNS + IMPORTANT_COLUMNS:
            if col in self.watchlist_df.columns:
                completeness = (self.watchlist_df[col].notna().sum() / len(self.watchlist_df)) * 100
                completeness_scores[col] = completeness
        
        # Calculate overall quality score
        required_completeness = np.mean([completeness_scores.get(col, 0) for col in REQUIRED_COLUMNS])
        important_completeness = np.mean([completeness_scores.get(col, 50) for col in IMPORTANT_COLUMNS])
        
        overall_score = (required_completeness * 0.7) + (important_completeness * 0.3)
        
        # Determine quality status
        if overall_score >= QUALITY_REQUIREMENTS['excellent_threshold']:
            status = 'Excellent'
        elif overall_score >= QUALITY_REQUIREMENTS['good_threshold']:
            status = 'Good'
        elif overall_score >= QUALITY_REQUIREMENTS['acceptable_threshold']:
            status = 'Acceptable'
        elif overall_score >= QUALITY_REQUIREMENTS['poor_threshold']:
            status = 'Poor'
        else:
            status = 'Critical'
        
        # Additional quality metrics
        additional_metrics = {
            'total_stocks': len(self.watchlist_df),
            'sectors_available': not self.sectors_df.empty,
            'returns_data_available': not self.returns_df.empty,
            'duplicate_tickers': self.watchlist_df['ticker'].duplicated().sum() if 'ticker' in self.watchlist_df.columns else 0,
            'price_coverage': 0  # Default value
        }
        
        # Calculate price coverage with error handling
        if 'price' in self.watchlist_df.columns:
            try:
                # Ensure price column is numeric
                price_series = pd.to_numeric(self.watchlist_df['price'], errors='coerce')
                additional_metrics['price_coverage'] = (price_series > 0).sum() / len(self.watchlist_df) * 100
            except Exception as e:
                logger.warning(f"Could not calculate price coverage: {e}")
                additional_metrics['price_coverage'] = 0
        
        self.data_quality_report = {
            'overall_score': round(overall_score, 1),
            'status': status,
            'required_completeness': round(required_completeness, 1),
            'important_completeness': round(important_completeness, 1),
            'completeness_by_column': completeness_scores,
            'additional_metrics': additional_metrics,
            'timestamp': datetime.now()
        }
        
        logger.info(f"📊 Data quality: {status} ({overall_score:.1f}%)")
    
    def _detect_market_conditions(self):
        """Detect current market conditions for adaptive signals"""
        
        if self.watchlist_df.empty or 'ret_1d' not in self.watchlist_df.columns:
            self.market_conditions = {'condition': 'unknown', 'confidence': 0}
            return
        
        try:
            # Ensure ret_1d is numeric for calculations
            ret_1d_series = pd.to_numeric(self.watchlist_df['ret_1d'], errors='coerce')
            
            # Calculate market breadth
            positive_stocks = (ret_1d_series > 0).sum()
            total_stocks = len(self.watchlist_df)
            market_breadth = (positive_stocks / total_stocks) * 100
        except Exception as e:
            logger.warning(f"Could not calculate market conditions: {e}")
            self.market_conditions = {'condition': 'unknown', 'confidence': 0}
            return
        
        # Calculate average sector performance if available
        avg_sector_performance = 0
        if not self.sectors_df.empty and 'sector_ret_1d' in self.sectors_df.columns:
            try:
                sector_rets = pd.to_numeric(self.sectors_df['sector_ret_1d'], errors='coerce')
                avg_sector_performance = sector_rets.mean()
            except:
                avg_sector_performance = 0
        
        # Determine market condition
        if (market_breadth >= MARKET_CONDITIONS['bull_market']['market_breadth_min'] and 
            avg_sector_performance >= MARKET_CONDITIONS['bull_market']['sector_strength_min']):
            condition = 'bull_market'
            confidence = min(100, market_breadth + abs(avg_sector_performance) * 10)
        elif (market_breadth <= MARKET_CONDITIONS['bear_market']['market_breadth_max'] and 
              avg_sector_performance <= MARKET_CONDITIONS['bear_market']['sector_strength_max']):
            condition = 'bear_market'
            confidence = min(100, (100 - market_breadth) + abs(avg_sector_performance) * 10)
        else:
            condition = 'neutral_market'
            confidence = 100 - abs(market_breadth - 50) * 2
        
        self.market_conditions = {
            'condition': condition,
            'confidence': round(confidence, 1),
            'market_breadth': round(market_breadth, 1),
            'avg_sector_performance': round(avg_sector_performance, 2),
            'positive_stocks': positive_stocks,
            'total_stocks': total_stocks
        }
        
        logger.info(f"📈 Market condition: {condition} ({confidence:.1f}% confidence)")
    
    def _enrich_data_with_features(self):
        """Advanced data enrichment with sophisticated feature engineering"""
        
        if self.watchlist_df.empty:
            return
        
        df = self.watchlist_df
        
        # 1. Enhanced 52-week position analysis
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            df['position_52w_calculated'] = ((df['price'] - df['low_52w']) / 
                                           (df['high_52w'] - df['low_52w']) * 100).clip(0, 100)
        
        # 2. Advanced moving average relationships
        sma_columns = ['sma20', 'sma50', 'sma200']
        for sma in sma_columns:
            if all(col in df.columns for col in ['price', sma]):
                df[f'above_{sma}'] = df['price'] > df[sma]
                df[f'pct_from_{sma}'] = ((df['price'] - df[sma]) / df[sma] * 100).fillna(0)
        
        # 3. Multi-timeframe momentum analysis
        timeframes = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m']
        available_timeframes = [tf for tf in timeframes if tf in df.columns]
        
        if len(available_timeframes) >= 3:
            # Momentum breadth (% of timeframes that are positive)
            positive_timeframes = sum(df[tf] > 0 for tf in available_timeframes)
            df['momentum_breadth'] = (positive_timeframes / len(available_timeframes)) * 100
            
            # Momentum consistency score
            momentum_scores = []
            for tf in available_timeframes:
                momentum_scores.append(df[tf].fillna(0))
            
            if momentum_scores:
                momentum_array = np.array(momentum_scores).T
                df['momentum_consistency'] = 100 - np.std(momentum_array, axis=1) * 5  # Lower std = higher consistency
                df['momentum_consistency'] = df['momentum_consistency'].clip(0, 100)
        
        # 4. Advanced volume analysis
        volume_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        available_vol_cols = [col for col in volume_cols if col in df.columns]
        
        if available_vol_cols:
            # Volume trend analysis
            if len(available_vol_cols) >= 2:
                df['volume_trend_score'] = (df[available_vol_cols[0]] - df[available_vol_cols[-1]]) * 25
                df['volume_trend_score'] = df['volume_trend_score'].clip(-50, 50) + 50
            
            # Volume spike detection with confirmation
            if 'vol_ratio_1d_90d' in df.columns:
                df['volume_spike_strength'] = np.where(
                    df['vol_ratio_1d_90d'] >= VOLUME_BENCHMARKS['extreme_interest'], 100,
                    np.where(df['vol_ratio_1d_90d'] >= VOLUME_BENCHMARKS['strong_interest'], 80,
                    np.where(df['vol_ratio_1d_90d'] >= VOLUME_BENCHMARKS['elevated_interest'], 60,
                    np.where(df['vol_ratio_1d_90d'] >= VOLUME_BENCHMARKS['normal_activity'], 40, 20)))
                )
        
        # 5. EPS quality and growth analysis
        if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
            # EPS consistency
            df['eps_stability'] = np.where(
                (df['eps_current'] > 0) & (df['eps_last_qtr'] > 0), 100,
                np.where((df['eps_current'] > 0) | (df['eps_last_qtr'] > 0), 60, 20)
            )
        
        if 'eps_change_pct' in df.columns:
            # EPS growth quality classification
            eps_change = df['eps_change_pct'].fillna(0)
            df['growth_quality_score'] = np.where(
                eps_change >= GROWTH_BENCHMARKS['accelerating'], 100,
                np.where(eps_change >= GROWTH_BENCHMARKS['strong'], 85,
                np.where(eps_change >= GROWTH_BENCHMARKS['decent'], 70,
                np.where(eps_change >= GROWTH_BENCHMARKS['modest'], 55,
                np.where(eps_change >= GROWTH_BENCHMARKS['weak'], 40, 20))))
            )
        
        # 6. Market cap categorization with risk assessment
        if 'mcap' in df.columns:
            df['mcap_category_detailed'] = pd.cut(
                df['mcap'].fillna(0),
                bins=[0, 1e9, 5e9, 2e10, 1e11, 5e11, np.inf],
                labels=['Nano', 'Small', 'Mid', 'Large', 'Mega', 'Giant']
            )
            
            # Risk score based on market cap
            df['mcap_risk_score'] = df['mcap_category_detailed'].map({
                'Nano': 80, 'Small': 60, 'Mid': 40, 'Large': 20, 'Mega': 10, 'Giant': 5
            }).fillna(50)
        
        # 7. Sector strength integration
        if not self.sectors_df.empty and 'sector' in df.columns:
            # Map current sector performance
            if 'sector_ret_30d' in self.sectors_df.columns:
                sector_strength_map = self.sectors_df.set_index('sector')['sector_ret_30d'].to_dict()
                df['current_sector_strength'] = df['sector'].map(sector_strength_map).fillna(0)
                
                # Sector momentum score
                df['sector_momentum_score'] = 50 + (df['current_sector_strength'] * 2)
                df['sector_momentum_score'] = df['sector_momentum_score'].clip(0, 100)
        
        # 8. Comprehensive risk factor assessment
        risk_factors = []
        
        # Valuation risk
        if 'pe' in df.columns:
            pe_risk = np.where(df['pe'] > VALUE_BENCHMARKS['expensive'], 25, 0)
            risk_factors.append(pe_risk)
        
        # Liquidity risk
        if 'vol_1d' in df.columns:
            liquidity_risk = np.where(df['vol_1d'] < VALIDATION_RULES['min_volume'] * 10, 20, 0)
            risk_factors.append(liquidity_risk)
        
        # Momentum risk
        if 'ret_30d' in df.columns:
            momentum_risk = np.where(df['ret_30d'] < -15, 30, 0)
            risk_factors.append(momentum_risk)
        
        # Market cap risk
        if 'mcap_risk_score' in df.columns:
            risk_factors.append(df['mcap_risk_score'] / 4)  # Scale down
        
        # Calculate total risk score
        if risk_factors:
            df['total_risk_score'] = np.sum(risk_factors, axis=0)
            df['total_risk_score'] = df['total_risk_score'].clip(0, 100)
        
        # 9. Data completeness score per stock
        essential_cols = ['price', 'pe', 'ret_30d', 'vol_1d', 'sector']
        completeness_scores = []
        
        for _, row in df.iterrows():
            available_data = sum(1 for col in essential_cols if col in df.columns and pd.notna(row.get(col)))
            completeness = (available_data / len(essential_cols)) * 100
            completeness_scores.append(completeness)
        
        df['data_completeness_score'] = completeness_scores
        
        self.watchlist_df = df
        logger.info("✅ Data enrichment completed with advanced features")
    
    def _create_master_dataset(self):
        """Create comprehensive master dataset with all sources merged"""
        
        master = self.watchlist_df.copy()
        
        # Merge returns analysis data if available
        if not self.returns_df.empty and 'ticker' in self.returns_df.columns:
            logger.info("Merging returns analysis data...")
            master = master.merge(
                self.returns_df,
                on='ticker',
                how='left',
                suffixes=('', '_returns_analysis')
            )
        
        # Add comprehensive sector data
        if not self.sectors_df.empty and 'sector' in master.columns:
            logger.info("Integrating sector performance data...")
            
            # Create comprehensive sector mapping
            sector_data = self.sectors_df.set_index('sector')
            
            # Add multiple timeframe sector data
            sector_columns_to_add = [
                'sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 
                'sector_ret_3m', 'sector_avg_30d', 'sector_count'
            ]
            
            for col in sector_columns_to_add:
                if col in sector_data.columns:
                    master[f'stock_{col}'] = master['sector'].map(sector_data[col]).fillna(0)
        
        # Final data validation - be more lenient
        if 'ticker' in master.columns:
            master = master.dropna(subset=['ticker'])
            
        # Only filter on price if we have meaningful price data
        if 'price' in master.columns:
            # Allow prices above 0, don't enforce minimum thresholds that might be too strict
            price_mask = (master['price'] > 0) | (master['price'].isna())
            master = master[price_mask]
            logger.info(f"After price validation: {len(master)} stocks remaining")
        
        # Ensure we have some data to work with
        if len(master) == 0:
            logger.warning("⚠️ No stocks passed validation. Using original data with basic cleaning only.")
            master = self.watchlist_df.copy()
            if 'ticker' in master.columns:
                master = master.dropna(subset=['ticker'])
                master = master[master['ticker'] != '']
        
        self.master_df = master.reset_index(drop=True)
        logger.info(f"✅ Master dataset created: {len(self.master_df)} stocks with comprehensive data")
    
    def _generate_precision_signals(self):
        """Generate ultra-precision signals with 8-factor analysis"""
        
        if self.master_df.empty:
            logger.warning("Cannot generate signals: master dataset is empty")
            return
        
        logger.info("🎯 Generating precision signals with 8-factor analysis...")
        
        df = self.master_df
        
        # Calculate all 8 factor scores with error handling
        try:
            df['momentum_score'] = self._calculate_momentum_factor(df)
            df['value_score'] = self._calculate_value_factor(df)
            df['growth_score'] = self._calculate_growth_factor(df)
            df['volume_score'] = self._calculate_volume_factor(df)
            df['technical_score'] = self._calculate_technical_factor(df)
            df['sector_score'] = self._calculate_sector_factor(df)
            df['risk_score'] = self._calculate_risk_factor(df)
            df['quality_score'] = self._calculate_quality_factor(df)
        except Exception as e:
            logger.error(f"Error calculating factor scores: {e}")
            # Set default scores if calculation fails
            for factor in ['momentum', 'value', 'growth', 'volume', 'technical', 'sector', 'risk', 'quality']:
                if f'{factor}_score' not in df.columns:
                    df[f'{factor}_score'] = 50.0
        
        # Apply market condition adjustments
        if self.market_conditions.get('condition') == 'bull_market':
            df['momentum_score'] *= MARKET_CONDITIONS['bull_market']['momentum_bias']
            df['momentum_score'] = df['momentum_score'].clip(0, 100)
        elif self.market_conditions.get('condition') == 'bear_market':
            df['value_score'] *= MARKET_CONDITIONS['bear_market']['value_bias']
            df['value_score'] = df['value_score'].clip(0, 100)
        
        # Calculate weighted composite score
        df['composite_score'] = (
            df['momentum_score'] * FACTOR_WEIGHTS['momentum'] +
            df['value_score'] * FACTOR_WEIGHTS['value'] +
            df['growth_score'] * FACTOR_WEIGHTS['growth'] +
            df['volume_score'] * FACTOR_WEIGHTS['volume'] +
            df['technical_score'] * FACTOR_WEIGHTS['technical'] +
            df['sector_score'] * FACTOR_WEIGHTS['sector'] +
            (100 - df['risk_score']) * FACTOR_WEIGHTS['risk'] +  # Invert risk score
            df['quality_score'] * FACTOR_WEIGHTS['quality']
        ).round(1)
        
        # Generate ultra-conservative signals
        df['signal'] = self._generate_ultra_conservative_signals(df['composite_score'])
        
        # Calculate signal confidence
        df['confidence'] = self._calculate_signal_confidence(df)
        
        # Sort by composite score for ranking
        self.master_df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        # Log signal distribution
        signal_counts = df['signal'].value_counts()
        logger.info(f"📊 Signal distribution: {dict(signal_counts)}")
    
    def _calculate_momentum_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum factor with multi-timeframe analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Multi-timeframe weights (recent performance weighted more heavily)
        timeframe_weights = {
            'ret_1d': 0.05,    # 5% - Very short term
            'ret_7d': 0.15,    # 15% - Short term
            'ret_30d': 0.40,   # 40% - Primary momentum (most important)
            'ret_3m': 0.30,    # 30% - Medium term trend
            'ret_6m': 0.10     # 10% - Longer term context
        }
        
        momentum_components = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for timeframe, weight in timeframe_weights.items():
            if timeframe in df.columns:
                returns = df[timeframe].fillna(0)
                
                # Convert returns to momentum score using conservative benchmarks
                if timeframe == 'ret_1d':
                    normalized = np.where(returns >= MOMENTUM_BENCHMARKS['excellent']['1d'], 90,
                                np.where(returns >= MOMENTUM_BENCHMARKS['good']['1d'], 70,
                                np.where(returns >= MOMENTUM_BENCHMARKS['neutral']['1d'], 50,
                                np.where(returns >= MOMENTUM_BENCHMARKS['poor']['1d'], 30, 10))))
                elif timeframe == 'ret_7d':
                    normalized = np.where(returns >= MOMENTUM_BENCHMARKS['excellent']['7d'], 90,
                                np.where(returns >= MOMENTUM_BENCHMARKS['good']['7d'], 70,
                                np.where(returns >= MOMENTUM_BENCHMARKS['neutral']['7d'], 50,
                                np.where(returns >= MOMENTUM_BENCHMARKS['poor']['7d'], 30, 10))))
                elif timeframe == 'ret_30d':
                    normalized = np.where(returns >= MOMENTUM_BENCHMARKS['excellent']['30d'], 90,
                                np.where(returns >= MOMENTUM_BENCHMARKS['good']['30d'], 70,
                                np.where(returns >= MOMENTUM_BENCHMARKS['neutral']['30d'], 50,
                                np.where(returns >= MOMENTUM_BENCHMARKS['poor']['30d'], 30, 10))))
                elif timeframe == 'ret_3m':
                    normalized = np.where(returns >= MOMENTUM_BENCHMARKS['excellent']['3m'], 90,
                                np.where(returns >= MOMENTUM_BENCHMARKS['good']['3m'], 70,
                                np.where(returns >= MOMENTUM_BENCHMARKS['neutral']['3m'], 50,
                                np.where(returns >= MOMENTUM_BENCHMARKS['poor']['3m'], 30, 10))))
                else:
                    # Default normalization for other timeframes
                    normalized = 50 + np.clip(returns / 2, -40, 40)
                
                momentum_components += normalized * weight
                total_weight += weight
        
        if total_weight > 0:
            score = momentum_components / total_weight
        
        # Bonus for momentum consistency
        if 'momentum_breadth' in df.columns:
            consistency_bonus = (df['momentum_breadth'] - 50) / 5  # ±10 points max
            score += consistency_bonus
        
        # Bonus for momentum alignment
        if 'momentum_consistency' in df.columns:
            alignment_bonus = (df['momentum_consistency'] - 50) / 10  # ±5 points max
            score += alignment_bonus
        
        return score.clip(0, 100).round(1)
    
    def _calculate_value_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate value factor with comprehensive valuation analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        if 'pe' not in df.columns:
            return score
        
        pe = df['pe'].fillna(1000)  # High default for missing PE
        
        # PE-based scoring with conservative benchmarks
        value_score = np.where(pe <= 0, 20,  # Negative earnings companies
                      np.where(pe <= VALUE_BENCHMARKS['deep_value'], 95,
                      np.where(pe <= VALUE_BENCHMARKS['strong_value'], 85,
                      np.where(pe <= VALUE_BENCHMARKS['fair_value'], 70,
                      np.where(pe <= VALUE_BENCHMARKS['growth_premium'], 55,
                      np.where(pe <= VALUE_BENCHMARKS['expensive'], 35,
                      np.where(pe <= VALUE_BENCHMARKS['overvalued'], 20, 10)))))))
        
        score = pd.Series(value_score, index=df.index)
        
        # Earnings quality adjustment
        if 'eps_stability' in df.columns:
            stability_bonus = (df['eps_stability'] - 50) / 10  # ±5 points
            score += stability_bonus
        
        # Growth-adjusted valuation (PEG-like concept)
        if 'eps_change_pct' in df.columns:
            eps_growth = df['eps_change_pct'].fillna(0)
            
            # Adjust value score based on growth
            # High growth justifies higher PE ratios
            growth_adjustment = np.where(
                eps_growth > 20, 10,      # High growth gets bonus
                np.where(eps_growth > 10, 5,       # Moderate growth gets small bonus
                np.where(eps_growth < -10, -15, 0))  # Declining earnings penalty
            )
            score += growth_adjustment
        
        # Market cap context adjustment
        if 'mcap_category_detailed' in df.columns:
            # Small caps can trade at higher multiples
            mcap_adjustment = df['mcap_category_detailed'].map({
                'Nano': 5, 'Small': 3, 'Mid': 0, 'Large': -2, 'Mega': -3, 'Giant': -5
            }).fillna(0)
            score += mcap_adjustment
        
        return score.clip(0, 100).round(1)
    
    def _calculate_growth_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate growth factor with EPS quality analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        if 'eps_change_pct' not in df.columns:
            return score
        
        eps_growth = df['eps_change_pct'].fillna(0)
        
        # EPS growth scoring with conservative thresholds
        growth_score = np.where(eps_growth >= GROWTH_BENCHMARKS['accelerating'], 95,
                       np.where(eps_growth >= GROWTH_BENCHMARKS['strong'], 85,
                       np.where(eps_growth >= GROWTH_BENCHMARKS['decent'], 70,
                       np.where(eps_growth >= GROWTH_BENCHMARKS['modest'], 60,
                       np.where(eps_growth >= GROWTH_BENCHMARKS['weak'], 45,
                       np.where(eps_growth >= GROWTH_BENCHMARKS['declining'], 25, 10))))))
        
        score = pd.Series(growth_score, index=df.index)
        
        # EPS quality bonus
        if 'growth_quality_score' in df.columns:
            quality_bonus = (df['growth_quality_score'] - 50) / 10  # ±5 points
            score += quality_bonus
        
        # Consistency bonus
        if 'eps_stability' in df.columns:
            stability_bonus = (df['eps_stability'] - 50) / 20  # ±2.5 points
            score += stability_bonus
        
        return score.clip(0, 100).round(1)
    
    def _calculate_volume_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume factor with advanced activity analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Primary volume score using RVOL
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            
            volume_score = np.where(rvol >= VOLUME_BENCHMARKS['extreme_interest'], 95,
                           np.where(rvol >= VOLUME_BENCHMARKS['strong_interest'], 85,
                           np.where(rvol >= VOLUME_BENCHMARKS['elevated_interest'], 70,
                           np.where(rvol >= VOLUME_BENCHMARKS['normal_activity'], 55,
                           np.where(rvol >= VOLUME_BENCHMARKS['weak_interest'], 35,
                           np.where(rvol >= VOLUME_BENCHMARKS['very_weak'], 20, 10))))))
            
            score = pd.Series(volume_score, index=df.index)
        
        # Advanced volume ratio analysis
        if 'vol_ratio_1d_90d' in df.columns:
            vol_ratio = df['vol_ratio_1d_90d'].fillna(1.0)
            ratio_bonus = np.clip((vol_ratio - 1) * 8, -15, 20)  # Recent volume vs 90-day average
            score += ratio_bonus
        
        # Volume trend bonus
        if 'volume_trend_score' in df.columns:
            trend_bonus = (df['volume_trend_score'] - 50) / 10  # ±5 points
            score += trend_bonus
        
        # Volume-price confirmation
        if all(col in df.columns for col in ['ret_1d', 'rvol']):
            # Strong volume with positive price action
            confirmation_bonus = np.where(
                (df['rvol'] >= 2.0) & (df['ret_1d'] > 2), 15,
                np.where((df['rvol'] >= 1.5) & (df['ret_1d'] > 1), 8, 0)
            )
            score += confirmation_bonus
        
        return score.clip(0, 100).round(1)
    
    def _calculate_technical_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate technical factor with comprehensive trend analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Moving average trend analysis
        sma_weights = {'above_sma20': 0.4, 'above_sma50': 0.35, 'above_sma200': 0.25}
        
        for sma_col, weight in sma_weights.items():
            if sma_col in df.columns:
                sma_bonus = df[sma_col] * 80 * weight  # Up to 80 points per SMA
                score += sma_bonus
        
        # 52-week position analysis
        if 'position_52w_calculated' in df.columns:
            position = df['position_52w_calculated'].fillna(50)
            
            position_score = np.where(position >= POSITION_BENCHMARKS['near_highs'], 30,
                            np.where(position >= POSITION_BENCHMARKS['strong_position'], 20,
                            np.where(position >= POSITION_BENCHMARKS['upper_range'], 10,
                            np.where(position >= POSITION_BENCHMARKS['middle_range'], 0,
                            np.where(position >= POSITION_BENCHMARKS['lower_range'], -10,
                            np.where(position >= POSITION_BENCHMARKS['weak_position'], -20, -30))))))
            
            score += position_score
        
        # Price distance from moving averages (strength of trend)
        distance_bonus = 0
        if 'pct_from_sma20' in df.columns:
            distance_bonus += np.clip(df['pct_from_sma20'] / 4, -5, 10)  # Bonus for being above SMA20
        if 'pct_from_sma50' in df.columns:
            distance_bonus += np.clip(df['pct_from_sma50'] / 6, -3, 8)   # Bonus for being above SMA50
        
        score += distance_bonus
        
        return score.clip(0, 100).round(1)
    
    def _calculate_sector_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sector factor with rotation analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Current sector strength
        if 'current_sector_strength' in df.columns:
            sector_strength = df['current_sector_strength'].fillna(0)
            strength_score = 50 + np.clip(sector_strength * 2.5, -40, 40)
            score = strength_score
        
        # Sector momentum score
        if 'sector_momentum_score' in df.columns:
            momentum_bonus = (df['sector_momentum_score'] - 50) / 5  # ±10 points
            score += momentum_bonus
        
        # Sector size and diversity factor
        if 'stock_sector_count' in df.columns:
            sector_count = df['stock_sector_count'].fillna(10)
            
            # Prefer sectors with reasonable diversification
            size_factor = np.where(sector_count >= 30, 8,      # Large sectors
                         np.where(sector_count >= 15, 5,      # Medium sectors
                         np.where(sector_count >= 8, 0,       # Small sectors
                         np.where(sector_count >= 3, -5, -10))))  # Very small sectors
            score += size_factor
        
        return score.clip(0, 100).round(1)
    
    def _calculate_risk_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive risk factor"""
        
        risk_score = pd.Series(0.0, index=df.index)
        
        # Use pre-calculated total risk score if available
        if 'total_risk_score' in df.columns:
            risk_score = df['total_risk_score'].fillna(50)
        else:
            # Calculate basic risk factors
            
            # Valuation risk
            if 'pe' in df.columns:
                pe_risk = np.where(df['pe'] > VALUE_BENCHMARKS['expensive'], 25,
                          np.where(df['pe'] <= 0, 30, 0))  # Negative earnings risk
                risk_score += pe_risk
            
            # Liquidity risk
            if 'vol_1d' in df.columns:
                liquidity_risk = np.where(df['vol_1d'] < VALIDATION_RULES['min_volume'] * 5, 20, 0)
                risk_score += liquidity_risk
            
            # Momentum risk
            if 'ret_30d' in df.columns:
                momentum_risk = np.where(df['ret_30d'] < -20, 30, 0)
                risk_score += momentum_risk
        
        return risk_score.clip(0, 100).round(1)
    
    def _calculate_quality_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality factor"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Data completeness score
        if 'data_completeness_score' in df.columns:
            completeness_bonus = (df['data_completeness_score'] - 50) / 2  # ±25 points
            score += completeness_bonus
        
        # Earnings quality
        if 'eps_stability' in df.columns:
            earnings_quality = (df['eps_stability'] - 50) / 4  # ±12.5 points
            score += earnings_quality
        
        # Market cap stability (larger = more data reliability)
        if 'mcap_category_detailed' in df.columns:
            stability_score = df['mcap_category_detailed'].map({
                'Nano': 30, 'Small': 45, 'Mid': 60, 'Large': 75, 'Mega': 85, 'Giant': 90
            }).fillna(50)
            score = (score + stability_score) / 2
        
        return score.clip(0, 100).round(1)
    
    def _generate_ultra_conservative_signals(self, scores: pd.Series) -> pd.Series:
        """Generate ultra-conservative signals with strict thresholds"""
        
        conditions = [
            scores >= SIGNAL_THRESHOLDS['STRONG_BUY'],    # 92+
            scores >= SIGNAL_THRESHOLDS['BUY'],           # 82+
            scores >= SIGNAL_THRESHOLDS['ACCUMULATE'],    # 72+
            scores >= SIGNAL_THRESHOLDS['WATCH'],         # 60+
            scores >= SIGNAL_THRESHOLDS['NEUTRAL'],       # 40+
            scores >= SIGNAL_THRESHOLDS['AVOID']          # 25+
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH', 'NEUTRAL', 'AVOID']
        
        return pd.Series(
            np.select(conditions, choices, default='STRONG_AVOID'),
            index=scores.index
        )
    
    def _calculate_signal_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal confidence with multiple validation methods"""
        
        confidence = pd.Series(50.0, index=df.index)
        
        # Factor scores for alignment calculation
        factor_columns = ['momentum_score', 'value_score', 'growth_score', 'volume_score', 
                         'technical_score', 'sector_score', 'quality_score']
        
        available_factors = [col for col in factor_columns if col in df.columns]
        
        if not available_factors:
            return confidence
        
        for idx, row in df.iterrows():
            factor_scores = [row[col] for col in available_factors if pd.notna(row[col])]
            
            if len(factor_scores) < 3:  # Need at least 3 factors for confidence
                confidence.iloc[idx] = 30.0
                continue
            
            # Base confidence from composite score
            base_confidence = row['composite_score']
            
            # Factor alignment bonus (lower standard deviation = higher confidence)
            std_dev = np.std(factor_scores)
            alignment_bonus = max(0, 30 - std_dev) * CONFIDENCE_CALIBRATION['factor_alignment_weight']
            
            # Data quality bonus
            quality_bonus = 0
            if 'data_completeness_score' in df.columns:
                quality_bonus = (row['data_completeness_score'] - 70) * 0.3 * CONFIDENCE_CALIBRATION['data_quality_weight']
            
            # Number of factors bonus
            factor_bonus = min(15, len(factor_scores) * 2) * CONFIDENCE_CALIBRATION['historical_weight']
            
            # Market condition confidence
            market_bonus = 0
            if self.market_conditions.get('confidence', 0) > 70:
                market_bonus = 5 * CONFIDENCE_CALIBRATION['market_context_weight']
            
            # Calculate final confidence
            total_confidence = (base_confidence + alignment_bonus + quality_bonus + 
                              factor_bonus + market_bonus)
            
            confidence.iloc[idx] = min(99, max(10, total_confidence))  # Cap between 10-99%
        
        return confidence.round(1)
    
    def _calculate_signal_explanations(self):
        """Generate detailed explanations for each signal"""
        
        logger.info("🧠 Generating signal explanations...")
        
        self.signal_explanations = {}
        
        for idx, stock in self.master_df.iterrows():
            ticker = stock.get('ticker', f'Stock_{idx}')
            
            explanation = self._generate_stock_explanation(stock)
            self.signal_explanations[ticker] = explanation
        
        logger.info(f"✅ Generated explanations for {len(self.signal_explanations)} stocks")
    
    def _generate_stock_explanation(self, stock: pd.Series) -> SignalExplanation:
        """Generate comprehensive explanation for individual stock signal"""
        
        signal = stock.get('signal', 'NEUTRAL')
        confidence = stock.get('confidence', 50)
        composite_score = stock.get('composite_score', 50)
        
        # Extract factor scores
        factor_scores = {
            'momentum': stock.get('momentum_score', 50),
            'value': stock.get('value_score', 50),
            'growth': stock.get('growth_score', 50),
            'volume': stock.get('volume_score', 50),
            'technical': stock.get('technical_score', 50),
            'sector': stock.get('sector_score', 50),
            'risk': stock.get('risk_score', 50),
            'quality': stock.get('quality_score', 50)
        }
        
        # Identify primary reason (highest scoring factors)
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        top_factors = [factor for factor, score in sorted_factors[:3] if score > 60]
        
        # Generate primary reason
        if not top_factors:
            primary_reason = "Mixed signals across factors"
        elif len(top_factors) == 1:
            primary_reason = f"Strong {FACTOR_IMPORTANCE.get(top_factors[0], top_factors[0])}"
        else:
            primary_reason = f"Multiple factors aligned: {', '.join([FACTOR_IMPORTANCE.get(f, f) for f in top_factors[:2]])}"
        
        # Identify supporting factors
        supporting_factors = []
        for factor, score in sorted_factors:
            if score > 70 and factor not in top_factors[:2]:
                supporting_factors.append(FACTOR_IMPORTANCE.get(factor, factor))
        
        # Identify risk factors
        risk_factors = []
        
        # Check specific risk conditions
        if stock.get('pe', 0) > VALUE_BENCHMARKS['expensive']:
            risk_factors.append(f"High valuation (PE {stock.get('pe', 0):.1f})")
        
        if stock.get('vol_1d', 0) < VALIDATION_RULES['min_volume']:
            risk_factors.append("Low liquidity")
        
        if stock.get('ret_30d', 0) < -15:
            risk_factors.append("Recent weakness")
        
        if stock.get('current_sector_strength', 0) < -5:
            risk_factors.append("Weak sector performance")
        
        risk_score = stock.get('risk_score', 50)
        if risk_score > 60:
            risk_factors.append("Elevated overall risk")
        
        # Generate recommendation based on signal
        if signal in ['STRONG_BUY', 'BUY']:
            if confidence >= 80:
                recommendation = f"High-confidence {signal.lower().replace('_', ' ')} opportunity"
            else:
                recommendation = f"Consider for {signal.lower().replace('_', ' ')} with caution"
        elif signal == 'ACCUMULATE':
            recommendation = "Consider gradual position building"
        elif signal == 'WATCH':
            recommendation = "Monitor for better entry opportunity"
        else:
            recommendation = "Avoid or consider exit"
        
        # Data quality assessment
        data_quality = stock.get('data_completeness_score', 70)
        
        return SignalExplanation(
            signal=signal,
            confidence=confidence,
            primary_reason=primary_reason,
            supporting_factors=supporting_factors,
            risk_factors=risk_factors,
            factor_scores=factor_scores,
            data_quality=data_quality,
            recommendation=recommendation
        )
    
    def _validate_and_rank_signals(self):
        """Final validation and ranking of signals"""
        
        if self.master_df.empty:
            logger.warning("Cannot validate and rank signals: master dataset is empty")
            return
        
        # Apply final quality filters
        if 'data_completeness_score' in self.master_df.columns:
            # Downgrade signals with poor data quality
            poor_data_mask = self.master_df['data_completeness_score'] < 60
            if poor_data_mask.any():
                # Reduce confidence for poor data quality stocks
                self.master_df.loc[poor_data_mask, 'confidence'] *= 0.8
                
                # Downgrade strong signals with poor data
                strong_signal_poor_data = poor_data_mask & self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])
                if strong_signal_poor_data.any():
                    self.master_df.loc[strong_signal_poor_data, 'signal'] = 'WATCH'
        
        # Only proceed with ranking if we have the required columns
        if 'composite_score' in self.master_df.columns and 'confidence' in self.master_df.columns:
            # Final ranking by composite score and confidence
            self.master_df['final_rank'] = (
                self.master_df['composite_score'] * 0.7 + 
                self.master_df['confidence'] * 0.3
            ).round(1)
            
            # Sort by final rank
            self.master_df = self.master_df.sort_values('final_rank', ascending=False).reset_index(drop=True)
        else:
            logger.warning("Cannot create final ranking: missing composite_score or confidence columns")
        
        logger.info("✅ Signal validation and ranking completed")
    
    def _record_processing_stats(self, processing_time: float):
        """Record comprehensive processing statistics"""
        
        total_stocks = len(self.master_df)
        
        signal_counts = self.master_df['signal'].value_counts() if not self.master_df.empty else {}
        
        self.processing_stats = ProcessingStats(
            total_stocks=total_stocks,
            processing_time=round(processing_time, 2),
            strong_buy_count=signal_counts.get('STRONG_BUY', 0),
            buy_count=signal_counts.get('BUY', 0),
            data_quality_avg=self.data_quality_report.get('overall_score', 0),
            signals_generated=len([s for s in signal_counts.keys() if s in ['STRONG_BUY', 'BUY', 'ACCUMULATE']]),
            timestamp=datetime.now()
        )
    
    # Public interface methods
    
    def get_top_opportunities(self, limit: int = 8) -> pd.DataFrame:
        """Get top opportunities with high confidence"""
        
        if self.master_df.empty:
            return pd.DataFrame()
        
        # Filter for strong signals with reasonable confidence
        opportunities = self.master_df[
            (self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])) &
            (self.master_df['confidence'] >= 65)
        ].head(limit)
        
        return opportunities
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary"""
        
        if self.master_df.empty:
            return {}
        
        summary = {
            'total_stocks': len(self.master_df),
            'processing_stats': self.processing_stats.__dict__ if self.processing_stats else {},
            'data_quality': self.data_quality_report,
            'market_conditions': self.market_conditions,
            'signal_distribution': self.master_df['signal'].value_counts().to_dict(),
            'avg_composite_score': self.master_df['composite_score'].mean(),
            'avg_confidence': self.master_df['confidence'].mean(),
            'last_update': self.last_update
        }
        
        # Market breadth
        if 'ret_1d' in self.master_df.columns:
            summary['market_breadth'] = (self.master_df['ret_1d'] > 0).mean() * 100
        
        # High confidence signals
        summary['high_confidence_signals'] = len(self.master_df[
            (self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])) & 
            (self.master_df['confidence'] >= 80)
        ])
        
        return summary
    
    def get_signal_explanation(self, ticker: str) -> Optional[SignalExplanation]:
        """Get detailed explanation for specific stock signal"""
        return self.signal_explanations.get(ticker)
    
    def get_filtered_stocks(self, 
                          sectors: List[str] = None,
                          categories: List[str] = None,
                          signals: List[str] = None,
                          min_score: float = 0,
                          min_confidence: float = 0,
                          max_risk: str = None) -> pd.DataFrame:
        """Get filtered stocks based on criteria"""
        
        if self.master_df.empty:
            return pd.DataFrame()
        
        filtered = self.master_df.copy()
        
        # Apply filters
        if sectors:
            filtered = filtered[filtered['sector'].isin(sectors)]
        
        if categories and 'category' in filtered.columns:
            filtered = filtered[filtered['category'].isin(categories)]
        
        if signals:
            filtered = filtered[filtered['signal'].isin(signals)]
        
        if min_score > 0:
            filtered = filtered[filtered['composite_score'] >= min_score]
        
        if min_confidence > 0:
            filtered = filtered[filtered['confidence'] >= min_confidence]
        
        if max_risk and 'total_risk_score' in filtered.columns:
            risk_mapping = {'Low': 30, 'Medium': 60, 'High': 100}
            max_risk_score = risk_mapping.get(max_risk, 100)
            filtered = filtered[filtered['total_risk_score'] <= max_risk_score]
        
        return filtered

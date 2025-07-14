"""
core.py - M.A.N.T.R.A. ULTIMATE Engine
======================================
Enterprise-grade signal processing using ALL sophisticated data
Built for maximum accuracy with 2200+ stocks at lightning speed
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Tuple, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

from config import *

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltimateMANTRAEngine:
    """
    Ultimate MANTRA Engine - Enterprise-grade signal processing
    
    Features:
    - 7-factor advanced scoring system
    - Comprehensive data utilization
    - Multi-timeframe analysis (1d to 5y)
    - Advanced volume pattern recognition
    - EPS growth trend analysis
    - Sector rotation intelligence
    - Signal confidence calculation
    - Optimized for 2200+ stocks
    """
    
    def __init__(self):
        self.watchlist_df = pd.DataFrame()
        self.returns_df = pd.DataFrame()
        self.sectors_df = pd.DataFrame()
        self.master_df = pd.DataFrame()
        self.last_update = None
        self.data_quality = {}
        self.processing_stats = {}
    
    def load_and_process(self) -> Tuple[bool, str]:
        """
        Ultimate data loading and processing pipeline
        Returns: (success, message)
        """
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting Ultimate MANTRA processing...")
            
            # Step 1: Parallel data loading
            success, message = self._load_all_data()
            if not success:
                return False, message
            
            # Step 2: Data cleaning and validation
            self._clean_and_validate_data()
            
            # Step 3: Advanced data enrichment
            self._enrich_data()
            
            # Step 4: Master dataframe creation with all data merged
            self._create_master_dataframe()
            
            # Step 5: Ultimate 7-factor signal generation
            self._calculate_ultimate_signals()
            
            # Step 6: Signal confidence and validation
            self._calculate_signal_confidence()
            
            # Step 7: Quality assessment and performance metrics
            self._assess_ultimate_quality()
            
            # Record processing time
            processing_time = time.time() - start_time
            self.processing_stats = {
                'processing_time': round(processing_time, 2),
                'stocks_processed': len(self.master_df),
                'signals_generated': len(self.master_df[self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])]),
                'timestamp': datetime.now()
            }
            
            self.last_update = time.time()
            logger.info(f"✅ Ultimate processing complete: {len(self.master_df)} stocks in {processing_time:.2f}s")
            
            return True, f"🎯 {len(self.master_df)} stocks processed with ultimate precision in {processing_time:.1f}s"
            
        except Exception as e:
            error_msg = f"❌ Ultimate processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _load_all_data(self) -> Tuple[bool, str]:
        """Advanced parallel data loading from all sheets"""
        
        def fetch_sheet(name: str, url: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Fetch individual sheet with enhanced error handling"""
            try:
                logger.info(f"Loading {name}...")
                response = requests.get(url, timeout=45)
                response.raise_for_status()
                
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"✅ {name}: {len(df)} rows loaded")
                return name, df
                
            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
                return name, None
        
        # Parallel loading with all three sheets
        results = {}
        with ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG['parallel_workers']) as executor:
            futures = {
                executor.submit(fetch_sheet, name, url): name 
                for name, url in DATA_URLS.items()
            }
            
            for future in as_completed(futures):
                name, df = future.result()
                if df is not None:
                    results[name] = df
        
        # Validate essential data
        if 'watchlist' not in results:
            return False, "❌ Critical: Watchlist data failed to load"
        
        # Store loaded data
        self.watchlist_df = results.get('watchlist', pd.DataFrame())
        self.returns_df = results.get('returns', pd.DataFrame())
        self.sectors_df = results.get('sectors', pd.DataFrame())
        
        loaded_sheets = len(results)
        return True, f"✅ {loaded_sheets}/3 data sheets loaded successfully"
    
    def _clean_and_validate_data(self):
        """Ultimate data cleaning with comprehensive validation"""
        
        # Clean watchlist data (primary dataset)
        if not self.watchlist_df.empty:
            self.watchlist_df = self._clean_watchlist_data(self.watchlist_df)
        
        # Clean returns analysis data
        if not self.returns_df.empty:
            self.returns_df = self._clean_returns_data(self.returns_df)
        
        # Clean sector data
        if not self.sectors_df.empty:
            self.sectors_df = self._clean_sector_data(self.sectors_df)
    
    def _clean_watchlist_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced watchlist data cleaning"""
        
        # 1. Rename columns to standard mapping
        df = df.rename(columns={k: v for k, v in WATCHLIST_COLUMNS.items() if k in df.columns})
        
        # 2. Keep available essential columns
        available_cols = [col for col in WATCHLIST_COLUMNS.values() if col in df.columns]
        df = df[available_cols]
        
        # 3. Clean ticker symbols (critical for merging)
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[~df['ticker'].isin(['NAN', 'NONE', '', 'NULL'])]
            df = df.dropna(subset=['ticker'])
            df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # 4. Advanced numeric conversion with error handling
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
        
        # 5. Data validation and filtering
        if 'price' in df.columns:
            df = df[df['price'] >= VALIDATION_RULES['min_price']]
        
        if 'pe' in df.columns:
            df.loc[df['pe'] > VALIDATION_RULES['max_pe'], 'pe'] = np.nan
        
        # 6. Smart defaults for missing critical data
        defaults = {
            'price': 0, 'pe': 0, 'eps_current': 0, 'eps_change_pct': 0,
            'vol_1d': 0, 'rvol': 1.0, 'from_low_pct': 50, 'from_high_pct': -50,
            'ret_1d': 0, 'ret_7d': 0, 'ret_30d': 0, 'ret_3m': 0
        }
        
        for col, default_val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
        
        return df.reset_index(drop=True)
    
    def _clean_returns_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean returns analysis data"""
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in RETURNS_COLUMNS.items() if k in df.columns})
        
        # Keep available columns
        available_cols = [col for col in RETURNS_COLUMNS.values() if col in df.columns]
        df = df[available_cols]
        
        # Clean ticker for merging
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if 'ret' in col or 'avg' in col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.reset_index(drop=True)
    
    def _clean_sector_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean comprehensive sector data"""
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in SECTOR_COLUMNS.items() if k in df.columns})
        
        # Keep available columns
        available_cols = [col for col in SECTOR_COLUMNS.values() if col in df.columns]
        df = df[available_cols]
        
        # Clean sector names
        if 'sector' in df.columns:
            df['sector'] = df['sector'].astype(str).str.strip()
            df = df[df['sector'] != '']
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if 'ret' in col or 'avg' in col or 'count' in col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.reset_index(drop=True)
    
    def _parse_market_cap_advanced(self, series: pd.Series) -> pd.Series:
        """Advanced market cap parsing with enhanced Indian notation support"""
        
        if series.dtype not in ['object', 'string']:
            return pd.to_numeric(series, errors='coerce')
        
        # Convert to string and clean
        s = series.astype(str).str.upper().str.strip()
        s = s.str.replace(',', '')  # Remove commas
        
        result = pd.Series(0.0, index=series.index)
        
        # Handle Crores (most common in Indian markets)
        cr_mask = s.str.contains('CR', na=False)
        if cr_mask.any():
            cr_values = s[cr_mask].str.extract(r'([\d.]+)', expand=False)
            result[cr_mask] = pd.to_numeric(cr_values, errors='coerce') * 1e7
        
        # Handle Lakhs
        lakh_mask = s.str.contains('L', na=False) & ~cr_mask
        if lakh_mask.any():
            lakh_values = s[lakh_mask].str.extract(r'([\d.]+)', expand=False)
            result[lakh_mask] = pd.to_numeric(lakh_values, errors='coerce') * 1e5
        
        # Handle K (thousands)
        k_mask = s.str.contains('K', na=False) & ~cr_mask & ~lakh_mask
        if k_mask.any():
            k_values = s[k_mask].str.extract(r'([\d.]+)', expand=False)
            result[k_mask] = pd.to_numeric(k_values, errors='coerce') * 1e3
        
        # Handle direct numbers
        num_mask = ~cr_mask & ~lakh_mask & ~k_mask
        if num_mask.any():
            result[num_mask] = pd.to_numeric(s[num_mask], errors='coerce')
        
        return result
    
    def _enrich_data(self):
        """Advanced data enrichment with sophisticated calculations"""
        
        if self.watchlist_df.empty:
            return
        
        df = self.watchlist_df
        
        # 1. Enhanced 52-week position analysis
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            df['position_52w'] = (df['price'] - df['low_52w']) / (df['high_52w'] - df['low_52w']) * 100
            df['position_52w'] = df['position_52w'].clip(0, 100)
        
        # 2. Advanced moving average analysis
        for sma in ['sma20', 'sma50', 'sma200']:
            if all(col in df.columns for col in ['price', sma]):
                df[f'above_{sma}'] = df['price'] > df[sma]
                df[f'pct_above_{sma}'] = ((df['price'] - df[sma]) / df[sma] * 100).fillna(0)
        
        # 3. Multi-timeframe momentum trends
        timeframes = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
        available_timeframes = [tf for tf in timeframes if tf in df.columns]
        
        if len(available_timeframes) >= 3:
            # Count positive timeframes
            positive_count = sum(df[tf] > 0 for tf in available_timeframes)
            df['momentum_breadth'] = positive_count / len(available_timeframes) * 100
            
            # Momentum acceleration (short-term vs long-term)
            if all(tf in df.columns for tf in ['ret_7d', 'ret_30d']):
                df['momentum_acceleration'] = df['ret_7d'] - df['ret_30d']
        
        # 4. Advanced volume analysis
        volume_ratios = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        available_ratios = [vr for vr in volume_ratios if vr in df.columns]
        
        if available_ratios:
            # Volume trend (increasing vs decreasing activity)
            if len(available_ratios) >= 2:
                df['volume_trend'] = df[available_ratios[0]] - df[available_ratios[-1]]
            
            # Volume spike detection
            if 'vol_ratio_1d_90d' in df.columns:
                df['volume_spike'] = df['vol_ratio_1d_90d'] >= VOLUME_BENCHMARKS['strong_spike']
        
        # 5. EPS quality analysis
        if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
            # EPS consistency (avoiding one-time gains)
            df['eps_consistency'] = np.where(
                (df['eps_current'] > 0) & (df['eps_last_qtr'] > 0),
                100, 50
            )
        
        if 'eps_change_pct' in df.columns:
            # EPS growth stability
            df['eps_growth_quality'] = pd.cut(
                df['eps_change_pct'].fillna(0),
                bins=[-np.inf, -10, 0, 10, 25, np.inf],
                labels=['Declining', 'Weak', 'Stable', 'Growing', 'Accelerating']
            )
        
        # 6. Market cap categorization with precise thresholds
        if 'mcap' in df.columns:
            df['mcap_category_precise'] = pd.cut(
                df['mcap'].fillna(0),
                bins=[0, 2e9, 1e10, 5e10, 2e11, np.inf],
                labels=['Nano', 'Small', 'Mid', 'Large', 'Mega']
            )
        
        # 7. Sector strength calculation (if sector data available)
        if not self.sectors_df.empty and 'sector' in df.columns:
            sector_strength = self.sectors_df.set_index('sector')['sector_ret_30d'].to_dict()
            df['sector_strength'] = df['sector'].map(sector_strength).fillna(0)
        
        # 8. Risk indicators
        risk_factors = []
        
        # PE-based risk
        if 'pe' in df.columns:
            pe_risk = np.where(df['pe'] > VALUE_BENCHMARKS['expensive'], 1, 0)
            risk_factors.append(pe_risk)
        
        # Volume-based risk
        if 'vol_1d' in df.columns:
            volume_risk = np.where(df['vol_1d'] < VALIDATION_RULES['min_volume'], 1, 0)
            risk_factors.append(volume_risk)
        
        # Momentum-based risk
        if 'ret_30d' in df.columns:
            momentum_risk = np.where(df['ret_30d'] < -15, 1, 0)
            risk_factors.append(momentum_risk)
        
        if risk_factors:
            df['total_risk_factors'] = np.sum(risk_factors, axis=0)
        
        self.watchlist_df = df
    
    def _create_master_dataframe(self):
        """Create unified master dataframe with all data merged"""
        
        master = self.watchlist_df.copy()
        
        # Merge returns analysis data if available
        if not self.returns_df.empty and 'ticker' in self.returns_df.columns:
            master = master.merge(
                self.returns_df, 
                on='ticker', 
                how='left', 
                suffixes=('', '_returns')
            )
        
        # Add sector data as additional columns
        if not self.sectors_df.empty and 'sector' in master.columns:
            sector_data = self.sectors_df.set_index('sector')
            
            # Add key sector metrics to each stock
            for col in ['sector_ret_30d', 'sector_avg_30d', 'sector_count']:
                if col in sector_data.columns:
                    master[f'stock_{col}'] = master['sector'].map(sector_data[col]).fillna(0)
        
        self.master_df = master
    
    def _calculate_ultimate_signals(self):
        """Ultimate 7-factor signal generation system"""
        
        if self.master_df.empty:
            return
        
        df = self.master_df
        
        # Calculate all 7 factor scores
        df['momentum_score'] = self._calculate_ultimate_momentum_score(df)
        df['value_score'] = self._calculate_ultimate_value_score(df)
        df['growth_score'] = self._calculate_ultimate_growth_score(df)
        df['volume_score'] = self._calculate_ultimate_volume_score(df)
        df['technical_score'] = self._calculate_ultimate_technical_score(df)
        df['sector_score'] = self._calculate_ultimate_sector_score(df)
        df['quality_score'] = self._calculate_ultimate_quality_score(df)
        
        # Calculate weighted composite score
        df['composite_score'] = (
            df['momentum_score'] * WEIGHTS['momentum'] +
            df['value_score'] * WEIGHTS['value'] +
            df['growth_score'] * WEIGHTS['growth'] +
            df['volume_score'] * WEIGHTS['volume'] +
            df['technical_score'] * WEIGHTS['technical'] +
            df['sector_score'] * WEIGHTS['sector'] +
            df['quality_score'] * WEIGHTS['quality']
        ).round(1)
        
        # Generate signals based on composite score
        df['signal'] = self._generate_ultimate_signals(df['composite_score'])
        
        # Calculate risk levels
        df['risk_level'] = self._calculate_ultimate_risk(df)
        
        # Position strength indicator
        df['position_strength'] = self._calculate_position_strength(df)
        
        self.master_df = df.sort_values('composite_score', ascending=False)
    
    def _calculate_ultimate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Advanced multi-timeframe momentum analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Multi-timeframe momentum with sophisticated weighting
        timeframe_weights = {
            'ret_1d': 0.05,    # 5% - Very short term
            'ret_3d': 0.10,    # 10% - Short term reaction
            'ret_7d': 0.15,    # 15% - Weekly trend
            'ret_30d': 0.35,   # 35% - Monthly trend (most important)
            'ret_3m': 0.25,    # 25% - Quarterly trend
            'ret_6m': 0.10     # 10% - Semi-annual trend
        }
        
        momentum_components = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for timeframe, weight in timeframe_weights.items():
            if timeframe in df.columns:
                returns = df[timeframe].fillna(0)
                
                # Convert to score scale with non-linear scaling
                normalized = 50 + np.tanh(returns / 20) * 40  # Tanh for non-linear scaling
                momentum_components += normalized * weight
                total_weight += weight
        
        if total_weight > 0:
            score = momentum_components / total_weight
        
        # Momentum consistency bonus
        if 'momentum_breadth' in df.columns:
            consistency_bonus = (df['momentum_breadth'] - 50) / 5  # ±10 points
            score += consistency_bonus
        
        # Momentum acceleration factor
        if 'momentum_acceleration' in df.columns:
            acceleration_factor = np.clip(df['momentum_acceleration'] / 2, -5, 5)
            score += acceleration_factor
        
        return score.clip(0, 100).round(1)
    
    def _calculate_ultimate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Comprehensive value analysis with market context"""
        
        score = pd.Series(50.0, index=df.index)
        
        if 'pe' not in df.columns:
            return score
        
        pe = df['pe'].fillna(0)
        
        # Advanced PE-based scoring with market context
        conditions = [
            (pe > 0) & (pe <= VALUE_BENCHMARKS['deep_value']),      # Deep value
            (pe > VALUE_BENCHMARKS['deep_value']) & (pe <= VALUE_BENCHMARKS['fair_value']),  # Fair value
            (pe > VALUE_BENCHMARKS['fair_value']) & (pe <= VALUE_BENCHMARKS['growth_premium']),  # Growth premium
            (pe > VALUE_BENCHMARKS['growth_premium']) & (pe <= VALUE_BENCHMARKS['expensive']),  # Expensive
            (pe > VALUE_BENCHMARKS['expensive']) & (pe <= VALUE_BENCHMARKS['overvalued']),  # Overvalued
            pe > VALUE_BENCHMARKS['overvalued'],  # Significantly overvalued
            pe <= 0  # No earnings or negative
        ]
        
        scores = [95, 80, 65, 45, 25, 10, 20]
        pe_score = pd.Series(np.select(conditions, scores, default=50), index=df.index)
        
        # Earnings quality adjustment
        if 'eps_current' in df.columns:
            profitable_bonus = np.where(df['eps_current'] > 0, 10, -10)
            pe_score += profitable_bonus
        
        # Market cap adjusted expectations
        if 'mcap_category_precise' in df.columns:
            # Small caps can have higher PE multiples
            mcap_adjustment = df['mcap_category_precise'].map({
                'Nano': 5, 'Small': 3, 'Mid': 0, 'Large': -2, 'Mega': -5
            }).fillna(0)
            pe_score += mcap_adjustment
        
        return pe_score.clip(0, 100).round(1)
    
    def _calculate_ultimate_growth_score(self, df: pd.DataFrame) -> pd.Series:
        """Advanced EPS growth and earnings quality analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        if 'eps_change_pct' not in df.columns:
            return score
        
        eps_growth = df['eps_change_pct'].fillna(0)
        
        # EPS growth scoring with non-linear scaling
        conditions = [
            eps_growth >= GROWTH_BENCHMARKS['excellent'],     # Excellent growth
            eps_growth >= GROWTH_BENCHMARKS['strong'],        # Strong growth
            eps_growth >= GROWTH_BENCHMARKS['decent'],        # Decent growth
            eps_growth >= GROWTH_BENCHMARKS['weak'],          # Weak growth
            eps_growth >= GROWTH_BENCHMARKS['declining'],     # Declining but not terrible
            eps_growth < GROWTH_BENCHMARKS['declining']       # Significantly declining
        ]
        
        scores = [95, 80, 65, 45, 25, 10]
        growth_score = pd.Series(np.select(conditions, scores, default=50), index=df.index)
        
        # EPS consistency bonus
        if 'eps_consistency' in df.columns:
            consistency_bonus = (df['eps_consistency'] - 50) / 10  # ±5 points
            growth_score += consistency_bonus
        
        # Growth quality assessment
        if 'eps_growth_quality' in df.columns:
            quality_bonus = df['eps_growth_quality'].map({
                'Accelerating': 15, 'Growing': 10, 'Stable': 0, 
                'Weak': -5, 'Declining': -15
            }).fillna(0)
            growth_score += quality_bonus
        
        return growth_score.clip(0, 100).round(1)
    
    def _calculate_ultimate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """Advanced volume pattern analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Primary volume score using RVOL
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            
            conditions = [
                rvol >= VOLUME_BENCHMARKS['extreme_spike'],    # Extreme volume
                rvol >= VOLUME_BENCHMARKS['strong_spike'],     # Strong volume spike
                rvol >= VOLUME_BENCHMARKS['elevated'],         # Elevated volume
                rvol >= VOLUME_BENCHMARKS['normal'],           # Normal volume
                rvol >= VOLUME_BENCHMARKS['weak'],             # Weak volume
                rvol < VOLUME_BENCHMARKS['weak']               # Very weak volume
            ]
            
            scores = [95, 85, 70, 55, 35, 15]
            volume_score = pd.Series(np.select(conditions, scores, default=50), index=df.index)
        else:
            volume_score = score.copy()
        
        # Advanced volume ratio analysis
        if 'vol_ratio_1d_90d' in df.columns:
            vol_ratio = df['vol_ratio_1d_90d'].fillna(1.0)
            ratio_bonus = np.clip((vol_ratio - 1) * 10, -15, 15)
            volume_score += ratio_bonus
        
        # Volume trend analysis
        if 'volume_trend' in df.columns:
            trend_factor = np.clip(df['volume_trend'] * 5, -10, 10)
            volume_score += trend_factor
        
        # Volume-price confirmation
        if all(col in df.columns for col in ['ret_1d', 'volume_spike']):
            # Positive price action with volume spike
            confirmation = df['volume_spike'] & (df['ret_1d'] > 1)
            volume_score += confirmation * 10
        
        return volume_score.clip(0, 100).round(1)
    
    def _calculate_ultimate_technical_score(self, df: pd.DataFrame) -> pd.Series:
        """Advanced technical analysis with multiple indicators"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Moving average trend analysis
        sma_weights = {'above_sma20': 0.4, 'above_sma50': 0.35, 'above_sma200': 0.25}
        
        for sma_col, weight in sma_weights.items():
            if sma_col in df.columns:
                sma_score = df[sma_col] * 100 * weight
                score += sma_score
        
        # 52-week position analysis
        if 'position_52w' in df.columns:
            position = df['position_52w'].fillna(50)
            
            # Non-linear position scoring (favor higher positions)
            position_score = np.where(
                position >= POSITION_BENCHMARKS['near_highs'], 25,
                np.where(position >= POSITION_BENCHMARKS['upper_range'], 15,
                np.where(position >= POSITION_BENCHMARKS['middle_range'], 0,
                np.where(position >= POSITION_BENCHMARKS['lower_range'], -10, -20)))
            )
            score += position_score
        
        # Percentage above moving averages (strength indicator)
        for pct_col in ['pct_above_sma20', 'pct_above_sma50', 'pct_above_sma200']:
            if pct_col in df.columns:
                pct_strength = np.clip(df[pct_col] / 2, -5, 5)  # ±5 points max
                score += pct_strength
        
        return score.clip(0, 100).round(1)
    
    def _calculate_ultimate_sector_score(self, df: pd.DataFrame) -> pd.Series:
        """Sector rotation and relative strength analysis"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Sector relative performance
        if 'sector_strength' in df.columns:
            sector_strength = df['sector_strength'].fillna(0)
            # Convert sector returns to score
            strength_score = 50 + np.clip(sector_strength * 2, -40, 40)
            score = strength_score
        
        # Sector average performance
        if 'stock_sector_avg_30d' in df.columns:
            avg_performance = df['stock_sector_avg_30d'].fillna(0)
            avg_bonus = np.clip(avg_performance, -10, 10)
            score += avg_bonus
        
        # Sector size factor (diversification)
        if 'stock_sector_count' in df.columns:
            # Prefer sectors with reasonable number of stocks
            sector_count = df['stock_sector_count'].fillna(10)
            size_factor = np.where(
                sector_count >= 20, 5,      # Large, diversified sectors
                np.where(sector_count >= 10, 0,  # Medium sectors
                np.where(sector_count >= 5, -3, -5))  # Small sectors
            )
            score += size_factor
        
        return score.clip(0, 100).round(1)
    
    def _calculate_ultimate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Financial quality and stability assessment"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Earnings quality
        if 'eps_consistency' in df.columns:
            earnings_quality = (df['eps_consistency'] - 50) / 2  # Scale to ±25 points
            score += earnings_quality
        
        # Market cap stability (larger companies = more stable)
        if 'mcap_category_precise' in df.columns:
            stability_score = df['mcap_category_precise'].map({
                'Nano': 20, 'Small': 35, 'Mid': 50, 'Large': 70, 'Mega': 85
            }).fillna(50)
            score = (score + stability_score) / 2  # Average with stability
        
        # Risk factor assessment
        if 'total_risk_factors' in df.columns:
            risk_penalty = df['total_risk_factors'] * 15  # -15 points per risk factor
            score -= risk_penalty
        
        # Volume liquidity quality
        if 'vol_1d' in df.columns:
            liquidity_bonus = np.where(df['vol_1d'] >= 100000, 10, 0)  # Liquid stocks bonus
            score += liquidity_bonus
        
        return score.clip(0, 100).round(1)
    
    def _generate_ultimate_signals(self, scores: pd.Series) -> pd.Series:
        """Generate sophisticated trading signals"""
        
        conditions = [
            scores >= SIGNALS['STRONG_BUY'],
            scores >= SIGNALS['BUY'], 
            scores >= SIGNALS['ACCUMULATE'],
            scores >= SIGNALS['WATCH'],
            scores >= SIGNALS['NEUTRAL'],
            scores >= SIGNALS['AVOID']
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH', 'NEUTRAL', 'AVOID']
        
        return pd.Series(
            np.select(conditions, choices, default='STRONG_AVOID'),
            index=scores.index
        )
    
    def _calculate_ultimate_risk(self, df: pd.DataFrame) -> pd.Series:
        """Comprehensive risk assessment"""
        
        risk_score = pd.Series(0.0, index=df.index)
        
        # Valuation risk
        if 'pe' in df.columns:
            pe_risk = np.where(df['pe'] > VALUE_BENCHMARKS['expensive'], 30, 0)
            risk_score += pe_risk
        
        # Liquidity risk
        if 'vol_1d' in df.columns:
            liquidity_risk = np.where(df['vol_1d'] < 50000, 25, 0)
            risk_score += liquidity_risk
        
        # Momentum risk
        if 'ret_30d' in df.columns:
            momentum_risk = np.where(df['ret_30d'] < -20, 35, 0)
            risk_score += momentum_risk
        
        # Earnings quality risk
        if 'eps_change_pct' in df.columns:
            earnings_risk = np.where(df['eps_change_pct'] < -15, 20, 0)
            risk_score += earnings_risk
        
        # Market cap risk (smaller = riskier)
        if 'mcap_category_precise' in df.columns:
            size_risk = df['mcap_category_precise'].map({
                'Nano': 40, 'Small': 25, 'Mid': 10, 'Large': 5, 'Mega': 0
            }).fillna(15)
            risk_score += size_risk
        
        # Convert to risk categories
        conditions = [
            risk_score <= 30,
            risk_score <= 60,
            risk_score <= 90,
            risk_score > 90
        ]
        
        categories = ['Low', 'Medium', 'High', 'Extreme']
        
        return pd.Series(
            np.select(conditions, categories, default='Medium'),
            index=df.index
        )
    
    def _calculate_position_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall position strength indicator"""
        
        strength = pd.Series(50.0, index=df.index)
        
        # 52-week position
        if 'position_52w' in df.columns:
            strength += (df['position_52w'] - 50) / 2
        
        # Momentum strength
        if 'momentum_score' in df.columns:
            strength += (df['momentum_score'] - 50) / 4
        
        # Technical strength
        if 'technical_score' in df.columns:
            strength += (df['technical_score'] - 50) / 4
        
        # Volume confirmation
        if 'volume_score' in df.columns:
            strength += (df['volume_score'] - 50) / 8
        
        return strength.clip(0, 100).round(1)
    
    def _calculate_signal_confidence(self):
        """Calculate confidence levels for each signal"""
        
        df = self.master_df
        
        if df.empty:
            return
        
        # Calculate confidence based on factor alignment
        scores = ['momentum_score', 'value_score', 'growth_score', 'volume_score', 
                 'technical_score', 'sector_score', 'quality_score']
        
        available_scores = [col for col in scores if col in df.columns]
        
        if not available_scores:
            df['confidence'] = 50.0
            return
        
        confidence_list = []
        
        for idx, row in df.iterrows():
            factor_scores = [row[col] for col in available_scores if pd.notna(row[col])]
            
            if not factor_scores:
                confidence_list.append(50.0)
                continue
            
            # High confidence when factors are aligned (low standard deviation)
            mean_score = np.mean(factor_scores)
            std_dev = np.std(factor_scores)
            
            # Base confidence from mean score
            base_confidence = mean_score
            
            # Alignment bonus (lower deviation = higher confidence)
            alignment_bonus = max(0, 25 - std_dev)
            
            # Number of factors bonus (more factors = higher confidence)
            factor_bonus = min(10, len(factor_scores) * 1.5)
            
            total_confidence = min(100, base_confidence + alignment_bonus + factor_bonus)
            confidence_list.append(round(total_confidence, 1))
        
        df['confidence'] = confidence_list
        self.master_df = df
    
    def _assess_ultimate_quality(self):
        """Comprehensive data quality assessment"""
        
        if self.master_df.empty:
            self.data_quality = {'score': 0, 'status': 'No Data', 'details': {}}
            return
        
        # Essential columns for quality assessment
        critical_columns = [
            'ticker', 'price', 'sector', 'pe', 'ret_30d', 
            'vol_1d', 'composite_score', 'signal'
        ]
        
        completeness_scores = []
        details = {}
        
        for col in critical_columns:
            if col in self.master_df.columns:
                completeness = (self.master_df[col].notna().sum() / len(self.master_df)) * 100
                completeness_scores.append(completeness)
                details[col] = round(completeness, 1)
        
        # Calculate overall quality score
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
        
        # Determine quality status
        if avg_completeness >= QUALITY_THRESHOLDS['excellent']:
            status = 'Excellent'
        elif avg_completeness >= QUALITY_THRESHOLDS['good']:
            status = 'Good'
        elif avg_completeness >= QUALITY_THRESHOLDS['acceptable']:
            status = 'Acceptable'
        elif avg_completeness >= QUALITY_THRESHOLDS['poor']:
            status = 'Poor'
        else:
            status = 'Critical'
        
        self.data_quality = {
            'score': round(avg_completeness, 1),
            'status': status,
            'details': {
                'total_stocks': len(self.master_df),
                'signals_generated': len(self.master_df[self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])]),
                'data_completeness': details,
                'sheets_loaded': {
                    'watchlist': not self.watchlist_df.empty,
                    'returns': not self.returns_df.empty,
                    'sectors': not self.sectors_df.empty
                }
            }
        }
    
    def get_ultimate_opportunities(self, limit: int = 16) -> pd.DataFrame:
        """Get top investment opportunities with ultimate analysis"""
        
        if self.master_df.empty:
            return pd.DataFrame()
        
        # Filter for strong signals with high confidence
        strong_signals = self.master_df[
            (self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])) &
            (self.master_df['confidence'] >= 60)
        ]
        
        return strong_signals.nlargest(limit, 'composite_score')
    
    def get_ultimate_summary(self) -> Dict:
        """Get comprehensive market summary with ultimate insights"""
        
        if self.master_df.empty:
            return {}
        
        df = self.master_df
        total_stocks = len(df)
        
        summary = {
            'total_stocks': total_stocks,
            'buy_signals': len(df[df['signal'].isin(['STRONG_BUY', 'BUY'])]),
            'accumulate_signals': len(df[df['signal'] == 'ACCUMULATE']),
            'avg_composite_score': df['composite_score'].mean(),
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
            'data_quality': self.data_quality,
            'processing_stats': self.processing_stats,
            'last_update': self.last_update
        }
        
        # Market breadth analysis
        if 'ret_1d' in df.columns:
            summary['market_breadth'] = (df['ret_1d'] > 0).sum() / total_stocks * 100
            summary['strong_momentum'] = len(df[df['momentum_score'] > 80])
        
        # Volume activity
        if 'volume_spike' in df.columns:
            summary['volume_spikes'] = df['volume_spike'].sum()
        
        # High confidence opportunities
        if 'confidence' in df.columns:
            summary['high_confidence_signals'] = len(df[
                (df['signal'].isin(['STRONG_BUY', 'BUY'])) & 
                (df['confidence'] >= 80)
            ])
        
        # Risk distribution
        if 'risk_level' in df.columns:
            risk_dist = df['risk_level'].value_counts()
            summary['risk_distribution'] = risk_dist.to_dict()
        
        # Sector insights
        if 'sector' in df.columns:
            sector_signals = df[df['signal'].isin(['STRONG_BUY', 'BUY'])]['sector'].value_counts()
            summary['top_sectors'] = sector_signals.head(5).to_dict()
        
        return summary
    
    def get_filtered_data(self, sectors: List[str] = None, 
                         categories: List[str] = None,
                         min_score: float = 0,
                         signals: List[str] = None,
                         risk_levels: List[str] = None) -> pd.DataFrame:
        """Get filtered data based on criteria"""
        
        if self.master_df.empty:
            return pd.DataFrame()
        
        filtered_df = self.master_df.copy()
        
        # Apply filters
        if sectors:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        if categories:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['composite_score'] >= min_score]
        
        if signals:
            filtered_df = filtered_df[filtered_df['signal'].isin(signals)]
        
        if risk_levels:
            filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]
        
        return filtered_df

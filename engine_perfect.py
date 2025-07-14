"""
engine_perfect.py - M.A.N.T.R.A. Version 3 FINAL - Ultimate Signal Engine
========================================================================
Perfect signal generation system with 8-factor precision analysis
Bulletproof data handling and explainable AI for ultra-high confidence
Built for permanent use with comprehensive error handling
"""

import pandas as pd
import numpy as np
import requests
import logging
import time
import io
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config_ultimate import CONFIG
from quality_ultimate import UltimateQualityController, ProcessingResult

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [ENGINE] %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SignalExplanation:
    """Container for signal reasoning and explanation"""
    ticker: str
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

@dataclass
class MarketCondition:
    """Container for market condition analysis"""
    condition: str  # 'bull_market', 'bear_market', 'neutral_market'
    confidence: float
    market_breadth: float
    avg_sector_performance: float
    supporting_evidence: List[str]
    timestamp: datetime

# =============================================================================
# ULTIMATE SIGNAL ENGINE
# =============================================================================

class UltimateSignalEngine:
    """
    Ultimate precision signal engine with explainable AI
    
    Features:
    - 8-factor precision analysis
    - Ultra-conservative thresholds (92+ for STRONG_BUY)
    - Bulletproof data handling
    - Explainable AI for every signal
    - Market condition adaptation
    - Comprehensive error handling
    """
    
    def __init__(self):
        self.config = CONFIG
        self.quality_controller = UltimateQualityController()
        
        # Data storage
        self.watchlist_df = pd.DataFrame()
        self.returns_df = pd.DataFrame()
        self.sectors_df = pd.DataFrame()
        self.master_df = pd.DataFrame()
        
        # Analysis results
        self.processing_stats = None
        self.market_condition = None
        self.signal_explanations = {}
        self.quality_reports = {}
        
        self.last_update = None
        
        logger.info("🔱 Ultimate Signal Engine initialized")
    
    def load_and_process(self) -> Tuple[bool, str]:
        """
        Main processing pipeline with bulletproof error handling
        Returns: (success, status_message)
        """
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting ultimate precision processing...")
            
            # Step 1: Load all data sources
            success, message = self._load_all_data_sources()
            if not success:
                return False, message
            
            # Step 2: Create master dataset
            success, message = self._create_master_dataset()
            if not success:
                return False, message
            
            # Step 3: Detect market conditions
            self._detect_market_conditions()
            
            # Step 4: Generate precision signals
            self._generate_precision_signals()
            
            # Step 5: Calculate explanations
            self._calculate_signal_explanations()
            
            # Step 6: Final validation and ranking
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
        """Load data from all Google Sheets sources with bulletproof handling"""
        
        def fetch_sheet_data(sheet_name: str, url: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Fetch individual sheet with comprehensive error handling"""
            try:
                logger.info(f"Loading {sheet_name} data from Google Sheets...")
                
                # Create session with retry logic
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=requests.packages.urllib3.util.retry.Retry(
                        total=self.config.data_source.MAX_RETRIES,
                        backoff_factor=self.config.data_source.BACKOFF_FACTOR,
                        status_forcelist=[500, 502, 503, 504]
                    )
                )
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                
                response = session.get(url, timeout=self.config.data_source.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))
                
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
        
        # Load all sheets in parallel for maximum performance
        results = {}
        urls = {
            'watchlist': self.config.data_source.get_sheet_url('watchlist'),
            'returns': self.config.data_source.get_sheet_url('returns'),
            'sectors': self.config.data_source.get_sheet_url('sectors')
        }
        
        with ThreadPoolExecutor(max_workers=self.config.performance.PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(fetch_sheet_data, name, url): name 
                for name, url in urls.items()
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
        
        # Process each dataset with quality control
        for sheet_name, df in results.items():
            if not df.empty:
                logger.info(f"🔍 Processing {sheet_name} with quality control...")
                processing_result = self.quality_controller.process_dataframe(df, sheet_name)
                
                if processing_result.success:
                    results[sheet_name] = processing_result.dataframe
                    self.quality_reports[sheet_name] = processing_result.quality_report
                    logger.info(f"✅ {sheet_name} processed: {processing_result.quality_report.status} quality")
                else:
                    logger.warning(f"⚠️ {sheet_name} processing issues: {processing_result.message}")
                    results[sheet_name] = processing_result.dataframe  # Use processed data anyway
                    self.quality_reports[sheet_name] = processing_result.quality_report
        
        # Store processed data
        self.watchlist_df = results.get('watchlist', pd.DataFrame())
        self.returns_df = results.get('returns', pd.DataFrame())
        self.sectors_df = results.get('sectors', pd.DataFrame())
        
        sheets_loaded = len([df for df in results.values() if not df.empty])
        return True, f"✅ Data sources loaded and processed: {sheets_loaded}/3 sheets successful"
    
    def _create_master_dataset(self) -> Tuple[bool, str]:
        """Create comprehensive master dataset with bulletproof merging"""
        
        try:
            logger.info("🔧 Creating master dataset...")
            
            if self.watchlist_df.empty:
                return False, "Cannot create master dataset: watchlist data is empty"
            
            master = self.watchlist_df.copy()
            
            # Merge returns analysis data if available
            if not self.returns_df.empty and 'ticker' in self.returns_df.columns:
                logger.info("Merging returns analysis data...")
                try:
                    # Remove duplicate columns before merge
                    returns_clean = self.returns_df.drop(columns=['company_name'], errors='ignore')
                    
                    master = master.merge(
                        returns_clean,
                        on='ticker',
                        how='left',
                        suffixes=('', '_returns')
                    )
                    logger.info(f"✅ Returns data merged: {len(returns_clean)} records")
                except Exception as e:
                    logger.warning(f"⚠️ Returns data merge failed: {e}")
            
            # Add comprehensive sector data
            if not self.sectors_df.empty and 'sector' in master.columns:
                logger.info("Integrating sector performance data...")
                try:
                    # Create sector lookup
                    sector_data = self.sectors_df.set_index('sector')
                    
                    # Add sector performance columns
                    sector_columns_to_add = [
                        'sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 
                        'sector_ret_3m', 'sector_avg_30d', 'sector_count'
                    ]
                    
                    for col in sector_columns_to_add:
                        if col in sector_data.columns:
                            mapped_values = master['sector'].map(sector_data[col])
                            master[f'current_{col}'] = pd.to_numeric(mapped_values, errors='coerce').fillna(0)
                    
                    logger.info(f"✅ Sector data integrated: {len(sector_columns_to_add)} metrics added")
                except Exception as e:
                    logger.warning(f"⚠️ Sector data integration failed: {e}")
            
            # Add derived features for signal generation
            master = self._add_derived_features(master)
            
            # Final data validation
            if 'ticker' in master.columns:
                master = master.dropna(subset=['ticker'])
                master = master[master['ticker'] != '']
            
            # Ensure we have usable data
            if len(master) == 0:
                return False, "Master dataset is empty after processing"
            
            self.master_df = master.reset_index(drop=True)
            logger.info(f"✅ Master dataset created: {len(self.master_df)} stocks with comprehensive data")
            
            return True, f"Master dataset ready: {len(self.master_df)} stocks"
            
        except Exception as e:
            logger.error(f"❌ Master dataset creation failed: {e}")
            return False, f"Failed to create master dataset: {str(e)}"
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for comprehensive analysis"""
        
        logger.info("🔧 Adding derived features for signal generation...")
        
        try:
            # 1. Enhanced 52-week position analysis
            if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
                price = pd.to_numeric(df['price'], errors='coerce')
                low_52w = pd.to_numeric(df['low_52w'], errors='coerce')
                high_52w = pd.to_numeric(df['high_52w'], errors='coerce')
                
                range_52w = high_52w - low_52w
                range_52w = range_52w.replace(0, np.nan)  # Avoid division by zero
                
                df['position_52w_pct'] = ((price - low_52w) / range_52w * 100).fillna(50).clip(0, 100)
            
            # 2. Moving average relationships
            sma_columns = ['sma20', 'sma50', 'sma200']
            available_smas = [col for col in sma_columns if col in df.columns]
            
            if 'price' in df.columns and available_smas:
                price = pd.to_numeric(df['price'], errors='coerce')
                
                for sma in available_smas:
                    sma_values = pd.to_numeric(df[sma], errors='coerce')
                    df[f'above_{sma}'] = (price > sma_values).fillna(False)
                    df[f'pct_from_{sma}'] = ((price - sma_values) / sma_values * 100).fillna(0).clip(-50, 50)
            
            # 3. Multi-timeframe momentum consistency
            momentum_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m']
            available_momentum = [col for col in momentum_cols if col in df.columns]
            
            if len(available_momentum) >= 2:
                positive_counts = pd.Series(0, index=df.index)
                momentum_values = []
                
                for col in available_momentum:
                    ret_values = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    positive_counts += (ret_values > 0).astype(int)
                    momentum_values.append(ret_values)
                
                df['momentum_breadth'] = (positive_counts / len(available_momentum)) * 100
                
                if len(momentum_values) >= 3:
                    momentum_array = np.array(momentum_values).T
                    momentum_std = pd.Series([np.std(row) for row in momentum_array], index=df.index)
                    df['momentum_consistency'] = (100 - momentum_std * 5).clip(0, 100)
                else:
                    df['momentum_consistency'] = 50.0
            
            # 4. Volume analysis
            if 'rvol' in df.columns:
                rvol = pd.to_numeric(df['rvol'], errors='coerce').fillna(1)
                
                df['volume_strength'] = np.where(
                    rvol >= self.config.benchmarks.VOLUME_BENCHMARKS['extreme_interest'], 100,
                    np.where(rvol >= self.config.benchmarks.VOLUME_BENCHMARKS['strong_interest'], 80,
                    np.where(rvol >= self.config.benchmarks.VOLUME_BENCHMARKS['elevated_interest'], 60,
                    np.where(rvol >= self.config.benchmarks.VOLUME_BENCHMARKS['normal_activity'], 40, 20)))
                )
            
            # 5. EPS quality analysis
            if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
                eps_current = pd.to_numeric(df['eps_current'], errors='coerce')
                eps_last_qtr = pd.to_numeric(df['eps_last_qtr'], errors='coerce')
                
                df['eps_stability'] = np.where(
                    (eps_current > 0) & (eps_last_qtr > 0), 100,
                    np.where((eps_current > 0) | (eps_last_qtr > 0), 60, 20)
                )
            
            # 6. Growth quality assessment
            if 'eps_change_pct' in df.columns:
                eps_change = pd.to_numeric(df['eps_change_pct'], errors='coerce').fillna(0)
                
                df['growth_quality'] = np.where(
                    eps_change >= self.config.benchmarks.GROWTH_BENCHMARKS['accelerating'], 100,
                    np.where(eps_change >= self.config.benchmarks.GROWTH_BENCHMARKS['strong'], 85,
                    np.where(eps_change >= self.config.benchmarks.GROWTH_BENCHMARKS['decent'], 70,
                    np.where(eps_change >= self.config.benchmarks.GROWTH_BENCHMARKS['modest'], 55,
                    np.where(eps_change >= self.config.benchmarks.GROWTH_BENCHMARKS['weak'], 40, 20))))
                )
            
            # 7. Risk assessment
            risk_factors = []
            
            # Valuation risk
            if 'pe' in df.columns:
                pe = pd.to_numeric(df['pe'], errors='coerce')
                valuation_risk = np.where(pe > self.config.benchmarks.VALUE_BENCHMARKS['expensive'], 25, 0)
                risk_factors.append(valuation_risk)
            
            # Liquidity risk
            if 'vol_1d' in df.columns:
                vol_1d = pd.to_numeric(df['vol_1d'], errors='coerce')
                liquidity_risk = np.where(vol_1d < 50000, 20, 0)
                risk_factors.append(liquidity_risk)
            
            # Momentum risk
            if 'ret_30d' in df.columns:
                ret_30d = pd.to_numeric(df['ret_30d'], errors='coerce')
                momentum_risk = np.where(ret_30d < -15, 30, 0)
                risk_factors.append(momentum_risk)
            
            # Calculate total risk score
            if risk_factors:
                df['total_risk_score'] = np.sum(risk_factors, axis=0)
                df['total_risk_score'] = df['total_risk_score'].clip(0, 100)
            else:
                df['total_risk_score'] = 25.0
            
            # 8. Data completeness score
            essential_cols = ['price', 'pe', 'ret_30d', 'vol_1d', 'sector']
            available_essential = [col for col in essential_cols if col in df.columns]
            
            if available_essential:
                completeness_scores = []
                for _, row in df.iterrows():
                    available_data = sum(1 for col in available_essential if pd.notna(row.get(col)))
                    completeness = (available_data / len(available_essential)) * 100
                    completeness_scores.append(completeness)
                df['data_completeness_score'] = completeness_scores
            else:
                df['data_completeness_score'] = 80.0
            
            logger.info("✅ Derived features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Derived features calculation failed: {e}")
            # Ensure required columns exist
            if 'data_completeness_score' not in df.columns:
                df['data_completeness_score'] = 80.0
            if 'total_risk_score' not in df.columns:
                df['total_risk_score'] = 25.0
            return df
    
    def _detect_market_conditions(self):
        """Detect current market conditions for signal adaptation"""
        
        try:
            logger.info("📈 Detecting market conditions...")
            
            if self.master_df.empty or 'ret_1d' not in self.master_df.columns:
                self.market_condition = MarketCondition(
                    condition='neutral_market',
                    confidence=50.0,
                    market_breadth=50.0,
                    avg_sector_performance=0.0,
                    supporting_evidence=['Insufficient data for analysis'],
                    timestamp=datetime.now()
                )
                return
            
            # Calculate market breadth
            ret_1d = pd.to_numeric(self.master_df['ret_1d'], errors='coerce')
            positive_stocks = (ret_1d > 0).sum()
            total_stocks = len(self.master_df)
            market_breadth = (positive_stocks / total_stocks) * 100 if total_stocks > 0 else 50
            
            # Calculate average sector performance
            avg_sector_performance = 0
            if not self.sectors_df.empty and 'sector_ret_1d' in self.sectors_df.columns:
                sector_rets = pd.to_numeric(self.sectors_df['sector_ret_1d'], errors='coerce')
                avg_sector_performance = sector_rets.mean()
            
            # Determine market condition
            supporting_evidence = []
            
            bull_conditions = self.config.market.MARKET_CONDITIONS['bull_market']
            bear_conditions = self.config.market.MARKET_CONDITIONS['bear_market']
            
            if (market_breadth >= bull_conditions['market_breadth_min'] and 
                avg_sector_performance >= bull_conditions['sector_strength_min']):
                condition = 'bull_market'
                confidence = min(100, market_breadth + abs(avg_sector_performance) * 10)
                supporting_evidence = [
                    f"{market_breadth:.1f}% of stocks are positive",
                    f"Average sector performance: {avg_sector_performance:+.1f}%",
                    "Broad-based strength across markets"
                ]
            elif (market_breadth <= bear_conditions['market_breadth_max'] and 
                  avg_sector_performance <= bear_conditions['sector_strength_max']):
                condition = 'bear_market'
                confidence = min(100, (100 - market_breadth) + abs(avg_sector_performance) * 10)
                supporting_evidence = [
                    f"Only {market_breadth:.1f}% of stocks are positive",
                    f"Average sector performance: {avg_sector_performance:+.1f}%",
                    "Widespread weakness across markets"
                ]
            else:
                condition = 'neutral_market'
                confidence = 100 - abs(market_breadth - 50) * 2
                supporting_evidence = [
                    f"Mixed market with {market_breadth:.1f}% positive stocks",
                    f"Sector performance: {avg_sector_performance:+.1f}%",
                    "No clear directional trend"
                ]
            
            self.market_condition = MarketCondition(
                condition=condition,
                confidence=round(confidence, 1),
                market_breadth=round(market_breadth, 1),
                avg_sector_performance=round(avg_sector_performance, 2),
                supporting_evidence=supporting_evidence,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 Market condition: {condition} ({confidence:.1f}% confidence)")
            
        except Exception as e:
            logger.error(f"❌ Market condition detection failed: {e}")
            self.market_condition = MarketCondition(
                condition='neutral_market',
                confidence=50.0,
                market_breadth=50.0,
                avg_sector_performance=0.0,
                supporting_evidence=['Analysis failed - assuming neutral conditions'],
                timestamp=datetime.now()
            )
    
    def _generate_precision_signals(self):
        """Generate ultra-precision signals with 8-factor analysis"""
        
        if self.master_df.empty:
            logger.warning("Cannot generate signals: master dataset is empty")
            return
        
        try:
            logger.info("🎯 Generating ultra-precision signals with 8-factor analysis...")
            
            df = self.master_df.copy()
            
            # Calculate all 8 factor scores
            factor_scores = {}
            
            factor_scores['momentum'] = self._calculate_momentum_factor(df)
            factor_scores['value'] = self._calculate_value_factor(df)
            factor_scores['growth'] = self._calculate_growth_factor(df)
            factor_scores['volume'] = self._calculate_volume_factor(df)
            factor_scores['technical'] = self._calculate_technical_factor(df)
            factor_scores['sector'] = self._calculate_sector_factor(df)
            factor_scores['risk'] = self._calculate_risk_factor(df)
            factor_scores['quality'] = self._calculate_quality_factor(df)
            
            # Add individual factor scores to dataframe
            for factor, scores in factor_scores.items():
                df[f'{factor}_score'] = scores
            
            # Apply market condition adjustments
            if self.market_condition:
                if self.market_condition.condition == 'bull_market':
                    momentum_bias = self.config.market.MARKET_CONDITIONS['bull_market']['momentum_bias']
                    df['momentum_score'] = (df['momentum_score'] * momentum_bias).clip(0, 100)
                elif self.market_condition.condition == 'bear_market':
                    value_bias = self.config.market.MARKET_CONDITIONS['bear_market']['value_bias']
                    df['value_score'] = (df['value_score'] * value_bias).clip(0, 100)
            
            # Calculate weighted composite score
            weights = self.config.signals.FACTOR_WEIGHTS
            df['composite_score'] = (
                df['momentum_score'] * weights['momentum'] +
                df['value_score'] * weights['value'] +
                df['growth_score'] * weights['growth'] +
                df['volume_score'] * weights['volume'] +
                df['technical_score'] * weights['technical'] +
                df['sector_score'] * weights['sector'] +
                (100 - df['risk_score']) * weights['risk'] +  # Invert risk score
                df['quality_score'] * weights['quality']
            ).round(1)
            
            # Generate ultra-conservative signals
            df['signal'] = self._generate_ultra_conservative_signals(df['composite_score'])
            
            # Calculate signal confidence with factor alignment
            df['confidence'] = self._calculate_signal_confidence(df)
            
            # Sort by composite score for ranking
            self.master_df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
            
            # Log signal distribution
            signal_counts = df['signal'].value_counts()
            logger.info(f"📊 Signal distribution: {dict(signal_counts)}")
            
            # Validate signal quality
            strong_buy_count = signal_counts.get('STRONG_BUY', 0)
            buy_count = signal_counts.get('BUY', 0)
            
            if strong_buy_count + buy_count == 0:
                logger.warning("⚠️ No actionable signals generated - consider reviewing thresholds")
            else:
                logger.info(f"✅ Generated {strong_buy_count} STRONG_BUY and {buy_count} BUY signals")
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            # Ensure we have basic scoring even if calculation fails
            if len(self.master_df) > 0:
                self.master_df['composite_score'] = 50.0
                self.master_df['signal'] = 'WATCH'
                self.master_df['confidence'] = 50.0
                # Add default factor scores
                for factor in ['momentum', 'value', 'growth', 'volume', 'technical', 'sector', 'risk', 'quality']:
                    self.master_df[f'{factor}_score'] = 50.0
    
    def _calculate_momentum_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum factor score with multi-timeframe analysis"""
        
        try:
            score = pd.Series(50.0, index=df.index)
            
            # Multi-timeframe weights
            timeframe_weights = {
                'ret_1d': 0.10, 'ret_7d': 0.20, 'ret_30d': 0.40, 'ret_3m': 0.30
            }
            
            momentum_components = pd.Series(0.0, index=df.index)
            total_weight = 0
            
            benchmarks = self.config.benchmarks.MOMENTUM_BENCHMARKS
            
            for timeframe, weight in timeframe_weights.items():
                if timeframe in df.columns:
                    returns = pd.to_numeric(df[timeframe], errors='coerce').fillna(0)
                    
                    # Score based on benchmarks
                    if timeframe == 'ret_1d':
                        thresholds = benchmarks['excellent']['1d'], benchmarks['good']['1d'], benchmarks['poor']['1d']
                    elif timeframe == 'ret_7d':
                        thresholds = benchmarks['excellent']['7d'], benchmarks['good']['7d'], benchmarks['poor']['7d']
                    elif timeframe == 'ret_30d':
                        thresholds = benchmarks['excellent']['30d'], benchmarks['good']['30d'], benchmarks['poor']['30d']
                    else:  # ret_3m
                        thresholds = benchmarks['excellent']['3m'], benchmarks['good']['3m'], benchmarks['poor']['3m']
                    
                    normalized = np.where(returns >= thresholds[0], 90,
                                np.where(returns >= thresholds[1], 70,
                                np.where(returns >= 0, 50,
                                np.where(returns >= thresholds[2], 30, 10))))
                    
                    momentum_components += normalized * weight
                    total_weight += weight
            
            if total_weight > 0:
                score = momentum_components / total_weight
            
            # Bonus for momentum consistency
            if 'momentum_breadth' in df.columns:
                consistency_bonus = (pd.to_numeric(df['momentum_breadth'], errors='coerce').fillna(50) - 50) / 10
                score += consistency_bonus
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_value_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate value factor score based on PE and earnings quality"""
        
        try:
            score = pd.Series(50.0, index=df.index)
            
            if 'pe' not in df.columns:
                return score
            
            pe = pd.to_numeric(df['pe'], errors='coerce')
            benchmarks = self.config.benchmarks.VALUE_BENCHMARKS
            
            # PE-based scoring
            value_score = np.where(pe <= 0, 20,  # Negative earnings
                          np.where(pe <= benchmarks['deep_value'], 95,
                          np.where(pe <= benchmarks['strong_value'], 85,
                          np.where(pe <= benchmarks['fair_value'], 70,
                          np.where(pe <= benchmarks['growth_premium'], 55,
                          np.where(pe <= benchmarks['expensive'], 35, 20))))))
            
            score = pd.Series(value_score, index=df.index)
            
            # Earnings quality adjustment
            if 'eps_stability' in df.columns:
                stability_bonus = (pd.to_numeric(df['eps_stability'], errors='coerce').fillna(50) - 50) / 10
                score += stability_bonus
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Value calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_growth_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate growth factor score based on EPS trends"""
        
        try:
            score = pd.Series(50.0, index=df.index)
            
            if 'eps_change_pct' not in df.columns:
                return score
            
            eps_growth = pd.to_numeric(df['eps_change_pct'], errors='coerce').fillna(0)
            benchmarks = self.config.benchmarks.GROWTH_BENCHMARKS
            
            growth_score = np.where(eps_growth >= benchmarks['accelerating'], 95,
                           np.where(eps_growth >= benchmarks['strong'], 85,
                           np.where(eps_growth >= benchmarks['decent'], 70,
                           np.where(eps_growth >= benchmarks['modest'], 60,
                           np.where(eps_growth >= benchmarks['weak'], 45, 25)))))
            
            score = pd.Series(growth_score, index=df.index)
            
            # Quality bonus
            if 'growth_quality' in df.columns:
                quality_bonus = (pd.to_numeric(df['growth_quality'], errors='coerce').fillna(50) - 50) / 10
                score += quality_bonus
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Growth calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_volume_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume factor score based on trading interest"""
        
        try:
            score = pd.Series(50.0, index=df.index)
            
            if 'rvol' in df.columns:
                rvol = pd.to_numeric(df['rvol'], errors='coerce').fillna(1)
                benchmarks = self.config.benchmarks.VOLUME_BENCHMARKS
                
                volume_score = np.where(rvol >= benchmarks['extreme_interest'], 95,
                               np.where(rvol >= benchmarks['strong_interest'], 85,
                               np.where(rvol >= benchmarks['elevated_interest'], 70,
                               np.where(rvol >= benchmarks['normal_activity'], 55, 35))))
                
                score = pd.Series(volume_score, index=df.index)
            
            # Volume strength bonus
            if 'volume_strength' in df.columns:
                strength_bonus = (pd.to_numeric(df['volume_strength'], errors='coerce').fillna(50) - 50) / 20
                score += strength_bonus
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Volume calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_technical_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate technical factor score based on chart patterns"""
        
        try:
            score = pd.Series(50.0, index=df.index)
            
            # Moving average trend analysis
            sma_weights = {'above_sma20': 0.4, 'above_sma50': 0.35, 'above_sma200': 0.25}
            
            for sma_col, weight in sma_weights.items():
                if sma_col in df.columns:
                    sma_bonus = df[sma_col].astype(float) * 30 * weight
                    score += sma_bonus
            
            # 52-week position analysis
            if 'position_52w_pct' in df.columns:
                position = pd.to_numeric(df['position_52w_pct'], errors='coerce').fillna(50)
                benchmarks = self.config.benchmarks.POSITION_BENCHMARKS
                
                position_score = np.where(position >= benchmarks['near_highs'], 30,
                                np.where(position >= benchmarks['strong_position'], 20,
                                np.where(position >= benchmarks['upper_range'], 10,
                                np.where(position >= benchmarks['middle_range'], 0, -10))))
                
                score += position_score
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Technical calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_sector_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sector factor score based on industry performance"""
        
        try:
            score = pd.Series(50.0, index=df.index)
            
            if 'current_sector_ret_30d' in df.columns:
                sector_strength = pd.to_numeric(df['current_sector_ret_30d'], errors='coerce').fillna(0)
                strength_score = 50 + np.clip(sector_strength * 2.5, -40, 40)
                score = strength_score
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Sector calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_risk_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk factor score (higher score = higher risk)"""
        
        try:
            if 'total_risk_score' in df.columns:
                return pd.to_numeric(df['total_risk_score'], errors='coerce').fillna(25).clip(0, 100)
            else:
                return pd.Series(25.0, index=df.index)  # Default moderate risk
                
        except Exception as e:
            logger.warning(f"Risk calculation failed: {e}")
            return pd.Series(25.0, index=df.index)
    
    def _calculate_quality_factor(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quality factor score based on data completeness"""
        
        try:
            score = pd.Series(80.0, index=df.index)
            
            if 'data_completeness_score' in df.columns:
                completeness = pd.to_numeric(df['data_completeness_score'], errors='coerce').fillna(80)
                score = completeness
            
            return score.clip(0, 100).round(1)
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return pd.Series(80.0, index=df.index)
    
    def _generate_ultra_conservative_signals(self, scores: pd.Series) -> pd.Series:
        """Generate ultra-conservative signals with strict thresholds"""
        
        thresholds = self.config.signals.SIGNAL_THRESHOLDS
        
        conditions = [
            scores >= thresholds['STRONG_BUY'],    # 92+
            scores >= thresholds['BUY'],           # 82+
            scores >= thresholds['ACCUMULATE'],    # 72+
            scores >= thresholds['WATCH'],         # 60+
            scores >= thresholds['NEUTRAL'],       # 40+
            scores >= thresholds['AVOID']          # 25+
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH', 'NEUTRAL', 'AVOID']
        
        return pd.Series(
            np.select(conditions, choices, default='STRONG_AVOID'),
            index=scores.index
        )
    
    def _calculate_signal_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal confidence based on factor alignment and quality"""
        
        try:
            confidence = pd.Series(50.0, index=df.index)
            
            factor_columns = ['momentum_score', 'value_score', 'growth_score', 'volume_score', 
                             'technical_score', 'sector_score', 'quality_score']
            
            available_factors = [col for col in factor_columns if col in df.columns]
            
            if not available_factors:
                return confidence
            
            for idx in df.index:
                try:
                    row = df.loc[idx]
                    factor_scores = [row[col] for col in available_factors if pd.notna(row[col])]
                    
                    if len(factor_scores) < 3:
                        confidence.loc[idx] = 30.0
                        continue
                    
                    # Base confidence from composite score
                    base_confidence = row.get('composite_score', 50)
                    
                    # Factor alignment bonus (lower std dev = higher alignment = higher confidence)
                    std_dev = np.std(factor_scores) if len(factor_scores) > 1 else 20
                    alignment_bonus = max(0, 35 - std_dev) * 0.5
                    
                    # Data quality bonus
                    quality_bonus = 0
                    if 'data_completeness_score' in df.columns:
                        quality_bonus = (row.get('data_completeness_score', 70) - 70) * 0.3
                    
                    # Market condition bonus
                    market_bonus = 0
                    if self.market_condition and self.market_condition.confidence > 70:
                        market_bonus = 5
                    
                    # Calculate final confidence
                    total_confidence = base_confidence + alignment_bonus + quality_bonus + market_bonus
                    confidence.loc[idx] = min(99, max(15, total_confidence))
                    
                except Exception as e:
                    confidence.loc[idx] = 50.0
            
            return confidence.round(1)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return pd.Series(50.0, index=df.index)
    
    def _calculate_signal_explanations(self):
        """Generate detailed explanations for each signal"""
        
        logger.info("🧠 Generating signal explanations...")
        
        self.signal_explanations = {}
        
        for idx, stock in self.master_df.iterrows():
            ticker = stock.get('ticker', f'Stock_{idx}')
            
            try:
                explanation = self._generate_stock_explanation(stock)
                self.signal_explanations[ticker] = explanation
            except Exception as e:
                logger.warning(f"Explanation generation failed for {ticker}: {e}")
                # Create basic explanation as fallback
                self.signal_explanations[ticker] = SignalExplanation(
                    ticker=ticker,
                    signal=stock.get('signal', 'WATCH'),
                    confidence=stock.get('confidence', 50),
                    primary_reason="Analysis based on available data",
                    supporting_factors=["Comprehensive factor analysis"],
                    risk_factors=["Standard market risks apply"],
                    factor_scores={},
                    data_quality=stock.get('data_completeness_score', 70),
                    recommendation="Proceed with appropriate risk management"
                )
        
        logger.info(f"✅ Generated explanations for {len(self.signal_explanations)} stocks")
    
    def _generate_stock_explanation(self, stock: pd.Series) -> SignalExplanation:
        """Generate comprehensive explanation for individual stock signal"""
        
        signal = stock.get('signal', 'NEUTRAL')
        confidence = stock.get('confidence', 50)
        ticker = stock.get('ticker', 'Unknown')
        
        # Extract factor scores
        factor_scores = {}
        for factor in ['momentum', 'value', 'growth', 'volume', 'technical', 'sector', 'risk', 'quality']:
            factor_scores[factor] = stock.get(f'{factor}_score', 50)
        
        # Identify primary reason (strongest factor)
        strong_factors = [(factor, score) for factor, score in factor_scores.items() 
                         if score > 70 and factor != 'risk']
        
        if strong_factors:
            primary_factor = max(strong_factors, key=lambda x: x[1])
            primary_reason = f"Strong {primary_factor[0]} factor (score: {primary_factor[1]:.0f})"
        else:
            primary_reason = "Mixed signal analysis"
        
        # Supporting factors
        supporting_factors = []
        for factor, score in factor_scores.items():
            if score > 75 and factor != 'risk':
                supporting_factors.append(f"{factor.title()}: {score:.0f}/100")
        
        if not supporting_factors:
            supporting_factors = ["Multiple factors evaluated"]
        
        # Risk factors
        risk_factors = []
        
        pe = stock.get('pe', 0)
        if pe > self.config.benchmarks.VALUE_BENCHMARKS['expensive']:
            risk_factors.append(f"High valuation (P/E: {pe:.1f})")
        
        vol_1d = stock.get('vol_1d', 0)
        if vol_1d < 50000:
            risk_factors.append("Low liquidity concern")
        
        ret_30d = stock.get('ret_30d', 0)
        if ret_30d < -15:
            risk_factors.append(f"Recent weakness ({ret_30d:+.1f}% in 30d)")
        
        risk_score = factor_scores.get('risk', 25)
        if risk_score > 60:
            risk_factors.append("Elevated overall risk profile")
        
        if not risk_factors:
            risk_factors = ["Standard market risks"]
        
        # Recommendation
        if signal in ['STRONG_BUY', 'BUY']:
            recommendation = f"Consider position in {ticker} with {confidence:.0f}% confidence"
        elif signal == 'ACCUMULATE':
            recommendation = f"Gradual accumulation of {ticker} recommended"
        elif signal == 'WATCH':
            recommendation = f"Monitor {ticker} for better entry opportunity"
        else:
            recommendation = f"Avoid or reduce exposure to {ticker}"
        
        return SignalExplanation(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            primary_reason=primary_reason,
            supporting_factors=supporting_factors[:5],  # Limit to 5
            risk_factors=risk_factors[:4],  # Limit to 4
            factor_scores=factor_scores,
            data_quality=stock.get('data_completeness_score', 70),
            recommendation=recommendation
        )
    
    def _validate_and_rank_signals(self):
        """Final validation and ranking of signals"""
        
        if self.master_df.empty:
            logger.warning("Cannot validate and rank signals: master dataset is empty")
            return
        
        try:
            # Apply final quality filters for high-confidence signals
            if 'data_completeness_score' in self.master_df.columns:
                poor_data_mask = self.master_df['data_completeness_score'] < 60
                if poor_data_mask.any():
                    # Reduce confidence for poor data quality
                    self.master_df.loc[poor_data_mask, 'confidence'] *= 0.8
                    
                    # Downgrade strong signals with poor data
                    strong_signal_poor_data = poor_data_mask & self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])
                    if strong_signal_poor_data.any():
                        self.master_df.loc[strong_signal_poor_data, 'signal'] = 'WATCH'
                        logger.info(f"Downgraded {strong_signal_poor_data.sum()} signals due to poor data quality")
            
            # Calculate final ranking score
            if all(col in self.master_df.columns for col in ['composite_score', 'confidence']):
                self.master_df['final_rank'] = (
                    self.master_df['composite_score'] * 0.7 + 
                    self.master_df['confidence'] * 0.3
                ).round(1)
                
                # Sort by final rank
                self.master_df = self.master_df.sort_values('final_rank', ascending=False).reset_index(drop=True)
            
            logger.info("✅ Signal validation and ranking completed")
            
        except Exception as e:
            logger.error(f"❌ Signal validation failed: {e}")
    
    def _record_processing_stats(self, processing_time: float):
        """Record comprehensive processing statistics"""
        
        try:
            total_stocks = len(self.master_df)
            
            signal_counts = self.master_df['signal'].value_counts() if not self.master_df.empty else {}
            
            # Calculate average data quality
            data_quality_avg = 70.0  # Default
            if 'watchlist' in self.quality_reports:
                data_quality_avg = self.quality_reports['watchlist'].overall_score
            
            self.processing_stats = ProcessingStats(
                total_stocks=total_stocks,
                processing_time=round(processing_time, 2),
                strong_buy_count=signal_counts.get('STRONG_BUY', 0),
                buy_count=signal_counts.get('BUY', 0),
                data_quality_avg=data_quality_avg,
                signals_generated=len([s for s in signal_counts.keys() if s in ['STRONG_BUY', 'BUY', 'ACCUMULATE']]),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Stats recording failed: {e}")
            self.processing_stats = ProcessingStats(
                total_stocks=0, processing_time=0, strong_buy_count=0, buy_count=0,
                data_quality_avg=0, signals_generated=0, timestamp=datetime.now()
            )
    
    # =============================================================================
    # PUBLIC INTERFACE METHODS
    # =============================================================================
    
    def get_top_opportunities(self, limit: int = 8) -> pd.DataFrame:
        """Get top opportunities with high confidence"""
        
        if self.master_df.empty:
            return pd.DataFrame()
        
        try:
            opportunities = self.master_df[
                (self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])) &
                (self.master_df['confidence'] >= 65)
            ].head(limit)
            
            return opportunities
        except Exception as e:
            logger.warning(f"Get top opportunities failed: {e}")
            return pd.DataFrame()
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary"""
        
        if self.master_df.empty:
            return {}
        
        try:
            summary = {
                'total_stocks': len(self.master_df),
                'processing_stats': self.processing_stats.__dict__ if self.processing_stats else {},
                'quality_reports': {k: v.__dict__ for k, v in self.quality_reports.items()},
                'market_condition': self.market_condition.__dict__ if self.market_condition else {},
                'signal_distribution': self.master_df['signal'].value_counts().to_dict(),
                'avg_composite_score': self.master_df['composite_score'].mean(),
                'avg_confidence': self.master_df['confidence'].mean(),
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
            # Market breadth
            if 'ret_1d' in self.master_df.columns:
                ret_1d = pd.to_numeric(self.master_df['ret_1d'], errors='coerce')
                summary['market_breadth'] = (ret_1d > 0).mean() * 100
            
            # High confidence signals
            summary['high_confidence_signals'] = len(self.master_df[
                (self.master_df['signal'].isin(['STRONG_BUY', 'BUY'])) & 
                (self.master_df['confidence'] >= 80)
            ])
            
            return summary
        except Exception as e:
            logger.warning(f"Market summary failed: {e}")
            return {}
    
    def get_signal_explanation(self, ticker: str) -> Optional[SignalExplanation]:
        """Get detailed explanation for specific stock signal"""
        return self.signal_explanations.get(ticker)
    
    def get_filtered_stocks(self, 
                          sectors: List[str] = None,
                          categories: List[str] = None,
                          signals: List[str] = None,
                          min_score: float = 0,
                          min_confidence: float = 0) -> pd.DataFrame:
        """Get filtered stocks based on criteria"""
        
        if self.master_df.empty:
            return pd.DataFrame()
        
        try:
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
            
            return filtered
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
            return pd.DataFrame()

# Export the main class
__all__ = ['UltimateSignalEngine']

if __name__ == "__main__":
    print("✅ Ultimate Signal Engine loaded successfully")

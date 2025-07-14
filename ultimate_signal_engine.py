"""
ultimate_signal_engine.py - M.A.N.T.R.A. ULTIMATE SIGNAL ENGINE
===============================================================
The most intelligent signal engine - Better than human analysis
Combines 8-factor analysis with pattern recognition and market intelligence
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# INTELLIGENT COMPONENTS
# ============================================================================

class MarketRegimeDetector:
    """Detect current market regime for adaptive analysis"""
    
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime using multiple indicators"""
        
        # Market breadth
        if 'ret_1d' in df.columns:
            advancing = (df['ret_1d'] > 0).sum()
            declining = (df['ret_1d'] < 0).sum()
            breadth = advancing / (advancing + declining) * 100 if (advancing + declining) > 0 else 50
        else:
            breadth = 50
        
        # Volatility measure
        if 'ret_30d' in df.columns:
            volatility = df['ret_30d'].std()
            extreme_moves = ((df['ret_30d'] > 30) | (df['ret_30d'] < -20)).sum()
        else:
            volatility = 15
            extreme_moves = 0
        
        # Volume activity
        if 'rvol' in df.columns:
            high_volume_stocks = (df['rvol'] > 2).sum()
            volume_surge = high_volume_stocks / len(df) * 100 if len(df) > 0 else 10
        else:
            volume_surge = 10
        
        # Determine regime
        if breadth > 65 and volatility < 20 and volume_surge > 15:
            regime = "BULL_TRENDING"
            characteristics = "Strong uptrend with broad participation"
            strategy = "Momentum and breakouts"
        elif breadth < 35 and volatility > 25:
            regime = "BEAR_VOLATILE"
            characteristics = "Downtrend with high volatility"
            strategy = "Value and defensive plays"
        elif volatility > 30 and extreme_moves > len(df) * 0.1:
            regime = "HIGH_VOLATILITY"
            characteristics = "Extreme volatility regime"
            strategy = "Quality and low-risk stocks"
        elif 45 <= breadth <= 55 and volatility < 15:
            regime = "RANGE_BOUND"
            characteristics = "Sideways market"
            strategy = "Sector rotation and value"
        else:
            regime = "TRANSITIONAL"
            characteristics = "Market in transition"
            strategy = "Balanced approach"
        
        return {
            'regime': regime,
            'breadth': breadth,
            'volatility': volatility,
            'volume_surge': volume_surge,
            'characteristics': characteristics,
            'strategy': strategy
        }

class SmartPatternRecognizer:
    """Advanced pattern recognition using price, volume, and fundamentals"""
    
    def __init__(self):
        self.patterns = {
            'GOLDEN_CROSS_PLUS': self._check_golden_cross_plus,
            'ACCUMULATION_BREAKOUT': self._check_accumulation_breakout,
            'EARNINGS_MOMENTUM': self._check_earnings_momentum,
            'SECTOR_ROTATION': self._check_sector_rotation,
            'VALUE_EMERGENCE': self._check_value_emergence,
            'VOLUME_CLIMAX': self._check_volume_climax,
            'TREND_CONTINUATION': self._check_trend_continuation,
            'REVERSAL_SETUP': self._check_reversal_setup
        }
    
    def detect_patterns(self, stock: pd.Series, market_regime: str) -> List[str]:
        """Detect all applicable patterns for a stock"""
        detected = []
        
        for pattern_name, check_func in self.patterns.items():
            if check_func(stock, market_regime):
                detected.append(pattern_name)
        
        return detected
    
    def _check_golden_cross_plus(self, stock: pd.Series, regime: str) -> bool:
        """Golden cross with volume and momentum confirmation"""
        return (
            stock.get('above_sma_50d', 0) == 1 and
            stock.get('above_sma_200d', 0) == 1 and
            stock.get('pct_from_sma_50d', 0) > 0 and
            stock.get('pct_from_sma_50d', 0) < 10 and
            stock.get('rvol', 1) > 1.5 and
            stock.get('momentum_score', 0) > 70
        )
    
    def _check_accumulation_breakout(self, stock: pd.Series, regime: str) -> bool:
        """Smart money accumulation followed by breakout"""
        return (
            stock.get('position_52w', 50) > 70 and
            stock.get('volume_30d', 0) > stock.get('volume_3m', 1) * 1.2 and
            stock.get('ret_7d', 0) > 5 and
            stock.get('rvol', 1) > 2
        )
    
    def _check_earnings_momentum(self, stock: pd.Series, regime: str) -> bool:
        """Strong earnings growth with price momentum"""
        return (
            stock.get('eps_change_pct', 0) > 40 and
            stock.get('eps_current', 0) > 0 and
            stock.get('ret_30d', 0) > 10 and
            stock.get('pe', 100) < 35
        )
    
    def _check_sector_rotation(self, stock: pd.Series, regime: str) -> bool:
        """Leading stock in a rotating sector"""
        return (
            stock.get('sector_relative_strength', 0) > 10 and
            stock.get('sector_score', 50) > 70 and
            stock.get('ret_7d', 0) > stock.get('sector_ret_7d', 0) * 1.5
        )
    
    def _check_value_emergence(self, stock: pd.Series, regime: str) -> bool:
        """Undervalued stock starting to move"""
        return (
            stock.get('pe', 100) > 0 and
            stock.get('pe', 100) < 15 and
            stock.get('position_52w', 50) < 40 and
            stock.get('ret_7d', 0) > 3 and
            stock.get('eps_change_pct', 0) > 0
        )
    
    def _check_volume_climax(self, stock: pd.Series, regime: str) -> bool:
        """Extreme volume indicating major move"""
        return (
            stock.get('rvol', 1) > 5 and
            abs(stock.get('ret_1d', 0)) > 3 and
            regime in ['BULL_TRENDING', 'HIGH_VOLATILITY']
        )
    
    def _check_trend_continuation(self, stock: pd.Series, regime: str) -> bool:
        """Strong trend likely to continue"""
        return (
            stock.get('momentum_breadth', 0) >= 75 and
            stock.get('ret_30d', 0) > 15 and
            stock.get('ret_7d', 0) > 0 and
            stock.get('technical_score', 50) > 75
        )
    
    def _check_reversal_setup(self, stock: pd.Series, regime: str) -> bool:
        """Potential trend reversal setup"""
        return (
            stock.get('position_52w', 50) < 25 and
            stock.get('ret_7d', 0) > 5 and
            stock.get('rvol', 1) > 2 and
            stock.get('pe', 100) > 0 and
            stock.get('pe', 100) < 20
        )

class IntelligentScorer:
    """Adaptive scoring based on market conditions and stock characteristics"""
    
    def __init__(self):
        self.regime_weights = {
            'BULL_TRENDING': {
                'momentum': 0.30, 'technical': 0.20, 'volume': 0.15,
                'growth': 0.15, 'sector': 0.10, 'value': 0.08, 
                'risk': 0.01, 'quality': 0.01
            },
            'BEAR_VOLATILE': {
                'value': 0.30, 'quality': 0.20, 'risk': 0.15,
                'momentum': 0.10, 'growth': 0.10, 'technical': 0.08,
                'volume': 0.05, 'sector': 0.02
            },
            'RANGE_BOUND': {
                'value': 0.25, 'sector': 0.20, 'technical': 0.15,
                'momentum': 0.15, 'growth': 0.10, 'volume': 0.10,
                'risk': 0.03, 'quality': 0.02
            },
            'HIGH_VOLATILITY': {
                'quality': 0.25, 'value': 0.25, 'risk': 0.20,
                'technical': 0.10, 'momentum': 0.08, 'growth': 0.07,
                'volume': 0.03, 'sector': 0.02
            },
            'TRANSITIONAL': CONFIG.FACTOR_WEIGHTS  # Use default weights
        }
    
    def calculate_adaptive_score(self, stock: pd.Series, regime: str, patterns: List[str]) -> float:
        """Calculate score adapted to market regime"""
        
        # Get regime-specific weights
        weights = self.regime_weights.get(regime, CONFIG.FACTOR_WEIGHTS)
        
        # Calculate base score
        base_score = (
            stock.get('momentum_score', 50) * weights['momentum'] +
            stock.get('value_score', 50) * weights['value'] +
            stock.get('growth_score', 50) * weights['growth'] +
            stock.get('volume_score', 50) * weights['volume'] +
            stock.get('technical_score', 50) * weights['technical'] +
            stock.get('sector_score', 50) * weights['sector'] +
            (100 - stock.get('risk_score', 50)) * weights['risk'] +
            stock.get('quality_score', 50) * weights['quality']
        )
        
        # Pattern bonuses
        pattern_bonuses = {
            'GOLDEN_CROSS_PLUS': 5,
            'ACCUMULATION_BREAKOUT': 8,
            'EARNINGS_MOMENTUM': 7,
            'SECTOR_ROTATION': 6,
            'VALUE_EMERGENCE': 6,
            'VOLUME_CLIMAX': 4,
            'TREND_CONTINUATION': 5,
            'REVERSAL_SETUP': 5
        }
        
        pattern_bonus = sum(pattern_bonuses.get(p, 0) for p in patterns)
        
        # Regime-specific adjustments
        regime_bonus = 0
        if regime == 'BULL_TRENDING' and stock.get('momentum_score', 50) > 80:
            regime_bonus = 5
        elif regime == 'BEAR_VOLATILE' and stock.get('value_score', 50) > 80:
            regime_bonus = 5
        elif regime == 'RANGE_BOUND' and stock.get('sector_relative_strength', 0) > 10:
            regime_bonus = 4
        
        # Final score
        final_score = base_score + pattern_bonus + regime_bonus
        
        return min(100, final_score)

class SmartInsightGenerator:
    """Generate intelligent, actionable insights"""
    
    def generate_insights(self, stock: pd.Series, patterns: List[str], 
                         regime: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Generate multi-dimensional insights"""
        
        insights = {
            'volume_intelligence': self._analyze_volume_intelligence(stock),
            'momentum_analysis': self._analyze_momentum_quality(stock),
            'value_proposition': self._analyze_value_proposition(stock),
            'risk_assessment': self._analyze_risk_factors(stock),
            'entry_strategy': self._determine_entry_strategy(stock, patterns, score),
            'key_catalyst': self._identify_catalyst(stock, patterns)
        }
        
        return insights
    
    def _analyze_volume_intelligence(self, stock: pd.Series) -> Dict[str, Any]:
        """Analyze volume patterns for smart money activity"""
        
        rvol = stock.get('rvol', 1)
        vol_30d_ratio = stock.get('vol_ratio_30d_90d', 1)
        
        if rvol > 5 and abs(stock.get('ret_1d', 0)) < 2:
            return {
                'detected': True,
                'type': 'ACCUMULATION',
                'icon': '🎯',
                'message': f"Heavy accumulation detected - {rvol:.1f}x volume with minimal price movement"
            }
        elif rvol > 3 and stock.get('ret_1d', 0) > 3:
            return {
                'detected': True,
                'type': 'BREAKOUT',
                'icon': '🚀',
                'message': f"Breakout volume - {rvol:.1f}x average with {stock.get('ret_1d', 0):.1f}% gain"
            }
        elif vol_30d_ratio > 1.3:
            return {
                'detected': True,
                'type': 'SUSTAINED',
                'icon': '📈',
                'message': "Sustained institutional interest - 30-day volume 30% above average"
            }
        else:
            return {'detected': False}
    
    def _analyze_momentum_quality(self, stock: pd.Series) -> Dict[str, Any]:
        """Analyze momentum characteristics"""
        
        ret_7d = stock.get('ret_7d', 0)
        ret_30d = stock.get('ret_30d', 0)
        momentum_score = stock.get('momentum_score', 50)
        
        if ret_7d > ret_30d / 4 and ret_7d > 0 and momentum_score > 80:
            return {
                'detected': True,
                'type': 'ACCELERATING',
                'icon': '⚡',
                'message': "Momentum accelerating - recent gains outpacing monthly trend"
            }
        elif ret_30d > 20 and ret_7d > 0:
            return {
                'detected': True,
                'type': 'STRONG_TREND',
                'icon': '💪',
                'message': f"Strong uptrend - {ret_30d:.0f}% gain over 30 days"
            }
        else:
            return {'detected': False}
    
    def _analyze_value_proposition(self, stock: pd.Series) -> Dict[str, Any]:
        """Analyze value characteristics"""
        
        pe = stock.get('pe', 100)
        eps_growth = stock.get('eps_change_pct', 0)
        
        if pe > 0 and pe < 15 and eps_growth > 30:
            return {
                'detected': True,
                'type': 'GROWTH_AT_VALUE',
                'icon': '💎',
                'message': f"Exceptional value - PE {pe:.1f} with {eps_growth:.0f}% earnings growth"
            }
        elif pe > 0 and pe < 20 and stock.get('position_52w', 50) < 30:
            return {
                'detected': True,
                'type': 'DEEP_VALUE',
                'icon': '🎁',
                'message': f"Deep value opportunity - PE {pe:.1f} near 52-week lows"
            }
        else:
            return {'detected': False}
    
    def _analyze_risk_factors(self, stock: pd.Series) -> Dict[str, Any]:
        """Assess risk factors"""
        
        risk_score = stock.get('risk_score', 50)
        pe = stock.get('pe', 100)
        volatility = abs(stock.get('ret_30d', 0))
        
        if risk_score > 70:
            return {
                'detected': True,
                'level': 'HIGH',
                'icon': '⚠️',
                'message': "High risk - consider position sizing carefully"
            }
        elif pe > 50 or pe <= 0:
            return {
                'detected': True,
                'level': 'MODERATE',
                'icon': '⚡',
                'message': "Valuation risk - earnings multiple extended"
            }
        else:
            return {
                'detected': False,
                'level': 'LOW',
                'icon': '✅',
                'message': "Risk within acceptable parameters"
            }
    
    def _determine_entry_strategy(self, stock: pd.Series, patterns: List[str], score: float) -> Dict[str, Any]:
        """Determine optimal entry strategy"""
        
        position_52w = stock.get('position_52w', 50)
        rvol = stock.get('rvol', 1)
        
        if 'ACCUMULATION_BREAKOUT' in patterns or score > 90:
            return {
                'type': 'HOT_ZONE',
                'label': '🔥 Hot Zone',
                'action': 'Buy immediately',
                'reason': 'High conviction setup with multiple confirmations'
            }
        elif position_52w > 80 and rvol > 2:
            return {
                'type': 'BREAKOUT_ZONE',
                'label': '🚀 Breakout',
                'action': 'Buy on strength',
                'reason': 'Breaking to new highs with volume'
            }
        elif score > 75 and position_52w < 40:
            return {
                'type': 'ACCUMULATION_ZONE',
                'label': '📈 Accumulate',
                'action': 'Scale in',
                'reason': 'High score at attractive levels'
            }
        elif 'VALUE_EMERGENCE' in patterns:
            return {
                'type': 'VALUE_ZONE',
                'label': '💎 Value Entry',
                'action': 'Buy in phases',
                'reason': 'Value emerging from oversold'
            }
        else:
            return {
                'type': 'WATCH_ZONE',
                'label': '👀 Watch',
                'action': 'Wait for setup',
                'reason': 'Monitor for better entry'
            }
    
    def _identify_catalyst(self, stock: pd.Series, patterns: List[str]) -> str:
        """Identify the primary catalyst"""
        
        if 'EARNINGS_MOMENTUM' in patterns:
            return f"Earnings explosion - {stock.get('eps_change_pct', 0):.0f}% EPS growth"
        elif 'SECTOR_ROTATION' in patterns:
            return f"Sector leadership - outperforming sector by {stock.get('sector_relative_strength', 0):.0f}%"
        elif stock.get('rvol', 1) > 3:
            return f"Volume surge - {stock.get('rvol', 1):.1f}x average volume"
        elif stock.get('momentum_score', 50) > 85:
            return "Strong momentum across all timeframes"
        else:
            return "Multiple technical factors aligning"

# ============================================================================
# MAIN ENGINE
# ============================================================================

class UltimateSignalEngine:
    """The ultimate signal engine - smarter than human analysis"""
    
    def __init__(self):
        self.config = CONFIG
        self.regime_detector = MarketRegimeDetector()
        self.pattern_recognizer = SmartPatternRecognizer()
        self.intelligent_scorer = IntelligentScorer()
        self.insight_generator = SmartInsightGenerator()
        
    def analyze(self, stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform ultimate analysis with all intelligence layers
        
        Args:
            stocks_df: Stocks dataframe
            sector_df: Sector performance dataframe
            
        Returns:
            Enhanced dataframe with signals and insights
        """
        
        start_time = pd.Timestamp.now()
        logger.info(f"🧠 Starting ULTIMATE analysis for {len(stocks_df)} stocks...")
        
        # Copy dataframe
        df = stocks_df.copy()
        
        # Detect market regime
        market_regime = self.regime_detector.detect_regime(df)
        regime_type = market_regime['regime']
        
        logger.info(f"📊 Market Regime: {regime_type} - {market_regime['characteristics']}")
        logger.info(f"📈 Strategy: {market_regime['strategy']}")
        
        # Add sector performance if available
        if sector_df is not None and not sector_df.empty:
            self._add_sector_metrics(df, sector_df)
        
        # Calculate base factor scores
        df = self._calculate_factor_scores(df)
        
        # Detect patterns for each stock
        patterns_list = []
        composite_scores = []
        insights_list = []
        
        for idx, stock in df.iterrows():
            # Detect patterns
            patterns = self.pattern_recognizer.detect_patterns(stock, regime_type)
            patterns_list.append(patterns)
            
            # Calculate adaptive score
            score = self.intelligent_scorer.calculate_adaptive_score(stock, regime_type, patterns)
            composite_scores.append(score)
            
            # Generate insights
            insights = self.insight_generator.generate_insights(stock, patterns, market_regime, score)
            insights_list.append(insights)
        
        # Add results to dataframe
        df['patterns'] = patterns_list
        df['composite_score'] = composite_scores
        df['smart_insights'] = insights_list
        df['confidence'] = df['composite_score']  # For compatibility
        
        # Generate signals based on adaptive scoring
        df = self._generate_smart_signals(df)
        
        # Add market regime info
        df['market_regime'] = regime_type
        df['regime_strategy'] = market_regime['strategy']
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)
        
        # Log summary
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        signal_summary = df['signal'].value_counts()
        pattern_count = sum(len(p) > 0 for p in patterns_list)
        
        logger.info(f"✅ Ultimate analysis complete in {elapsed:.2f}s")
        logger.info(f"📊 Signals: {dict(signal_summary)}")
        logger.info(f"🎯 {pattern_count} stocks with patterns detected")
        
        return df
    
    def _add_sector_metrics(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> None:
        """Add sector performance metrics"""
        
        # Create sector lookup
        sector_metrics = {}
        for _, row in sector_df.iterrows():
            sector = row['sector']
            sector_metrics[sector] = {
                'sector_ret_1d': row.get('sector_ret_1d', 0),
                'sector_ret_7d': row.get('sector_ret_7d', 0),
                'sector_ret_30d': row.get('sector_ret_30d', 0),
                'sector_performance': row.get('sector_ret_30d', 0)  # Primary metric
            }
        
        # Apply to stocks
        for metric_name in ['sector_ret_1d', 'sector_ret_7d', 'sector_ret_30d', 'sector_performance']:
            df[metric_name] = df['sector'].map(lambda s: sector_metrics.get(s, {}).get(metric_name, 0))
        
        # Calculate relative strength
        df['sector_relative_strength'] = df['ret_30d'] - df['sector_performance']
    
    def _calculate_factor_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all factor scores"""
        
        # Momentum score
        df['momentum_score'] = self._calculate_momentum_score(df)
        
        # Value score
        df['value_score'] = self._calculate_value_score(df)
        
        # Growth score
        df['growth_score'] = self._calculate_growth_score(df)
        
        # Volume score
        df['volume_score'] = self._calculate_volume_score(df)
        
        # Technical score
        df['technical_score'] = self._calculate_technical_score(df)
        
        # Sector score
        df['sector_score'] = self._calculate_sector_score(df)
        
        # Risk score
        df['risk_score'] = self._calculate_risk_score(df)
        
        # Quality score
        df['quality_score'] = self._calculate_quality_score(df)
        
        # Add momentum quality indicator
        df['momentum_quality'] = df.apply(self._assess_momentum_quality, axis=1)
        
        return df
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced momentum scoring"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Multi-timeframe momentum with adaptive weights
        timeframes = {
            'ret_1d': {'weight': 0.10, 'threshold': 2},
            'ret_7d': {'weight': 0.20, 'threshold': 5},
            'ret_30d': {'weight': 0.40, 'threshold': 10},
            'ret_3m': {'weight': 0.30, 'threshold': 20}
        }
        
        for col, params in timeframes.items():
            if col in df.columns:
                returns = df[col].fillna(0)
                
                # Non-linear scoring
                col_score = np.where(returns > params['threshold'] * 2, 90,
                            np.where(returns > params['threshold'], 75,
                            np.where(returns > 0, 60,
                            np.where(returns > -params['threshold'], 40, 20))))
                
                score += (col_score - 50) * params['weight']
        
        # Consistency bonus
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            all_positive = ((df['ret_1d'] > 0) & (df['ret_7d'] > 0) & (df['ret_30d'] > 0))
            score += all_positive * 10
        
        return score.clip(0, 100)
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced value scoring"""
        
        if 'pe' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        pe = df['pe'].fillna(30)
        
        # Non-linear PE scoring with consideration for growth
        base_score = np.where(pe <= 0, 20,  # Negative earnings
                     np.where(pe <= 10, 95,
                     np.where(pe <= 15, 85,
                     np.where(pe <= 20, 75,
                     np.where(pe <= 25, 65,
                     np.where(pe <= 30, 55,
                     np.where(pe <= 40, 40,
                     np.where(pe <= 50, 25, 10))))))))
        
        score = pd.Series(base_score, index=df.index)
        
        # Adjust for growth (PEG concept)
        if 'eps_change_pct' in df.columns:
            eps_growth = df['eps_change_pct'].fillna(0)
            peg_adjustment = np.where((pe > 0) & (pe < 50) & (eps_growth > 0),
                                     np.minimum(20, eps_growth / pe * 10), 0)
            score += peg_adjustment
        
        # Position bonus for value
        if 'position_52w' in df.columns:
            value_position_bonus = np.where(df['position_52w'] < 30, 10, 0)
            score += value_position_bonus
        
        return score.clip(0, 100)
    
    def _calculate_growth_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced growth scoring"""
        
        score = pd.Series(50.0, index=df.index)
        
        # EPS growth
        if 'eps_change_pct' in df.columns:
            eps_growth = df['eps_change_pct'].fillna(0)
            eps_score = np.where(eps_growth >= 100, 95,
                        np.where(eps_growth >= 50, 85,
                        np.where(eps_growth >= 30, 75,
                        np.where(eps_growth >= 20, 65,
                        np.where(eps_growth >= 10, 55,
                        np.where(eps_growth >= 0, 45, 25))))))
            score = eps_score
        
        # Revenue/price momentum as growth proxy
        if 'ret_3m' in df.columns:
            price_momentum = df['ret_3m'].fillna(0)
            growth_momentum = np.where(price_momentum > 30, 10,
                              np.where(price_momentum > 15, 5, 0))
            score += growth_momentum
        
        return score.clip(0, 100)
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced volume scoring"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Relative volume
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1)
            rvol_score = np.where(rvol >= 10, 95,
                         np.where(rvol >= 5, 90,
                         np.where(rvol >= 3, 80,
                         np.where(rvol >= 2, 70,
                         np.where(rvol >= 1.5, 60, 40)))))
            score = rvol_score
        
        # Sustained volume
        if 'vol_ratio_30d_90d' in df.columns:
            sustained = df['vol_ratio_30d_90d'].fillna(1)
            sustained_bonus = np.where(sustained > 1.3, 10,
                              np.where(sustained > 1.1, 5, 0))
            score += sustained_bonus
        
        return score.clip(0, 100)
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced technical scoring"""
        
        score = pd.Series(50.0, index=df.index)
        
        # Moving average alignment
        ma_score = 0
        if 'above_sma_20d' in df.columns:
            ma_score += df['above_sma_20d'] * 15
        if 'above_sma_50d' in df.columns:
            ma_score += df['above_sma_50d'] * 15
        if 'above_sma_200d' in df.columns:
            ma_score += df['above_sma_200d'] * 20
        
        score = 50 + ma_score
        
        # 52-week position
        if 'position_52w' in df.columns:
            position = df['position_52w']
            position_score = np.where(position >= 80, 20,
                            np.where(position >= 60, 10,
                            np.where(position >= 40, 0,
                            np.where(position >= 20, -10, -20))))
            score += position_score
        
        # Trend strength bonus
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            trend_strength = ((df['ret_7d'] > 0) & (df['ret_30d'] > 10))
            score += trend_strength * 10
        
        return score.clip(0, 100)
    
    def _calculate_sector_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced sector scoring"""
        
        if 'sector_performance' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        # Base sector score
        sector_perf = df['sector_performance'].fillna(0)
        base_score = 50 + (sector_perf * 2)
        
        # Relative strength bonus
        if 'sector_relative_strength' in df.columns:
            rel_strength = df['sector_relative_strength']
            bonus = np.where(rel_strength > 20, 20,
                    np.where(rel_strength > 10, 15,
                    np.where(rel_strength > 5, 10,
                    np.where(rel_strength > 0, 5, 0))))
            base_score += bonus
        
        return base_score.clip(0, 100)
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced risk scoring (higher = more risk)"""
        
        risk = pd.Series(25.0, index=df.index)  # Base risk
        
        # Valuation risk
        if 'pe' in df.columns:
            pe = df['pe'].fillna(25)
            val_risk = np.where(pe > 50, 30,
                       np.where(pe > 35, 20,
                       np.where(pe <= 0, 25, 0)))
            risk += val_risk
        
        # Volatility risk
        if 'ret_30d' in df.columns:
            ret_30d = df['ret_30d'].fillna(0)
            vol_risk = np.where(ret_30d < -30, 40,
                       np.where(ret_30d < -20, 30,
                       np.where(ret_30d > 50, 20,
                       np.where(ret_30d > 40, 10, 0))))
            risk += vol_risk
        
        # Liquidity risk
        if 'volume_1d' in df.columns:
            volume = df['volume_1d'].fillna(100000)
            liq_risk = np.where(volume < 50000, 25,
                       np.where(volume < 100000, 15,
                       np.where(volume < 500000, 5, 0)))
            risk += liq_risk
        
        return risk.clip(0, 100)
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced quality scoring"""
        
        quality = pd.Series(50.0, index=df.index)
        
        # Data completeness
        important_fields = ['price', 'pe', 'eps_current', 'ret_30d', 'volume_1d', 'sector']
        available = [f for f in important_fields if f in df.columns]
        
        if available:
            completeness = df[available].notna().sum(axis=1) / len(available) * 50
            quality = 50 + completeness
        
        # Earnings quality bonus
        if 'eps_current' in df.columns:
            positive_eps = (df['eps_current'] > 0) * 10
            quality += positive_eps
        
        # Consistency bonus
        if 'eps_change_pct' in df.columns:
            consistent_growth = (df['eps_change_pct'] > 0) * 10
            quality += consistent_growth
        
        return quality.clip(0, 100)
    
    def _assess_momentum_quality(self, stock: pd.Series) -> str:
        """Assess momentum quality for each stock"""
        
        ret_7d = stock.get('ret_7d', 0)
        ret_30d = stock.get('ret_30d', 0)
        
        if ret_7d > 5 and ret_30d < 0:
            return 'REVERSAL'
        elif ret_7d > ret_30d / 4 and ret_7d > 0:
            return 'ACCELERATING'
        elif ret_7d > 0 and ret_30d > 10:
            return 'STEADY'
        else:
            return 'NEUTRAL'
    
    def _generate_smart_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate intelligent signals based on adaptive scoring"""
        
        # Dynamic thresholds based on market regime
        regime = df['market_regime'].iloc[0] if len(df) > 0 else 'TRANSITIONAL'
        
        # Adjust thresholds based on regime
        if regime == 'BULL_TRENDING':
            # Slightly lower thresholds in bull market
            thresholds = {
                'STRONG_BUY': 88,
                'BUY': 78,
                'ACCUMULATE': 68,
                'WATCH': 55,
                'NEUTRAL': 40
            }
        elif regime == 'BEAR_VOLATILE':
            # Higher thresholds in bear market
            thresholds = {
                'STRONG_BUY': 95,
                'BUY': 85,
                'ACCUMULATE': 75,
                'WATCH': 65,
                'NEUTRAL': 50
            }
        else:
            # Default thresholds
            thresholds = {
                'STRONG_BUY': 92,
                'BUY': 82,
                'ACCUMULATE': 72,
                'WATCH': 60,
                'NEUTRAL': 40
            }
        
        # Generate signals
        conditions = [
            df['composite_score'] >= thresholds['STRONG_BUY'],
            df['composite_score'] >= thresholds['BUY'],
            df['composite_score'] >= thresholds['ACCUMULATE'],
            df['composite_score'] >= thresholds['WATCH'],
            df['composite_score'] >= thresholds['NEUTRAL']
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH', 'NEUTRAL']
        
        df['signal'] = np.select(conditions, choices, default='AVOID')
        
        # Add entry zone from insights
        df['entry_zone'] = df['smart_insights'].apply(
            lambda x: x.get('entry_strategy', {}) if isinstance(x, dict) else {}
        )
        
        return df

# Export
__all__ = ['UltimateSignalEngine']

"""
signal_engine.py - M.A.N.T.R.A. Version 3 FINAL Signal Engine
=============================================================
8-Factor precision signal generation with explainable AI
Ultra-conservative thresholds for maximum confidence
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [SIGNAL] %(levelname)s - %(message)s"))
    logger.addHandler(handler)

class SignalEngine:
    """8-Factor precision signal engine"""
    
    def __init__(self):
        self.config = CONFIG
        self.stocks_df = None
        self.sector_df = None
        self.signals_df = None
        self.market_condition = None
        
    def analyze(self, stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main analysis method - generates signals for all stocks
        
        Args:
            stocks_df: Main stocks dataframe
            sector_df: Sector performance dataframe
            
        Returns:
            DataFrame with signals and scores
        """
        start_time = pd.Timestamp.now()
        
        self.stocks_df = stocks_df.copy()
        self.sector_df = sector_df
        
        logger.info(f"🎯 Starting signal analysis for {len(stocks_df)} stocks...")
        
        # Detect market condition
        self._detect_market_condition()
        
        # Calculate all 8 factors
        logger.info("📊 Calculating 8-factor scores...")
        self._calculate_momentum_factor()
        self._calculate_value_factor()
        self._calculate_growth_factor()
        self._calculate_volume_factor()
        self._calculate_technical_factor()
        self._calculate_sector_factor()
        self._calculate_risk_factor()
        self._calculate_quality_factor()
        
        # Calculate composite score
        self._calculate_composite_score()
        
        # Generate signals
        self._generate_signals()
        
        # Add explanations
        self._add_explanations()
        
        # Sort by composite score
        self.signals_df = self.stocks_df.sort_values('composite_score', ascending=False)
        
        # Log summary
        signal_counts = self.signals_df['signal'].value_counts()
        logger.info(f"✅ Signal analysis complete in {(pd.Timestamp.now() - start_time).total_seconds():.2f}s")
        logger.info(f"📈 Signals: {dict(signal_counts)}")
        
        return self.signals_df
    
    def _detect_market_condition(self):
        """Detect overall market condition"""
        
        if 'ret_1d' not in self.stocks_df.columns:
            self.market_condition = 'neutral'
            return
        
        # Calculate market breadth
        positive_stocks = (self.stocks_df['ret_1d'] > 0).sum()
        total_stocks = len(self.stocks_df)
        breadth = (positive_stocks / total_stocks * 100) if total_stocks > 0 else 50
        
        # Determine condition
        if breadth >= 65:
            self.market_condition = 'bullish'
        elif breadth <= 35:
            self.market_condition = 'bearish'
        else:
            self.market_condition = 'neutral'
        
        logger.info(f"📊 Market condition: {self.market_condition} (breadth: {breadth:.1f}%)")
    
    def _calculate_momentum_factor(self):
        """Calculate momentum factor (25% weight)"""
        
        self.stocks_df['momentum_score'] = 50.0  # Default
        
        # Multi-timeframe momentum
        momentum_cols = {
            'ret_1d': 0.10,
            'ret_7d': 0.20,
            'ret_30d': 0.40,
            'ret_3m': 0.30
        }
        
        weighted_momentum = pd.Series(0.0, index=self.stocks_df.index)
        total_weight = 0
        
        for col, weight in momentum_cols.items():
            if col in self.stocks_df.columns:
                returns = self.stocks_df[col].fillna(0)
                
                # Score based on performance
                if col == 'ret_30d':  # Most important
                    score = np.where(returns >= 20, 90,
                           np.where(returns >= 10, 80,
                           np.where(returns >= 5, 70,
                           np.where(returns >= 0, 60,
                           np.where(returns >= -10, 40, 20)))))
                else:
                    score = 50 + (returns * 2)  # Simple scaling
                
                weighted_momentum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            self.stocks_df['momentum_score'] = (weighted_momentum / total_weight).clip(0, 100)
        
        # Bonus for momentum consistency
        if 'momentum_breadth' in self.stocks_df.columns:
            consistency_bonus = (self.stocks_df['momentum_breadth'] - 50) / 10
            self.stocks_df['momentum_score'] = (self.stocks_df['momentum_score'] + consistency_bonus).clip(0, 100)
    
    def _calculate_value_factor(self):
        """Calculate value factor (20% weight)"""
        
        self.stocks_df['value_score'] = 50.0  # Default
        
        if 'pe' not in self.stocks_df.columns:
            return
        
        pe = self.stocks_df['pe'].fillna(25)
        
        # PE-based scoring (lower is better for value)
        value_score = np.where(pe <= 0, 30,      # Negative earnings
                      np.where(pe <= 10, 95,     # Deep value
                      np.where(pe <= 15, 85,     # Strong value
                      np.where(pe <= 20, 75,     # Good value
                      np.where(pe <= 25, 65,     # Fair value
                      np.where(pe <= 35, 50,     # Growth premium
                      np.where(pe <= 50, 35,     # Expensive
                      20)))))))                  # Very expensive
        
        self.stocks_df['value_score'] = value_score
        
        # Adjust for EPS quality
        if 'eps_current' in self.stocks_df.columns:
            positive_eps = (self.stocks_df['eps_current'] > 0).astype(int)
            self.stocks_df['value_score'] += positive_eps * 5
            self.stocks_df['value_score'] = self.stocks_df['value_score'].clip(0, 100)
    
    def _calculate_growth_factor(self):
        """Calculate growth factor (18% weight)"""
        
        self.stocks_df['growth_score'] = 50.0  # Default
        
        if 'eps_change_pct' not in self.stocks_df.columns:
            return
        
        eps_growth = self.stocks_df['eps_change_pct'].fillna(0)
        
        # EPS growth scoring
        growth_score = np.where(eps_growth >= 50, 95,      # Exceptional growth
                       np.where(eps_growth >= 30, 85,      # Strong growth
                       np.where(eps_growth >= 20, 75,      # Good growth
                       np.where(eps_growth >= 10, 65,      # Decent growth
                       np.where(eps_growth >= 5, 55,       # Modest growth
                       np.where(eps_growth >= 0, 45,       # Stable
                       np.where(eps_growth >= -10, 35,     # Slight decline
                       20)))))))                           # Significant decline
        
        self.stocks_df['growth_score'] = growth_score
        
        # Bonus for consistent growth
        if 'eps_growing' in self.stocks_df.columns:
            self.stocks_df['growth_score'] += self.stocks_df['eps_growing'] * 5
            self.stocks_df['growth_score'] = self.stocks_df['growth_score'].clip(0, 100)
    
    def _calculate_volume_factor(self):
        """Calculate volume factor (15% weight)"""
        
        self.stocks_df['volume_score'] = 50.0  # Default
        
        if 'rvol' not in self.stocks_df.columns:
            return
        
        rvol = self.stocks_df['rvol'].fillna(1)
        
        # Volume scoring (higher relative volume is better)
        volume_score = np.where(rvol >= 5, 95,      # Extreme interest
                       np.where(rvol >= 3, 85,      # Very high interest
                       np.where(rvol >= 2, 75,      # High interest
                       np.where(rvol >= 1.5, 65,    # Elevated interest
                       np.where(rvol >= 1.2, 55,    # Above average
                       np.where(rvol >= 0.8, 45,    # Normal
                       30))))))                     # Below average
        
        self.stocks_df['volume_score'] = volume_score
        
        # Bonus for volume spike
        if 'volume_spike' in self.stocks_df.columns:
            self.stocks_df['volume_score'] += self.stocks_df['volume_spike'] * 5
            self.stocks_df['volume_score'] = self.stocks_df['volume_score'].clip(0, 100)
    
    def _calculate_technical_factor(self):
        """Calculate technical factor (12% weight)"""
        
        self.stocks_df['technical_score'] = 50.0  # Default
        
        # Start with base score
        tech_score = pd.Series(50.0, index=self.stocks_df.index)
        
        # Moving average analysis
        for ma in ['20d', '50d', '200d']:
            col = f'above_sma_{ma}'
            if col in self.stocks_df.columns:
                tech_score += self.stocks_df[col] * 10  # +10 for each MA above
        
        # 52-week position
        if 'position_52w' in self.stocks_df.columns:
            position = self.stocks_df['position_52w']
            position_bonus = np.where(position >= 80, 20,    # Near highs
                            np.where(position >= 60, 10,     # Upper range
                            np.where(position >= 40, 0,      # Middle range
                            np.where(position >= 20, -10,    # Lower range
                            -20))))                          # Near lows
            tech_score += position_bonus
        
        self.stocks_df['technical_score'] = tech_score.clip(0, 100)
    
    def _calculate_sector_factor(self):
        """Calculate sector factor (6% weight)"""
        
        self.stocks_df['sector_score'] = 50.0  # Default
        
        if self.sector_df is None or self.sector_df.empty:
            return
        
        # Create sector performance lookup
        sector_perf = {}
        if 'sector_ret_30d' in self.sector_df.columns:
            for _, row in self.sector_df.iterrows():
                sector_perf[row['sector']] = row['sector_ret_30d']
        
        # Apply sector performance to stocks
        if 'sector' in self.stocks_df.columns and sector_perf:
            self.stocks_df['sector_performance'] = self.stocks_df['sector'].map(sector_perf).fillna(0)
            
            # Score based on sector strength
            sector_score = 50 + (self.stocks_df['sector_performance'] * 2)
            self.stocks_df['sector_score'] = sector_score.clip(0, 100)
    
    def _calculate_risk_factor(self):
        """Calculate risk factor (3% weight) - higher score = higher risk"""
        
        risk_score = pd.Series(25.0, index=self.stocks_df.index)  # Base risk
        
        # Valuation risk
        if 'pe' in self.stocks_df.columns:
            pe = self.stocks_df['pe'].fillna(25)
            valuation_risk = np.where(pe > 50, 25,
                            np.where(pe > 35, 15,
                            np.where(pe <= 0, 20, 0)))  # Negative PE is risky
            risk_score += valuation_risk
        
        # Volatility risk (based on return patterns)
        if 'ret_30d' in self.stocks_df.columns:
            ret_30d = self.stocks_df['ret_30d'].fillna(0)
            volatility_risk = np.where(ret_30d < -20, 30,
                             np.where(ret_30d < -10, 20,
                             np.where(ret_30d > 50, 15, 0)))  # Too hot is risky too
            risk_score += volatility_risk
        
        # Low volume risk
        if 'volume_1d' in self.stocks_df.columns:
            volume = self.stocks_df['volume_1d'].fillna(0)
            liquidity_risk = np.where(volume < 50000, 20,
                            np.where(volume < 100000, 10, 0))
            risk_score += liquidity_risk
        
        self.stocks_df['risk_score'] = risk_score.clip(0, 100)
    
    def _calculate_quality_factor(self):
        """Calculate quality factor (1% weight) - data completeness"""
        
        # Count non-null important fields
        important_fields = ['price', 'pe', 'eps_current', 'ret_30d', 'volume_1d', 'sector']
        available_fields = [f for f in important_fields if f in self.stocks_df.columns]
        
        if available_fields:
            non_null_counts = self.stocks_df[available_fields].notna().sum(axis=1)
            completeness = (non_null_counts / len(available_fields)) * 100
            self.stocks_df['quality_score'] = completeness
        else:
            self.stocks_df['quality_score'] = 80.0  # Default
    
    def _calculate_composite_score(self):
        """Calculate weighted composite score"""
        
        weights = self.config.FACTOR_WEIGHTS
        
        # Calculate weighted sum
        self.stocks_df['composite_score'] = (
            self.stocks_df['momentum_score'] * weights['momentum'] +
            self.stocks_df['value_score'] * weights['value'] +
            self.stocks_df['growth_score'] * weights['growth'] +
            self.stocks_df['volume_score'] * weights['volume'] +
            self.stocks_df['technical_score'] * weights['technical'] +
            self.stocks_df['sector_score'] * weights['sector'] +
            (100 - self.stocks_df['risk_score']) * weights['risk'] +  # Invert risk
            self.stocks_df['quality_score'] * weights['quality']
        ).round(1)
        
        # Apply market condition adjustments
        if self.market_condition == 'bullish':
            # Boost momentum in bull market
            self.stocks_df['composite_score'] += self.stocks_df['momentum_score'] * 0.05
        elif self.market_condition == 'bearish':
            # Boost value in bear market
            self.stocks_df['composite_score'] += self.stocks_df['value_score'] * 0.05
        
        # Ensure score is within bounds
        self.stocks_df['composite_score'] = self.stocks_df['composite_score'].clip(0, 100)
    
    def _generate_signals(self):
        """Generate trading signals based on composite score"""
        
        thresholds = self.config.SIGNAL_THRESHOLDS
        
        conditions = [
            self.stocks_df['composite_score'] >= thresholds['STRONG_BUY'],
            self.stocks_df['composite_score'] >= thresholds['BUY'],
            self.stocks_df['composite_score'] >= thresholds['ACCUMULATE'],
            self.stocks_df['composite_score'] >= thresholds['WATCH'],
            self.stocks_df['composite_score'] >= thresholds['NEUTRAL'],
            self.stocks_df['composite_score'] >= thresholds['AVOID']
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'WATCH', 'NEUTRAL', 'AVOID']
        
        self.stocks_df['signal'] = np.select(conditions, choices, default='STRONG_AVOID')
        
        # Add confidence (same as composite score for simplicity)
        self.stocks_df['confidence'] = self.stocks_df['composite_score']
    
    def _add_explanations(self):
        """Add explanations for each signal"""
        
        explanations = []
        
        for idx, row in self.stocks_df.iterrows():
            # Find top factors
            factors = {
                'Momentum': row['momentum_score'],
                'Value': row['value_score'],
                'Growth': row['growth_score'],
                'Volume': row['volume_score'],
                'Technical': row['technical_score'],
                'Sector': row['sector_score']
            }
            
            # Sort factors by score
            sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
            top_factors = [f"{name} ({score:.0f})" for name, score in sorted_factors[:3]]
            
            # Create explanation
            if row['signal'] == 'STRONG_BUY':
                explanation = f"Ultra-high confidence with {', '.join(top_factors)}"
            elif row['signal'] == 'BUY':
                explanation = f"High confidence with {', '.join(top_factors)}"
            elif row['signal'] == 'ACCUMULATE':
                explanation = f"Good opportunity with {', '.join(top_factors)}"
            else:
                explanation = f"Signal based on {', '.join(top_factors)}"
            
            explanations.append(explanation)
        
        self.stocks_df['explanation'] = explanations
    
    def get_top_opportunities(self, limit: int = 10) -> pd.DataFrame:
        """Get top opportunities"""
        
        if self.signals_df is None:
            return pd.DataFrame()
        
        # Filter for actionable signals
        actionable = self.signals_df[
            self.signals_df['signal'].isin(['STRONG_BUY', 'BUY', 'ACCUMULATE'])
        ]
        
        return actionable.head(limit)
    
    def get_signal_summary(self) -> Dict[str, int]:
        """Get signal distribution summary"""
        
        if self.signals_df is None:
            return {}
        
        return self.signals_df['signal'].value_counts().to_dict()
    
    def get_market_breadth(self) -> float:
        """Get market breadth percentage"""
        
        if self.signals_df is None or 'ret_1d' not in self.signals_df.columns:
            return 50.0
        
        positive = (self.signals_df['ret_1d'] > 0).sum()
        total = len(self.signals_df)
        
        return (positive / total * 100) if total > 0 else 50.0

# Export
# For backward compatibility, import from enhanced version
from enhanced_signal_engine import EnhancedSignalEngine as SignalEngine

__all__ = ['SignalEngine']

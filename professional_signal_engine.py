"""
professional_signal_engine.py - Professional Signal Engine
==========================================================
Clean, simple, bug-free signal generation
Focus on reliability and clarity
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class ProfessionalSignalEngine:
    """Professional signal engine with robust error handling"""
    
    def __init__(self):
        self.config = CONFIG
        self.market_condition = 'NEUTRAL'
        
    def analyze(self, stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main analysis method - clean and robust
        
        Args:
            stocks_df: Stock data
            sector_df: Sector data
            
        Returns:
            Enhanced dataframe with signals
        """
        try:
            logger.info(f"Starting analysis for {len(stocks_df)} stocks")
            
            # Create a copy to avoid modifying original
            df = stocks_df.copy()
            
            # Step 1: Detect market condition
            self._detect_market_condition(df)
            
            # Step 2: Calculate factor scores
            logger.info("Calculating factor scores...")
            df = self._calculate_all_factors(df, sector_df)
            
            # Step 3: Calculate composite score
            df['composite_score'] = self._calculate_composite_score(df)
            
            # Step 4: Generate signals
            df['signal'] = self._generate_signals(df)
            
            # Step 5: Add confidence (same as composite score)
            df['confidence'] = df['composite_score'].round(1)
            
            # Step 6: Add key insights
            df['key_insights'] = df.apply(self._generate_insights, axis=1)
            
            # Sort by composite score
            df = df.sort_values('composite_score', ascending=False)
            
            logger.info(f"Analysis complete. Signals: {df['signal'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            # Return original dataframe with default values
            stocks_df['signal'] = 'HOLD'
            stocks_df['confidence'] = 50
            stocks_df['composite_score'] = 50
            return stocks_df
    
    def _detect_market_condition(self, df: pd.DataFrame) -> None:
        """Detect overall market condition"""
        try:
            if 'ret_1d' not in df.columns:
                self.market_condition = 'NEUTRAL'
                return
            
            # Calculate market breadth
            positive = (df['ret_1d'] > 0).sum()
            total = len(df)
            breadth = (positive / total * 100) if total > 0 else 50
            
            # Determine condition
            if breadth >= 65:
                self.market_condition = 'BULLISH'
            elif breadth <= 35:
                self.market_condition = 'BEARISH'
            else:
                self.market_condition = 'NEUTRAL'
                
            logger.info(f"Market condition: {self.market_condition} (breadth: {breadth:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error detecting market condition: {e}")
            self.market_condition = 'NEUTRAL'
    
    def _calculate_all_factors(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all factor scores with error handling"""
        
        # Initialize all scores with defaults
        factor_scores = {
            'momentum_score': 50.0,
            'value_score': 50.0,
            'growth_score': 50.0,
            'volume_score': 50.0,
            'technical_score': 50.0,
            'sector_score': 50.0,
            'risk_score': 50.0,
            'quality_score': 50.0
        }
        
        # Calculate each factor safely
        try:
            df['momentum_score'] = self._calculate_momentum_score(df)
        except Exception as e:
            logger.error(f"Momentum calculation failed: {e}")
            df['momentum_score'] = factor_scores['momentum_score']
        
        try:
            df['value_score'] = self._calculate_value_score(df)
        except Exception as e:
            logger.error(f"Value calculation failed: {e}")
            df['value_score'] = factor_scores['value_score']
        
        try:
            df['growth_score'] = self._calculate_growth_score(df)
        except Exception as e:
            logger.error(f"Growth calculation failed: {e}")
            df['growth_score'] = factor_scores['growth_score']
        
        try:
            df['volume_score'] = self._calculate_volume_score(df)
        except Exception as e:
            logger.error(f"Volume calculation failed: {e}")
            df['volume_score'] = factor_scores['volume_score']
        
        try:
            df['technical_score'] = self._calculate_technical_score(df)
        except Exception as e:
            logger.error(f"Technical calculation failed: {e}")
            df['technical_score'] = factor_scores['technical_score']
        
        try:
            df['sector_score'] = self._calculate_sector_score(df, sector_df)
        except Exception as e:
            logger.error(f"Sector calculation failed: {e}")
            df['sector_score'] = factor_scores['sector_score']
        
        try:
            df['risk_score'] = self._calculate_risk_score(df)
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            df['risk_score'] = factor_scores['risk_score']
        
        try:
            df['quality_score'] = self._calculate_quality_score(df)
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            df['quality_score'] = factor_scores['quality_score']
        
        return df
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score (simplified and robust)"""
        score = pd.Series(50.0, index=df.index)
        
        # Use available return columns
        if 'ret_30d' in df.columns:
            ret_30d = df['ret_30d'].fillna(0)
            score = 50 + (ret_30d * 1.5).clip(-50, 50)
        
        # Add bonus for consistent positive returns
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            consistent = ((df['ret_1d'] > 0) & (df['ret_7d'] > 0) & (df['ret_30d'] > 0))
            score = score + (consistent * 10)
        
        return score.clip(0, 100)
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate value score (simplified and robust)"""
        if 'pe' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        pe = df['pe'].fillna(25)
        
        # Simple PE-based scoring
        score = np.where(pe <= 0, 30,        # Negative PE
                np.where(pe <= 15, 90,       # Great value
                np.where(pe <= 25, 70,       # Good value
                np.where(pe <= 35, 50,       # Fair value
                np.where(pe <= 50, 30, 10)))))  # Expensive
        
        return pd.Series(score, index=df.index)
    
    def _calculate_growth_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate growth score (simplified and robust)"""
        if 'eps_change_pct' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        eps_growth = df['eps_change_pct'].fillna(0)
        
        # Simple growth scoring
        score = np.where(eps_growth >= 50, 90,
                np.where(eps_growth >= 30, 80,
                np.where(eps_growth >= 20, 70,
                np.where(eps_growth >= 10, 60,
                np.where(eps_growth >= 0, 50, 30)))))
        
        return pd.Series(score, index=df.index)
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume score (simplified and robust)"""
        if 'rvol' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        rvol = df['rvol'].fillna(1)
        
        # Simple volume scoring
        score = np.where(rvol >= 5, 90,
                np.where(rvol >= 3, 80,
                np.where(rvol >= 2, 70,
                np.where(rvol >= 1.5, 60, 50))))
        
        return pd.Series(score, index=df.index)
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate technical score (simplified and robust)"""
        score = pd.Series(50.0, index=df.index)
        
        # Price vs moving averages
        ma_count = 0
        for ma in ['sma_20d', 'sma_50d', 'sma_200d']:
            if ma in df.columns and 'price' in df.columns:
                above_ma = (df['price'] > df[ma])
                score = score + (above_ma * 10)
                ma_count += 1
        
        # 52-week position
        if 'position_52w' in df.columns:
            position = df['position_52w'].fillna(50)
            position_bonus = (position - 50) / 5  # +/-10 points
            score = score + position_bonus
        
        return score.clip(0, 100)
    
    def _calculate_sector_score(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.Series:
        """Calculate sector score (simplified and robust)"""
        score = pd.Series(50.0, index=df.index)
        
        if sector_df is None or sector_df.empty or 'sector' not in df.columns:
            return score
        
        try:
            # Create sector performance lookup
            if 'sector_ret_30d' in sector_df.columns:
                sector_perf = dict(zip(sector_df['sector'], sector_df['sector_ret_30d']))
                df['sector_performance'] = df['sector'].map(sector_perf).fillna(0)
                
                # Score based on sector strength
                score = 50 + (df['sector_performance'] * 2).clip(-50, 50)
        except Exception as e:
            logger.error(f"Sector score calculation error: {e}")
        
        return score.clip(0, 100)
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk score (higher = more risk)"""
        risk = pd.Series(30.0, index=df.index)  # Base risk
        
        # High PE risk
        if 'pe' in df.columns:
            pe = df['pe'].fillna(25)
            high_pe_risk = np.where(pe > 50, 30, np.where(pe > 35, 15, 0))
            risk = risk + high_pe_risk
        
        # High volatility risk
        if 'ret_30d' in df.columns:
            ret_30d = df['ret_30d'].fillna(0)
            volatility_risk = np.where(abs(ret_30d) > 50, 20, 
                             np.where(abs(ret_30d) > 30, 10, 0))
            risk = risk + volatility_risk
        
        return risk.clip(0, 100)
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quality score based on data completeness"""
        # Important fields
        important_fields = ['price', 'pe', 'eps_current', 'ret_30d', 'volume_1d']
        available = [f for f in important_fields if f in df.columns]
        
        if not available:
            return pd.Series(80.0, index=df.index)
        
        # Score based on non-null values
        non_null_count = df[available].notna().sum(axis=1)
        score = (non_null_count / len(available)) * 100
        
        return score.clip(0, 100)
    
    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite score"""
        weights = self.config.FACTOR_WEIGHTS
        
        # Calculate weighted sum
        composite = (
            df['momentum_score'] * weights['momentum'] +
            df['value_score'] * weights['value'] +
            df['growth_score'] * weights['growth'] +
            df['volume_score'] * weights['volume'] +
            df['technical_score'] * weights['technical'] +
            df['sector_score'] * weights['sector'] +
            (100 - df['risk_score']) * weights['risk'] +
            df['quality_score'] * weights['quality']
        )
        
        # Market condition adjustment
        if self.market_condition == 'BULLISH':
            composite = composite + (df['momentum_score'] * 0.05)
        elif self.market_condition == 'BEARISH':
            composite = composite + (df['value_score'] * 0.05)
        
        return composite.clip(0, 100).round(1)
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on composite score"""
        score = df['composite_score']
        
        # Simple, clear thresholds
        conditions = [
            score >= 85,  # STRONG_BUY
            score >= 70,  # BUY
            score >= 40,  # HOLD
            score >= 0    # SELL
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL']
        
        return pd.Series(np.select(conditions[:-1], choices[:-1], default=choices[-1]), index=df.index)
    
    def _generate_insights(self, row: pd.Series) -> str:
        """Generate simple, actionable insights"""
        insights = []
        
        # Momentum insight
        if row.get('momentum_score', 50) > 80:
            insights.append("Strong momentum")
        elif row.get('momentum_score', 50) < 30:
            insights.append("Weak momentum")
        
        # Value insight
        if row.get('pe', 100) > 0 and row.get('pe', 100) < 15:
            insights.append(f"Attractive valuation (PE: {row['pe']:.1f})")
        
        # Volume insight
        if row.get('rvol', 1) > 3:
            insights.append(f"High volume ({row['rvol']:.1f}x average)")
        
        # Growth insight
        if row.get('eps_change_pct', 0) > 30:
            insights.append(f"Strong earnings growth ({row['eps_change_pct']:.0f}%)")
        
        # Return top 2 insights
        if insights:
            return " | ".join(insights[:2])
        else:
            return f"Score: {row.get('composite_score', 50):.0f}/100"

# Export
__all__ = ['ProfessionalSignalEngine']

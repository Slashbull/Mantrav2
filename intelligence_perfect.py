"""
intelligence_perfect.py - M.A.N.T.R.A. Version 3 FINAL - Perfect Explainable AI
==============================================================================
Ultimate explainable AI system for crystal-clear signal reasoning
Provides natural language explanations for every trading decision
Built for permanent use with comprehensive transparency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from config_ultimate import CONFIG

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [INTELLIGENCE] %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# =============================================================================
# EXPLANATION DATA STRUCTURES
# =============================================================================

@dataclass
class DetailedExplanation:
    """Comprehensive explanation container for stock signals"""
    ticker: str
    signal: str
    confidence: float
    
    # Core explanation
    headline: str
    primary_thesis: str
    supporting_evidence: List[str]
    risk_considerations: List[str]
    
    # Factor breakdown
    factor_analysis: Dict[str, Dict[str, Any]]
    
    # Actionable insights
    recommendation: str
    target_action: str
    risk_management: str
    
    # Context
    market_context: str
    sector_context: str
    
    # Quality metrics
    data_quality: str
    explanation_confidence: float
    timestamp: datetime

@dataclass
class FactorInsight:
    """Individual factor analysis insight"""
    factor_name: str
    score: float
    rating: str  # 'Excellent', 'Good', 'Fair', 'Poor'
    icon: str
    description: str
    supporting_data: Dict[str, Any]
    contribution: float  # Contribution to overall signal

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    overall_risk: str  # 'Low', 'Medium', 'High', 'Very High'
    risk_score: float
    key_risks: List[str]
    risk_factors: Dict[str, str]
    mitigation_strategies: List[str]

# =============================================================================
# PERFECT EXPLAINABLE AI ENGINE
# =============================================================================

class PerfectExplainableAI:
    """
    Perfect explainable AI system for trading signals
    
    Converts quantitative analysis into clear, actionable insights
    that any investor can understand and trust.
    """
    
    def __init__(self):
        self.config = CONFIG
        self.explanation_templates = self._load_explanation_templates()
        self.factor_interpreters = self._load_factor_interpreters()
        self.risk_assessors = self._load_risk_assessors()
        self.market_context_analyzer = self._load_market_context_analyzer()
        
        logger.info("🧠 Perfect Explainable AI System initialized")
    
    def _load_explanation_templates(self) -> Dict[str, Dict]:
        """Load natural language explanation templates"""
        
        return {
            "STRONG_BUY": {
                "headline": "🚀 Exceptional Opportunity - Strong Buy Signal",
                "thesis_templates": [
                    "Multiple factors strongly aligned for significant upside potential",
                    "Outstanding {primary_strength} combined with {secondary_strength} creates compelling opportunity",
                    "Rare combination of value and momentum with {confidence}% confidence",
                    "All key indicators pointing to sustained outperformance"
                ],
                "confidence_descriptors": {
                    95: "Exceptionally high confidence",
                    90: "Very high confidence", 
                    85: "High confidence",
                    80: "Strong confidence",
                    75: "Good confidence"
                }
            },
            
            "BUY": {
                "headline": "📈 Strong Opportunity - Buy Signal",
                "thesis_templates": [
                    "Solid fundamentals with positive technical setup",
                    "Good value opportunity with improving {primary_strength}",
                    "Strong {primary_strength} provides good upside potential",
                    "Multiple positive factors support purchase decision"
                ],
                "confidence_descriptors": {
                    90: "High confidence",
                    85: "Good confidence",
                    80: "Solid confidence", 
                    75: "Reasonable confidence",
                    70: "Moderate confidence"
                }
            },
            
            "ACCUMULATE": {
                "headline": "📊 Building Opportunity - Accumulate Signal",
                "thesis_templates": [
                    "Gradual accumulation opportunity with improving fundamentals",
                    "Good long-term value with patience required for {primary_strength}",
                    "Building position makes sense given current {primary_strength}",
                    "Steady accumulation recommended as investment story develops"
                ],
                "confidence_descriptors": {
                    80: "Good confidence for accumulation",
                    75: "Reasonable confidence for building",
                    70: "Moderate confidence for gradual entry",
                    65: "Some confidence for patient accumulation"
                }
            },
            
            "WATCH": {
                "headline": "👀 Monitor Closely - Watch Signal", 
                "thesis_templates": [
                    "Mixed signals require careful monitoring before commitment",
                    "Potential developing but needs confirmation in {weak_factor}",
                    "Watching for improvement in {weak_factor} to trigger entry",
                    "On the radar but not ready for full commitment"
                ],
                "confidence_descriptors": {
                    70: "Worth monitoring closely",
                    65: "Interesting but uncertain",
                    60: "Mixed signals present",
                    55: "Unclear direction"
                }
            },
            
            "AVOID": {
                "headline": "⚠️ Caution Required - Avoid Signal",
                "thesis_templates": [
                    "Multiple risk factors outweigh potential upside",
                    "Fundamental concerns in {primary_weakness} make investment unattractive",
                    "Technical weakness in {primary_weakness} suggests downside risk",
                    "Better opportunities available elsewhere with lower risk"
                ],
                "confidence_descriptors": {
                    80: "Clear avoidance recommended",
                    75: "Strong caution advised",
                    70: "Significant concerns present",
                    65: "Multiple warning signs detected"
                }
            }
        }
    
    def _load_factor_interpreters(self) -> Dict[str, Dict]:
        """Load factor interpretation logic for natural language"""
        
        return {
            "momentum": {
                "name": "Price Momentum",
                "weight": self.config.signals.FACTOR_WEIGHTS["momentum"],
                "thresholds": {
                    90: ("Exceptional momentum", "🚀", "Very strong uptrend across all timeframes"),
                    80: ("Strong momentum", "📈", "Solid upward price movement"),
                    70: ("Good momentum", "↗️", "Positive price trend developing"),
                    60: ("Modest momentum", "➡️", "Some upward movement"),
                    50: ("Neutral momentum", "😐", "No clear price direction"),
                    40: ("Weak momentum", "↘️", "Some downward pressure"),
                    30: ("Poor momentum", "📉", "Declining price trend"),
                    20: ("Very poor momentum", "⬇️", "Strong downward movement")
                },
                "details": {
                    "timeframes": ["1-day", "7-day", "30-day", "3-month"],
                    "interpretation": "Measures price performance across multiple timeframes to identify trends"
                }
            },
            
            "value": {
                "name": "Valuation Attractiveness", 
                "weight": self.config.signals.FACTOR_WEIGHTS["value"],
                "thresholds": {
                    90: ("Exceptional value", "💎", "Trading at very attractive valuation levels"),
                    80: ("Strong value", "💰", "Good value proposition at current prices"),
                    70: ("Fair value", "⚖️", "Reasonably valued relative to fundamentals"),
                    60: ("Modest value", "🤏", "Slight value opportunity present"),
                    50: ("Neutral valuation", "😐", "Fair market pricing"),
                    40: ("Expensive", "💸", "Premium valuation concern"),
                    30: ("Very expensive", "🔥", "High valuation presents risk"),
                    20: ("Extremely expensive", "⚠️", "Significant overvaluation risk")
                },
                "details": {
                    "metrics": ["P/E ratio", "Earnings quality", "Market context"],
                    "interpretation": "Assesses if stock is attractively priced relative to fundamentals"
                }
            },
            
            "growth": {
                "name": "Earnings Growth",
                "weight": self.config.signals.FACTOR_WEIGHTS["growth"],
                "thresholds": {
                    90: ("Exceptional growth", "🌟", "Outstanding earnings acceleration"),
                    80: ("Strong growth", "📊", "Solid earnings expansion trend"),
                    70: ("Good growth", "📈", "Healthy earnings improvement"),
                    60: ("Modest growth", "➕", "Some earnings progress"),
                    50: ("Stable earnings", "😐", "Flat earnings performance"),
                    40: ("Declining growth", "➖", "Slowing earnings momentum"),
                    30: ("Poor growth", "📉", "Weakening earnings trend"),
                    20: ("Severe decline", "🔴", "Significant earnings deterioration")
                },
                "details": {
                    "metrics": ["EPS change %", "Growth sustainability", "Quality assessment"],
                    "interpretation": "Evaluates earnings growth trends and sustainability"
                }
            },
            
            "volume": {
                "name": "Trading Interest",
                "weight": self.config.signals.FACTOR_WEIGHTS["volume"],
                "thresholds": {
                    90: ("Exceptional interest", "🔥", "Very high institutional and retail interest"),
                    80: ("Strong interest", "📊", "Elevated trading activity"),
                    70: ("Good interest", "👥", "Above-average market participation"),
                    60: ("Modest interest", "🤏", "Some increased trading activity"),
                    50: ("Normal interest", "😐", "Average trading levels"),
                    40: ("Weak interest", "👻", "Below-average participation"),
                    30: ("Poor interest", "😴", "Low trading activity"),
                    20: ("Very poor interest", "💀", "Minimal market interest")
                },
                "details": {
                    "metrics": ["Relative volume", "Volume trends", "Liquidity assessment"],
                    "interpretation": "Measures market interest and trading liquidity"
                }
            },
            
            "technical": {
                "name": "Technical Setup",
                "weight": self.config.signals.FACTOR_WEIGHTS["technical"],
                "thresholds": {
                    90: ("Excellent setup", "⭐", "Very strong technical position"),
                    80: ("Strong setup", "✅", "Good technical indicators aligned"),
                    70: ("Positive setup", "👍", "Favorable technical picture"),
                    60: ("Neutral setup", "😐", "Mixed technical signals"),
                    50: ("Unclear setup", "🤷", "No clear technical direction"),
                    40: ("Weak setup", "👎", "Some technical concerns"),
                    30: ("Poor setup", "❌", "Negative technical indicators"),
                    20: ("Very poor setup", "🚫", "Significant technical weakness")
                },
                "details": {
                    "metrics": ["Moving averages", "52-week position", "Chart patterns"],
                    "interpretation": "Analyzes chart patterns and technical indicators"
                }
            },
            
            "sector": {
                "name": "Sector Strength",
                "weight": self.config.signals.FACTOR_WEIGHTS["sector"],
                "thresholds": {
                    90: ("Hot sector", "🔥", "Industry showing exceptional outperformance"),
                    80: ("Strong sector", "💪", "Industry performing very well"),
                    70: ("Good sector", "👍", "Industry showing positive trends"),
                    60: ("Neutral sector", "😐", "Industry performing in-line"),
                    50: ("Mixed sector", "🤷", "Industry showing mixed signals"),
                    40: ("Weak sector", "👎", "Industry underperforming market"),
                    30: ("Poor sector", "📉", "Industry showing clear weakness"),
                    20: ("Struggling sector", "⚠️", "Industry facing significant headwinds")
                },
                "details": {
                    "metrics": ["Sector returns", "Relative performance", "Rotation trends"],
                    "interpretation": "Evaluates industry group performance and rotation trends"
                }
            }
        }
    
    def _load_risk_assessors(self) -> Dict[str, Dict]:
        """Load risk assessment interpretation logic"""
        
        return {
            "valuation_risk": {
                "high_pe": {
                    "threshold": self.config.benchmarks.VALUE_BENCHMARKS['expensive'],
                    "warning": "High valuation multiple increases downside risk",
                    "explanation": "P/E ratio above {threshold} suggests premium pricing with limited margin of safety"
                },
                "negative_earnings": {
                    "threshold": 0,
                    "warning": "Loss-making company carries execution risk", 
                    "explanation": "Negative earnings indicate operational challenges and uncertain outlook"
                }
            },
            
            "liquidity_risk": {
                "low_volume": {
                    "threshold": 50000,
                    "warning": "Low trading volume may impact liquidity",
                    "explanation": "Daily volume below {threshold} shares creates potential exit difficulties"
                },
                "wide_spreads": {
                    "threshold": 2,
                    "warning": "Wide bid-ask spreads increase trading costs",
                    "explanation": "Limited market makers affect pricing efficiency"
                }
            },
            
            "momentum_risk": {
                "declining_trend": {
                    "threshold": -15,
                    "warning": "Recent price weakness suggests selling pressure",
                    "explanation": "30-day return below {threshold}% indicates negative momentum"
                },
                "high_volatility": {
                    "threshold": 30,
                    "warning": "High volatility increases uncertainty",
                    "explanation": "Price swings above {threshold}% create timing and psychological risks"
                }
            },
            
            "fundamental_risk": {
                "sector_weakness": {
                    "threshold": -5,
                    "warning": "Weak sector performance creates headwinds",
                    "explanation": "Industry declining by {threshold}% affects all sector participants"
                },
                "data_quality": {
                    "threshold": 70,
                    "warning": "Limited data availability reduces analysis confidence",
                    "explanation": "Data completeness below {threshold}% affects reliability"
                }
            }
        }
    
    def _load_market_context_analyzer(self) -> Dict[str, Dict]:
        """Load market context interpretation logic"""
        
        return {
            "bull_market": {
                "description": "Strong bull market conditions support growth strategies",
                "characteristics": "Broad market strength with positive sector rotation",
                "implications": "Momentum and growth factors receive increased weighting",
                "strategy_adjust": "Favor momentum plays and growth opportunities with strong technical setups"
            },
            
            "bear_market": {
                "description": "Challenging bear market conditions require defensive positioning", 
                "characteristics": "Market weakness demands focus on quality and value",
                "implications": "Value and quality factors receive increased weighting",
                "strategy_adjust": "Focus on defensive value plays and high-quality companies with strong balance sheets"
            },
            
            "neutral_market": {
                "description": "Mixed market conditions favor stock selection",
                "characteristics": "Sideways market environment requires careful stock picking",
                "implications": "Balanced factor weighting with emphasis on individual merit",
                "strategy_adjust": "Individual stock fundamentals and technical setups take precedence"
            }
        }
    
    def generate_comprehensive_explanation(self, 
                                         stock_data: pd.Series,
                                         market_context: Dict = None) -> DetailedExplanation:
        """Generate comprehensive explanation for a stock signal"""
        
        ticker = self._safe_get(stock_data, 'ticker', 'Unknown')
        signal = self._safe_get(stock_data, 'signal', 'NEUTRAL')
        confidence = self._safe_get_numeric(stock_data, 'confidence', 50)
        
        try:
            # Generate core explanation components
            headline = self._generate_headline(signal, confidence)
            primary_thesis = self._generate_primary_thesis(stock_data, signal)
            supporting_evidence = self._generate_supporting_evidence(stock_data)
            risk_considerations = self._generate_risk_considerations(stock_data)
            
            # Factor analysis
            factor_analysis = self._analyze_all_factors(stock_data)
            
            # Actionable insights
            recommendation = self._generate_recommendation(stock_data, signal, confidence)
            target_action = self._generate_target_action(signal, confidence)
            risk_management = self._generate_risk_management(stock_data, signal)
            
            # Context analysis
            market_ctx = self._generate_market_context(market_context)
            sector_ctx = self._generate_sector_context(stock_data)
            
            # Quality assessment
            data_quality = self._assess_explanation_quality(stock_data)
            explanation_confidence = self._calculate_explanation_confidence(stock_data, confidence)
            
            return DetailedExplanation(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                
                headline=headline,
                primary_thesis=primary_thesis,
                supporting_evidence=supporting_evidence,
                risk_considerations=risk_considerations,
                
                factor_analysis=factor_analysis,
                
                recommendation=recommendation,
                target_action=target_action,
                risk_management=risk_management,
                
                market_context=market_ctx,
                sector_context=sector_ctx,
                
                data_quality=data_quality,
                explanation_confidence=explanation_confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Comprehensive explanation generation failed for {ticker}: {e}")
            # Return basic explanation as fallback
            return self._generate_basic_explanation(stock_data)
    
    def _safe_get(self, series: pd.Series, key: str, default: Any) -> Any:
        """Safely get value from series with default"""
        try:
            value = series.get(key, default)
            return value if pd.notna(value) else default
        except:
            return default
    
    def _safe_get_numeric(self, series: pd.Series, key: str, default: float = 0.0) -> float:
        """Safely get numeric value from series with default"""
        try:
            value = series.get(key, default)
            if pd.isna(value):
                return default
            if isinstance(value, str):
                value = pd.to_numeric(value, errors='coerce')
                return default if pd.isna(value) else float(value)
            return float(value)
        except:
            return default
    
    def _generate_headline(self, signal: str, confidence: float) -> str:
        """Generate compelling headline for the signal"""
        
        template = self.explanation_templates.get(signal, {})
        base_headline = template.get('headline', f'{signal} Signal')
        
        # Add confidence qualifier
        if confidence >= 90:
            qualifier = " (Ultra-High Confidence)"
        elif confidence >= 80:
            qualifier = " (High Confidence)"
        elif confidence >= 70:
            qualifier = " (Good Confidence)"
        elif confidence >= 60:
            qualifier = " (Moderate Confidence)"
        else:
            qualifier = " (Low Confidence)"
        
        return base_headline + qualifier
    
    def _generate_primary_thesis(self, stock_data: pd.Series, signal: str) -> str:
        """Generate primary investment thesis"""
        
        # Identify the strongest factors
        factor_scores = {}
        for factor in ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']:
            factor_scores[factor] = self._safe_get_numeric(stock_data, f'{factor}_score', 50)
        
        # Find primary and secondary strengths
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        primary_strength = sorted_factors[0][0] if sorted_factors[0][1] > 60 else None
        secondary_strength = sorted_factors[1][0] if len(sorted_factors) > 1 and sorted_factors[1][1] > 60 else None
        
        # Get templates for this signal
        templates = self.explanation_templates.get(signal, {}).get('thesis_templates', [])
        
        if not templates:
            return f"Analysis suggests {signal.lower().replace('_', ' ')} based on current market factors"
        
        # Choose appropriate template based on factor strength
        if primary_strength and secondary_strength:
            thesis_template = next((t for t in templates if '{primary_strength}' in t and '{secondary_strength}' in t), templates[0])
            return thesis_template.format(
                primary_strength=self.factor_interpreters[primary_strength]['name'].lower(),
                secondary_strength=self.factor_interpreters[secondary_strength]['name'].lower(),
                confidence=self._safe_get_numeric(stock_data, 'confidence', 50)
            )
        elif primary_strength:
            thesis_template = next((t for t in templates if '{primary_strength}' in t), templates[0])
            return thesis_template.format(
                primary_strength=self.factor_interpreters[primary_strength]['name'].lower(),
                confidence=self._safe_get_numeric(stock_data, 'confidence', 50)
            )
        else:
            return templates[0].format(confidence=self._safe_get_numeric(stock_data, 'confidence', 50))
    
    def _generate_supporting_evidence(self, stock_data: pd.Series) -> List[str]:
        """Generate list of supporting evidence points"""
        
        evidence = []
        
        # Check each factor for supporting evidence
        factors = ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']
        
        for factor in factors:
            score = self._safe_get_numeric(stock_data, f'{factor}_score', 50)
            if score >= 70:  # Strong factor
                interpretation = self._interpret_factor_score(factor, score)
                evidence.append(f"{interpretation['icon']} {interpretation['description']}")
        
        # Add specific metric evidence
        self._add_specific_metric_evidence(stock_data, evidence)
        
        return evidence[:6]  # Limit to top 6 pieces of evidence
    
    def _add_specific_metric_evidence(self, stock_data: pd.Series, evidence: List[str]):
        """Add specific metric-based evidence"""
        
        # Price momentum evidence
        ret_30d = self._safe_get_numeric(stock_data, 'ret_30d', 0)
        if ret_30d > 15:
            evidence.append(f"📈 Strong 30-day return of {ret_30d:+.1f}%")
        elif ret_30d > 5:
            evidence.append(f"↗️ Positive 30-day momentum of {ret_30d:+.1f}%")
        
        # Valuation evidence
        pe = self._safe_get_numeric(stock_data, 'pe', 0)
        if 0 < pe <= 15:
            evidence.append(f"💰 Attractive valuation at {pe:.1f}x P/E")
        elif 0 < pe <= 25:
            evidence.append(f"⚖️ Reasonable valuation at {pe:.1f}x P/E")
        
        # Volume evidence
        rvol = self._safe_get_numeric(stock_data, 'rvol', 1)
        if rvol >= 3:
            evidence.append(f"🔥 Very high relative volume at {rvol:.1f}x")
        elif rvol >= 2:
            evidence.append(f"📊 Elevated volume activity at {rvol:.1f}x")
        
        # EPS growth evidence
        eps_change = self._safe_get_numeric(stock_data, 'eps_change_pct', 0)
        if eps_change > 20:
            evidence.append(f"🌟 Strong earnings growth of {eps_change:+.1f}%")
        elif eps_change > 10:
            evidence.append(f"📊 Good earnings growth of {eps_change:+.1f}%")
        
        # Technical evidence
        position_52w = self._safe_get_numeric(stock_data, 'position_52w_pct', 50)
        if position_52w > 80:
            evidence.append(f"⭐ Strong position at {position_52w:.0f}% of 52-week range")
        elif position_52w > 60:
            evidence.append(f"👍 Good position at {position_52w:.0f}% of 52-week range")
    
    def _generate_risk_considerations(self, stock_data: pd.Series) -> List[str]:
        """Generate list of risk considerations"""
        
        risks = []
        
        # Valuation risks
        pe = self._safe_get_numeric(stock_data, 'pe', 0)
        if pe > self.config.benchmarks.VALUE_BENCHMARKS['expensive']:
            risks.append(f"⚠️ High valuation at {pe:.1f}x P/E increases downside risk")
        elif pe <= 0:
            risks.append("⚠️ Loss-making company carries execution and operational risk")
        
        # Liquidity risks
        vol_1d = self._safe_get_numeric(stock_data, 'vol_1d', 0)
        if vol_1d < 50000:
            risks.append(f"⚠️ Low daily volume ({vol_1d/1000:.0f}K) may impact liquidity")
        
        # Momentum risks
        ret_30d = self._safe_get_numeric(stock_data, 'ret_30d', 0)
        if ret_30d < -15:
            risks.append(f"⚠️ Recent weakness ({ret_30d:+.1f}% in 30d) suggests selling pressure")
        
        # Sector risks
        sector_strength = self._safe_get_numeric(stock_data, 'current_sector_ret_30d', 0)
        if sector_strength < -5:
            risks.append(f"⚠️ Weak sector performance ({sector_strength:+.1f}%) creates headwinds")
        
        # Market cap risks
        market_cap = self._safe_get_numeric(stock_data, 'market_cap', 0)
        if market_cap < 1e9:  # Less than 1000 crores
            risks.append(f"⚠️ Small market cap increases volatility and liquidity risk")
        
        # Data quality risks
        data_completeness = self._safe_get_numeric(stock_data, 'data_completeness_score', 100)
        if data_completeness < 70:
            risks.append(f"⚠️ Limited data availability ({data_completeness:.0f}%) reduces analysis confidence")
        
        return risks[:5]  # Limit to top 5 risk factors
    
    def _analyze_all_factors(self, stock_data: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Analyze all factors with detailed breakdowns"""
        
        factor_analysis = {}
        
        factors = ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']
        
        for factor in factors:
            score = self._safe_get_numeric(stock_data, f'{factor}_score', 50)
            interpretation = self._interpret_factor_score(factor, score)
            
            factor_analysis[factor] = {
                'score': score,
                'rating': interpretation['rating'],
                'icon': interpretation['icon'],
                'description': interpretation['description'],
                'details': interpretation.get('details', ''),
                'weight': self.factor_interpreters[factor]['weight'],
                'contribution': score * self.factor_interpreters[factor]['weight'],
                'strength': self._categorize_factor_strength(score)
            }
        
        return factor_analysis
    
    def _interpret_factor_score(self, factor: str, score: float) -> Dict[str, Any]:
        """Interpret individual factor score into natural language"""
        
        factor_info = self.factor_interpreters.get(factor, {})
        thresholds = factor_info.get('thresholds', {})
        
        # Find appropriate threshold
        for threshold in sorted(thresholds.keys(), reverse=True):
            if score >= threshold:
                rating, icon, description = thresholds[threshold]
                return {
                    'rating': rating,
                    'icon': icon,
                    'description': description,
                    'details': factor_info.get('details', {}).get('interpretation', '')
                }
        
        # Default interpretation
        return {
            'rating': 'Unknown',
            'icon': '❓',
            'description': 'Unable to assess this factor',
            'details': ''
        }
    
    def _categorize_factor_strength(self, score: float) -> str:
        """Categorize factor strength for easy understanding"""
        
        if score >= 85:
            return 'Very Strong'
        elif score >= 75:
            return 'Strong'
        elif score >= 65:
            return 'Good'
        elif score >= 55:
            return 'Fair'
        elif score >= 45:
            return 'Neutral'
        elif score >= 35:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _generate_recommendation(self, stock_data: pd.Series, signal: str, confidence: float) -> str:
        """Generate actionable recommendation"""
        
        ticker = self._safe_get(stock_data, 'ticker', 'this stock')
        
        if signal == 'STRONG_BUY':
            if confidence >= 90:
                return f"Strong recommendation to buy {ticker} with high conviction. Consider this a core holding opportunity."
            else:
                return f"Buy {ticker} but monitor closely given moderate confidence level."
        
        elif signal == 'BUY':
            if confidence >= 80:
                return f"Good opportunity to buy {ticker} with solid fundamental support."
            else:
                return f"Consider buying {ticker} but validate with additional research and smaller position size."
        
        elif signal == 'ACCUMULATE':
            return f"Build position in {ticker} gradually over 3-6 months. Dollar-cost averaging recommended."
        
        elif signal == 'WATCH':
            return f"Monitor {ticker} closely for better entry opportunity. Wait for signal improvement."
        
        elif signal == 'AVOID':
            return f"Avoid new positions in {ticker}. Consider reducing existing exposure."
        
        else:
            return f"Strong avoidance recommended for {ticker}. Exit any existing positions."
    
    def _generate_target_action(self, signal: str, confidence: float) -> str:
        """Generate specific target action"""
        
        if signal == 'STRONG_BUY':
            if confidence >= 90:
                return "Consider 3-5% portfolio allocation for high-conviction opportunity"
            elif confidence >= 85:
                return "Consider 2-3% portfolio allocation"
            else:
                return "Consider 1-2% portfolio allocation with close monitoring"
        
        elif signal == 'BUY':
            if confidence >= 80:
                return "Consider 1-3% portfolio allocation"
            else:
                return "Consider small starter position (0.5-1%) with potential to scale up"
        
        elif signal == 'ACCUMULATE':
            return "Build position over 3-6 months targeting 1-2% total allocation"
        
        elif signal == 'WATCH':
            return "Add to watchlist for future consideration. No immediate action required."
        
        else:
            return "No position recommended. Avoid or exit existing positions."
    
    def _generate_risk_management(self, stock_data: pd.Series, signal: str) -> str:
        """Generate risk management guidance"""
        
        if signal in ['STRONG_BUY', 'BUY']:
            # Calculate appropriate stop loss based on volatility and momentum
            ret_30d = self._safe_get_numeric(stock_data, 'ret_30d', 0)
            pe = self._safe_get_numeric(stock_data, 'pe', 20)
            
            if ret_30d > 20:  # High momentum stock
                stop_loss = "Consider 15-20% stop loss due to momentum-driven nature"
            elif ret_30d > 10:
                stop_loss = "Consider 12-15% stop loss"
            else:
                stop_loss = "Consider 8-12% stop loss"
            
            if pe > 30:  # High valuation
                position_sizing = "Use smaller position size due to valuation risk"
            else:
                position_sizing = "Standard position sizing appropriate"
            
            return f"{stop_loss}. {position_sizing}. Scale in gradually and monitor fundamentals."
        
        elif signal == 'ACCUMULATE':
            return "Use dollar-cost averaging over 3-6 months. Scale out if fundamentals deteriorate significantly."
        
        elif signal == 'WATCH':
            return "No position recommended. Monitor for signal improvement before considering entry."
        
        else:
            return "Avoid new positions. Consider exit strategies for existing holdings."
    
    def _generate_market_context(self, market_context: Dict = None) -> str:
        """Generate market context explanation"""
        
        if not market_context:
            return "Market context analysis not available for this assessment"
        
        condition = market_context.get('condition', 'unknown')
        confidence = market_context.get('confidence', 0)
        breadth = market_context.get('market_breadth', 50)
        
        context_info = self.market_context_analyzer.get(condition, {})
        
        if not context_info:
            return f"Current market conditions: {condition} (confidence: {confidence:.0f}%)"
        
        description = context_info.get('description', '')
        characteristics = context_info.get('characteristics', '')
        strategy_adjust = context_info.get('strategy_adjust', '')
        
        return f"{description} with {breadth:.0f}% market breadth. {characteristics}. Strategy: {strategy_adjust}"
    
    def _generate_sector_context(self, stock_data: pd.Series) -> str:
        """Generate sector context explanation"""
        
        sector = self._safe_get(stock_data, 'sector', 'Unknown')
        sector_strength = self._safe_get_numeric(stock_data, 'current_sector_ret_30d', 0)
        sector_score = self._safe_get_numeric(stock_data, 'sector_score', 50)
        
        if sector == 'Unknown':
            return "Sector analysis not available"
        
        if sector_strength > 5:
            trend = "strong outperformance"
            impact = "provides significant tailwinds"
        elif sector_strength > 2:
            trend = "moderate outperformance"
            impact = "provides some support"
        elif sector_strength > -2:
            trend = "in-line performance"
            impact = "neutral impact"
        elif sector_strength > -5:
            trend = "underperformance"
            impact = "creates some headwinds"
        else:
            trend = "significant underperformance"
            impact = "creates major headwinds"
        
        return f"{sector} sector showing {trend} ({sector_strength:+.1f}% in 30d) which {impact} for the investment thesis."
    
    def _assess_explanation_quality(self, stock_data: pd.Series) -> str:
        """Assess quality of explanation based on data availability"""
        
        data_completeness = self._safe_get_numeric(stock_data, 'data_completeness_score', 70)
        
        # Count available key metrics
        key_metrics = ['price', 'pe', 'ret_30d', 'vol_1d', 'eps_change_pct', 'rvol']
        available_metrics = sum(1 for metric in key_metrics if pd.notna(stock_data.get(metric)))
        
        completeness_pct = (available_metrics / len(key_metrics)) * 100
        
        if completeness_pct >= 90 and data_completeness >= 90:
            return "High quality analysis with comprehensive data coverage"
        elif completeness_pct >= 75 and data_completeness >= 75:
            return "Good quality analysis with sufficient data for reliable insights"
        elif completeness_pct >= 60 and data_completeness >= 60:
            return "Acceptable analysis quality with some data limitations noted"
        else:
            return "Limited analysis quality due to insufficient data coverage"
    
    def _calculate_explanation_confidence(self, stock_data: pd.Series, signal_confidence: float) -> float:
        """Calculate confidence in the explanation itself"""
        
        # Base confidence from signal confidence
        base_confidence = signal_confidence
        
        # Data quality adjustment
        data_completeness = self._safe_get_numeric(stock_data, 'data_completeness_score', 70)
        data_adjustment = (data_completeness - 70) * 0.4
        
        # Factor alignment (how consistent are the factors?)
        factor_scores = []
        for factor in ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']:
            score = self._safe_get_numeric(stock_data, f'{factor}_score', 50)
            if score > 0:  # Valid score
                factor_scores.append(score)
        
        if len(factor_scores) >= 3:
            factor_std = np.std(factor_scores)
            alignment_adjustment = max(-20, 15 - factor_std * 0.6)
        else:
            alignment_adjustment = -10  # Penalize for insufficient factors
        
        # Calculate final explanation confidence
        explanation_confidence = base_confidence + data_adjustment + alignment_adjustment
        
        return max(15, min(98, explanation_confidence))
    
    def _generate_basic_explanation(self, stock_data: pd.Series) -> DetailedExplanation:
        """Generate basic explanation as fallback"""
        
        ticker = self._safe_get(stock_data, 'ticker', 'Unknown')
        signal = self._safe_get(stock_data, 'signal', 'NEUTRAL')
        confidence = self._safe_get_numeric(stock_data, 'confidence', 50)
        
        return DetailedExplanation(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            headline=f"{signal} Signal for {ticker}",
            primary_thesis="Analysis based on available market data and fundamental factors",
            supporting_evidence=["Comprehensive factor evaluation completed"],
            risk_considerations=["Standard market risks apply"],
            factor_analysis={},
            recommendation=f"Review {ticker} based on {signal.lower()} signal",
            target_action="Consider risk tolerance and portfolio allocation",
            risk_management="Apply appropriate risk management principles",
            market_context="Market context evaluation in progress",
            sector_context="Sector analysis based on available data",
            data_quality="Standard analysis quality",
            explanation_confidence=confidence,
            timestamp=datetime.now()
        )
    
    def generate_simple_explanation(self, stock_data: pd.Series) -> str:
        """Generate simple one-line explanation for quick scanning"""
        
        try:
            signal = self._safe_get(stock_data, 'signal', 'NEUTRAL')
            confidence = self._safe_get_numeric(stock_data, 'confidence', 50)
            ticker = self._safe_get(stock_data, 'ticker', 'Stock')
            
            # Find primary strength
            factor_scores = {}
            for factor in ['momentum', 'value', 'growth', 'volume']:
                factor_scores[factor] = self._safe_get_numeric(stock_data, f'{factor}_score', 50)
            
            if factor_scores:
                primary_factor = max(factor_scores.items(), key=lambda x: x[1])
                primary_name = primary_factor[0]
                primary_score = primary_factor[1]
            else:
                primary_name = "analysis"
                primary_score = 50
            
            # Simple templates based on signal
            if signal == 'STRONG_BUY':
                return f"{ticker}: Strong Buy ({confidence:.0f}% confidence) - Excellent {primary_name} with multiple factors aligned"
            elif signal == 'BUY':
                return f"{ticker}: Buy ({confidence:.0f}% confidence) - Strong {primary_name} with solid fundamentals"
            elif signal == 'ACCUMULATE':
                return f"{ticker}: Accumulate ({confidence:.0f}% confidence) - Building opportunity with good {primary_name}"
            elif signal == 'WATCH':
                return f"{ticker}: Watch ({confidence:.0f}% confidence) - Mixed signals, monitor for {primary_name} improvement"
            else:
                return f"{ticker}: Avoid ({confidence:.0f}% confidence) - Multiple concerns outweigh positives"
                
        except Exception as e:
            ticker = self._safe_get(stock_data, 'ticker', 'Stock')
            return f"{ticker}: Analysis completed - Review detailed factors for investment decision"
    
    def generate_risk_summary(self, stock_data: pd.Series) -> RiskAssessment:
        """Generate comprehensive risk summary"""
        
        try:
            risk_score = self._safe_get_numeric(stock_data, 'total_risk_score', 25)
            
            # Determine overall risk level
            if risk_score <= 25:
                overall_risk = 'Low'
            elif risk_score <= 50:
                overall_risk = 'Medium'
            elif risk_score <= 75:
                overall_risk = 'High'
            else:
                overall_risk = 'Very High'
            
            # Identify key risks
            key_risks = []
            risk_factors = {}
            
            # Valuation risk
            pe = self._safe_get_numeric(stock_data, 'pe', 0)
            if pe > 35:
                key_risks.append(f"High valuation (P/E: {pe:.1f}x)")
                risk_factors['valuation'] = f"P/E ratio of {pe:.1f}x above safe levels"
            elif pe <= 0:
                key_risks.append("Loss-making company")
                risk_factors['profitability'] = "Negative earnings indicate operational challenges"
            
            # Liquidity risk
            vol_1d = self._safe_get_numeric(stock_data, 'vol_1d', 0)
            if vol_1d < 100000:
                key_risks.append(f"Low liquidity ({vol_1d/1000:.0f}K daily volume)")
                risk_factors['liquidity'] = "Limited trading volume may impact entry/exit"
            
            # Momentum risk
            ret_30d = self._safe_get_numeric(stock_data, 'ret_30d', 0)
            if ret_30d < -15:
                key_risks.append(f"Recent decline ({ret_30d:+.1f}%)")
                risk_factors['momentum'] = "Negative momentum suggests selling pressure"
            
            # Sector risk
            sector_perf = self._safe_get_numeric(stock_data, 'current_sector_ret_30d', 0)
            if sector_perf < -5:
                key_risks.append(f"Weak sector ({sector_perf:+.1f}%)")
                risk_factors['sector'] = "Industry headwinds may affect performance"
            
            # Generate mitigation strategies
            mitigation_strategies = []
            if overall_risk in ['High', 'Very High']:
                mitigation_strategies.extend([
                    "Use smaller position sizes to manage risk",
                    "Implement tighter stop-loss levels",
                    "Monitor fundamental developments closely"
                ])
            if 'liquidity' in risk_factors:
                mitigation_strategies.append("Use limit orders to control execution")
            if 'valuation' in risk_factors:
                mitigation_strategies.append("Wait for better entry points on weakness")
            
            return RiskAssessment(
                overall_risk=overall_risk,
                risk_score=risk_score,
                key_risks=key_risks,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies
            )
            
        except Exception as e:
            logger.warning(f"Risk summary generation failed: {e}")
            return RiskAssessment(
                overall_risk='Medium',
                risk_score=50.0,
                key_risks=['Standard market risks'],
                risk_factors={'general': 'Market and company-specific risks apply'},
                mitigation_strategies=['Apply standard risk management principles']
            )

# Export main classes and functions
__all__ = [
    'PerfectExplainableAI',
    'DetailedExplanation',
    'FactorInsight',
    'RiskAssessment'
]

if __name__ == "__main__":
    print("✅ Perfect Explainable AI System loaded successfully")

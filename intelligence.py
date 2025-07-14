"""
intelligence.py - M.A.N.T.R.A. Version 3 FINAL Intelligence System
==================================================================
Explainable AI system for crystal-clear signal reasoning
Provides natural language explanations for every trading decision
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from config_final import *

logger = logging.getLogger(__name__)

@dataclass
class DetailedExplanation:
    """Comprehensive explanation container"""
    ticker: str
    signal: str
    confidence: float
    
    # Core explanation
    headline: str
    primary_thesis: str
    supporting_evidence: List[str]
    risk_considerations: List[str]
    
    # Factor breakdown
    factor_analysis: Dict[str, Dict]
    
    # Actionable insights
    recommendation: str
    target_action: str
    risk_management: str
    
    # Context
    market_context: str
    sector_context: str
    
    # Metadata
    data_quality: str
    explanation_confidence: float
    timestamp: datetime

class MANTRAIntelligence:
    """
    Advanced intelligence system for explainable trading signals
    
    Converts quantitative analysis into clear, actionable insights
    that any investor can understand and trust.
    """
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
        self.factor_interpreters = self._load_factor_interpreters()
        self.risk_assessors = self._load_risk_assessors()
        self.market_context_analyzer = self._load_market_context_analyzer()
        
        logger.info("🧠 MANTRA Intelligence System initialized")
    
    def _load_explanation_templates(self) -> Dict:
        """Load natural language explanation templates"""
        
        return {
            "STRONG_BUY": {
                "headline": "🚀 Exceptional Opportunity - Strong Buy Signal",
                "thesis_templates": [
                    "Multiple factors strongly aligned for significant upside potential",
                    "Outstanding {primary_strength} combined with {secondary_strength}",
                    "Rare combination of value and momentum creates compelling opportunity",
                    "All key indicators pointing to sustained outperformance"
                ],
                "confidence_language": {
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
                    "Good value opportunity with improving momentum",
                    "Strong {primary_strength} provides good upside potential",
                    "Multiple positive factors support purchase decision"
                ],
                "confidence_language": {
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
                    "Good long-term value with patience required",
                    "Building position makes sense given {primary_strength}",
                    "Steady accumulation recommended as story develops"
                ],
                "confidence_language": {
                    80: "Good confidence for accumulation",
                    75: "Reasonable confidence for building",
                    70: "Moderate confidence for gradual entry",
                    65: "Some confidence for patient accumulation"
                }
            },
            
            "WATCH": {
                "headline": "👀 Monitor Closely - Watch Signal", 
                "thesis_templates": [
                    "Mixed signals require careful monitoring",
                    "Potential developing but needs confirmation",
                    "Watching for {catalyst} to trigger entry",
                    "On the radar but not ready for commitment"
                ],
                "confidence_language": {
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
                    "Fundamental concerns make investment unattractive",
                    "Technical weakness suggests downside risk",
                    "Better opportunities available elsewhere"
                ],
                "confidence_language": {
                    80: "Clear avoidance recommended",
                    75: "Strong caution advised",
                    70: "Significant concerns present",
                    65: "Multiple warning signs"
                }
            }
        }
    
    def _load_factor_interpreters(self) -> Dict:
        """Load factor interpretation logic for natural language"""
        
        return {
            "momentum": {
                "name": "Price Momentum",
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
                    "interpretation": "Measures price performance across multiple timeframes"
                }
            },
            
            "value": {
                "name": "Valuation Attractiveness", 
                "thresholds": {
                    90: ("Exceptional value", "💎", "Trading at very attractive valuation"),
                    80: ("Strong value", "💰", "Good value at current prices"),
                    70: ("Fair value", "⚖️", "Reasonably valued"),
                    60: ("Modest value", "🤏", "Slight value opportunity"),
                    50: ("Neutral valuation", "😐", "Fair market pricing"),
                    40: ("Expensive", "💸", "Premium valuation"),
                    30: ("Very expensive", "🔥", "High valuation concern"),
                    20: ("Extremely expensive", "⚠️", "Significant overvaluation risk")
                },
                "details": {
                    "metrics": ["P/E ratio", "Earnings quality", "Market context"],
                    "interpretation": "Assesses if stock is attractively priced relative to fundamentals"
                }
            },
            
            "growth": {
                "name": "Earnings Growth",
                "thresholds": {
                    90: ("Exceptional growth", "🌟", "Outstanding earnings acceleration"),
                    80: ("Strong growth", "📊", "Solid earnings expansion"),
                    70: ("Good growth", "📈", "Healthy earnings trend"),
                    60: ("Modest growth", "➕", "Some earnings improvement"),
                    50: ("Stable earnings", "😐", "Flat earnings performance"),
                    40: ("Declining growth", "➖", "Slowing earnings growth"),
                    30: ("Poor growth", "📉", "Weakening earnings"),
                    20: ("Severe decline", "🔴", "Significant earnings deterioration")
                },
                "details": {
                    "metrics": ["EPS change %", "Growth quality", "Sustainability"],
                    "interpretation": "Evaluates earnings growth trends and quality"
                }
            },
            
            "volume": {
                "name": "Trading Interest",
                "thresholds": {
                    90: ("Exceptional interest", "🔥", "Very high trading activity"),
                    80: ("Strong interest", "📊", "Elevated trading volume"),
                    70: ("Good interest", "👥", "Above-average activity"),
                    60: ("Modest interest", "🤏", "Some increased activity"),
                    50: ("Normal interest", "😐", "Average trading levels"),
                    40: ("Weak interest", "👻", "Below-average volume"),
                    30: ("Poor interest", "😴", "Low trading activity"),
                    20: ("Very poor interest", "💀", "Minimal trading volume")
                },
                "details": {
                    "metrics": ["Relative volume", "Volume ratios", "Activity trends"],
                    "interpretation": "Measures market interest and liquidity"
                }
            },
            
            "technical": {
                "name": "Technical Setup",
                "thresholds": {
                    90: ("Excellent setup", "⭐", "Very strong technical position"),
                    80: ("Strong setup", "✅", "Good technical indicators"),
                    70: ("Positive setup", "👍", "Favorable technical picture"),
                    60: ("Neutral setup", "😐", "Mixed technical signals"),
                    50: ("Unclear setup", "🤷", "No clear technical direction"),
                    40: ("Weak setup", "👎", "Some technical concerns"),
                    30: ("Poor setup", "❌", "Negative technical indicators"),
                    20: ("Very poor setup", "🚫", "Significant technical weakness")
                },
                "details": {
                    "metrics": ["Moving averages", "52-week position", "Trend strength"],
                    "interpretation": "Analyzes chart patterns and technical indicators"
                }
            },
            
            "sector": {
                "name": "Sector Strength",
                "thresholds": {
                    90: ("Hot sector", "🔥", "Industry showing exceptional strength"),
                    80: ("Strong sector", "💪", "Industry performing well"),
                    70: ("Good sector", "👍", "Industry showing positive trends"),
                    60: ("Neutral sector", "😐", "Industry performing in-line"),
                    50: ("Mixed sector", "🤷", "Industry showing mixed signals"),
                    40: ("Weak sector", "👎", "Industry underperforming"),
                    30: ("Poor sector", "📉", "Industry showing weakness"),
                    20: ("Struggling sector", "⚠️", "Industry facing significant headwinds")
                },
                "details": {
                    "metrics": ["Sector returns", "Relative performance", "Rotation trends"],
                    "interpretation": "Evaluates industry group performance and trends"
                }
            }
        }
    
    def _load_risk_assessors(self) -> Dict:
        """Load risk assessment interpretation logic"""
        
        return {
            "valuation_risk": {
                "high_pe": {
                    "threshold": 35,
                    "warning": "High valuation multiple increases downside risk",
                    "explanation": "P/E ratio above {pe} suggests premium pricing"
                },
                "negative_earnings": {
                    "threshold": 0,
                    "warning": "Loss-making company carries execution risk", 
                    "explanation": "Negative earnings indicate business challenges"
                }
            },
            
            "liquidity_risk": {
                "low_volume": {
                    "threshold": 50000,
                    "warning": "Low trading volume may impact liquidity",
                    "explanation": "Daily volume below {volume} shares creates exit risk"
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
                    "explanation": "30-day return below {return}% indicates negative momentum"
                },
                "volatility": {
                    "threshold": 30,
                    "warning": "High volatility increases uncertainty",
                    "explanation": "Price swings above {volatility}% create timing risk"
                }
            },
            
            "fundamental_risk": {
                "sector_weakness": {
                    "threshold": -5,
                    "warning": "Weak sector performance creates headwinds",
                    "explanation": "Industry declining by {return}% affects all players"
                },
                "data_quality": {
                    "threshold": 70,
                    "warning": "Limited data availability reduces confidence",
                    "explanation": "Data completeness below {quality}% affects analysis reliability"
                }
            }
        }
    
    def _load_market_context_analyzer(self) -> Dict:
        """Load market context interpretation logic"""
        
        return {
            "bull_market": {
                "description": "Strong bull market conditions",
                "characteristics": "Broad market strength supports growth strategies",
                "implications": "Momentum and growth factors receive higher weighting",
                "strategy_adjust": "Favor momentum plays and growth opportunities"
            },
            
            "bear_market": {
                "description": "Challenging bear market conditions", 
                "characteristics": "Market weakness requires defensive positioning",
                "implications": "Value and quality factors receive higher weighting",
                "strategy_adjust": "Focus on defensive value plays and quality companies"
            },
            
            "neutral_market": {
                "description": "Mixed market conditions",
                "characteristics": "Sideways market requires stock selection",
                "implications": "Balanced factor weighting for stock picking",
                "strategy_adjust": "Individual stock merit takes precedence"
            }
        }
    
    def generate_comprehensive_explanation(self, 
                                         stock_data: pd.Series,
                                         market_context: Dict = None) -> DetailedExplanation:
        """Generate comprehensive explanation for a stock signal"""
        
        ticker = stock_data.get('ticker', 'Unknown')
        signal = stock_data.get('signal', 'NEUTRAL')
        confidence = stock_data.get('confidence', 50)
        
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
    
    def _generate_headline(self, signal: str, confidence: float) -> str:
        """Generate compelling headline for the signal"""
        
        template = self.explanation_templates.get(signal, {})
        base_headline = template.get('headline', f'{signal} Signal')
        
        # Add confidence qualifier
        if confidence >= 90:
            qualifier = " (Very High Confidence)"
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
        factor_scores = {
            'momentum': stock_data.get('momentum_score', 50),
            'value': stock_data.get('value_score', 50),
            'growth': stock_data.get('growth_score', 50),
            'volume': stock_data.get('volume_score', 50),
            'technical': stock_data.get('technical_score', 50),
            'sector': stock_data.get('sector_score', 50)
        }
        
        # Find primary and secondary strengths
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        primary_strength = sorted_factors[0][0] if sorted_factors[0][1] > 60 else None
        secondary_strength = sorted_factors[1][0] if len(sorted_factors) > 1 and sorted_factors[1][1] > 60 else None
        
        # Get templates for this signal
        templates = self.explanation_templates.get(signal, {}).get('thesis_templates', [])
        
        if not templates:
            return f"Analysis suggests {signal.lower().replace('_', ' ')} based on current factors"
        
        # Choose appropriate template based on factor strength
        if primary_strength and secondary_strength:
            thesis_template = next((t for t in templates if '{primary_strength}' in t and '{secondary_strength}' in t), templates[0])
            return thesis_template.format(
                primary_strength=self.factor_interpreters[primary_strength]['name'].lower(),
                secondary_strength=self.factor_interpreters[secondary_strength]['name'].lower()
            )
        elif primary_strength:
            thesis_template = next((t for t in templates if '{primary_strength}' in t), templates[0])
            return thesis_template.format(
                primary_strength=self.factor_interpreters[primary_strength]['name'].lower()
            )
        else:
            return templates[0]
    
    def _generate_supporting_evidence(self, stock_data: pd.Series) -> List[str]:
        """Generate list of supporting evidence points"""
        
        evidence = []
        
        # Check each factor for supporting evidence
        factors = ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']
        
        for factor in factors:
            score = stock_data.get(f'{factor}_score', 50)
            if score >= 70:  # Strong factor
                interpretation = self._interpret_factor_score(factor, score)
                evidence.append(f"{interpretation['icon']} {interpretation['description']}")
        
        # Add specific metric evidence
        self._add_specific_metric_evidence(stock_data, evidence)
        
        return evidence[:5]  # Limit to top 5 pieces of evidence
    
    def _add_specific_metric_evidence(self, stock_data: pd.Series, evidence: List[str]):
        """Add specific metric-based evidence"""
        
        # Price momentum evidence
        ret_30d = stock_data.get('ret_30d', 0)
        if ret_30d > 15:
            evidence.append(f"📈 Strong 30-day return of {ret_30d:+.1f}%")
        elif ret_30d > 5:
            evidence.append(f"↗️ Positive 30-day return of {ret_30d:+.1f}%")
        
        # Valuation evidence
        pe = stock_data.get('pe', 0)
        if 0 < pe <= 15:
            evidence.append(f"💰 Attractive valuation at {pe:.1f}x P/E")
        elif 0 < pe <= 25:
            evidence.append(f"⚖️ Reasonable valuation at {pe:.1f}x P/E")
        
        # Volume evidence
        rvol = stock_data.get('rvol', 1)
        if rvol >= 3:
            evidence.append(f"🔥 Very high relative volume at {rvol:.1f}x")
        elif rvol >= 2:
            evidence.append(f"📊 Elevated volume activity at {rvol:.1f}x")
        
        # EPS growth evidence
        eps_change = stock_data.get('eps_change_pct', 0)
        if eps_change > 20:
            evidence.append(f"🌟 Strong earnings growth of {eps_change:+.1f}%")
        elif eps_change > 10:
            evidence.append(f"📊 Good earnings growth of {eps_change:+.1f}%")
        
        # Technical evidence
        from_low_pct = stock_data.get('from_low_pct', 50)
        if from_low_pct > 80:
            evidence.append(f"⭐ Strong position at {from_low_pct:.0f}% of 52-week range")
        elif from_low_pct > 60:
            evidence.append(f"👍 Good position at {from_low_pct:.0f}% of 52-week range")
    
    def _generate_risk_considerations(self, stock_data: pd.Series) -> List[str]:
        """Generate list of risk considerations"""
        
        risks = []
        
        # Valuation risks
        pe = stock_data.get('pe', 0)
        if pe > 35:
            risks.append(f"⚠️ High valuation at {pe:.1f}x P/E increases downside risk")
        elif pe <= 0:
            risks.append("⚠️ Loss-making company carries execution risk")
        
        # Liquidity risks
        vol_1d = stock_data.get('vol_1d', 0)
        if vol_1d < 50000:
            risks.append(f"⚠️ Low daily volume ({vol_1d/1000:.0f}K) may impact liquidity")
        
        # Momentum risks
        ret_30d = stock_data.get('ret_30d', 0)
        if ret_30d < -15:
            risks.append(f"⚠️ Recent weakness ({ret_30d:+.1f}% in 30d) suggests selling pressure")
        
        # Sector risks
        sector_strength = stock_data.get('current_sector_strength', 0)
        if sector_strength < -5:
            risks.append(f"⚠️ Weak sector performance ({sector_strength:+.1f}%) creates headwinds")
        
        # Market cap risks
        mcap_category = stock_data.get('mcap_category_detailed', 'Unknown')
        if mcap_category in ['Nano', 'Small']:
            risks.append(f"⚠️ {mcap_category} cap stock carries higher volatility risk")
        
        # Data quality risks
        data_completeness = stock_data.get('data_completeness_score', 100)
        if data_completeness < 70:
            risks.append(f"⚠️ Limited data availability ({data_completeness:.0f}%) reduces confidence")
        
        return risks[:4]  # Limit to top 4 risk factors
    
    def _analyze_all_factors(self, stock_data: pd.Series) -> Dict[str, Dict]:
        """Analyze all factors with detailed breakdowns"""
        
        factor_analysis = {}
        
        factors = ['momentum', 'value', 'growth', 'volume', 'technical', 'sector']
        
        for factor in factors:
            score = stock_data.get(f'{factor}_score', 50)
            interpretation = self._interpret_factor_score(factor, score)
            
            factor_analysis[factor] = {
                'score': score,
                'rating': interpretation['rating'],
                'icon': interpretation['icon'],
                'description': interpretation['description'],
                'details': interpretation.get('details', ''),
                'strength': self._categorize_factor_strength(score)
            }
        
        return factor_analysis
    
    def _interpret_factor_score(self, factor: str, score: float) -> Dict:
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
        
        if score >= 80:
            return 'Very Strong'
        elif score >= 70:
            return 'Strong'
        elif score >= 60:
            return 'Good'
        elif score >= 50:
            return 'Neutral'
        elif score >= 40:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _generate_recommendation(self, stock_data: pd.Series, signal: str, confidence: float) -> str:
        """Generate actionable recommendation"""
        
        ticker = stock_data.get('ticker', 'this stock')
        
        if signal == 'STRONG_BUY':
            if confidence >= 85:
                return f"Strong recommendation to buy {ticker} with high conviction"
            else:
                return f"Buy {ticker} but monitor closely due to moderate confidence"
        
        elif signal == 'BUY':
            if confidence >= 80:
                return f"Good opportunity to buy {ticker} with solid fundamentals"
            else:
                return f"Consider buying {ticker} but validate with additional research"
        
        elif signal == 'ACCUMULATE':
            return f"Build position in {ticker} gradually over time"
        
        elif signal == 'WATCH':
            return f"Monitor {ticker} closely for better entry opportunity"
        
        elif signal == 'AVOID':
            return f"Avoid or consider reducing exposure to {ticker}"
        
        else:
            return f"No clear action recommended for {ticker} at this time"
    
    def _generate_target_action(self, signal: str, confidence: float) -> str:
        """Generate specific target action"""
        
        if signal in ['STRONG_BUY', 'BUY']:
            if confidence >= 85:
                return "Consider 2-3% portfolio allocation"
            elif confidence >= 75:
                return "Consider 1-2% portfolio allocation"
            else:
                return "Consider small starter position (0.5-1%)"
        
        elif signal == 'ACCUMULATE':
            return "Build position over 3-6 months (0.5-2% total)"
        
        elif signal == 'WATCH':
            return "Add to watchlist for future consideration"
        
        else:
            return "No position recommended"
    
    def _generate_risk_management(self, stock_data: pd.Series, signal: str) -> str:
        """Generate risk management guidance"""
        
        if signal in ['STRONG_BUY', 'BUY']:
            # Calculate stop loss based on volatility and support levels
            ret_30d = stock_data.get('ret_30d', 0)
            price = stock_data.get('price', 0)
            
            if ret_30d > 20:  # High momentum stock
                stop_loss = "Consider 15-20% stop loss due to momentum nature"
            elif ret_30d > 10:
                stop_loss = "Consider 10-15% stop loss"
            else:
                stop_loss = "Consider 8-12% stop loss"
            
            position_sizing = "Start with smaller position and add on strength"
            
            return f"{stop_loss}. {position_sizing}."
        
        elif signal == 'ACCUMULATE':
            return "Use dollar-cost averaging. Scale out if fundamentals deteriorate."
        
        elif signal == 'WATCH':
            return "No position recommended. Monitor for signal improvement."
        
        else:
            return "Avoid new positions. Consider exit if currently held."
    
    def _generate_market_context(self, market_context: Dict = None) -> str:
        """Generate market context explanation"""
        
        if not market_context:
            return "Market context analysis not available"
        
        condition = market_context.get('condition', 'unknown')
        confidence = market_context.get('confidence', 0)
        breadth = market_context.get('market_breadth', 50)
        
        context_info = self.market_context_analyzer.get(condition, {})
        
        if not context_info:
            return f"Current market conditions: {condition} (confidence: {confidence:.0f}%)"
        
        description = context_info.get('description', '')
        characteristics = context_info.get('characteristics', '')
        strategy_adjust = context_info.get('strategy_adjust', '')
        
        return f"{description} with {breadth:.0f}% market breadth. {characteristics}. {strategy_adjust}."
    
    def _generate_sector_context(self, stock_data: pd.Series) -> str:
        """Generate sector context explanation"""
        
        sector = stock_data.get('sector', 'Unknown')
        sector_strength = stock_data.get('current_sector_strength', 0)
        sector_score = stock_data.get('sector_score', 50)
        
        if sector == 'Unknown':
            return "Sector analysis not available"
        
        if sector_strength > 5:
            trend = "strong outperformance"
        elif sector_strength > 2:
            trend = "moderate outperformance"
        elif sector_strength > -2:
            trend = "in-line performance"
        elif sector_strength > -5:
            trend = "underperformance"
        else:
            trend = "significant underperformance"
        
        return f"{sector} sector showing {trend} ({sector_strength:+.1f}%). This {'supports' if sector_strength > 0 else 'challenges'} the investment thesis."
    
    def _assess_explanation_quality(self, stock_data: pd.Series) -> str:
        """Assess quality of explanation based on data availability"""
        
        data_completeness = stock_data.get('data_completeness_score', 70)
        
        # Count available key metrics
        key_metrics = ['price', 'pe', 'ret_30d', 'vol_1d', 'eps_change_pct', 'rvol']
        available_metrics = sum(1 for metric in key_metrics if pd.notna(stock_data.get(metric, np.nan)))
        
        completeness_pct = (available_metrics / len(key_metrics)) * 100
        
        if completeness_pct >= 90 and data_completeness >= 90:
            return "High quality analysis with comprehensive data"
        elif completeness_pct >= 75 and data_completeness >= 75:
            return "Good quality analysis with sufficient data"
        elif completeness_pct >= 60 and data_completeness >= 60:
            return "Acceptable analysis with some data limitations"
        else:
            return "Limited analysis due to insufficient data"
    
    def _calculate_explanation_confidence(self, stock_data: pd.Series, signal_confidence: float) -> float:
        """Calculate confidence in the explanation itself"""
        
        # Base confidence from signal confidence
        base_confidence = signal_confidence
        
        # Data quality adjustment
        data_completeness = stock_data.get('data_completeness_score', 70)
        data_adjustment = (data_completeness - 70) * 0.3
        
        # Factor alignment (how consistent are the factors?)
        factor_scores = [
            stock_data.get('momentum_score', 50),
            stock_data.get('value_score', 50),
            stock_data.get('growth_score', 50),
            stock_data.get('volume_score', 50),
            stock_data.get('technical_score', 50),
            stock_data.get('sector_score', 50)
        ]
        
        factor_std = np.std([s for s in factor_scores if pd.notna(s)])
        alignment_adjustment = max(-15, 10 - factor_std * 0.5)
        
        # Calculate final explanation confidence
        explanation_confidence = base_confidence + data_adjustment + alignment_adjustment
        
        return max(10, min(95, explanation_confidence))
    
    def generate_simple_explanation(self, stock_data: pd.Series) -> str:
        """Generate simple one-line explanation for quick scanning"""
        
        signal = stock_data.get('signal', 'NEUTRAL')
        confidence = stock_data.get('confidence', 50)
        ticker = stock_data.get('ticker', 'Stock')
        
        # Find primary strength
        factor_scores = {
            'momentum': stock_data.get('momentum_score', 50),
            'value': stock_data.get('value_score', 50),
            'growth': stock_data.get('growth_score', 50),
            'volume': stock_data.get('volume_score', 50)
        }
        
        primary_factor = max(factor_scores.items(), key=lambda x: x[1])
        primary_name = primary_factor[0]
        primary_score = primary_factor[1]
        
        # Simple templates
        if signal == 'STRONG_BUY':
            return f"{ticker}: Strong Buy ({confidence:.0f}% confidence) - Excellent {primary_name} + multiple factors aligned"
        elif signal == 'BUY':
            return f"{ticker}: Buy ({confidence:.0f}% confidence) - Strong {primary_name} with good fundamentals"
        elif signal == 'ACCUMULATE':
            return f"{ticker}: Accumulate ({confidence:.0f}% confidence) - Building opportunity, good {primary_name}"
        elif signal == 'WATCH':
            return f"{ticker}: Watch ({confidence:.0f}% confidence) - Mixed signals, monitor for clarity"
        else:
            return f"{ticker}: Avoid ({confidence:.0f}% confidence) - Multiple concerns outweigh positives"
    
    def generate_risk_summary(self, stock_data: pd.Series) -> Dict[str, str]:
        """Generate comprehensive risk summary"""
        
        risk_summary = {
            'overall_risk': 'Medium',
            'key_risks': [],
            'risk_score': stock_data.get('total_risk_score', 50),
            'risk_factors': {}
        }
        
        total_risk = stock_data.get('total_risk_score', 50)
        
        # Overall risk categorization
        if total_risk <= 25:
            risk_summary['overall_risk'] = 'Low'
        elif total_risk <= 50:
            risk_summary['overall_risk'] = 'Medium'
        elif total_risk <= 75:
            risk_summary['overall_risk'] = 'High'
        else:
            risk_summary['overall_risk'] = 'Very High'
        
        # Identify specific risk factors
        pe = stock_data.get('pe', 0)
        if pe > 35:
            risk_summary['risk_factors']['valuation'] = f"High P/E of {pe:.1f}x"
        
        vol_1d = stock_data.get('vol_1d', 0)
        if vol_1d < 100000:
            risk_summary['risk_factors']['liquidity'] = f"Low volume ({vol_1d/1000:.0f}K daily)"
        
        ret_30d = stock_data.get('ret_30d', 0)
        if ret_30d < -15:
            risk_summary['risk_factors']['momentum'] = f"Recent decline ({ret_30d:+.1f}%)"
        
        mcap = stock_data.get('mcap_category_detailed', 'Unknown')
        if mcap in ['Nano', 'Small']:
            risk_summary['risk_factors']['size'] = f"{mcap} cap volatility"
        
        return risk_summary
    
    def format_explanation_for_display(self, explanation: DetailedExplanation) -> Dict[str, str]:
        """Format explanation for clean UI display"""
        
        return {
            'header': explanation.headline,
            'thesis': explanation.primary_thesis,
            'confidence': f"{explanation.confidence:.0f}% confidence",
            'recommendation': explanation.recommendation,
            'action': explanation.target_action,
            'risks': ' • '.join(explanation.risk_considerations),
            'evidence': ' • '.join(explanation.supporting_evidence),
            'context': explanation.market_context,
            'quality': explanation.data_quality
        }
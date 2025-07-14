"""
intelligence.py - M.A.N.T.R.A. Explainable AI
============================================
Simple explanations for signals
"""

import pandas as pd
from typing import Dict, List, Optional

class PerfectExplainableAI:
    """Simple explainable AI"""
    
    def __init__(self):
        pass
    
    def generate_explanation(self, stock_data: pd.Series) -> str:
        """Generate simple explanation"""
        signal = stock_data.get('signal', 'NEUTRAL')
        confidence = stock_data.get('confidence', 50)
        
        if signal == 'STRONG_BUY':
            return f"Ultra-high confidence ({confidence:.0f}%) with multiple factors aligned"
        elif signal == 'BUY':
            return f"High confidence ({confidence:.0f}%) opportunity"
        else:
            return f"{signal} signal with {confidence:.0f}% confidence"

__all__ = ['PerfectExplainableAI']

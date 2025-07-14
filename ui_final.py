"""
ui_final.py - M.A.N.T.R.A. UI Components
========================================
Simple but beautiful UI components
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional

class PerfectUIComponents:
    """Simple UI components"""
    
    def __init__(self):
        pass
    
    def render_header(self):
        """Render header"""
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🔱 M.A.N.T.R.A. Version 3 Final</h1>
            <p>Market Analysis Neural Trading Research Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    def metric_card(self, label: str, value: str, delta: str = ""):
        """Display metric card"""
        st.metric(label, value, delta)
    
    def success_message(self, message: str):
        """Display success message"""
        st.success(message)
    
    def error_message(self, message: str):
        """Display error message"""
        st.error(message)
    
    def info_message(self, message: str):
        """Display info message"""
        st.info(message)

__all__ = ['PerfectUIComponents']

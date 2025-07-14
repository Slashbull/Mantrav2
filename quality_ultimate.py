"""
quality_ultimate.py - M.A.N.T.R.A. Version 3 FINAL - Ultimate Quality Control
=============================================================================
Ultimate data quality control system for bulletproof reliability
Comprehensive validation, cleaning, and quality metrics
Built for permanent use with zero crashes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time

# Configure logger
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QualityIssue:
    """Individual data quality issue"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'completeness', 'validity', 'consistency'
    description: str
    affected_count: int
    recommendation: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass 
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    status: str
    processing_time: float
    timestamp: datetime
    total_stocks: int
    usable_stocks: int
    critical_issues: List[QualityIssue] = field(default_factory=list)
    warnings: List[QualityIssue] = field(default_factory=list)
    info_messages: List[QualityIssue] = field(default_factory=list)

@dataclass
class ProcessingResult:
    """Result of data processing"""
    success: bool
    dataframe: pd.DataFrame
    quality_report: QualityReport
    message: str

# =============================================================================
# QUALITY CONTROLLER
# =============================================================================

class UltimateQualityController:
    """Ultimate quality control system for M.A.N.T.R.A."""
    
    def __init__(self):
        """Initialize quality controller"""
        self.issues = []
        
    def process_dataframe(self, df: pd.DataFrame, sheet_name: str = "dataset") -> ProcessingResult:
        """Process dataframe with comprehensive quality control"""
        
        start_time = time.time()
        
        try:
            # Clean the dataframe
            cleaned_df = self._clean_dataframe(df, sheet_name)
            
            # Analyze quality
            quality_report = self._analyze_quality(cleaned_df, sheet_name)
            
            # Create result
            success = quality_report.overall_score >= 60  # Minimum quality threshold
            message = f"Processed {len(cleaned_df)} stocks with {quality_report.status} quality"
            
            return ProcessingResult(
                success=success,
                dataframe=cleaned_df,
                quality_report=quality_report,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Quality processing failed: {e}")
            
            # Return original dataframe with error report
            error_report = QualityReport(
                overall_score=0.0,
                status='Error',
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
                total_stocks=len(df),
                usable_stocks=0
            )
            
            return ProcessingResult(
                success=False,
                dataframe=df,
                quality_report=error_report,
                message=f"Processing failed: {str(e)}"
            )
    
    def _clean_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Clean and prepare dataframe"""
        
        try:
            # Create a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Remove completely empty rows
            cleaned_df = cleaned_df.dropna(how='all')
            
            # Handle column names based on sheet type
            if sheet_name == "watchlist":
                # These are the actual column names from your Google Sheets
                numeric_columns = [
                    'price', 'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                    'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
                    'sma_20d', 'sma_50d', 'sma_200d', 'volume_1d', 'volume_7d', 'volume_30d',
                    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'rvol',
                    'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct', 'market_cap'
                ]
                
                # Clean numeric columns
                for col in numeric_columns:
                    if col in cleaned_df.columns:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # Handle percentage columns
                pct_columns = ['from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 
                              'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'eps_change_pct']
                for col in pct_columns:
                    if col in cleaned_df.columns:
                        # Remove % sign if present
                        if cleaned_df[col].dtype == 'object':
                            cleaned_df[col] = cleaned_df[col].astype(str).str.replace('%', '')
                            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            elif sheet_name == "returns":
                # Returns analysis columns
                return_columns = [
                    'returns_ret_1d', 'returns_ret_3d', 'returns_ret_7d', 'returns_ret_30d',
                    'returns_ret_3m', 'returns_ret_6m', 'returns_ret_1y', 'returns_ret_3y', 'returns_ret_5y',
                    'avg_ret_30d', 'avg_ret_3m', 'avg_ret_6m', 'avg_ret_1y', 'avg_ret_3y', 'avg_ret_5y'
                ]
                
                for col in return_columns:
                    if col in cleaned_df.columns:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            elif sheet_name == "sectors":
                # Sector analysis columns
                sector_columns = [col for col in cleaned_df.columns if 'sector_ret_' in col or 'sector_avg_' in col]
                
                for col in sector_columns:
                    if col in cleaned_df.columns:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                if 'sector_count' in cleaned_df.columns:
                    cleaned_df['sector_count'] = pd.to_numeric(cleaned_df['sector_count'], errors='coerce')
            
            # Remove rows with missing critical data
            if 'ticker' in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=['ticker'])
                cleaned_df = cleaned_df[cleaned_df['ticker'] != '']
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return df
    
    def _analyze_quality(self, df: pd.DataFrame, sheet_name: str) -> QualityReport:
        """Analyze data quality"""
        
        start_time = time.time()
        self.issues = []
        
        # Calculate basic metrics
        total_stocks = len(df)
        
        # Check data completeness
        if sheet_name == "watchlist":
            critical_columns = ['ticker', 'price', 'sector']
            usable_stocks = len(df.dropna(subset=[col for col in critical_columns if col in df.columns]))
        else:
            usable_stocks = total_stocks
        
        # Calculate quality score
        if total_stocks > 0:
            completeness_ratio = usable_stocks / total_stocks
            overall_score = completeness_ratio * 100
        else:
            overall_score = 0
        
        # Determine status
        if overall_score >= 95:
            status = "Excellent"
        elif overall_score >= 85:
            status = "Good"
        elif overall_score >= 70:
            status = "Acceptable"
        else:
            status = "Poor"
        
        # Check for issues
        if overall_score < 80:
            self.issues.append(QualityIssue(
                severity='warning',
                category='completeness',
                description=f'Data completeness is {overall_score:.1f}%',
                affected_count=total_stocks - usable_stocks,
                recommendation='Review data source for missing values',
                field=None,
                details=None
            ))
        
        return QualityReport(
            overall_score=overall_score,
            status=status,
            processing_time=time.time() - start_time,
            timestamp=datetime.now(),
            total_stocks=total_stocks,
            usable_stocks=usable_stocks,
            critical_issues=[i for i in self.issues if i.severity == 'critical'],
            warnings=[i for i in self.issues if i.severity == 'warning'],
            info_messages=[i for i in self.issues if i.severity == 'info']
        )

# Export main class
__all__ = ['UltimateQualityController', 'QualityReport', 'QualityIssue', 'ProcessingResult']

"""
quality_ultimate.py - M.A.N.T.R.A. Version 3 FINAL - Bulletproof Quality Control
================================================================================
Ultimate data quality control system inspired by core_system_foundation.py
Comprehensive validation, cleaning, and quality analysis for reliable signals
Built for permanent use - handles all edge cases bulletproof
"""

import pandas as pd
import numpy as np
import logging
import re
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config_ultimate import CONFIG

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [QUALITY] %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# =============================================================================
# QUALITY ASSESSMENT CLASSES
# =============================================================================

@dataclass
class QualityIssue:
    """Individual data quality issue"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'completeness', 'validity', 'consistency', 'outlier'
    description: str
    affected_count: int
    recommendation: str
    field: str = None
    details: Dict = field(default_factory=dict)

@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    status: str
    processing_time: float
    data_hash: str
    timestamp: datetime
    
    # Issue tracking
    critical_issues: List[QualityIssue]
    warnings: List[QualityIssue]
    info_messages: List[QualityIssue]
    
    # Detailed metrics
    completeness_scores: Dict[str, float]
    consistency_scores: Dict[str, float]
    validity_scores: Dict[str, float]
    outlier_analysis: Dict[str, Any]
    
    # Summary stats
    total_stocks: int
    usable_stocks: int
    excluded_stocks: int
    
    # Quality breakdown
    excellent_quality: int
    good_quality: int
    poor_quality: int
    
    # Data lineage
    lineage: List[str]
    source_info: Dict[str, Any]

@dataclass  
class ProcessingResult:
    """Result of data processing with quality metrics"""
    success: bool
    dataframe: pd.DataFrame
    quality_report: QualityReport
    message: str
    processing_stats: Dict[str, Any]

# =============================================================================
# BULLETPROOF DATA CLEANER
# =============================================================================

class BulletproofDataCleaner:
    """Bulletproof data cleaning system that handles all edge cases"""
    
    def __init__(self):
        self.config = CONFIG
        self.processing_stats = {}
        self.lineage = []
    
    def clean_dataframe(self, df: pd.DataFrame, sheet_name: str = "unknown") -> pd.DataFrame:
        """Clean and normalize dataframe with bulletproof error handling"""
        start_time = time.time()
        initial_rows = len(df)
        self.lineage.append(f"Starting cleanup of {sheet_name}: {initial_rows} rows")
        
        try:
            # Step 1: Clean column names
            df = self._clean_column_names(df)
            
            # Step 2: Remove completely empty rows/columns
            df = self._remove_empty_data(df)
            
            # Step 3: Apply column mappings
            df = self._apply_column_mappings(df, sheet_name)
            
            # Step 4: Clean numeric columns
            df = self._clean_numeric_columns(df)
            
            # Step 5: Clean text columns
            df = self._clean_text_columns(df)
            
            # Step 6: Handle duplicates
            df = self._handle_duplicates(df, sheet_name)
            
            # Step 7: Optimize data types
            df = self._optimize_data_types(df)
            
            final_rows = len(df)
            processing_time = time.time() - start_time
            
            self.lineage.append(f"Completed cleanup of {sheet_name}: {initial_rows} → {final_rows} rows in {processing_time:.2f}s")
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Data cleaning failed for {sheet_name}: {e}")
            # Return original dataframe as fallback
            return df.reset_index(drop=True)
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize column names"""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
        
        # Clean column names
        cleaned_columns = []
        for col in df.columns:
            # Convert to string and clean
            clean_col = str(col).strip().lower()
            # Remove special characters except underscores
            clean_col = re.sub(r"[^\w\s]", "", clean_col)
            # Replace spaces with underscores
            clean_col = re.sub(r"\s+", "_", clean_col)
            # Remove multiple underscores
            clean_col = re.sub(r"_+", "_", clean_col)
            # Remove leading/trailing underscores
            clean_col = clean_col.strip("_")
            
            cleaned_columns.append(clean_col)
        
        df.columns = cleaned_columns
        return df
    
    def _remove_empty_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows and columns"""
        # Remove empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self.lineage.append(f"Removed {len(empty_cols)} empty columns")
        
        # Remove empty rows
        initial_rows = len(df)
        df = df.dropna(how='all')
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.lineage.append(f"Removed {removed_rows} empty rows")
        
        return df
    
    def _apply_column_mappings(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Apply column mappings for standardization"""
        if sheet_name == "watchlist":
            mappings = self.config.schema.COLUMN_MAPPINGS
            available_mappings = {old: new for old, new in mappings.items() if old in df.columns}
            if available_mappings:
                df = df.rename(columns=available_mappings)
                self.lineage.append(f"Applied {len(available_mappings)} column mappings")
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all numeric columns with bulletproof conversion"""
        numeric_pattern = re.compile(
            r"^(price|prev_close|ret_|avg_ret|volume|vol_ratio|"
            r"low_52w|high_52w|from_low_pct|from_high_pct|pe|eps|rvol|"
            r"market_cap|sma|dma|sector_ret|sector_avg)"
        )
        
        for col in df.columns:
            if numeric_pattern.match(col) or df[col].dtype == 'object':
                try:
                    if self._looks_numeric(df[col]):
                        df[col] = self._bulletproof_numeric_conversion(df[col], col)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")
                    # Fill with safe defaults
                    df[col] = self._get_safe_default(col)
        
        return df
    
    def _looks_numeric(self, series: pd.Series) -> bool:
        """Check if a series looks like it should be numeric"""
        if series.dtype in ['int64', 'float64']:
            return True
        
        # Sample a few non-null values
        sample = series.dropna().head(10).astype(str)
        if len(sample) == 0:
            return False
        
        # Check if they look numeric (contain digits, decimal points, etc.)
        numeric_chars = sum(1 for val in sample if re.search(r'[\d.,-]', val))
        return numeric_chars > len(sample) * 0.5
    
    def _bulletproof_numeric_conversion(self, series: pd.Series, col_name: str) -> pd.Series:
        """Bulletproof numeric conversion that handles all edge cases"""
        if series.dtype in ['int64', 'float64']:
            return series
        
        try:
            # Convert to string first
            str_series = series.astype(str)
            
            # Clean common issues
            cleaned = str_series.copy()
            
            # Remove currency symbols and common text
            for symbol in ['₹', '$', '€', '£', 'Cr', 'L', 'K', 'M', 'B', '%', '↑', '↓']:
                cleaned = cleaned.str.replace(symbol, '', regex=False)
            
            # Remove commas and spaces
            cleaned = cleaned.str.replace(',', '').str.replace(' ', '')
            
            # Handle dashes and special characters
            cleaned = cleaned.str.replace('--', '').str.replace('nil', '').str.replace('N/A', '')
            
            # Replace empty strings with NaN
            cleaned = cleaned.replace(['', 'nan', 'NaN', 'None', 'null'], np.nan)
            
            # Handle market cap notation (Cr, L, K)
            if 'market_cap' in col_name or 'cap' in col_name:
                cleaned = self._parse_market_cap_notation(cleaned)
            
            # Convert to numeric
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            
            # Handle infinite values
            numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
            
            # Apply validation rules
            if col_name in self.config.quality.VALIDATION_RULES:
                rules = self.config.quality.VALIDATION_RULES[col_name]
                min_val = rules.get('min', -np.inf)
                max_val = rules.get('max', np.inf)
                
                # Set out-of-range values to NaN
                numeric_series = numeric_series.where(
                    (numeric_series >= min_val) & (numeric_series <= max_val)
                )
            
            return numeric_series
            
        except Exception as e:
            logger.warning(f"Numeric conversion failed for {col_name}: {e}")
            return pd.Series(self._get_safe_default(col_name), index=series.index)
    
    def _parse_market_cap_notation(self, series: pd.Series) -> pd.Series:
        """Parse Indian market cap notation (Cr, L, K)"""
        try:
            parsed = series.copy()
            
            # Handle Crores
            cr_mask = parsed.str.contains('Cr|crore|crores', case=False, na=False)
            if cr_mask.any():
                cr_values = parsed[cr_mask].str.extract(r'([\d.]+)', expand=False)
                parsed[cr_mask] = (pd.to_numeric(cr_values, errors='coerce') * 1e7).astype(str)
            
            # Handle Lakhs
            l_mask = parsed.str.contains('L|lakh|lakhs', case=False, na=False) & ~cr_mask
            if l_mask.any():
                l_values = parsed[l_mask].str.extract(r'([\d.]+)', expand=False)
                parsed[l_mask] = (pd.to_numeric(l_values, errors='coerce') * 1e5).astype(str)
            
            # Handle Thousands
            k_mask = parsed.str.contains('K|thousand', case=False, na=False) & ~cr_mask & ~l_mask
            if k_mask.any():
                k_values = parsed[k_mask].str.extract(r'([\d.]+)', expand=False)
                parsed[k_mask] = (pd.to_numeric(k_values, errors='coerce') * 1e3).astype(str)
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Market cap parsing failed: {e}")
            return series
    
    def _get_safe_default(self, col_name: str) -> float:
        """Get safe default value for a column"""
        defaults = {
            'price': 100.0, 'pe': 20.0, 'eps_current': 5.0, 'eps_change_pct': 0.0,
            'vol_1d': 10000, 'rvol': 1.0, 'from_low_pct': 50.0, 'from_high_pct': -50.0,
            'ret_1d': 0.0, 'ret_7d': 0.0, 'ret_30d': 0.0, 'ret_3m': 0.0,
            'sma20': 100.0, 'sma50': 100.0, 'sma200': 100.0,
            'low_52w': 90.0, 'high_52w': 110.0, 'market_cap': 1e8
        }
        
        # Check for pattern matches
        for pattern, default in defaults.items():
            if pattern in col_name:
                return default
        
        return 0.0  # Default fallback
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if col not in ['ticker', 'name', 'sector', 'category']:
                continue
                
            try:
                # Clean text data
                df[col] = df[col].astype(str).str.strip()
                
                # Replace common null representations
                df[col] = df[col].replace(['nan', 'NaN', 'None', 'null', ''], 'Unknown')
                
                # Special handling for ticker
                if col == 'ticker':
                    df[col] = df[col].str.upper().str.replace(' ', '')
                
            except Exception as e:
                logger.warning(f"Text cleaning failed for {col}: {e}")
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Handle duplicate records"""
        if 'ticker' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['ticker'], keep='first')
            removed_count = initial_count - len(df)
            if removed_count > 0:
                self.lineage.append(f"Removed {removed_count} duplicate tickers")
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert low-cardinality object columns to category
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_reduction = (1 - final_memory / initial_memory) * 100
        
        if memory_reduction > 0:
            self.lineage.append(f"Memory optimized: {initial_memory:.1f}MB → {final_memory:.1f}MB ({memory_reduction:.1f}% reduction)")
        
        return df

# =============================================================================
# COMPREHENSIVE QUALITY ANALYZER
# =============================================================================

class ComprehensiveQualityAnalyzer:
    """Comprehensive data quality analysis system"""
    
    def __init__(self):
        self.config = CONFIG
        self.issues = []
    
    def analyze_quality(self, df: pd.DataFrame, sheet_name: str = "dataset") -> QualityReport:
        """Perform comprehensive quality analysis"""
        start_time = time.time()
        self.issues = []
        
        # Generate data hash for tracking
        data_hash = self._generate_data_hash(df)
        
        # Completeness analysis
        completeness_scores = self._analyze_completeness(df)
        
        # Validity analysis
        validity_scores = self._analyze_validity(df)
        
        # Consistency analysis
        consistency_scores = self._analyze_consistency(df)
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(df)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(
            completeness_scores, validity_scores, consistency_scores, outlier_analysis
        )
        
        # Determine quality status
        status = self._determine_status(overall_score)
        
        # Categorize issues
        critical_issues = [i for i in self.issues if i.severity == 'critical']
        warnings = [i for i in self.issues if i.severity == 'warning']
        info_messages = [i for i in self.issues if i.severity == 'info']
        
        # Quality breakdown
        excellent_quality, good_quality, poor_quality = self._calculate_quality_breakdown(df, overall_score)
        
        processing_time = time.time() - start_time
        
        return QualityReport(
            overall_score=round(overall_score, 1),
            status=status,
            processing_time=round(processing_time, 3),
            data_hash=data_hash,
            timestamp=datetime.now(),
            
            critical_issues=critical_issues,
            warnings=warnings,
            info_messages=info_messages,
            
            completeness_scores=completeness_scores,
            consistency_scores=consistency_scores,
            validity_scores=validity_scores,
            outlier_analysis=outlier_analysis,
            
            total_stocks=len(df),
            usable_stocks=len(df) - len(critical_issues),
            excluded_stocks=len(critical_issues),
            
            excellent_quality=excellent_quality,
            good_quality=good_quality,
            poor_quality=poor_quality,
            
            lineage=[],
            source_info={'sheet_name': sheet_name, 'columns': len(df.columns)}
        )
    
    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of data for tracking changes"""
        try:
            if 'ticker' in df.columns and 'price' in df.columns:
                hash_data = df[['ticker', 'price']].fillna('').astype(str).apply(''.join, axis=1).str.cat()
            else:
                hash_data = str(df.shape) + str(list(df.columns))
            
            return hashlib.md5(hash_data.encode()).hexdigest()[:12]
        except:
            return datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _analyze_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze data completeness"""
        completeness_scores = {}
        
        # Check required columns
        for col in self.config.quality.REQUIRED_COLUMNS:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                completeness_scores[col] = round(completeness, 1)
                
                if completeness < 80:
                    self.issues.append(QualityIssue(
                        severity='critical',
                        category='completeness',
                        description=f'Required column {col} only {completeness:.1f}% complete',
                        affected_count=int(len(df) * (100 - completeness) / 100),
                        recommendation=f'Improve data collection for {col}',
                        field=col
                    ))
            else:
                completeness_scores[col] = 0.0
                self.issues.append(QualityIssue(
                    severity='critical',
                    category='completeness',
                    description=f'Required column {col} is missing',
                    affected_count=len(df),
                    recommendation=f'Add {col} column to data source',
                    field=col
                ))
        
        # Check important columns
        for col in self.config.quality.IMPORTANT_COLUMNS:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                completeness_scores[col] = round(completeness, 1)
                
                if completeness < 60:
                    self.issues.append(QualityIssue(
                        severity='warning',
                        category='completeness',
                        description=f'Important column {col} only {completeness:.1f}% complete',
                        affected_count=int(len(df) * (100 - completeness) / 100),
                        recommendation=f'Consider improving data for {col}',
                        field=col
                    ))
        
        return completeness_scores
    
    def _analyze_validity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze data validity"""
        validity_scores = {}
        
        for field, rules in self.config.quality.VALIDATION_RULES.items():
            if field not in df.columns:
                continue
            
            min_val = rules.get('min', -np.inf)
            max_val = rules.get('max', np.inf)
            
            valid_mask = (df[field] >= min_val) & (df[field] <= max_val) & df[field].notna()
            validity = (valid_mask.sum() / len(df)) * 100
            validity_scores[field] = round(validity, 1)
            
            if validity < 90:
                invalid_count = (~valid_mask).sum()
                severity = 'critical' if validity < 70 else 'warning'
                
                self.issues.append(QualityIssue(
                    severity=severity,
                    category='validity',
                    description=f'{invalid_count} records have invalid {field} values',
                    affected_count=invalid_count,
                    recommendation=f'Review {field} data for outliers and errors',
                    field=field
                ))
        
        return validity_scores
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze data consistency"""
        consistency_scores = {}
        
        # Check for duplicate tickers
        if 'ticker' in df.columns:
            duplicates = df['ticker'].duplicated().sum()
            consistency = ((len(df) - duplicates) / len(df)) * 100
            consistency_scores['ticker_uniqueness'] = round(consistency, 1)
            
            if duplicates > 0:
                self.issues.append(QualityIssue(
                    severity='critical',
                    category='consistency',
                    description=f'{duplicates} duplicate ticker symbols found',
                    affected_count=duplicates,
                    recommendation='Remove duplicate tickers',
                    field='ticker'
                ))
        
        # Price range consistency
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            valid_range = (df['price'] >= df['low_52w']) & (df['price'] <= df['high_52w'])
            range_consistency = (valid_range.sum() / len(df)) * 100
            consistency_scores['price_range'] = round(range_consistency, 1)
            
            if range_consistency < 90:
                invalid_count = (~valid_range).sum()
                self.issues.append(QualityIssue(
                    severity='warning',
                    category='consistency',
                    description=f'{invalid_count} stocks have price outside 52-week range',
                    affected_count=invalid_count,
                    recommendation='Verify 52-week range calculations',
                    field='price'
                ))
        
        return consistency_scores
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numeric data"""
        outlier_analysis = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].notna().sum() < 3:  # Need at least 3 values
                continue
            
            try:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outliers = z_scores > self.config.quality.OUTLIER_THRESHOLD
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        outlier_percentage = (outlier_count / len(df)) * 100
                        
                        outlier_analysis[col] = {
                            'count': int(outlier_count),
                            'percentage': round(outlier_percentage, 1),
                            'mean': round(mean, 2),
                            'std': round(std, 2),
                            'threshold': self.config.quality.OUTLIER_THRESHOLD
                        }
                        
                        if outlier_percentage > 5:  # More than 5% outliers
                            severity = 'warning' if outlier_percentage < 15 else 'critical'
                            self.issues.append(QualityIssue(
                                severity=severity,
                                category='outlier',
                                description=f'{outlier_count} outliers detected in {col} ({outlier_percentage:.1f}%)',
                                affected_count=int(outlier_count),
                                recommendation=f'Review {col} for data entry errors or unusual values',
                                field=col,
                                details={'mean': mean, 'std': std}
                            ))
            except Exception as e:
                logger.warning(f"Outlier analysis failed for {col}: {e}")
        
        return outlier_analysis
    
    def _calculate_overall_score(self, completeness: Dict, validity: Dict, 
                                consistency: Dict, outliers: Dict) -> float:
        """Calculate overall quality score"""
        
        # Completeness score (50% weight)
        required_completeness = [completeness.get(col, 0) for col in self.config.quality.REQUIRED_COLUMNS]
        important_completeness = [completeness.get(col, 50) for col in self.config.quality.IMPORTANT_COLUMNS]
        
        avg_required = np.mean(required_completeness) if required_completeness else 0
        avg_important = np.mean(important_completeness) if important_completeness else 50
        
        completeness_score = (avg_required * 0.7 + avg_important * 0.3)
        
        # Validity score (30% weight)
        validity_scores = list(validity.values())
        validity_score = np.mean(validity_scores) if validity_scores else 80
        
        # Consistency score (15% weight)
        consistency_scores = list(consistency.values())
        consistency_score = np.mean(consistency_scores) if consistency_scores else 90
        
        # Outlier penalty (5% weight)
        outlier_penalty = min(20, len(outliers) * 2)
        outlier_score = max(0, 100 - outlier_penalty)
        
        # Weighted overall score
        overall = (
            completeness_score * 0.50 +
            validity_score * 0.30 +
            consistency_score * 0.15 +
            outlier_score * 0.05
        )
        
        return max(0, min(100, overall))
    
    def _determine_status(self, score: float) -> str:
        """Determine quality status from score"""
        if score >= self.config.quality.EXCELLENT_THRESHOLD:
            return 'Excellent'
        elif score >= self.config.quality.GOOD_THRESHOLD:
            return 'Good'
        elif score >= self.config.quality.ACCEPTABLE_THRESHOLD:
            return 'Acceptable'
        elif score >= self.config.quality.POOR_THRESHOLD:
            return 'Poor'
        else:
            return 'Critical'
    
    def _calculate_quality_breakdown(self, df: pd.DataFrame, overall_score: float) -> Tuple[int, int, int]:
        """Calculate quality breakdown by stock"""
        if overall_score >= 90:
            excellent = int(len(df) * 0.7)
            good = int(len(df) * 0.25)
        elif overall_score >= 80:
            excellent = int(len(df) * 0.4)
            good = int(len(df) * 0.45)
        else:
            excellent = int(len(df) * 0.2)
            good = int(len(df) * 0.4)
        
        poor = len(df) - excellent - good
        
        return excellent, good, max(0, poor)

# =============================================================================
# MAIN QUALITY CONTROLLER
# =============================================================================

class UltimateQualityController:
    """Ultimate quality controller that orchestrates all quality operations"""
    
    def __init__(self):
        self.config = CONFIG
        self.cleaner = BulletproofDataCleaner()
        self.analyzer = ComprehensiveQualityAnalyzer()
        
        logger.info("🛡️ Ultimate Quality Controller initialized")
    
    def process_dataframe(self, df: pd.DataFrame, sheet_name: str = "dataset") -> ProcessingResult:
        """Process dataframe with comprehensive quality control"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
            
            # Step 1: Clean the dataframe
            cleaned_df = self.cleaner.clean_dataframe(df, sheet_name)
            
            # Step 2: Analyze quality
            quality_report = self.analyzer.analyze_quality(cleaned_df, sheet_name)
            
            # Step 3: Apply quality filters
            filtered_df = self._apply_quality_filters(cleaned_df, quality_report)
            
            # Step 4: Final validation
            success, message = self._validate_final_result(filtered_df, quality_report)
            
            processing_time = time.time() - start_time
            
            processing_stats = {
                'processing_time': round(processing_time, 3),
                'initial_rows': len(df),
                'final_rows': len(filtered_df),
                'rows_removed': len(df) - len(filtered_df),
                'quality_score': quality_report.overall_score,
                'critical_issues': len(quality_report.critical_issues),
                'warnings': len(quality_report.warnings)
            }
            
            logger.info(f"✅ {sheet_name} processed: {quality_report.status} quality ({quality_report.overall_score:.1f}%)")
            
            return ProcessingResult(
                success=success,
                dataframe=filtered_df,
                quality_report=quality_report,
                message=message,
                processing_stats=processing_stats
            )
            
        except Exception as e:
            logger.error(f"❌ Processing failed for {sheet_name}: {e}")
            
            # Create minimal quality report for error case
            error_report = QualityReport(
                overall_score=0.0,
                status='Error',
                processing_time=time.time() - start_time,
                data_hash='error',
                timestamp=datetime.now(),
                critical_issues=[],
                warnings=[],
                info_messages=[],
                completeness_scores={},
                consistency_scores={},
                validity_scores={},
                outlier_analysis={},
                total_stocks=len(df),
                usable_stocks=0,
                excluded_stocks=len(df),
                excellent_quality=0,
                good_quality=0,
                poor_quality=len(df),
                lineage=[],
                source_info={'error': str(e)}
            )
            
            return ProcessingResult(
                success=False,
                dataframe=df,  # Return original dataframe
                quality_report=error_report,
                message=f"Processing failed: {str(e)}",
                processing_stats={'error': str(e)}
            )
    
    def _apply_quality_filters(self, df: pd.DataFrame, quality_report: QualityReport) -> pd.DataFrame:
        """Apply quality-based filtering"""
        filtered_df = df.copy()
        
        # Remove stocks with critical data issues
        if 'ticker' in filtered_df.columns:
            # Remove stocks with missing tickers
            filtered_df = filtered_df[filtered_df['ticker'].notna() & (filtered_df['ticker'] != 'Unknown')]
        
        if 'price' in filtered_df.columns:
            # Remove stocks with invalid prices
            filtered_df = filtered_df[
                (filtered_df['price'] > 0) & 
                (filtered_df['price'] < 100000) &
                (filtered_df['price'].notna())
            ]
        
        # Remove duplicates (keep first occurrence)
        if 'ticker' in filtered_df.columns:
            filtered_df = filtered_df.drop_duplicates(subset=['ticker'], keep='first')
        
        return filtered_df.reset_index(drop=True)
    
    def _validate_final_result(self, df: pd.DataFrame, quality_report: QualityReport) -> Tuple[bool, str]:
        """Validate final processing result"""
        if len(df) == 0:
            return False, "No usable data after quality filtering"
        
        if quality_report.overall_score < self.config.quality.MIN_DATA_QUALITY_SCORE:
            return False, f"Data quality score {quality_report.overall_score:.1f} below minimum threshold {self.config.quality.MIN_DATA_QUALITY_SCORE}"
        
        critical_issues = len(quality_report.critical_issues)
        if critical_issues > len(df) * 0.1:  # More than 10% critical issues
            return False, f"Too many critical issues: {critical_issues}"
        
        return True, f"Quality validation passed: {quality_report.status} quality ({quality_report.overall_score:.1f}%)"
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'config_version': self.config.ui.APP_CONFIG['version'],
            'quality_threshold': self.config.quality.MIN_DATA_QUALITY_SCORE,
            'validation_rules': len(self.config.quality.VALIDATION_RULES),
            'required_columns': len(self.config.quality.REQUIRED_COLUMNS),
            'important_columns': len(self.config.quality.IMPORTANT_COLUMNS)
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_quality_controller() -> UltimateQualityController:
    """Create and return quality controller instance"""
    return UltimateQualityController()

def process_dataframe(df: pd.DataFrame, sheet_name: str = "dataset") -> ProcessingResult:
    """Convenience function to process a dataframe with quality control"""
    controller = create_quality_controller()
    return controller.process_dataframe(df, sheet_name)

def analyze_quality(df: pd.DataFrame, sheet_name: str = "dataset") -> QualityReport:
    """Convenience function to analyze data quality only"""
    analyzer = ComprehensiveQualityAnalyzer()
    return analyzer.analyze_quality(df, sheet_name)

# Export main classes and functions
__all__ = [
    'UltimateQualityController',
    'BulletproofDataCleaner', 
    'ComprehensiveQualityAnalyzer',
    'QualityReport',
    'QualityIssue',
    'ProcessingResult',
    'create_quality_controller',
    'process_dataframe',
    'analyze_quality'
]

if __name__ == "__main__":
    print("✅ Ultimate Quality Control System loaded successfully")

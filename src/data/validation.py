""
Data Validation Module
Checks for ethical compliance in features
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class EthicalDataChecker:
    """Check engineered features for ethical compliance"""
    
    def __init__(self):
        self.pii_patterns = [
            'email', 'phone', 'name', 'address', 'ssn', 'passport',
            'credit_card', 'bank_account', 'social_security'
        ]
        
        self.prohibited_patterns = [
            'sentiment', 'social', 'media', 'alternative',
            'proprietary', 'confidential', 'internal'
        ]
    
    def check_engineered_features(self, df) -> Dict[str, bool]:
        """Check if engineered features comply with ethical constraints"""
        checks = {}
        
        if df.empty:
            return {check: False for check in [
                'no_pii_features', 'no_prohibited_features',
                'within_feature_limit', 'appropriate_data_types'
            ]}
        
        # Check for PII patterns in column names
        column_names = [str(col).lower() for col in df.columns]
        checks['no_pii_features'] = not any(
            any(pii in col for pii in self.pii_patterns)
            for col in column_names
        )
        
        # Check for prohibited patterns
        checks['no_prohibited_features'] = not any(
            any(prohibited in col for prohibited in self.prohibited_patterns)
            for col in column_names
        )
        
        # Check feature count (Data Minimization)
        checks['within_feature_limit'] = len(df.columns) <= 50 # Reasonable limit
        
        # Check data types are appropriate
        checks['appropriate_data_types'] = self._check_data_types(df)
        
        # Check no extreme values that might indicate PII
        checks['no_extreme_values'] = self._check_extreme_values(df)
        
        # Check all values are finite
        checks['all_finite'] = df.replace([np.inf, -np.inf], np.nan).notna().all().all()
        
        return checks
    
    def _check_data_types(self, df) -> bool:
        """Check that data types are appropriate for financial features"""
        inappropriate_dtypes = ['object', 'datetime']
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            if any(inappropriate in dtype for inappropriate in inappropriate_dtypes):
                # Check if it's actually a timestamp (allowed)
                if col != 'timestamp':
                    return False
        
        return True
    
    def _check_extreme_values(self, df) -> bool:
        """Check for extreme values that might indicate PII"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return True
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for values that might be IDs or codes
            # (very large integers, or very precise decimals)
            if col_data.dtype in [np.int64, np.int32]:
                # Check for suspiciously large integers (could be IDs)
                max_val = col_data.abs().max()
                if max_val > 1e9: # Over 1 billion
                    return False
            
            # Check for binary data (0/1) - might be one-hot encoded PII
            unique_vals = col_data.unique()
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                # Check if column name suggests it might be encoded PII
                col_lower = col.lower()
                if any(pii in col_lower for pii in ['gender', 'race', 'ethnic', 'age_group']):
                    return False
        
        return True
    
    def validate_feature_importance(self, feature_importances: Dict[str, float]) -> Dict[str, bool]:
        """Validate that feature importances don't reveal sensitive information"""
        checks = {}
        
        # Check that no feature has disproportionate importance
        # (might indicate it's capturing sensitive information)
        if feature_importances:
            importances = list(feature_importances.values())
            max_importance = max(importances) if importances else 0
            mean_importance = np.mean(importances) if importances else 0
            
            checks['no_dominant_feature'] = max_importance < 0.5 # No single feature > 50%
            checks['balanced_importance'] = max_importance / (mean_importance + 1e-10) < 10
        
        # Check feature names don't reveal sensitive patterns
        feature_names = list(feature_importances.keys())
        checks['feature_names_appropriate'] = not any(
            any(pii in name.lower() for pii in self.pii_patterns)
            for name in feature_names
        )
        
        return checks

Sent via the Samsung Galaxy S25 Ultra, an AT&T 5G smartphone
Get Outlook for Android
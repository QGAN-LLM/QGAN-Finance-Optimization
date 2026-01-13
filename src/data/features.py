"""
Feature Engineering Module
Creates purpose-limited features adhering to Data Minimization
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Engineer features with ethical constraints"""
    
    def __init__(self, config):
        self.config = config['feature_engineering']
        self._validate_config()
    
    def _validate_config(self):
        """Validate config for ethical compliance"""
        # Check no prohibited categories are enabled
        prohibited = self.config['prohibited_categories']
        for category in prohibited:
            if self.config.get(category, {}).get('enabled', False):
                raise ValueError(f"Prohibited feature category '{category}' is enabled")
    
    def create_features(self, df):
        """Create all features based on config"""
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Price-based features
        if self.config['price_features']['returns']:
            result_df = self._add_price_features(result_df)
        
        # Technical indicators
        if self.config['technical_indicators']['enabled']:
            result_df = self._add_technical_indicators(result_df)
        
        # Volatility features
        if self.config['volatility']['enabled']:
            result_df = self._add_volatility_features(result_df)
        
        # Time features
        if self.config['time_features']['enabled']:
            result_df = self._add_time_features(result_df)
        
        # Apply transformations
        if self.config['transformations']['enabled']:
            result_df = self._apply_transformations(result_df)
        
        # Feature selection
        if self.config['feature_selection']['enabled']:
            result_df = self._select_features(result_df)
        
        return result_df
    
    def _add_price_features(self, df):
        """Add price-based features"""
        if 'close' not in df.columns:
            return df
        
        result_df = df.copy()
        
        # Returns
        result_df['returns'] = df['close'].pct_change()
        result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Lagged returns
        for lag in [1, 2, 3]:
            result_df[f'returns_lag{lag}'] = result_df['returns'].shift(lag)
        
        # High-low range
        if 'high' in df.columns and 'low' in df.columns:
            result_df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Apply naming convention
        naming = self.config['output']['naming']
        for col in ['returns', 'log_returns']:
            if col in result_df.columns:
                result_df.rename(columns={col: f"{naming['prefix_returns']}{col}"}, inplace=True)
        
        return result_df
    
    def _add_technical_indicators(self, df):
        """Add technical indicators (limited set)"""
        import pandas_ta as ta
        
        result_df = df.copy()
        ta_config = self.config['technical_indicators']
        
        # RSI
        if ta_config['momentum']['rsi']['enabled']:
            period = ta_config['momentum']['rsi']['period']
            result_df['rsi'] = ta.rsi(df['close'], length=period)
        
        # MACD
        if ta_config['momentum']['macd']['enabled']:
            config = ta_config['momentum']['macd']
            macd = ta.macd(df['close'], 
                          fast=config['fast_period'],
                          slow=config['slow_period'],
                          signal=config['signal_period'])
            result_df = pd.concat([result_df, macd], axis=1)
        
        # ATR
        if ta_config['volatility']['atr']['enabled']:
            period = ta_config['volatility']['atr']['period']
            result_df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=period)
        
        # Bollinger Bands
        if ta_config['volatility']['bollinger_bands']['enabled']:
            config = ta_config['volatility']['bollinger_bands']
            bb = ta.bbands(df['close'], 
                          length=config['period'],
                          std=config['std_dev'])
            result_df = pd.concat([result_df, bb], axis=1)
        
        # SMA
        if ta_config['trend']['sma']['enabled']:
            for period in ta_config['trend']['sma']['periods']:
                result_df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        
        # Apply naming convention
        naming = self.config['output']['naming']
        tech_cols = [c for c in result_df.columns if c not in df.columns]
        for col in tech_cols:
            result_df.rename(columns={col: f"{naming['prefix_technical']}{col}"}, inplace=True)
        
        return result_df
    
    def _add_volatility_features(self, df):
        """Add volatility features"""
        result_df = df.copy()
        
        # Realized volatility
        if self.config['volatility']['realized_volatility']['enabled']:
            window = self.config['volatility']['realized_volatility']['window']
            if 'log_returns' in result_df.columns or 'returns' in result_df.columns:
                returns_col = 'log_returns' if 'log_returns' in result_df.columns else 'returns'
                result_df['realized_vol'] = result_df[returns_col].rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility
        if self.config['volatility']['parkinson_volatility']['enabled']:
            window = self.config['volatility']['parkinson_volatility']['window']
            if 'high' in df.columns and 'low' in df.columns:
                log_hl = np.log(df['high'] / df['low'])
                result_df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * (log_hl**2).rolling(window).mean()) * np.sqrt(252)
        
        # Apply naming convention
        naming = self.config['output']['naming']
        vol_cols = [c for c in result_df.columns if 'vol' in c.lower() and c not in df.columns]
        for col in vol_cols:
            result_df.rename(columns={col: f"{naming['prefix_volatility']}{col}"}, inplace=True)
        
        return result_df
    
    def _add_time_features(self, df):
        """Add time-based features"""
        result_df = df.copy()
        
        if result_df.index.name == 'timestamp':
            index_series = result_df.index
        else:
            index_series = result_df['timestamp']
        
        # Hour of day
        if self.config['time_features']['hour_of_day']:
            result_df['hour'] = index_series.hour
        
        # Day of week
        if self.config['time_features']['day_of_week']:
            result_df['day_of_week'] = index_series.dayofweek
        
        # Is weekend
        if self.config['time_features']['is_weekend']:
            result_df['is_weekend'] = index_series.dayofweek.isin([5, 6]).astype(int)
        
        # Apply naming convention
        naming = self.config['output']['naming']
        time_cols = [c for c in result_df.columns if c in ['hour', 'day_of_week', 'is_weekend']]
        for col in time_cols:
            result_df.rename(columns={col: f"{naming['prefix_time']}{col}"}, inplace=True)
        
        return result_df
    
    def _apply_transformations(self, df):
        """Apply feature transformations"""
        result_df = df.copy()
        transform_config = self.config['transformations']
        
        if transform_config['normalization']['enabled']:
            method = transform_config['normalization']['method']
            result_df = self._normalize_features(result_df, method)
        
        return result_df
    
    def _normalize_features(self, df, method='standard'):
        """Normalize features using specified method"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # Select numeric columns (excluding timestamp if present)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return df
        
        # Fit and transform
        df_numeric = df[numeric_cols].copy()
        scaled_values = scaler.fit_transform(df_numeric)
        
        # Create new DataFrame with scaled values
        scaled_df = pd.DataFrame(scaled_values, 
                                columns=[f"{col}_scaled" for col in numeric_cols],
                                index=df.index)
        
        # Combine with non-numeric columns
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        result_df = pd.concat([df[non_numeric_cols], scaled_df], axis=1)
        
        return result_df
    
    def _select_features(self, df):
        """Select features based on config"""
        selection_config = self.config['feature_selection']
        
        if selection_config['method'] == 'correlation':
            return self._select_by_correlation(df, selection_config)
        else:
            # Default: return all features
            return df
    
    def _select_by_correlation(self, df, config):
        """Remove highly correlated features"""
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return df
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns 
                  if any(upper[column] > config['correlation_threshold'])]
        
        # Drop features
        result_df = df.drop(columns=to_drop)
        
        print(f"Feature selection: Dropped {len(to_drop)} highly correlated features")
        
        return result_df


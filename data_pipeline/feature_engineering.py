"""
Feature engineering for financial time series.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import pandas_ta as ta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineers features from cleaned financial data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.feature_names = []
    
    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform cleaned data into feature-rich dataset.
        
        Args:
            data: Cleaned DataFrame
            
        Returns:
            Tuple of (feature DataFrame, feature names)
        """
        logger.info(f"Starting feature engineering on {len(data)} rows")
        
        # Make a copy
        features = data.copy()
        
        # 1. Basic price features
        features = self._add_basic_features(features)
        
        # 2. Technical indicators
        features = self._add_technical_indicators(features)
        
        # 3. Statistical features
        features = self._add_statistical_features(features)
        
        # 4. Time-based features
        features = self._add_time_features(features)
        
        # 5. Volatility regime features
        features = self._add_volatility_features(features)
        
        # 6. Lagged features
        features = self._add_lagged_features(features)
        
        # 7. Interaction features
        features = self._add_interaction_features(features)
        
        # Remove NaN values from feature engineering
        initial_len = len(features)
        features = features.dropna()
        removed = initial_len - len(features)
        if removed > 0:
            logger.info(f"Removed {removed} rows with NaN after feature engineering")
        
        logger.info(f"Feature engineering complete. {len(self.feature_names)} features created")
        return features, self.feature_names
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features."""
        if 'close' in data.columns:
            # Price returns
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Price ranges
            data['range'] = data['high'] - data['low']
            data['range_pct'] = data['range'] / data['close']
            
            # Price position within range
            data['price_position'] = (data['close'] - data['low']) / data['range']
            
            self.feature_names.extend(['returns', 'log_returns', 'range', 
                                      'range_pct', 'price_position'])
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        if 'close' not in data.columns:
            return data
        
        # Moving averages
        periods = [5, 10, 20, 50, 100]
        for period in periods:
            data[f'sma_{period}'] = ta.sma(data['close'], length=period)
            data[f'ema_{period}'] = ta.ema(data['close'], length=period)
            self.feature_names.extend([f'sma_{period}', f'ema_{period}'])
        
        # RSI (Relative Strength Index)
        data['rsi'] = ta.rsi(data['close'], length=14)
        self.feature_names.append('rsi')
        
        # MACD
        macd = ta.macd(data['close'])
        if macd is not None:
            data['macd'] = macd['MACD_12_26_9']
            data['macd_signal'] = macd['MACDs_12_26_9']
            data['macd_hist'] = macd['MACDh_12_26_9']
            self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Bollinger Bands
        bb = ta.bbands(data['close'], length=20)
        if bb is not None:
            data['bb_upper'] = bb['BBU_20_2.0']
            data['bb_middle'] = bb['BBM_20_2.0']
            data['bb_lower'] = bb['BBL_20_2.0']
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            self.feature_names.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width'])
        
        # Average True Range (volatility)
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        self.feature_names.append('atr')
        
        # Stochastic Oscillator
        stoch = ta.stoch(data['high'], data['low'], data['close'])
        if stoch is not None:
            data['stoch_k'] = stoch['STOCHk_14_3_3']
            data['stoch_d'] = stoch['STOCHd_14_3_3']
            self.feature_names.extend(['stoch_k', 'stoch_d'])
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        if 'returns' not in data.columns:
            return data
        
        # Rolling statistics
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            # Rolling volatility
            data[f'volatility_{window}'] = data['returns'].rolling(window=window).std()
            
            # Rolling skewness and kurtosis
            data[f'skewness_{window}'] = data['returns'].rolling(window=window).skew()
            data[f'kurtosis_{window}'] = data['returns'].rolling(window=window).kurt()
            
            # Rolling quantiles
            data[f'q25_{window}'] = data['returns'].rolling(window=window).quantile(0.25)
            data[f'q75_{window}'] = data['returns'].rolling(window=window).quantile(0.75)
            
            self.feature_names.extend([
                f'volatility_{window}', f'skewness_{window}', f'kurtosis_{window}',
                f'q25_{window}', f'q75_{window}'
            ])
        
        return data
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        # Time of day features (cyclical encoding)
        data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
        
        # Day of week features
        data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
        data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
        
        # Month features
        data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
        
        # Trading session indicators
        data['asian_session'] = ((data.index.hour >= 0) & (data.index.hour < 9)).astype(int)
        data['european_session'] = ((data.index.hour >= 8) & (data.index.hour < 17)).astype(int)
        data['us_session'] = ((data.index.hour >= 13) & (data.index.hour < 22)).astype(int)
        
        self.feature_names.extend([
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'asian_session',
            'european_session', 'us_session'
        ])
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features."""
        if 'volatility' not in data.columns and 'returns' in data.columns:
            # Calculate 21-day volatility
            window = 21 * 24 * 60  # 21 days in minutes
            data['volatility'] = data['returns'].rolling(window=window).std()
        
        if 'volatility' in data.columns:
            # Volatility regime indicators
            thresholds = self.config.get('vix_thresholds', {'low': 15.0, 'high': 30.0})
            
            data['vol_regime_low'] = (data['volatility'] < thresholds['low']).astype(int)
            data['vol_regime_medium'] = ((data['volatility'] >= thresholds['low']) & 
                                         (data['volatility'] < thresholds['high'])).astype(int)
            data['vol_regime_high'] = (data['volatility'] >= thresholds['high']).astype(int)
            
            self.feature_names.extend(['vol_regime_low', 'vol_regime_medium', 'vol_regime_high'])
        
        return data
    
    def _add_lagged_features(self, data: pd.DataFrame, max_lags: int = 5) -> pd.DataFrame:
        """Add lagged features."""
        important_cols = ['returns', 'log_returns', 'volume', 'range_pct']
        available_cols = [col for col in important_cols if col in data.columns]
        
        for col in available_cols:
            for lag in range(1, max_lags + 1):
                data[f'{col}_lag{lag}'] = data[col].shift(lag)
                self.feature_names.append(f'{col}_lag{lag}')
        
        return data
    
    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between important variables."""
        interactions = []
        
        # Interaction between volatility and momentum
        if 'volatility' in data.columns and 'rsi' in data.columns:
            data['vol_rsi_interaction'] = data['volatility'] * data['rsi']
            interactions.append('vol_rsi_interaction')
        
        # Interaction between volume and price movement
        if 'volume' in data.columns and 'returns' in data.columns:
            data['volume_return_interaction'] = data['volume'] * data['returns'].abs()
            interactions.append('volume_return_interaction')
        
        # Polynomial features for key indicators
        if 'rsi' in data.columns:
            data['rsi_squared'] = data['rsi'] ** 2
            interactions.append('rsi_squared')
        
        if 'atr' in data.columns:
            data['atr_squared'] = data['atr'] ** 2
            interactions.append('atr_squared')
        
        self.feature_names.extend(interactions)
        return data
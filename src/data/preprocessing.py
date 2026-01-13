"""
Data Preprocessing Module for Dukascopy CSV files
Implements Data Minimization and quality assurance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DukascopyPreprocessor:
    """Preprocess Dukascopy CSV files with ethical constraints"""
    
    def __init__(self, config):
        self.config = config
        self.dukas_config = config['dukascopy']
        
    def load_csv(self, file_path):
        """
        Load Dukascopy CSV file with column mapping
        Implements Data Minimization: only keep necessary columns
        """
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Apply column mapping
            column_mapping = self.dukas_config['column_mapping']
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Data Minimization: Keep only necessary columns
            columns_to_keep = self.dukas_config['columns_to_keep']
            available_columns = [c for c in columns_to_keep if c in df.columns]
            df = df[available_columns]
            
            # Calculate derived columns if base columns exist
            if 'ask' in df.columns and 'bid' in df.columns:
                df['mid_price'] = (df['ask'] + df['bid']) / 2
                df['spread'] = df['ask'] - df['bid']
            
            if 'ask_volume' in df.columns and 'bid_volume' in df.columns:
                df['volume'] = df['ask_volume'] + df['bid_volume']
            
            # Drop original columns if not in keep list
            columns_to_drop = self.dukas_config['columns_to_drop']
            df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def resample_data(self, df, frequency='1H'):
        """Resample data to specified frequency"""
        if df.empty:
            return df
        
        # Resample OHLC
        ohlc_dict = {
            'mid_price': 'ohlc',
            'volume': 'sum'
        }
        
        resampled = df.resample(frequency).agg(ohlc_dict)
        
        # Flatten multi-index columns
        resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
        
        # Rename columns for clarity
        column_mapping = {
            'mid_price_open': 'open',
            'mid_price_high': 'high', 
            'mid_price_low': 'low',
            'mid_price_close': 'close',
            'volume_sum': 'volume'
        }
        
        resampled = resampled.rename(columns=column_mapping)
        return resampled
    
    def filter_trading_hours(self, df):
        """Filter to trading hours only"""
        if df.empty:
            return df
        
        trading_config = self.dukas_config.get('trading_hours', {})
        if not trading_config:
            return df
        
        # Filter by day of week
        days = trading_config.get('days', [])
        if days:
            df = df[df.index.day_name().isin(days)]
        
        # Filter by time of day
        start_time = trading_config.get('start', '00:00')
        end_time = trading_config.get('end', '23:59')
        
        time_mask = (df.index.time >= pd.to_datetime(start_time).time()) & \
                   (df.index.time <= pd.to_datetime(end_time).time())
        
        return df[time_mask]
    
    def handle_missing_values(self, df, method='interpolate'):
        """Handle missing values with specified method"""
        if df.empty:
            return df
        
        max_gap = self.dukas_config['quality']['max_gap_minutes']
        
        if method == 'interpolate':
            # Only interpolate small gaps
            df = df.interpolate(method='linear', limit=max_gap)
        
        # Forward/backward fill remaining
        df = df.ffill().bfill()
        
        return df
    
    def remove_outliers(self, df):
        """Remove outliers based on config constraints"""
        if df.empty:
            return df
        
        quality_config = self.dukas_config['quality']
        
        # Price bounds
        if 'mid_price' in df.columns:
            price_mask = (df['mid_price'] >= quality_config['min_price']) & \
                        (df['mid_price'] <= quality_config['max_price'])
            df = df[price_mask]
        
        # Spread bounds
        if 'spread' in df.columns:
            max_spread = quality_config['max_spread_pips'] / 10000
            spread_mask = df['spread'] <= max_spread
            df = df[spread_mask]
        
        # Volume bounds
        if 'volume' in df.columns:
            volume_mask = (df['volume'] >= quality_config['min_volume']) & \
                         (df['volume'] <= quality_config['max_volume'])
            df = df[volume_mask]
        
        return df


class DataValidator:
    """Validate data quality and ethical compliance"""
    
    def __init__(self, config):
        self.config = config
        self.quality_config = config['dukascopy']['quality']
    
    def validate_data(self, df):
        """Comprehensive data validation"""
        validation_report = {}
        
        if df.empty:
            return {
                'timestamp_continuity': False,
                'price_validity': False,
                'spread_validity': False,
                'missing_values': 1.0,
                'duplicate_rows': 0
            }
        
        # Check timestamp continuity
        validation_report['timestamp_continuity'] = self._check_timestamp_continuity(df)
        
        # Check price validity
        validation_report['price_validity'] = self._check_price_validity(df)
        
        # Check spread validity
        validation_report['spread_validity'] = self._check_spread_validity(df)
        
        # Check missing values
        validation_report['missing_values'] = df.isnull().mean().mean()
        
        # Check duplicates
        validation_report['duplicate_rows'] = df.duplicated().sum()
        
        return validation_report
    
    def _check_timestamp_continuity(self, df):
        """Check if timestamps are continuous within max gap"""
        if df.index.nunique() < 2:
            return False
        
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
        max_allowed_gap = self.quality_config['max_gap_minutes']
        
        return (time_diffs[1:] <= max_allowed_gap).all()
    
    def _check_price_validity(self, df):
        """Check if prices are within valid bounds"""
        if 'mid_price' not in df.columns:
            return False
        
        price_series = df['mid_price'].dropna()
        if len(price_series) == 0:
            return False
        
        min_price = self.quality_config['min_price']
        max_price = self.quality_config['max_price']
        
        return price_series.between(min_price, max_price).all()
    
    def _check_spread_validity(self, df):
        """Check if spreads are within valid bounds"""
        if 'spread' not in df.columns:
            return False
        
        spread_series = df['spread'].dropna()
        if len(spread_series) == 0:
            return False
        
        max_spread = self.quality_config['max_spread_pips'] / 10000
        
        return (spread_series >= 0).all() and (spread_series <= max_spread).all()


"""
Data acquisition from financial data sources.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataAcquirer:
    """Acquires financial market data from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data acquirer.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/raw'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources mapping
        self.sources = {
            'yfinance': self._download_yfinance,
            'dukascopy': self._download_dukascopy,
            'trader_made': self._download_trader_made
        }
    
    def download_data(self, source: Optional[str] = None) -> pd.DataFrame:
        """
        Download data from specified source.
        
        Args:
            source: Data source name (yfinance, dukascopy, trader_made)
            
        Returns:
            DataFrame with market data
        """
        source = source or self.config.get('data_source', 'yfinance')
        
        if source not in self.sources:
            raise ValueError(f"Unknown data source: {source}. Available: {list(self.sources.keys())}")
        
        logger.info(f"Downloading data from {source} for {self.config['currency_pair']}")
        data = self.sources[source]()
        
        # Save raw data
        filename = f"{self.config['currency_pair']}_{source}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = self.data_dir / filename
        data.to_parquet(filepath)
        logger.info(f"Saved raw data to {filepath}")
        
        return data
    
    def _download_yfinance(self) -> pd.DataFrame:
        """Download data using Yahoo Finance API."""
        ticker = self.config['currency_pair'].replace('/', '') + '=X'
        
        try:
            data = yf.download(
                ticker,
                start=self.config['start_date'],
                end=self.config['end_date'],
                interval=self.config.get('granularity', '1m'),
                progress=False
            )
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            
            # Add additional features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            logger.info(f"Downloaded {len(data)} rows from Yahoo Finance")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download from Yahoo Finance: {e}")
            raise
    
    def _download_dukascopy(self) -> pd.DataFrame:
        """Download data from Dukascopy (placeholder implementation)."""
        logger.warning("Dukascopy integration not implemented. Using simulated data.")
        
        # Generate simulated data for development
        dates = pd.date_range(
            start=self.config['start_date'],
            end=self.config['end_date'],
            freq=self.config.get('granularity', '1min')
        )
        
        n_samples = len(dates)
        np.random.seed(42)
        
        # Simulate EUR/USD price with realistic characteristics
        base_price = 1.10
        volatility = 0.0005  # 5 pips per minute
        
        returns = np.random.normal(0, volatility, n_samples)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add micro-structure noise
        noise = np.random.normal(0, volatility * 0.1, n_samples)
        prices += noise
        
        # Create OHLC data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.0001, 0.0001, n_samples)),
            'high': prices * (1 + np.random.uniform(0, 0.0002, n_samples)),
            'low': prices * (1 - np.random.uniform(0, 0.0002, n_samples)),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=dates)
        
        logger.info(f"Generated {len(data)} simulated rows for Dukascopy")
        return data
    
    def _download_trader_made(self) -> pd.DataFrame:
        """Download data from TraderMade API (placeholder)."""
        raise NotImplementedError("TraderMade API integration not yet implemented")
    
    @property
    def metadata_path(self) -> Path:
        """Path to metadata file."""
        return self.data_dir / "metadata.json"
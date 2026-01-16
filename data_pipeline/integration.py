"""
Data integration from multiple sources.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataIntegrator:
    """Integrates data from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data integrator.
        
        Args:
            config: Integration configuration
        """
        self.config = config
    
    def integrate(self, 
                  market_data: pd.DataFrame,
                  macroeconomic_data: Optional[pd.DataFrame] = None,
                  sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Integrate data from multiple sources.
        
        Args:
            market_data: Primary market data
            macroeconomic_data: Macroeconomic indicators
            sentiment_data: News/sentiment data
            
        Returns:
            Integrated DataFrame
        """
        logger.info("Starting data integration")
        
        # Start with market data
        integrated = market_data.copy()
        
        # Integrate macroeconomic data
        if macroeconomic_data is not None:
            integrated = self._integrate_macroeconomic(integrated, macroeconomic_data)
        
        # Integrate sentiment data
        if sentiment_data is not None:
            integrated = self._integrate_sentiment(integrated, sentiment_data)
        
        # Add derived cross-asset features
        integrated = self._add_cross_asset_features(integrated)
        
        logger.info(f"Integration complete. Final shape: {integrated.shape}")
        return integrated
    
    def _integrate_macroeconomic(self, 
                                market_data: pd.DataFrame,
                                macro_data: pd.DataFrame) -> pd.DataFrame:
        """Integrate macroeconomic data."""
        # Ensure both have datetime index
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)
        
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            macro_data.index = pd.to_datetime(macro_data.index)
        
        # Align frequencies (macro data is usually daily)
        market_data_daily = market_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Merge on date
        merged = market_data_daily.merge(
            macro_data,
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_macro')
        )
        
        # Forward fill macro data to minute frequency
        for col in macro_data.columns:
            market_data[col] = np.nan
            
            # Set values at daily frequency
            for date, value in merged[col].dropna().items():
                mask = (market_data.index.date == date.date())
                market_data.loc[mask, col] = value
            
            # Forward fill within days
            market_data[col] = market_data[col].ffill()
        
        return market_data
    
    def _integrate_sentiment(self,
                           market_data: pd.DataFrame,
                           sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Integrate sentiment data."""
        # This is a placeholder for sentiment integration
        # In practice, you would use NLP models to extract sentiment scores
        
        if 'sentiment_score' in sentiment_data.columns:
            # Simple merge based on nearest timestamp
            market_data['sentiment'] = np.nan
            
            # For each market timestamp, find closest sentiment score
            for idx in market_data.index:
                time_diff = abs((sentiment_data.index - idx).total_seconds())
                closest_idx = time_diff.idxmin()
                if time_diff.min() < 3600:  # Within 1 hour
                    market_data.loc[idx, 'sentiment'] = sentiment_data.loc[closest_idx, 'sentiment_score']
            
            # Forward fill sentiment
            market_data['sentiment'] = market_data['sentiment'].ffill()
        
        return market_data
    
    def _add_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features derived from other asset classes."""
        # Placeholder for cross-asset correlations
        # In practice, you would download data for correlated assets
        
        # Example: Add VIX if available
        try:
            import yfinance as yf
            vix = yf.download('^VIX', 
                             start=data.index[0].strftime('%Y-%m-%d'),
                             end=data.index[-1].strftime('%Y-%m-%d'),
                             interval='1d',
                             progress=False)
            
            if not vix.empty:
                # Resample to match market data frequency
                vix_daily = vix['Close'].rename('vix')
                
                # Merge VIX data
                data['vix'] = np.nan
                for date, vix_value in vix_daily.dropna().items():
                    mask = (data.index.date == date.date())
                    data.loc[mask, 'vix'] = vix_value
                
                # Forward fill VIX
                data['vix'] = data['vix'].ffill()
                
                # Add VIX-based features
                if 'vix' in data.columns:
                    data['vix_change'] = data['vix'].pct_change()
                    data['vix_regime'] = pd.cut(data['vix'], 
                                               bins=[0, 15, 30, 45, 100],
                                               labels=['very_low', 'low', 'medium', 'high'])
        
        except Exception as e:
            logger.warning(f"Could not download VIX data: {e}")
        
        return data
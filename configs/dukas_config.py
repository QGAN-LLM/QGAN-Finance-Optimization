# Dukascopy CSV Data Configuration
# Data Minimization: Only necessary columns for EUR/USD analysis

dukascopy:
  # File naming pattern for Dukascopy CSV exports
  file_pattern: "EURUSD_{year}.csv"
  years: [2020, 2021, 2022, 2023, 2024]
  
  # Column mapping for Dukascopy CSV format
  # Dukascopy typically provides: Local time, Ask, Bid, AskVolume, BidVolume
  column_mapping:
    Local time: "timestamp"
    Ask: "ask"
    Bid: "bid"
    AskVolume: "ask_volume"
    BidVolume: "bid_volume"
  
  # Data Minimization: We only keep necessary columns
  columns_to_keep:
    - "timestamp"
    - "ask"
    - "bid"
    # Derived columns will be calculated:
    - "mid_price" # (ask + bid) / 2
    - "spread" # ask - bid
    - "volume" # ask_volume + bid_volume
  
  # Excluded columns (Data Minimization principle)
  columns_to_drop:
    - "AskVolume"
    - "BidVolume"
    - "Local time" # After mapping to timestamp
  
  # Timezone handling
  timezone: "UTC"
  
  # Data quality constraints
  quality:
    # Price validity checks
    min_price: 0.5
    max_price: 2.0 # EUR/USD reasonable bounds
    max_spread_pips: 50 # Max 50 pips spread
    
    # Volume filters
    min_volume: 0
    max_volume: 1000000000
    
    # Time continuity
    max_gap_minutes: 60 # Maximum gap to interpolate
    
  # Sampling and frequency
  resample_frequency: "1H" # Resample to 1-hour candles
  trading_hours:
    start: "00:00"
    end: "23:59"
    days: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
  
  # Ethical data handling
  ethical_constraints:
    anonymize: true
    remove_outliers: true
    # Purpose limitation: Data only used for research
    usage_restrictions:
      - "academic_research_only"
      - "no_commercial_use"
      - "no_redistribution"
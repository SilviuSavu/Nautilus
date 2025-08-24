#!/usr/bin/env python3
"""
Historical Data Loader for Nautilus Trading Platform
Downloads historical stock data from multiple free sources and loads into PostgreSQL/TimescaleDB

Usage:
    python load_historical_data.py --source yfinance --symbols AAPL,MSFT,GOOGL --period 5y
    python load_historical_data.py --source kaggle --dataset stock-market-data
    python load_historical_data.py --load-sp500 --interval 1d --years 3
"""

import os
import sys
import argparse
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import asyncpg
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    """
    Historical data loader with multiple source support
    """
    
    def __init__(self, 
                 db_url: str = "postgresql://nautilus:nautilus123@localhost:5432/nautilus",
                 data_dir: str = "/app/data"):
        self.db_url = db_url
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.engine = create_engine(db_url)
        
    def get_sp500_symbols(self) -> List[str]:
        """Get current S&P 500 symbols from Wikipedia"""
        try:
            logger.info("Fetching S&P 500 symbols from Wikipedia...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].str.replace('.', '-').tolist()
            logger.info(f"Found {len(symbols)} S&P 500 symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
            # Fallback to most liquid stocks
            return [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
                'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
                'CRM', 'NFLX', 'PFE', 'WMT', 'TMO', 'ABT', 'KO', 'PEP', 'CVX',
                'ORCL', 'ACN', 'CSCO', 'INTC', 'AMD', 'QCOM', 'IBM', 'PYPL'
            ]
    
    def download_yfinance_data(self, 
                              symbols: List[str],
                              period: str = "5y",
                              interval: str = "1d") -> pd.DataFrame:
        """
        Download historical data using yfinance
        
        Args:
            symbols: List of stock symbols
            period: Data period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)
        """
        try:
            logger.info(f"Downloading {len(symbols)} symbols with {period} period, {interval} interval")
            
            # Download data in batches to avoid timeouts
            batch_size = 50
            all_data = []
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {len(batch_symbols)} symbols")
                
                try:
                    batch_data = yf.download(
                        batch_symbols,
                        period=period,
                        interval=interval,
                        group_by='ticker',
                        threads=True,
                        progress=False
                    )
                    
                    if not batch_data.empty:
                        # Reshape multi-level columns to flat format
                        if len(batch_symbols) == 1:
                            # Single symbol
                            batch_data.columns = [f"{batch_symbols[0]}_{col}" for col in batch_data.columns]
                        else:
                            # Multiple symbols - already has proper format
                            pass
                        
                        all_data.append(batch_data)
                        
                except Exception as e:
                    logger.error(f"Error downloading batch {batch_symbols}: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No data downloaded successfully")
            
            # Combine all batches
            combined_data = pd.concat(all_data, axis=1)
            logger.info(f"Downloaded {len(combined_data)} rows of data")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error downloading yfinance data: {e}")
            raise
    
    def process_yfinance_to_standard_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert yfinance multi-column format to standard format
        """
        try:
            logger.info("Processing yfinance data to standard format...")
            
            records = []
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-symbol data
                symbols = data.columns.get_level_values(0).unique()
                
                for symbol in symbols:
                    try:
                        symbol_data = data[symbol].dropna()
                        
                        for date, row in symbol_data.iterrows():
                            records.append({
                                'symbol': symbol,
                                'date': date.date() if hasattr(date, 'date') else date,
                                'open': float(row.get('Open', np.nan)),
                                'high': float(row.get('High', np.nan)),
                                'low': float(row.get('Low', np.nan)),
                                'close': float(row.get('Close', np.nan)),
                                'adj_close': float(row.get('Adj Close', row.get('Close', np.nan))),
                                'volume': int(row.get('Volume', 0)) if not pd.isna(row.get('Volume', np.nan)) else 0,
                                'source': 'yfinance',
                                'created_at': datetime.now()
                            })
                    except Exception as e:
                        logger.warning(f"Error processing symbol {symbol}: {e}")
                        continue
            else:
                # Single symbol or pre-processed data
                for date, row in data.iterrows():
                    # Try to extract symbol from column names
                    symbol = 'UNKNOWN'
                    for col in data.columns:
                        if '_' in str(col):
                            symbol = str(col).split('_')[0]
                            break
                    
                    records.append({
                        'symbol': symbol,
                        'date': date.date() if hasattr(date, 'date') else date,
                        'open': float(row.iloc[0]) if len(row) > 0 else np.nan,
                        'high': float(row.iloc[1]) if len(row) > 1 else np.nan,
                        'low': float(row.iloc[2]) if len(row) > 2 else np.nan,
                        'close': float(row.iloc[3]) if len(row) > 3 else np.nan,
                        'adj_close': float(row.iloc[4]) if len(row) > 4 else float(row.iloc[3]) if len(row) > 3 else np.nan,
                        'volume': int(row.iloc[5]) if len(row) > 5 and not pd.isna(row.iloc[5]) else 0,
                        'source': 'yfinance',
                        'created_at': datetime.now()
                    })
            
            df = pd.DataFrame(records)
            df = df.dropna(subset=['close'])  # Remove rows without closing prices
            
            logger.info(f"Processed {len(df)} price records for {df['symbol'].nunique()} symbols")
            return df
            
        except Exception as e:
            logger.error(f"Error processing yfinance data: {e}")
            raise
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to Parquet format"""
        try:
            filepath = self.data_dir / filename
            
            # Convert datetime columns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Save with compression
            df.to_parquet(filepath, compression='snappy', index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving to parquet: {e}")
            raise
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            logger.info("Creating database tables...")
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS historical_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(15, 6),
                high DECIMAL(15, 6),
                low DECIMAL(15, 6),
                close DECIMAL(15, 6) NOT NULL,
                adj_close DECIMAL(15, 6),
                volume BIGINT DEFAULT 0,
                source VARCHAR(50) DEFAULT 'unknown',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(symbol, date, source)
            );
            
            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_date 
            ON historical_prices (symbol, date);
            
            CREATE INDEX IF NOT EXISTS idx_historical_prices_date 
            ON historical_prices (date);
            
            CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol 
            ON historical_prices (symbol);
            
            -- Convert to hypertable if TimescaleDB is available
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                    -- Check if hypertable already exists
                    IF NOT EXISTS (SELECT 1 FROM _timescaledb_catalog.hypertable 
                                  WHERE table_name = 'historical_prices') THEN
                        PERFORM create_hypertable('historical_prices', 'date', 
                                                if_not_exists => true);
                        RAISE NOTICE 'Created TimescaleDB hypertable for historical_prices';
                    END IF;
                END IF;
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE NOTICE 'TimescaleDB not available or hypertable creation failed: %', SQLERRM;
            END $$;
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def load_to_database(self, df: pd.DataFrame, batch_size: int = 10000):
        """Load DataFrame to database in batches"""
        try:
            logger.info(f"Loading {len(df)} records to database...")
            
            # Ensure tables exist
            self.create_tables()
            
            # Load data in batches
            total_loaded = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size].copy()
                
                try:
                    # Use upsert to handle duplicates
                    batch.to_sql(
                        'historical_prices',
                        self.engine,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
                    
                    total_loaded += len(batch)
                    logger.info(f"Loaded batch {i//batch_size + 1}: {total_loaded}/{len(df)} records")
                    
                except SQLAlchemyError as e:
                    logger.warning(f"Error loading batch {i//batch_size + 1}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {total_loaded} records to database")
            
        except Exception as e:
            logger.error(f"Error loading to database: {e}")
            raise
    
    def download_and_load(self, 
                         source: str = "yfinance",
                         symbols: Optional[List[str]] = None,
                         period: str = "5y",
                         interval: str = "1d"):
        """Complete download and load process"""
        try:
            logger.info(f"Starting data download from {source}...")
            
            # Get symbols
            if symbols is None:
                symbols = self.get_sp500_symbols()[:50]  # Start with top 50
            
            # Download data
            if source == "yfinance":
                raw_data = self.download_yfinance_data(symbols, period, interval)
                processed_data = self.process_yfinance_to_standard_format(raw_data)
            else:
                raise ValueError(f"Unsupported source: {source}")
            
            # Save to parquet
            filename = f"historical_data_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            parquet_path = self.save_to_parquet(processed_data, filename)
            
            # Load to database
            self.load_to_database(processed_data)
            
            logger.info(f"Data loading complete! Parquet saved to: {parquet_path}")
            return parquet_path
            
        except Exception as e:
            logger.error(f"Error in download and load process: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Load historical stock data')
    parser.add_argument('--source', default='yfinance', choices=['yfinance', 'kaggle'],
                       help='Data source')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated list of symbols (default: S&P 500)')
    parser.add_argument('--period', default='5y',
                       help='Data period for yfinance (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)')
    parser.add_argument('--interval', default='1d',
                       help='Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)')
    parser.add_argument('--load-sp500', action='store_true',
                       help='Load all S&P 500 symbols')
    parser.add_argument('--years', type=int, default=5,
                       help='Number of years of data to download')
    parser.add_argument('--db-url',
                       default=os.getenv('DATABASE_URL', 'postgresql://nautilus:nautilus123@localhost:5432/nautilus'),
                       help='Database connection URL')
    parser.add_argument('--data-dir', default='/app/data',
                       help='Directory to save parquet files')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Create loader
    loader = HistoricalDataLoader(db_url=args.db_url, data_dir=args.data_dir)
    
    try:
        # Load data
        parquet_path = loader.download_and_load(
            source=args.source,
            symbols=symbols,
            period=args.period,
            interval=args.interval
        )
        
        print(f"\n✅ Success! Data loaded and saved to: {parquet_path}")
        print(f"📊 Check your database: SELECT COUNT(*) FROM historical_prices;")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
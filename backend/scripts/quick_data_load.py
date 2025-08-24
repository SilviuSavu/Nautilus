#!/usr/bin/env python3
"""
Quick Data Load Script for Nautilus Platform
Simple script to quickly populate database with sample historical data

Usage:
    python quick_data_load.py
    python quick_data_load.py --symbols AAPL,MSFT,GOOGL --days 30
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_load_sample_data(symbols=None, period="1y", save_parquet=True):
    """
    Quick function to load sample data for testing
    
    Args:
        symbols: List of symbols or None for default set
        period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        save_parquet: Whether to save data as parquet file
    """
    
    if symbols is None:
        # Default test symbols - highly liquid stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    logger.info(f"Loading sample data for {len(symbols)} symbols: {', '.join(symbols)}")
    
    try:
        # Download data from Yahoo Finance
        logger.info(f"Downloading {period} of data...")
        data = yf.download(
            symbols, 
            period=period,
            interval='1d',
            group_by='ticker',
            threads=True,
            progress=True
        )
        
        if data.empty:
            raise ValueError("No data downloaded")
        
        logger.info(f"Downloaded {len(data)} rows of data")
        
        # Process data into standard format
        records = []
        
        if len(symbols) == 1:
            # Single symbol
            symbol = symbols[0]
            for date, row in data.iterrows():
                records.append({
                    'symbol': symbol,
                    'date': date.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'adj_close': float(row['Adj Close']),
                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                    'source': 'yfinance_quick',
                    'created_at': datetime.now()
                })
        else:
            # Multiple symbols
            for symbol in symbols:
                try:
                    symbol_data = data[symbol].dropna()
                    for date, row in symbol_data.iterrows():
                        records.append({
                            'symbol': symbol,
                            'date': date.date(),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'adj_close': float(row['Adj Close']),
                            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                            'source': 'yfinance_quick',
                            'created_at': datetime.now()
                        })
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue
        
        df = pd.DataFrame(records)
        df = df.dropna(subset=['close'])
        
        logger.info(f"Processed {len(df)} price records for {df['symbol'].nunique()} symbols")
        
        # Save to parquet if requested
        if save_parquet:
            os.makedirs('/app/data', exist_ok=True)
            parquet_file = f"/app/data/quick_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(parquet_file, compression='snappy', index=False)
            logger.info(f"Saved data to {parquet_file}")
        
        # Connect to database
        db_url = os.getenv('DATABASE_URL', 'postgresql://nautilus:nautilus123@localhost:5432/nautilus')
        engine = create_engine(db_url)
        
        # Create table if it doesn't exist
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
        
        CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_date 
        ON historical_prices (symbol, date);
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
            logger.info("Database table created/verified")
        
        # Load data to database
        logger.info("Loading data to database...")
        df.to_sql('historical_prices', engine, if_exists='append', index=False)
        
        # Verify data loaded
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count, COUNT(DISTINCT symbol) as symbols FROM historical_prices WHERE source = 'yfinance_quick'"))
            row = result.fetchone()
            logger.info(f"Database now contains {row.count} records for {row.symbols} symbols from quick load")
        
        logger.info("✅ Quick data load completed successfully!")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error in quick data load: {e}")
        raise

def test_data_access():
    """Test accessing the loaded data"""
    try:
        db_url = os.getenv('DATABASE_URL', 'postgresql://nautilus:nautilus123@localhost:5432/nautilus')
        engine = create_engine(db_url)
        
        # Test queries
        with engine.connect() as conn:
            # Get basic stats
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    string_agg(DISTINCT symbol, ', ' ORDER BY symbol) as symbols
                FROM historical_prices 
                WHERE source = 'yfinance_quick'
            """))
            
            stats = result.fetchone()
            
            print("\n📊 Database Statistics:")
            print(f"   Total Records: {stats.total_records:,}")
            print(f"   Unique Symbols: {stats.unique_symbols}")
            print(f"   Date Range: {stats.earliest_date} to {stats.latest_date}")
            print(f"   Symbols: {stats.symbols}")
            
            # Get sample data
            result = conn.execute(text("""
                SELECT symbol, date, close, volume 
                FROM historical_prices 
                WHERE source = 'yfinance_quick'
                ORDER BY date DESC, symbol 
                LIMIT 10
            """))
            
            print("\n📈 Sample Recent Data:")
            for row in result:
                print(f"   {row.symbol}: {row.date} - Close: ${row.close:.2f}, Volume: {row.volume:,}")
        
        print("\n✅ Data access test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error testing data access: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick load sample data')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: AAPL,MSFT,GOOGL,AMZN,TSLA)')
    parser.add_argument('--period', default='1y', help='Period (1y, 2y, 5y, etc.)')
    parser.add_argument('--test-only', action='store_true', help='Only test data access, do not load')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_data_access()
    else:
        symbols = None
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
        # Load data
        quick_load_sample_data(symbols=symbols, period=args.period)
        
        # Test access
        test_data_access()
        
        print(f"\n🎉 Done! You now have sample data to test with.")
        print(f"💡 Try running: SELECT * FROM historical_prices LIMIT 10;")
#!/usr/bin/env python3
"""
Simple data loader - just pull data and put it in the database
Run this to get historical data loaded quickly for testing tomorrow
"""

import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import os

# Database connection
DB_URL = os.getenv('DATABASE_URL', 'postgresql://nautilus:nautilus123@localhost:5432/nautilus')

def load_data():
    print("🚀 Loading historical stock data...")
    
    # Top liquid stocks for testing
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC',
        'ADBE', 'CRM', 'NFLX', 'PFE', 'WMT', 'TMO', 'ABT', 'KO'
    ]
    
    print(f"📈 Downloading 2 years of data for {len(symbols)} symbols...")
    
    # Download data
    data = yf.download(symbols, period='2y', interval='1d', group_by='ticker', threads=True)
    
    # Process into records
    records = []
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
                    'source': 'yfinance',
                    'created_at': datetime.now()
                })
        except Exception as e:
            print(f"⚠️ Skipping {symbol}: {e}")
    
    df = pd.DataFrame(records)
    print(f"📊 Processed {len(df):,} records for {df['symbol'].nunique()} symbols")
    
    # Save parquet
    os.makedirs('/tmp/nautilus_data', exist_ok=True)
    parquet_file = f"/tmp/nautilus_data/historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df.to_parquet(parquet_file, compression='snappy')
    print(f"💾 Saved to: {parquet_file}")
    
    # Connect to database
    engine = create_engine(DB_URL)
    
    # Create table
    create_sql = """
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
    
    CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_date ON historical_prices (symbol, date);
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()
    
    # Load to database
    print("🔄 Loading to database...")
    df.to_sql('historical_prices', engine, if_exists='append', index=False)
    
    # Check what we loaded
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) as count, COUNT(DISTINCT symbol) as symbols FROM historical_prices")).fetchone()
        print(f"✅ Database now has {result.count:,} records for {result.symbols} symbols")
    
    print(f"🎉 Done! Data loaded and ready for testing tomorrow.")

if __name__ == "__main__":
    load_data()
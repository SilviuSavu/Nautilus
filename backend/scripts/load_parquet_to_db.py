#!/usr/bin/env python3
"""
Load existing parquet files to database
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os

DB_URL = 'postgresql://nautilus:nautilus123@postgres:5432/nautilus'

def load_parquet_files():
    engine = create_engine(DB_URL)
    
    # Create tables
    print("Creating database tables...")
    create_tables_sql = """
    -- Historical prices table
    CREATE TABLE IF NOT EXISTS historical_prices (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        date DATE NOT NULL,
        timeframe VARCHAR(20) DEFAULT 'daily',
        open DECIMAL(15, 6),
        high DECIMAL(15, 6),
        low DECIMAL(15, 6),
        close DECIMAL(15, 6) NOT NULL,
        adj_close DECIMAL(15, 6),
        volume BIGINT DEFAULT 0,
        source VARCHAR(50) DEFAULT 'unknown',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(symbol, date, timeframe, source)
    );
    
    -- Economic indicators table
    CREATE TABLE IF NOT EXISTS economic_indicators (
        id SERIAL PRIMARY KEY,
        series_id VARCHAR(50) NOT NULL,
        description TEXT,
        date DATE NOT NULL,
        value DECIMAL(15, 6),
        source VARCHAR(50) DEFAULT 'unknown',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(series_id, date)
    );
    
    -- Fundamental data table  
    CREATE TABLE IF NOT EXISTS fundamental_data (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        name TEXT,
        sector VARCHAR(100),
        industry VARCHAR(100),
        market_cap VARCHAR(50),
        pe_ratio VARCHAR(20),
        price_to_book VARCHAR(20),
        dividend_yield VARCHAR(20),
        eps VARCHAR(20),
        beta VARCHAR(20),
        profit_margin VARCHAR(20),
        operating_margin VARCHAR(20),
        return_on_assets VARCHAR(20),
        return_on_equity VARCHAR(20),
        revenue_ttm VARCHAR(50),
        source VARCHAR(50) DEFAULT 'unknown',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(symbol, source, DATE(created_at))
    );
    
    -- SEC filings table
    CREATE TABLE IF NOT EXISTS sec_filings (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        company_name TEXT,
        cik VARCHAR(20),
        form_type VARCHAR(20),
        filing_date DATE,
        accession_number VARCHAR(50),
        source VARCHAR(50) DEFAULT 'unknown',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(ticker, accession_number)
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_date ON historical_prices (symbol, date);
    CREATE INDEX IF NOT EXISTS idx_economic_indicators_series_date ON economic_indicators (series_id, date);
    CREATE INDEX IF NOT EXISTS idx_fundamental_data_symbol ON fundamental_data (symbol);
    CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings (ticker);
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_tables_sql))
        conn.commit()
    
    # Load parquet files
    parquet_files = {
        '/app/data/stock_data/daily_prices_20250824.parquet': 'historical_prices',
        '/app/data/fred_data/economic_indicators_20250824.parquet': 'economic_indicators', 
        '/app/data/alpha_vantage_data/fundamentals_20250824.parquet': 'fundamental_data',
        '/app/data/edgar_data/sec_filings_20250824.parquet': 'sec_filings'
    }
    
    total_loaded = 0
    
    for parquet_file, table_name in parquet_files.items():
        if os.path.exists(parquet_file):
            print(f"Loading {parquet_file} to {table_name}...")
            
            try:
                df = pd.read_parquet(parquet_file)
                print(f"  Read {len(df):,} records from parquet")
                
                # Clean data for database loading
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                if 'filing_date' in df.columns:
                    df['filing_date'] = pd.to_datetime(df['filing_date']).dt.date
                
                # Load to database
                df.to_sql(table_name, engine, if_exists='append', index=False, method='multi')
                
                print(f"  ✅ Loaded {len(df):,} records to {table_name}")
                total_loaded += len(df)
                
            except Exception as e:
                print(f"  ❌ Error loading {parquet_file}: {e}")
        else:
            print(f"  ⚠️ File not found: {parquet_file}")
    
    print(f"\n🎉 Loading complete! Total records loaded: {total_loaded:,}")
    
    # Verify data
    print("\n📊 Verification:")
    with engine.connect() as conn:
        for table_name in ['historical_prices', 'economic_indicators', 'fundamental_data', 'sec_filings']:
            try:
                count = conn.execute(text(f'SELECT COUNT(*) as count FROM {table_name}')).fetchone()
                print(f"  {table_name}: {count[0]:,} records")
            except Exception as e:
                print(f"  {table_name}: Error - {e}")

if __name__ == "__main__":
    load_parquet_files()
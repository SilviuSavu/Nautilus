#!/usr/bin/env python3
"""
Simple script to load parquet data to database
"""

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = 'postgresql://nautilus:nautilus123@postgres:5432/nautilus'

def load_data():
    engine = create_engine(DB_URL)
    
    print("🔄 Loading parquet data to database...")
    
    # Create simple table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(15, 6),
                high DECIMAL(15, 6),
                low DECIMAL(15, 6),
                close DECIMAL(15, 6),
                adj_close DECIMAL(15, 6),
                volume BIGINT,
                source VARCHAR(50),
                created_at TIMESTAMP
            )
        """))
        conn.commit()
    
    # Load stock data
    print("📈 Loading stock data...")
    df = pd.read_parquet('/app/data/stock_data/daily_prices_20250824.parquet')
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.to_sql('historical_prices', engine, if_exists='append', index=False)
    print(f"  ✅ {len(df):,} stock records loaded")
    
    # Create economic data table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id SERIAL PRIMARY KEY,
                series_id VARCHAR(50),
                description TEXT,
                date DATE,
                value DECIMAL(15, 6),
                source VARCHAR(50),
                created_at TIMESTAMP
            )
        """))
        conn.commit()
    
    # Load FRED data
    print("🏛️ Loading FRED economic data...")
    fred_df = pd.read_parquet('/app/data/fred_data/economic_indicators_20250824.parquet')
    fred_df['created_at'] = pd.to_datetime(fred_df['created_at'])
    fred_df.to_sql('economic_indicators', engine, if_exists='append', index=False)
    print(f"  ✅ {len(fred_df):,} FRED records loaded")
    
    # Create fundamentals table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                name TEXT,
                sector VARCHAR(100),
                industry VARCHAR(100),
                market_cap VARCHAR(50),
                pe_ratio VARCHAR(20),
                eps VARCHAR(20),
                beta VARCHAR(20),
                source VARCHAR(50),
                created_at TIMESTAMP
            )
        """))
        conn.commit()
    
    # Load Alpha Vantage data
    print("📊 Loading Alpha Vantage fundamental data...")
    av_df = pd.read_parquet('/app/data/alpha_vantage_data/fundamentals_20250824.parquet')
    av_df['created_at'] = pd.to_datetime(av_df['created_at'])
    av_df.to_sql('fundamental_data', engine, if_exists='append', index=False)
    print(f"  ✅ {len(av_df):,} fundamental records loaded")
    
    # Create SEC filings table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20),
                company_name TEXT,
                cik VARCHAR(20),
                form_type VARCHAR(20),
                filing_date DATE,
                accession_number VARCHAR(50),
                source VARCHAR(50),
                created_at TIMESTAMP
            )
        """))
        conn.commit()
    
    # Load EDGAR data
    print("🏢 Loading EDGAR SEC filing data...")
    edgar_df = pd.read_parquet('/app/data/edgar_data/sec_filings_20250824.parquet')
    edgar_df['created_at'] = pd.to_datetime(edgar_df['created_at'])
    edgar_df.to_sql('sec_filings', engine, if_exists='append', index=False)
    print(f"  ✅ {len(edgar_df):,} SEC filing records loaded")
    
    # Summary
    print("\n🎉 Data loading complete!")
    with engine.connect() as conn:
        for table in ['historical_prices', 'economic_indicators', 'fundamental_data', 'sec_filings']:
            count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).fetchone()
            print(f"  {table}: {count[0]:,} records")

if __name__ == "__main__":
    load_data()
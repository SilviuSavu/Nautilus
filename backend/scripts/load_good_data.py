#!/usr/bin/env python3
"""
Load the successfully downloaded institutional data
"""

import pandas as pd
from sqlalchemy import create_engine, text
import yfinance as yf
from datetime import datetime

DB_URL = 'postgresql://nautilus:nautilus123@postgres:5432/nautilus'

def load_institutional_data():
    engine = create_engine(DB_URL)
    
    print("🔄 Loading institutional data to database...")
    
    # 1. Load FRED Economic Data
    print("🏛️ Loading FRED economic data...")
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
    
    fred_df = pd.read_parquet('/app/data/fred_data/economic_indicators_20250824.parquet')
    fred_df.to_sql('economic_indicators', engine, if_exists='append', index=False)
    print(f"  ✅ {len(fred_df):,} FRED economic indicators loaded")
    
    # 2. Load Alpha Vantage Fundamental Data
    print("📊 Loading Alpha Vantage fundamental data...")
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
                peg_ratio VARCHAR(20),
                price_to_book VARCHAR(20),
                dividend_yield VARCHAR(20),
                eps VARCHAR(20),
                beta VARCHAR(20),
                profit_margin VARCHAR(20),
                source VARCHAR(50),
                created_at TIMESTAMP
            )
        """))
        conn.commit()
    
    av_df = pd.read_parquet('/app/data/alpha_vantage_data/fundamentals_20250824.parquet')
    av_df.to_sql('fundamental_data', engine, if_exists='append', index=False)
    print(f"  ✅ {len(av_df):,} fundamental data records loaded")
    
    # 3. Load EDGAR SEC Filing Data
    print("🏢 Loading EDGAR SEC filing data...")
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
    
    edgar_df = pd.read_parquet('/app/data/edgar_data/sec_filings_20250824.parquet')
    edgar_df.to_sql('sec_filings', engine, if_exists='append', index=False)
    print(f"  ✅ {len(edgar_df):,} SEC filing records loaded")
    
    # 4. Load Some Basic Stock Data
    print("📈 Loading basic stock data...")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                date DATE,
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
    
    # Download a few stocks directly
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    stock_records = []
    
    for symbol in symbols:
        try:
            print(f"  Downloading {symbol}...")
            data = yf.download(symbol, period='1y', interval='1d', progress=False)
            
            for date, row in data.iterrows():
                stock_records.append({
                    'symbol': symbol,
                    'date': date.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'adj_close': float(row['Adj Close']),
                    'volume': int(row['Volume']),
                    'source': 'yfinance_direct',
                    'created_at': datetime.now()
                })
        except Exception as e:
            print(f"    ⚠️ Error with {symbol}: {e}")
    
    if stock_records:
        stock_df = pd.DataFrame(stock_records)
        stock_df.to_sql('historical_prices', engine, if_exists='append', index=False)
        print(f"  ✅ {len(stock_df):,} stock price records loaded")
    
    # Create indexes
    print("🔧 Creating indexes...")
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_date ON historical_prices (symbol, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_economic_indicators_series_date ON economic_indicators (series_id, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fundamental_data_symbol ON fundamental_data (symbol)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings (ticker)"))
        conn.commit()
    
    # Summary
    print("\n🎉 Institutional data loading complete!")
    with engine.connect() as conn:
        tables = ['historical_prices', 'economic_indicators', 'fundamental_data', 'sec_filings']
        for table in tables:
            try:
                count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).fetchone()
                print(f"  {table}: {count[0]:,} records")
            except:
                print(f"  {table}: Table not found")
    
    print("\n📊 Sample data preview:")
    with engine.connect() as conn:
        # FRED sample
        result = conn.execute(text("SELECT series_id, description, value FROM economic_indicators WHERE series_id = 'FEDFUNDS' ORDER BY date DESC LIMIT 1")).fetchone()
        if result:
            print(f"  Latest Fed Funds Rate: {result[2]}% ({result[1]})")
        
        # Stock sample
        result = conn.execute(text("SELECT symbol, date, close FROM historical_prices ORDER BY date DESC LIMIT 1")).fetchone()
        if result:
            print(f"  Latest Stock Price: {result[0]} = ${result[2]:.2f} on {result[1]}")
        
        # Fundamentals sample
        result = conn.execute(text("SELECT symbol, name, pe_ratio FROM fundamental_data LIMIT 1")).fetchone()
        if result:
            print(f"  Sample Fundamental: {result[1]} ({result[0]}) P/E: {result[2]}")

if __name__ == "__main__":
    load_institutional_data()
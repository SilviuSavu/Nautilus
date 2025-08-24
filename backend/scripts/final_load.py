#!/usr/bin/env python3
"""
Final data loader - load what we can
"""

import pandas as pd
from sqlalchemy import create_engine, text
import yfinance as yf
from datetime import datetime

DB_URL = 'postgresql://nautilus:nautilus123@postgres:5432/nautilus'

def main():
    engine = create_engine(DB_URL)
    
    print("🎉 Loading institutional data for testing...")
    
    # 1. Load just the basic stock data
    print("📈 Getting fresh stock data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    # Create simple stock table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10),
                date DATE,
                open DECIMAL(10, 2),
                high DECIMAL(10, 2),
                low DECIMAL(10, 2),
                close DECIMAL(10, 2),
                volume BIGINT,
                source VARCHAR(20)
            )
        """))
        conn.commit()
    
    # Load stock data
    stock_data = []
    for symbol in symbols:
        try:
            print(f"  Loading {symbol}...")
            data = yf.download(symbol, period='6mo', progress=False)
            for date, row in data.iterrows():
                stock_data.append({
                    'symbol': symbol,
                    'date': date.date(),
                    'open': round(float(row['Open']), 2),
                    'high': round(float(row['High']), 2),
                    'low': round(float(row['Low']), 2),
                    'close': round(float(row['Close']), 2),
                    'volume': int(row['Volume']),
                    'source': 'yfinance'
                })
        except Exception as e:
            print(f"    ⚠️ Error with {symbol}: {e}")
    
    if stock_data:
        stock_df = pd.DataFrame(stock_data)
        # Use replace to handle duplicates
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM historical_prices WHERE source = 'yfinance'"))
            conn.commit()
        
        stock_df.to_sql('historical_prices', engine, if_exists='append', index=False)
        print(f"  ✅ Loaded {len(stock_df):,} stock records")
    
    # 2. Check if FRED data is there, if not load it
    with engine.connect() as conn:
        try:
            count = conn.execute(text("SELECT COUNT(*) FROM economic_indicators")).fetchone()
            print(f"📊 Economic indicators already loaded: {count[0]:,} records")
        except:
            print("📊 Loading FRED data from parquet...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id SERIAL PRIMARY KEY,
                    series_id VARCHAR(20),
                    description TEXT,
                    date DATE,
                    value DECIMAL(15, 6),
                    source VARCHAR(20)
                )
            """))
            conn.commit()
            
            fred_df = pd.read_parquet('/app/data/fred_data/economic_indicators_20250824.parquet')
            # Select only essential columns
            fred_df = fred_df[['series_id', 'description', 'date', 'value', 'source']]
            fred_df.to_sql('economic_indicators', engine, if_exists='append', index=False)
            print(f"  ✅ Loaded {len(fred_df):,} FRED records")
    
    # 3. Load simplified Alpha Vantage data
    print("💼 Loading fundamental data...")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10),
                name TEXT,
                sector VARCHAR(50),
                market_cap VARCHAR(20),
                pe_ratio VARCHAR(10),
                beta VARCHAR(10),
                source VARCHAR(20)
            )
        """))
        conn.commit()
    
    # Load and simplify Alpha Vantage data
    try:
        av_df = pd.read_parquet('/app/data/alpha_vantage_data/fundamentals_20250824.parquet')
        av_simple = av_df[['symbol', 'name', 'sector', 'market_cap', 'pe_ratio', 'beta', 'source']]
        av_simple.to_sql('fundamental_data', engine, if_exists='append', index=False)
        print(f"  ✅ Loaded {len(av_simple):,} fundamental records")
    except Exception as e:
        print(f"  ⚠️ Alpha Vantage data error: {e}")
    
    # 4. Load SEC filings
    print("🏢 Loading SEC filings...")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                company_name TEXT,
                form_type VARCHAR(10),
                filing_date DATE,
                source VARCHAR(20)
            )
        """))
        conn.commit()
    
    try:
        edgar_df = pd.read_parquet('/app/data/edgar_data/sec_filings_20250824.parquet')
        edgar_simple = edgar_df[['ticker', 'company_name', 'form_type', 'filing_date', 'source']]
        edgar_simple.to_sql('sec_filings', engine, if_exists='append', index=False)
        print(f"  ✅ Loaded {len(edgar_simple):,} SEC filing records")
    except Exception as e:
        print(f"  ⚠️ SEC data error: {e}")
    
    # Summary
    print("\n🎯 Data loading complete! Summary:")
    with engine.connect() as conn:
        tables = ['historical_prices', 'economic_indicators', 'fundamental_data', 'sec_filings']
        for table in tables:
            try:
                count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).fetchone()
                print(f"  📊 {table}: {count[0]:,} records")
            except Exception as e:
                print(f"  ❌ {table}: {e}")
    
    print("\n🚀 Ready for Toraniko factor testing!")
    print("💡 Try: SELECT symbol, date, close FROM historical_prices ORDER BY date DESC LIMIT 10;")

if __name__ == "__main__":
    main()
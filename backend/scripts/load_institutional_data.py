#!/usr/bin/env python3
"""
Institutional Data Loader for Nautilus Platform
Uses configured data sources: FRED, EDGAR, Alpha Vantage, Yahoo Finance
Saves parquet files in project data directory
"""

import os
import sys
import asyncio
import aiohttp
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import json
import time

# Project data directory
DATA_DIR = "/app/data"
DB_URL = os.getenv('DATABASE_URL', 'postgresql://nautilus:nautilus123@postgres:5432/nautilus')

# API Keys from your configuration
FRED_API_KEY = "1f1ba9c949e988e12796b7c1f6cce1bf"
ALPHA_VANTAGE_API_KEY = "271AHP91HVAPDRGP"

def ensure_data_dir():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{DATA_DIR}/stock_data", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/fred_data", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/edgar_data", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/alpha_vantage_data", exist_ok=True)

def load_stock_data():
    """Load basic stock data using Yahoo Finance"""
    print("📈 Loading stock market data...")
    
    # Institutional-grade symbols (most liquid, large cap)
    symbols = [
        # Tech giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 'CRM', 'ADBE',
        # Financial sector
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'V', 'MA', 'AXP',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
        # Consumer/Industrial
        'PG', 'KO', 'WMT', 'HD', 'DIS', 'MCD', 'NKE', 'COST', 'LOW', 'TGT',
        # Energy & Materials
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'BHP', 'FCX', 'NEM', 'AA'
    ]
    
    print(f"Downloading 3 years of data for {len(symbols)} institutional symbols...")
    
    # Download daily data
    daily_data = yf.download(symbols, period='3y', interval='1d', group_by='ticker', threads=True)
    
    # Process daily data
    daily_records = []
    for symbol in symbols:
        try:
            symbol_data = daily_data[symbol].dropna()
            for date, row in symbol_data.iterrows():
                daily_records.append({
                    'symbol': symbol,
                    'date': date.date(),
                    'timeframe': 'daily',
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
            print(f"⚠️ Error with {symbol}: {e}")
    
    daily_df = pd.DataFrame(daily_records)
    
    # Save daily data parquet
    daily_file = f"{DATA_DIR}/stock_data/daily_prices_{datetime.now().strftime('%Y%m%d')}.parquet"
    daily_df.to_parquet(daily_file, compression='snappy', index=False)
    print(f"💾 Saved daily data: {daily_file} ({len(daily_df):,} records)")
    
    # Download hourly data for last 2 months (smaller set)
    print("Downloading hourly data for top 20 symbols...")
    hourly_data = yf.download(symbols[:20], period='60d', interval='1h', group_by='ticker', threads=True)
    
    # Process hourly data
    hourly_records = []
    for symbol in symbols[:20]:
        try:
            symbol_data = hourly_data[symbol].dropna()
            for date, row in symbol_data.iterrows():
                hourly_records.append({
                    'symbol': symbol,
                    'datetime': date,
                    'timeframe': 'hourly',
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
            print(f"⚠️ Error with hourly {symbol}: {e}")
    
    hourly_df = pd.DataFrame(hourly_records)
    
    if not hourly_df.empty:
        hourly_file = f"{DATA_DIR}/stock_data/hourly_prices_{datetime.now().strftime('%Y%m%d')}.parquet"
        hourly_df.to_parquet(hourly_file, compression='snappy', index=False)
        print(f"💾 Saved hourly data: {hourly_file} ({len(hourly_df):,} records)")
    
    return daily_df, hourly_df

def load_fred_data():
    """Load FRED economic data using your configured API key"""
    print("🏛️ Loading FRED institutional economic data...")
    
    # Key economic indicators for institutional trading
    fred_series = {
        # GDP and Growth
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real GDP',
        'GDPPOT': 'Real Potential GDP',
        'INDPRO': 'Industrial Production Index',
        'PAYEMS': 'Nonfarm Payrolls',
        'UNRATE': 'Unemployment Rate',
        
        # Inflation and Monetary Policy
        'CPIAUCSL': 'Consumer Price Index',
        'CPILFESL': 'Core CPI',
        'FEDFUNDS': 'Federal Funds Rate',
        'DFF': 'Daily Federal Funds Rate',
        'DGS2': '2-Year Treasury Rate',
        'DGS5': '5-Year Treasury Rate',
        'DGS10': '10-Year Treasury Rate',
        'DGS30': '30-Year Treasury Rate',
        
        # Financial Markets
        'DEXUSEU': 'USD/EUR Exchange Rate',
        'VIXCLS': 'VIX Volatility Index',
        'DCOILWTICO': 'WTI Oil Price',
        'GOLDAMGBD228NLBM': 'Gold Price',
        
        # Money Supply and Credit
        'M2SL': 'Money Supply M2',
        'BOGMBASE': 'Monetary Base',
        'TOTALSA': 'Total Consumer Credit',
        
        # Labor Market
        'CIVPART': 'Labor Force Participation Rate',
        'AHETPI': 'Average Hourly Earnings',
        'ICSA': 'Initial Jobless Claims',
        'JTSJOL': 'Job Openings'
    }
    
    fred_records = []
    
    for series_id, description in fred_series.items():
        try:
            print(f"  Fetching {series_id}: {description}")
            
            # Build FRED API URL
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'start_date': '2020-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'observations' in data:
                    for obs in data['observations']:
                        if obs['value'] != '.' and obs['value'] is not None:
                            fred_records.append({
                                'series_id': series_id,
                                'description': description,
                                'date': obs['date'],
                                'value': float(obs['value']),
                                'source': 'fred',
                                'created_at': datetime.now()
                            })
                    
                    print(f"    ✅ {len([obs for obs in data['observations'] if obs['value'] != '.'])} observations")
            else:
                print(f"    ❌ Error: {response.status_code}")
            
            # Respect FRED rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    ⚠️ Error fetching {series_id}: {e}")
    
    fred_df = pd.DataFrame(fred_records)
    
    if not fred_df.empty:
        fred_file = f"{DATA_DIR}/fred_data/economic_indicators_{datetime.now().strftime('%Y%m%d')}.parquet"
        fred_df.to_parquet(fred_file, compression='snappy', index=False)
        print(f"💾 Saved FRED data: {fred_file} ({len(fred_df):,} records)")
    
    return fred_df

def load_alpha_vantage_data():
    """Load Alpha Vantage fundamental data using your configured API key"""
    print("📊 Loading Alpha Vantage institutional data...")
    
    # Major symbols for fundamental data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    av_records = []
    
    for symbol in symbols:
        try:
            print(f"  Fetching fundamental data for {symbol}")
            
            # Get company overview
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Symbol' in data and data['Symbol'] == symbol:
                    av_records.append({
                        'symbol': symbol,
                        'name': data.get('Name', ''),
                        'sector': data.get('Sector', ''),
                        'industry': data.get('Industry', ''),
                        'market_cap': data.get('MarketCapitalization', ''),
                        'pe_ratio': data.get('PERatio', ''),
                        'peg_ratio': data.get('PEGRatio', ''),
                        'price_to_book': data.get('PriceToBookRatio', ''),
                        'dividend_yield': data.get('DividendYield', ''),
                        'eps': data.get('EPS', ''),
                        'revenue_per_share': data.get('RevenuePerShareTTM', ''),
                        'profit_margin': data.get('ProfitMargin', ''),
                        'operating_margin': data.get('OperatingMarginTTM', ''),
                        'return_on_assets': data.get('ReturnOnAssetsTTM', ''),
                        'return_on_equity': data.get('ReturnOnEquityTTM', ''),
                        'revenue_ttm': data.get('RevenueTTM', ''),
                        'gross_profit_ttm': data.get('GrossProfitTTM', ''),
                        'beta': data.get('Beta', ''),
                        '52_week_high': data.get('52WeekHigh', ''),
                        '52_week_low': data.get('52WeekLow', ''),
                        'source': 'alpha_vantage',
                        'created_at': datetime.now()
                    })
                    print(f"    ✅ Got fundamental data for {symbol}")
                else:
                    print(f"    ❌ No data returned for {symbol}")
            else:
                print(f"    ❌ API Error: {response.status_code}")
            
            # Respect Alpha Vantage rate limits (5 per minute)
            time.sleep(12)
            
        except Exception as e:
            print(f"    ⚠️ Error fetching {symbol}: {e}")
    
    av_df = pd.DataFrame(av_records)
    
    if not av_df.empty:
        av_file = f"{DATA_DIR}/alpha_vantage_data/fundamentals_{datetime.now().strftime('%Y%m%d')}.parquet"
        av_df.to_parquet(av_file, compression='snappy', index=False)
        print(f"💾 Saved Alpha Vantage data: {av_file} ({len(av_df):,} records)")
    
    return av_df

def load_edgar_data():
    """Load EDGAR data using your configured endpoints"""
    print("🏢 Loading EDGAR institutional filing data...")
    
    edgar_records = []
    
    # Major companies for SEC data
    companies = [
        ('AAPL', 'Apple Inc'),
        ('MSFT', 'Microsoft Corporation'),
        ('GOOGL', 'Alphabet Inc'),
        ('AMZN', 'Amazon.com Inc'),
        ('TSLA', 'Tesla Inc'),
        ('META', 'Meta Platforms Inc'),
        ('JPM', 'JPMorgan Chase & Co'),
        ('JNJ', 'Johnson & Johnson'),
        ('V', 'Visa Inc'),
        ('PG', 'Procter & Gamble Company')
    ]
    
    for ticker, company_name in companies:
        try:
            print(f"  Fetching EDGAR data for {ticker}")
            
            # Use EDGAR API to get recent filings
            headers = {'User-Agent': 'Nautilus Trading Platform (noreply@nautilus.com)'}
            
            # Get company CIK first
            cik_url = f"https://www.sec.gov/files/company_tickers.json"
            cik_response = requests.get(cik_url, headers=headers)
            
            if cik_response.status_code == 200:
                companies_data = cik_response.json()
                
                # Find CIK for ticker
                cik = None
                for key, company_info in companies_data.items():
                    if company_info.get('ticker', '').upper() == ticker:
                        cik = str(company_info['cik_str']).zfill(10)
                        break
                
                if cik:
                    print(f"    Found CIK {cik} for {ticker}")
                    
                    # Get recent submissions
                    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                    sub_response = requests.get(submissions_url, headers=headers)
                    
                    if sub_response.status_code == 200:
                        submissions = sub_response.json()
                        
                        # Get recent filings
                        filings = submissions.get('filings', {}).get('recent', {})
                        forms = filings.get('form', [])
                        dates = filings.get('filingDate', [])
                        accession_numbers = filings.get('accessionNumber', [])
                        
                        for i, (form, date, accession) in enumerate(zip(forms[:10], dates[:10], accession_numbers[:10])):
                            edgar_records.append({
                                'ticker': ticker,
                                'company_name': company_name,
                                'cik': cik,
                                'form_type': form,
                                'filing_date': date,
                                'accession_number': accession,
                                'source': 'edgar',
                                'created_at': datetime.now()
                            })
                        
                        print(f"    ✅ Got {len(forms[:10])} recent filings")
                    else:
                        print(f"    ❌ Submissions API Error: {sub_response.status_code}")
                else:
                    print(f"    ❌ CIK not found for {ticker}")
            else:
                print(f"    ❌ CIK API Error: {cik_response.status_code}")
            
            # Respect SEC rate limits
            time.sleep(0.2)
            
        except Exception as e:
            print(f"    ⚠️ Error fetching {ticker}: {e}")
    
    edgar_df = pd.DataFrame(edgar_records)
    
    if not edgar_df.empty:
        edgar_file = f"{DATA_DIR}/edgar_data/sec_filings_{datetime.now().strftime('%Y%m%d')}.parquet"
        edgar_df.to_parquet(edgar_file, compression='snappy', index=False)
        print(f"💾 Saved EDGAR data: {edgar_file} ({len(edgar_df):,} records)")
    
    return edgar_df

def load_to_database(dataframes):
    """Load all data to database"""
    print("🔄 Loading data to PostgreSQL database...")
    
    engine = create_engine(DB_URL)
    
    # Create tables
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
    
    -- Hourly prices table
    CREATE TABLE IF NOT EXISTS hourly_prices (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        datetime TIMESTAMP WITH TIME ZONE NOT NULL,
        timeframe VARCHAR(20) DEFAULT 'hourly',
        open DECIMAL(15, 6),
        high DECIMAL(15, 6),
        low DECIMAL(15, 6),
        close DECIMAL(15, 6) NOT NULL,
        adj_close DECIMAL(15, 6),
        volume BIGINT DEFAULT 0,
        source VARCHAR(50) DEFAULT 'unknown',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(symbol, datetime, source)
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
    CREATE INDEX IF NOT EXISTS idx_hourly_prices_symbol_datetime ON hourly_prices (symbol, datetime);
    CREATE INDEX IF NOT EXISTS idx_economic_indicators_series_date ON economic_indicators (series_id, date);
    CREATE INDEX IF NOT EXISTS idx_fundamental_data_symbol ON fundamental_data (symbol);
    CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings (ticker);
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_tables_sql))
        conn.commit()
    
    # Load each dataframe
    total_loaded = 0
    
    for table_name, df in dataframes.items():
        if not df.empty:
            try:
                df.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"  ✅ Loaded {len(df):,} records to {table_name}")
                total_loaded += len(df)
            except Exception as e:
                print(f"  ❌ Error loading {table_name}: {e}")
    
    print(f"🎉 Database loading complete! Total: {total_loaded:,} records")

def main():
    """Main execution function"""
    print("🚀 Starting institutional data loading...")
    print(f"📁 Data will be saved to: {DATA_DIR}")
    
    ensure_data_dir()
    
    # Load all data sources
    daily_df, hourly_df = load_stock_data()
    fred_df = load_fred_data()
    av_df = load_alpha_vantage_data()
    edgar_df = load_edgar_data()
    
    # Prepare for database loading
    dataframes = {
        'historical_prices': daily_df,
        'hourly_prices': hourly_df,
        'economic_indicators': fred_df,
        'fundamental_data': av_df,
        'sec_filings': edgar_df
    }
    
    # Load to database
    load_to_database(dataframes)
    
    print("\n📊 Data Loading Summary:")
    print(f"  Stock Data (Daily): {len(daily_df):,} records")
    print(f"  Stock Data (Hourly): {len(hourly_df):,} records") 
    print(f"  FRED Economic: {len(fred_df):,} records")
    print(f"  Alpha Vantage Fundamentals: {len(av_df):,} records")
    print(f"  EDGAR SEC Filings: {len(edgar_df):,} records")
    print(f"\n🎯 Ready for factor analysis and strategy testing!")

if __name__ == "__main__":
    main()
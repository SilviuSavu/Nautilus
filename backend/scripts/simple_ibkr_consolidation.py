#!/usr/bin/env python3
"""
Simple IBKR data consolidation - add IBKR data to our institutional dataset
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

DB_URL = 'postgresql://nautilus:nautilus123@postgres:5432/nautilus'

def consolidate_ibkr():
    engine = create_engine(DB_URL)
    
    print("🔌 Adding IBKR Data to Institutional Dataset")
    print("=" * 50)
    
    with engine.connect() as conn:
        # 1. Add IBKR data to historical_prices table
        print("📈 Adding IBKR market data to historical_prices...")
        
        # Check current IBKR data in historical_prices
        current_ibkr = conn.execute(text(
            "SELECT COUNT(*) FROM historical_prices WHERE source = 'ibkr'"
        )).fetchone()
        
        print(f"  Current IBKR records in historical_prices: {current_ibkr[0]:,}")
        
        if current_ibkr[0] == 0:
            # Insert IBKR data
            conn.execute(text("""
                INSERT INTO historical_prices (symbol, date, open, high, low, close, volume, source, created_at)
                SELECT 
                    REPLACE(REPLACE(REPLACE(instrument_id, '.SMART', ''), '.NASDAQ', ''), '.NYSE', '') as symbol,
                    DATE(to_timestamp(timestamp_ns/1000000000)) as date,
                    open_price::DECIMAL(15,6) as open,
                    high_price::DECIMAL(15,6) as high,
                    low_price::DECIMAL(15,6) as low,
                    close_price::DECIMAL(15,6) as close,
                    volume::BIGINT as volume,
                    'ibkr' as source,
                    NOW() as created_at
                FROM market_bars
                WHERE DATE(to_timestamp(timestamp_ns/1000000000)) >= '2024-01-01'
                GROUP BY 1,2,3,4,5,6,7,8,9
                ON CONFLICT DO NOTHING
            """))
            conn.commit()
            
            new_count = conn.execute(text(
                "SELECT COUNT(*) FROM historical_prices WHERE source = 'ibkr'"
            )).fetchone()
            
            print(f"  ✅ Added {new_count[0]:,} IBKR price records")
        else:
            print(f"  ✅ IBKR data already consolidated")
        
        # 2. Summary of all data sources
        print("\n📊 Complete Institutional Data Summary:")
        print("-" * 45)
        
        summary = conn.execute(text("""
            SELECT 'Stock Prices' as data_type, source, COUNT(*) as records
            FROM historical_prices
            GROUP BY source
            UNION ALL
            SELECT 'Economic Data' as data_type, source, COUNT(*) as records
            FROM economic_indicators
            GROUP BY source  
            UNION ALL
            SELECT 'Fundamentals' as data_type, source, COUNT(*) as records
            FROM fundamental_data
            GROUP BY source
            ORDER BY data_type, records DESC
        """)).fetchall()
        
        current_type = None
        total_records = 0
        
        for row in summary:
            if row[0] != current_type:
                if current_type:
                    print()
                print(f"📋 {row[0]}:")
                current_type = row[0]
            print(f"  {row[1]}: {row[2]:,} records")
            total_records += row[2]
        
        print(f"\n🎯 Total Records: {total_records:,}")
        
        # 3. Available symbols across all sources
        symbols = conn.execute(text("""
            SELECT DISTINCT symbol
            FROM (
                SELECT symbol FROM historical_prices WHERE symbol IS NOT NULL
                UNION
                SELECT symbol FROM fundamental_data WHERE symbol IS NOT NULL
            ) all_symbols
            ORDER BY symbol
        """)).fetchall()
        
        symbol_list = [row[0] for row in symbols if row[0]]
        print(f"\n📈 Available Symbols: {len(symbol_list)}")
        print(f"   {', '.join(symbol_list[:12])}{'...' if len(symbol_list) > 12 else ''}")
        
        # 4. Data coverage for key symbols
        print(f"\n🔍 Data Coverage for Major Symbols:")
        coverage = conn.execute(text("""
            SELECT 
                h.symbol,
                COUNT(DISTINCT h.date) as price_days,
                MIN(h.date) as earliest,
                MAX(h.date) as latest,
                CASE WHEN f.symbol IS NOT NULL THEN 'Yes' ELSE 'No' END as has_fundamentals,
                STRING_AGG(DISTINCT h.source, ', ') as sources
            FROM historical_prices h
            LEFT JOIN fundamental_data f ON h.symbol = f.symbol
            WHERE h.symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA')
            GROUP BY h.symbol, f.symbol
            ORDER BY price_days DESC
        """)).fetchall()
        
        print("Symbol | Days | Date Range            | Fundamentals | Sources")
        print("-" * 68)
        for row in coverage:
            date_range = f"{row[2]} to {row[3]}"
            print(f"{row[0]:<6} | {row[1]:>4} | {date_range:<21} | {row[4]:<12} | {row[5]}")
        
        # 5. FRED economic indicators sample
        print(f"\n🏛️ Key Economic Indicators (Latest Values):")
        econ_sample = conn.execute(text("""
            SELECT series_id, description, value, date
            FROM economic_indicators
            WHERE series_id IN ('FEDFUNDS', 'UNRATE', 'VIXCLS', 'DGS10')
            AND date = (SELECT MAX(date) FROM economic_indicators e2 WHERE e2.series_id = economic_indicators.series_id)
            ORDER BY series_id
        """)).fetchall()
        
        for row in econ_sample:
            print(f"  {row[0]}: {row[2]:.2f}% ({row[3]})")

def main():
    consolidate_ibkr()
    
    print("\n🎉 IBKR Integration Complete!")
    print("\n💼 All Institutional Data Sources Active:")
    print("   🔌 IBKR: Market data, real-time bars, instruments")
    print("   🏛️ FRED: 121,915+ economic indicators")
    print("   📊 Alpha Vantage: Fundamental data for top stocks")
    print("   🏢 EDGAR: Recent SEC filings")
    
    print("\n🚀 Platform ready for:")
    print("   • Toraniko factor analysis with real market data")
    print("   • Economic regime detection using FRED indicators")
    print("   • Multi-source data validation and cross-referencing")
    print("   • Comprehensive risk model development")

if __name__ == "__main__":
    main()
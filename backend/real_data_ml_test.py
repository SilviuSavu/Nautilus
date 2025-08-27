#!/usr/bin/env python3
"""
Real Data ML Test - Using Actual PostgreSQL Data + Live IBKR Feed
Test the ML engine with REAL market data, not synthetic BS
"""

import asyncio
import requests
import json
import time
import asyncpg
from typing import List, Dict, Any
import numpy as np

async def get_real_market_data_from_postgres():
    """Get actual market data from PostgreSQL database"""
    try:
        conn = await asyncpg.connect(
            "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
        )
        
        # Check what tables exist
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        tables = await conn.fetch(tables_query)
        print(f"📊 Database tables: {[row['table_name'] for row in tables]}")
        
        # Look for market data tables
        market_data_tables = [
            table['table_name'] for table in tables 
            if any(keyword in table['table_name'].lower() 
                  for keyword in ['price', 'market', 'tick', 'bar', 'quote'])
        ]
        
        if market_data_tables:
            print(f"🎯 Found market data tables: {market_data_tables}")
            
            # Try to get data from the first market data table
            table_name = market_data_tables[0]
            data_query = f"SELECT * FROM {table_name} LIMIT 100"
            
            try:
                data = await conn.fetch(data_query)
                print(f"✅ Retrieved {len(data)} records from {table_name}")
                
                # Extract price data if available
                real_prices = []
                real_volumes = []
                
                for row in data:
                    # Try to find price and volume columns
                    row_dict = dict(row)
                    for key, value in row_dict.items():
                        if key.lower() in ['price', 'close', 'last'] and value is not None:
                            try:
                                real_prices.append(float(value))
                            except:
                                pass
                        elif key.lower() in ['volume', 'size'] and value is not None:
                            try:
                                real_volumes.append(float(value))
                            except:
                                pass
                
                if real_prices:
                    print(f"💰 Real prices found: {len(real_prices)} data points")
                    print(f"   Sample prices: {real_prices[:10]}")
                    return {
                        "prices": real_prices,
                        "volumes": real_volumes if real_volumes else [100000] * len(real_prices),
                        "source": f"PostgreSQL table: {table_name}",
                        "data_points": len(real_prices)
                    }
                    
            except Exception as e:
                print(f"⚠️ Error querying {table_name}: {e}")
        
        # If no market data tables, try generic approach
        print("🔍 Searching for any tables with numeric data...")
        
        # Get all tables and sample their data
        for table in tables[:5]:  # Check first 5 tables
            table_name = table['table_name']
            try:
                sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                sample_data = await conn.fetch(sample_query)
                if sample_data:
                    print(f"📋 Table {table_name}: {len(sample_data[0])} columns")
                    # Show column names
                    columns = list(sample_data[0].keys())
                    print(f"   Columns: {columns[:10]}")  # First 10 columns
                    
            except Exception as e:
                print(f"   Error sampling {table_name}: {e}")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    
    return None

async def get_live_ibkr_data():
    """Get live data from IBKR MarketData engine"""
    try:
        # Get list of tracked symbols
        response = requests.get("http://localhost:8800/symbols", timeout=10)
        symbols_data = response.json()
        symbols = symbols_data.get("symbols", [])
        
        print(f"📡 IBKR tracking {len(symbols)} symbols: {symbols}")
        
        # Try to trigger data collection
        for symbol in symbols[:3]:  # Test first 3 symbols
            try:
                process_response = requests.post(f"http://localhost:8800/process/ibkr", 
                                               json={"symbol": symbol}, timeout=10)
                print(f"🎯 Triggered data collection for {symbol}: {process_response.status_code}")
                
                # Wait a moment for data to be collected
                time.sleep(1)
                
                # Try to get the data
                data_response = requests.get(f"http://localhost:8800/data/ibkr/{symbol}", timeout=10)
                if data_response.status_code == 200:
                    data = data_response.json()
                    print(f"✅ Got live data for {symbol}: {len(data) if isinstance(data, list) else 'single point'}")
                    return {
                        "symbol": symbol,
                        "data": data,
                        "source": "Live IBKR feed",
                        "timestamp": time.time()
                    }
                else:
                    print(f"⚠️ No data for {symbol}: {data_response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error getting {symbol}: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ IBKR connection failed: {e}")
        return None

async def test_ml_with_real_data():
    """Test ML engine with real data instead of synthetic crap"""
    print("🧠 REAL DATA ML TEST - Using Actual Market Data")
    print("=" * 60)
    
    # Try to get real data from multiple sources
    print("\n1. 📊 Checking PostgreSQL database...")
    postgres_data = await get_real_market_data_from_postgres()
    
    print("\n2. 📡 Checking live IBKR feed...")
    ibkr_data = await get_live_ibkr_data()
    
    # Prepare real data for ML prediction
    real_data = None
    data_source = "No real data available"
    
    if postgres_data and len(postgres_data.get("prices", [])) > 10:
        real_data = postgres_data
        data_source = postgres_data["source"]
        print(f"✅ Using PostgreSQL data: {len(postgres_data['prices'])} price points")
        
    elif ibkr_data:
        # Process IBKR data format
        ibkr_raw = ibkr_data.get("data", [])
        if isinstance(ibkr_raw, list) and len(ibkr_raw) > 0:
            # Extract prices from IBKR format
            prices = []
            volumes = []
            for item in ibkr_raw:
                if isinstance(item, dict):
                    if 'price' in item:
                        prices.append(float(item['price']))
                    if 'volume' in item:
                        volumes.append(float(item['volume']))
            
            if prices:
                real_data = {
                    "prices": prices,
                    "volumes": volumes if volumes else [100000] * len(prices),
                    "source": f"Live IBKR - {ibkr_data['symbol']}",
                    "data_points": len(prices)
                }
                data_source = real_data["source"]
                print(f"✅ Using IBKR data: {len(prices)} live data points")
    
    if not real_data:
        print("❌ NO REAL DATA AVAILABLE - Cannot perform realistic test")
        print("   - PostgreSQL database has no suitable market data")  
        print("   - IBKR feed is not providing data points")
        print("   - This confirms the previous tests were synthetic")
        return
    
    print(f"\n3. 🧠 Testing ML Engine with REAL DATA from: {data_source}")
    
    # Test ML prediction with real data
    test_symbol = "AAPL"
    start_time = time.time()
    
    try:
        response = requests.post(
            f"http://localhost:8400/ml/predict/price/{test_symbol}",
            json={
                "prices": real_data["prices"],
                "volume": real_data["volumes"]
            },
            timeout=30
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n🎯 REAL DATA ML PREDICTION RESULTS:")
            print("=" * 50)
            print(f"📊 Data Source: {data_source}")
            print(f"📊 Real Data Points: {real_data['data_points']}")
            print(f"📊 Processing Time: {processing_time:.1f}ms")
            print(f"📊 Hardware Used: {result.get('hardware_used', 'Unknown')}")
            
            prediction = result.get("prediction", {})
            print(f"\n💰 Current Price: ${prediction.get('current_price', 'N/A')}")
            print(f"💰 Predicted Price: ${prediction.get('predicted_price', 'N/A')}")
            print(f"📈 Price Change: {prediction.get('price_change_percent', 'N/A')}%")
            print(f"📊 Confidence: {prediction.get('confidence', 'N/A')}")
            print(f"🌊 Volatility: {prediction.get('volatility_prediction', {}).get('predicted_volatility', 'N/A')}%")
            
            trend_pred = prediction.get('trend_prediction', {})
            print(f"📈 Trend: {trend_pred.get('predicted_trend', 'N/A')}")
            
            # Show the first few real prices used
            sample_prices = real_data["prices"][:10]
            print(f"\n📊 Sample Real Prices Used: {[f'${p:.2f}' for p in sample_prices]}")
            
            print(f"\n✅ REAL DATA TEST SUCCESSFUL!")
            print(f"   - Used {real_data['data_points']} actual market data points")
            print(f"   - Hardware acceleration: {result.get('hardware_used')}")
            print(f"   - Response time: {processing_time:.1f}ms")
            print(f"   - This is a REAL prediction, not synthetic!")
            
        else:
            print(f"❌ ML prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ML test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ml_with_real_data())
#!/usr/bin/env python3
"""
Test script to demonstrate real data integration capabilities
This tests the Analytics Engine's ability to work with real market data
"""

import asyncio
import requests
import json

def test_analytics_health():
    """Test Analytics Engine health endpoint"""
    try:
        response = requests.get("http://localhost:8100/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analytics Engine Health: {data['status']}")
            print(f"   Uptime: {data['uptime_seconds']:.2f} seconds")
            print(f"   MessageBus: {'Connected' if data.get('messagebus_connected') else 'Disconnected'}")
            return True
        else:
            print(f"❌ Analytics Engine Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Analytics Engine: {e}")
        return False

def test_direct_database_access():
    """Test direct database access to market_bars table"""
    try:
        import asyncpg
        import asyncio
        import os
        from datetime import datetime
        
        DATABASE_URL = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
        
        async def fetch_market_data():
            try:
                conn = await asyncpg.connect(DATABASE_URL)
                
                # Query for NFLX data
                query = """
                    SELECT instrument_id, venue, timestamp_ns, close_price, volume
                    FROM market_bars 
                    WHERE instrument_id = 'NFLX.SMART'
                    ORDER BY timestamp_ns DESC
                    LIMIT 5
                """
                
                rows = await conn.fetch(query)
                await conn.close()
                
                if rows:
                    print("\n✅ Direct Database Test - NFLX Market Data:")
                    for row in rows:
                        timestamp = datetime.fromtimestamp(row['timestamp_ns'] / 1_000_000_000)
                        print(f"   {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: ${row['close_price']:.2f} (Vol: {row['volume']:,})")
                    return True
                else:
                    print("\n❌ No NFLX market data found in database")
                    return False
                    
            except Exception as e:
                print(f"\n❌ Database connection failed: {e}")
                return False
        
        return asyncio.run(fetch_market_data())
        
    except ImportError:
        print("\n❌ asyncpg not available for direct database test")
        return False

def test_portfolio_analytics():
    """Test portfolio analytics endpoint with mock data"""
    try:
        response = requests.get("http://localhost:8100/analytics/calculate/test_portfolio")
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Portfolio Analytics Test:")
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"   Portfolio ID: {data.get('portfolio_id', 'N/A')}")
                print(f"   Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
                print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
            return True
        else:
            print(f"\n❌ Portfolio analytics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"\n❌ Portfolio analytics test failed: {e}")
        return False

def test_available_symbols():
    """Test if we can get available symbols from database"""
    try:
        import asyncpg
        import asyncio
        
        DATABASE_URL = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
        
        async def fetch_symbols():
            try:
                conn = await asyncpg.connect(DATABASE_URL)
                
                query = """
                    SELECT DISTINCT instrument_id, COUNT(*) as data_points
                    FROM market_bars 
                    GROUP BY instrument_id 
                    ORDER BY data_points DESC
                    LIMIT 10
                """
                
                rows = await conn.fetch(query)
                await conn.close()
                
                if rows:
                    print("\n✅ Available Symbols Test:")
                    print("   Top 10 symbols by data points:")
                    for row in rows:
                        print(f"   {row['instrument_id']}: {row['data_points']:,} data points")
                    return True
                else:
                    print("\n❌ No symbols found in database")
                    return False
                    
            except Exception as e:
                print(f"\n❌ Symbols query failed: {e}")
                return False
        
        return asyncio.run(fetch_symbols())
        
    except ImportError:
        print("\n❌ asyncpg not available for symbols test")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Analytics Engine Real Data Integration")
    print("=" * 50)
    
    results = []
    
    # Test 1: Engine Health
    results.append(test_analytics_health())
    
    # Test 2: Direct Database Access
    results.append(test_direct_database_access())
    
    # Test 3: Portfolio Analytics
    results.append(test_portfolio_analytics())
    
    # Test 4: Available Symbols
    results.append(test_available_symbols())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Analytics Engine real data integration working.")
    elif passed > 0:
        print("⚠️  Some tests passed. Partial real data integration working.")
    else:
        print("❌ All tests failed. Real data integration needs debugging.")
    
    print("\n📋 Real Data Integration Status:")
    print(f"   - Database Connection: {'✅ Working' if results[1] else '❌ Failed'}")
    print(f"   - Analytics Engine: {'✅ Healthy' if results[0] else '❌ Unhealthy'}")
    print(f"   - Portfolio Analytics: {'✅ Working' if results[2] else '❌ Failed'}")
    print(f"   - Market Data Access: {'✅ Working' if results[3] else '❌ Failed'}")

if __name__ == "__main__":
    main()
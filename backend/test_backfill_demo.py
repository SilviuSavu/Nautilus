#!/usr/bin/env python3
"""
Historical Data Backfill Demonstration Script
This script demonstrates the comprehensive backfill system that pulls missing
historical data from IB Gateway into PostgreSQL for all timeframes.
"""

import asyncio
import logging
import requests
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Base URL
API_BASE = "http://localhost:8000"

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_market_data_api():
    """Test if market data API is working"""
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/market-data/historical/bars",
            params={"symbol": "AAPL", "timeframe": "1d"},
            timeout=5
        )
        data = response.json()
        return "candles" in data and isinstance(data["candles"], list)
    except Exception as e:
        logger.error(f"Market data API test failed: {e}")
        return False

def test_gap_analysis(symbol="TSLA"):
    """Test gap analysis for missing data"""
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/historical/analyze-gaps/{symbol}",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Gap analysis failed: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Gap analysis error: {e}")
        return None

def add_backfill_request(symbol, timeframes=None, days_back=30):
    """Add a backfill request"""
    try:
        payload = {
            "symbol": symbol,
            "timeframes": timeframes or ["1d", "1h"],
            "days_back": days_back,
            "priority": 1
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/historical/backfill/add",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Backfill request failed: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Backfill request error: {e}")
        return None

def start_backfill_process():
    """Start the backfill process"""
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/historical/backfill/start",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Start backfill failed: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Start backfill error: {e}")
        return None

def get_backfill_status():
    """Get backfill status"""
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/historical/backfill/status",
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Status check failed: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return None

def main():
    """Main demonstration function"""
    print("🚀 Historical Data Backfill System Demonstration")
    print("=" * 60)
    
    # 1. Check API Health
    print("\n1. 🔍 Checking API Health...")
    if not check_api_health():
        print("❌ API is not healthy. Please start the backend server.")
        return
    print("✅ API is healthy")
    
    # 2. Test Market Data API
    print("\n2. 📊 Testing Market Data API...")
    if test_market_data_api():
        print("✅ Market Data API is working")
    else:
        print("❌ Market Data API test failed")
    
    # 3. Demonstrate Gap Analysis
    print("\n3. 🔍 Analyzing Missing Data Gaps for TSLA...")
    gaps = test_gap_analysis("TSLA")
    if gaps:
        print(f"✅ Gap analysis completed for TSLA")
        print(f"   Total gaps found: {gaps.get('total_gaps', 0)}")
        
        if gaps.get('missing_data_gaps'):
            print("   Missing data gaps by timeframe:")
            for timeframe, gap_list in gaps['missing_data_gaps'].items():
                print(f"     {timeframe}: {len(gap_list)} gaps")
    else:
        print("❌ Gap analysis failed")
    
    # 4. Add Sample Backfill Requests
    print("\n4. 📈 Adding Sample Backfill Requests...")
    
    # Popular instruments to backfill
    instruments = [
        {"symbol": "AAPL", "timeframes": ["1d", "1h", "15m"], "days": 90},
        {"symbol": "GOOGL", "timeframes": ["1d", "1h"], "days": 60},
        {"symbol": "TSLA", "timeframes": ["1d", "4h", "1h"], "days": 120},
        {"symbol": "SPY", "timeframes": ["1d", "1h"], "days": 180}
    ]
    
    successful_requests = 0
    for instrument in instruments:
        result = add_backfill_request(
            instrument["symbol"], 
            instrument["timeframes"], 
            instrument["days"]
        )
        if result:
            print(f"   ✅ Added backfill request for {instrument['symbol']}")
            successful_requests += 1
        else:
            print(f"   ❌ Failed to add request for {instrument['symbol']}")
    
    print(f"   Successfully added {successful_requests}/{len(instruments)} backfill requests")
    
    # 5. Check Backfill Status
    print("\n5. 📊 Checking Backfill Status...")
    status = get_backfill_status()
    if status:
        backfill_status = status.get('backfill_status', {})
        print(f"   Queue size: {backfill_status.get('queue_size', 0)}")
        print(f"   Running: {backfill_status.get('is_running', False)}")
        print(f"   Active requests: {backfill_status.get('active_requests', 0)}")
        print(f"   Completed requests: {backfill_status.get('completed_requests', 0)}")
    
    # 6. Demonstrate Starting Backfill Process
    print("\n6. 🚀 Starting Priority Backfill Process...")
    print("   Note: This will start pulling data from IB Gateway to PostgreSQL")
    
    # Uncomment the line below to actually start the backfill process
    # start_result = start_backfill_process()
    print("   ⚠️  Backfill start is commented out for demo purposes")
    print("   To actually start backfill, uncomment the line in the script")
    
    # 7. Summary and Next Steps
    print("\n" + "=" * 60)
    print("🎯 BACKFILL SYSTEM SUMMARY")
    print("=" * 60)
    print("""
The Historical Data Backfill System provides:

✅ COMPLETE SOLUTION FEATURES:
1. 🔍 Gap Analysis - Identifies missing data for any instrument/timeframe
2. 📊 Intelligent Backfill - Pulls only missing data, not duplicates  
3. ⚡ Rate Limited - Respects IB Gateway limits (50 req/min)
4. 🎯 Priority Queue - Handles multiple instruments efficiently
5. 📈 All Timeframes - Supports 1m to 1M timeframes
6. 💾 PostgreSQL Storage - Stores in normalized format
7. 🔄 Real-time Status - Track progress and errors
8. 🛡️ Error Recovery - Handles IB Gateway timeouts gracefully

📋 AVAILABLE API ENDPOINTS:
• POST /api/v1/historical/backfill/start - Start priority instruments backfill
• POST /api/v1/historical/backfill/add - Add custom backfill request  
• GET  /api/v1/historical/backfill/status - Get backfill status
• POST /api/v1/historical/backfill/stop - Stop backfill process
• GET  /api/v1/historical/analyze-gaps/{symbol} - Analyze missing data

🎯 PRIORITY INSTRUMENTS INCLUDED:
• Major US Stocks: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX
• Major Forex: EUR/USD, GBP/USD, JPY/USD, AUD/USD
• Major ETFs: SPY, QQQ, IWM

🚀 TO START FULL BACKFILL:
   curl -X POST http://localhost:8000/api/v1/historical/backfill/start
   
This will automatically backfill all priority instruments with complete
historical data across all timeframes, solving the chart data issues permanently!
""")

if __name__ == "__main__":
    main()
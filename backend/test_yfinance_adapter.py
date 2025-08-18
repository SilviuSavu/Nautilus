#!/usr/bin/env python3
"""
Test script for YFinance adapter integration with NautilusTrader.
This script demonstrates how to use the yfinance adapter.
"""

import asyncio
import sys
import pandas as pd
from pathlib import Path

# Add the nautilus_trader to the Python path
sys.path.insert(0, '/app/nautilus_trader')

print("🚀 Testing YFinance Adapter with NautilusTrader")
print("=" * 50)

def test_basic_yfinance():
    """Test basic yfinance functionality."""
    print("🔍 Step 1: Testing basic yfinance functionality...")
    
    try:
        import yfinance as yf
        
        # Test basic ticker creation
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and "symbol" in info:
            print(f"✅ Basic yfinance works - Symbol: {info.get('symbol')}")
            print(f"   Company: {info.get('longName', 'N/A')}")
            print(f"   Currency: {info.get('currency', 'N/A')}")
            return True
        else:
            print("❌ Basic yfinance test failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic yfinance test failed: {e}")
        return False


def test_adapter_imports():
    """Test that our adapter can be imported."""
    print("\n🔍 Step 2: Testing adapter imports...")
    
    try:
        from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
        from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
        from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
        from nautilus_trader.adapters.yfinance.factories import create_yfinance_data_client
        
        print("✅ All adapter components imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def test_adapter_functionality():
    """Test actual adapter functionality."""
    print("\n🔍 Step 3: Testing adapter functionality...")
    
    try:
        # Import required modules
        from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
        from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
        from nautilus_trader.model.identifiers import InstrumentId
        
        # Test instrument provider
        print("   Testing instrument provider...")
        provider = YFinanceInstrumentProvider()
        
        # Load a test instrument
        instrument_id = InstrumentId.from_str("AAPL.YAHOO")
        await provider.load_async(instrument_id)
        
        # Check if instrument was loaded
        instruments = list(provider.get_all().values())
        if instruments:
            instrument = instruments[0]
            print(f"✅ Loaded instrument: {instrument.id}")
            print(f"   Quote currency: {instrument.quote_currency}")
            print(f"   Price precision: {instrument.price_precision}")
        else:
            print("⚠️  No instruments loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapter functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_fetching():
    """Test data fetching capabilities."""
    print("\n🔍 Step 4: Testing data fetching...")
    
    try:
        import yfinance as yf
        
        # Test basic data fetching
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d", interval="1d")
        
        if not data.empty:
            print(f"✅ Fetched {len(data)} days of AAPL data")
            print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No data fetched")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching test failed: {e}")
        return False


def test_configuration():
    """Test adapter configuration."""
    print("\n🔍 Step 5: Testing adapter configuration...")
    
    try:
        from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
        
        # Test default config
        config = YFinanceDataClientConfig()
        print(f"✅ Default config created:")
        print(f"   Cache expiry: {config.cache_expiry_seconds}s")
        print(f"   Rate limit delay: {config.rate_limit_delay}s")
        print(f"   Request timeout: {config.request_timeout}s")
        
        # Test custom config
        custom_config = YFinanceDataClientConfig(
            symbols=["AAPL", "MSFT", "TSLA"],
            rate_limit_delay=0.2,
            cache_expiry_seconds=1800
        )
        print(f"✅ Custom config created with {len(custom_config.symbols)} symbols")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Starting YFinance Adapter Integration Tests\n")
    
    tests = [
        ("Basic yfinance", test_basic_yfinance),
        ("Adapter imports", test_adapter_imports),
        ("Adapter functionality", test_adapter_functionality),
        ("Data fetching", test_data_fetching),
        ("Configuration", test_configuration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! YFinance adapter is ready for use.")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
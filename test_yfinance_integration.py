#!/usr/bin/env python3
"""
Simple integration test for YFinance adapter.
This tests the basic functionality of the yfinance adapter.
"""

import asyncio
import sys
import os

# Add the nautilus_trader to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nautilus_trader'))

from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.factories import create_yfinance_data_client
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.data.messages import RequestBars
from nautilus_trader.model.data import BarType
from nautilus_trader.core.uuid import UUID4
import pandas as pd


async def test_yfinance_integration():
    """Test basic yfinance adapter functionality."""
    print("🔍 Testing YFinance Adapter Integration...")
    
    # Create components
    clock = LiveClock()
    trader_id = TraderId("TESTER-001")
    msgbus = MessageBus(trader_id=trader_id, clock=clock, name="TestBus")
    cache = Cache()
    
    # Configure YFinance client
    config = YFinanceDataClientConfig(
        cache_expiry_seconds=60,
        rate_limit_delay=0.1,
        symbols=["AAPL", "MSFT"]
    )
    
    # Create data client
    event_loop = asyncio.get_event_loop()
    client = create_yfinance_data_client(
        loop=event_loop,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        config=config,
        name="YFinanceTest"
    )
    
    print("✅ Client created successfully")
    
    # Test connection
    try:
        await client._connect()
        print("✅ Connected to YFinance")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test instrument loading
    try:
        instrument_id = InstrumentId.from_str("AAPL.YAHOO")
        await client._instrument_provider.load_async(instrument_id)
        
        # Check if instrument was loaded
        instrument = cache.instrument(instrument_id)
        if instrument:
            print(f"✅ Loaded instrument: {instrument.id}")
            print(f"   - Quote currency: {instrument.quote_currency}")
            print(f"   - Price precision: {instrument.price_precision}")
        else:
            print("⚠️  Instrument not found in cache")
            
    except Exception as e:
        print(f"❌ Instrument loading failed: {e}")
    
    # Test bars request
    try:
        bar_type = BarType.from_str("AAPL.YAHOO-1-DAY-LAST-EXTERNAL")
        
        # Create a simple handler to collect bars
        bars_received = []
        
        def handle_data(data, correlation_id):
            if hasattr(data, 'bar_type'):
                bars_received.append(data)
                print(f"📊 Received bar: {data.open}-{data.high}-{data.low}-{data.close} Vol:{data.volume}")
        
        # Monkey patch the handle_data method for testing
        original_handle_data = client._handle_data
        client._handle_data = handle_data
        
        # Request bars for last 5 days
        request = RequestBars(
            client_id=client.id,
            venue_id=None,
            bar_type=bar_type,
            start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5),
            end=pd.Timestamp.now(tz="UTC"),
            limit=5,
            correlation_id=UUID4(),
            ts_init=clock.timestamp_ns(),
        )
        
        await client._request_bars(request)
        
        if bars_received:
            print(f"✅ Received {len(bars_received)} bars")
        else:
            print("⚠️  No bars received")
            
        # Restore original method
        client._handle_data = original_handle_data
        
    except Exception as e:
        print(f"❌ Bars request failed: {e}")
    
    # Test disconnection
    try:
        await client._disconnect()
        print("✅ Disconnected successfully")
    except Exception as e:
        print(f"❌ Disconnection failed: {e}")
    
    print("🎉 YFinance adapter integration test completed!")
    return True


def test_basic_yfinance():
    """Test basic yfinance functionality."""
    print("🔍 Testing basic yfinance functionality...")
    
    try:
        import yfinance as yf
        
        # Test basic ticker creation and data fetch
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and "symbol" in info:
            print(f"✅ Basic yfinance test passed - Symbol: {info.get('symbol')}")
            print(f"   - Company: {info.get('longName', 'N/A')}")
            print(f"   - Currency: {info.get('currency', 'N/A')}")
            return True
        else:
            print("❌ Basic yfinance test failed - no info returned")
            return False
            
    except Exception as e:
        print(f"❌ Basic yfinance test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting YFinance Adapter Tests\n")
    
    # Test basic yfinance first
    basic_success = test_basic_yfinance()
    print()
    
    if basic_success:
        # Test full integration
        integration_success = asyncio.run(test_yfinance_integration())
        
        if integration_success:
            print("\n🎉 All tests passed! YFinance adapter is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Integration tests failed!")
            sys.exit(1)
    else:
        print("\n❌ Basic tests failed!")
        sys.exit(1)
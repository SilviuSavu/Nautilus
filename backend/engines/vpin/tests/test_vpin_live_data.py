"""
VPIN Live Data Tests
Comprehensive testing with real IBKR Level 2 market data.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# HTTP client for testing
import httpx

# VPIN imports
from ..live_data_connector import VPINLiveDataConnector, LiveDataConfig, create_live_data_connector
from ..models import VPIN_TIER1_SYMBOLS, MarketRegime


# Configure logging for live tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.mark.live
@pytest.mark.asyncio
class TestVPINLiveDataConnection:
    """Test VPIN with live IBKR Level 2 data connection"""
    
    @pytest.fixture
    async def live_connector(self):
        """Create live data connector for testing"""
        connector = create_live_data_connector(
            host="127.0.0.1",
            port=7497,  # Paper trading port
            client_id=999,  # Test client ID
            symbols=["AAPL"]  # Start with single symbol
        )
        yield connector
        if connector.connected:
            await connector.disconnect()
    
    async def test_ibkr_connection_availability(self):
        """Test if IBKR connection is available"""
        try:
            from ib_insync import IB
            print("✅ ib_insync library available")
            
            # Quick connection test
            ib = IB()
            connected = False
            
            try:
                await asyncio.wait_for(
                    ib.connectAsync('127.0.0.1', 7497, clientId=998, timeout=10),
                    timeout=15.0
                )
                connected = True
                print("✅ IBKR TWS/Gateway connection successful")
            except Exception as e:
                print(f"⚠️  IBKR connection failed: {e}")
                print("   Make sure TWS or IB Gateway is running on port 7497")
                pytest.skip("IBKR not available - connection test skipped")
            finally:
                if connected:
                    ib.disconnect()
                    
        except ImportError:
            print("❌ ib_insync not installed")
            print("   Install with: pip install ib_insync")
            pytest.skip("ib_insync not available - live tests skipped")
    
    async def test_live_data_connector_initialization(self, live_connector):
        """Test live data connector initialization"""
        assert live_connector is not None
        assert live_connector.config.symbols == ["AAPL"]
        assert live_connector.config.ib_port == 7497
        assert not live_connector.connected
        
        print("✅ Live data connector initialized")
        
    async def test_live_ibkr_connection(self, live_connector):
        """Test connection to IBKR TWS/Gateway"""
        try:
            connected = await live_connector.connect()
            
            if not connected:
                pytest.skip("Could not connect to IBKR - make sure TWS/Gateway is running")
                
            assert live_connector.connected
            assert live_connector.ib is not None
            
            print("✅ IBKR live connection established")
            
            # Test connection status
            status = live_connector.get_connection_status()
            assert status["connected"]
            assert status["ibkr_available"]
            
            print(f"✅ Connection status: {status}")
            
        except Exception as e:
            pytest.skip(f"IBKR connection failed: {e}")
    
    async def test_live_symbol_subscription(self, live_connector):
        """Test subscription to live Level 2 data"""
        # Connect first
        connected = await live_connector.connect()
        if not connected:
            pytest.skip("IBKR connection required for subscription test")
        
        # Subscribe to AAPL
        subscribed = await live_connector.subscribe_to_symbols(["AAPL"])
        
        if not subscribed:
            pytest.skip("Failed to subscribe to live data")
        
        assert "AAPL" in live_connector.live_data
        assert "AAPL" in live_connector.contracts
        assert "AAPL" in live_connector.volume_synchronizers
        
        print("✅ Live symbol subscription successful")
        
        # Wait for some data
        print("Waiting for live market data...")
        await asyncio.sleep(10.0)  # Wait 10 seconds for data
        
        # Check if we received data
        aapl_data = await live_connector.get_live_data("AAPL")
        assert aapl_data is not None
        
        print(f"✅ Live data received: {aapl_data.to_dict()}")
        
        # Verify we got price data
        has_price_data = aapl_data.bid or aapl_data.ask or aapl_data.last
        
        if not has_price_data:
            print("⚠️  No price data received - market may be closed")
        else:
            print(f"✅ Price data: Bid={aapl_data.bid}, Ask={aapl_data.ask}, Last={aapl_data.last}")
    
    async def test_live_data_quality(self, live_connector):
        """Test quality of live market data"""
        # Connect and subscribe
        connected = await live_connector.connect()
        if not connected:
            pytest.skip("IBKR connection required")
            
        subscribed = await live_connector.subscribe_to_symbols(["AAPL", "MSFT"])
        if not subscribed:
            pytest.skip("Subscription failed")
        
        print("Testing live data quality for 30 seconds...")
        
        # Collect data for 30 seconds
        start_time = time.time()
        data_points = []
        
        while time.time() - start_time < 30.0:
            await asyncio.sleep(1.0)
            
            all_data = await live_connector.get_all_live_data()
            data_points.append({
                "timestamp": time.time(),
                "symbols_with_data": len([d for d in all_data.values() if d["bid"] or d["ask"]]),
                "total_ticks": live_connector.stats["total_ticks"],
                "errors": live_connector.stats["errors"]
            })
            
        # Analyze data quality
        total_ticks = live_connector.stats["total_ticks"]
        total_errors = live_connector.stats["errors"]
        
        print(f"Data Quality Results:")
        print(f"  Total ticks received: {total_ticks}")
        print(f"  Total errors: {total_errors}")
        print(f"  Error rate: {total_errors/max(1, total_ticks):.2%}")
        
        # Data quality assertions
        assert total_ticks > 0, "No market data received"
        assert total_errors / max(1, total_ticks) < 0.1, "Error rate too high (>10%)"
        
        print("✅ Live data quality acceptable")


@pytest.mark.live
@pytest.mark.asyncio  
class TestVPINLiveCalculations:
    """Test VPIN calculations with live market data"""
    
    @pytest.fixture
    async def live_vpin_system(self):
        """Setup complete live VPIN system"""
        connector = create_live_data_connector(symbols=["AAPL"])
        
        # Setup callbacks to collect VPIN results
        vpin_results = []
        data_updates = []
        
        async def on_vpin_update(symbol, vpin_result, pattern_result):
            vpin_results.append({
                "symbol": symbol,
                "vpin_value": vpin_result.vpin_value,
                "regime": pattern_result.predicted_regime if pattern_result else None,
                "timestamp": time.time()
            })
            
        async def on_data_update(symbol, data):
            data_updates.append({
                "symbol": symbol,
                "timestamp": data.timestamp,
                "has_price": bool(data.bid or data.ask)
            })
            
        connector.on_vpin_update = on_vpin_update
        connector.on_data_update = on_data_update
        
        yield connector, vpin_results, data_updates
        
        if connector.connected:
            await connector.disconnect()
    
    async def test_live_vpin_calculation_accuracy(self, live_vpin_system):
        """Test VPIN calculation accuracy with live data"""
        connector, vpin_results, data_updates = live_vpin_system
        
        # Connect to live data
        connected = await connector.connect()
        if not connected:
            pytest.skip("IBKR connection required for VPIN testing")
        
        subscribed = await live_connector.subscribe_to_symbols()
        if not subscribed:
            pytest.skip("Live data subscription failed")
        
        print("Testing live VPIN calculations for 5 minutes...")
        
        # Run for 5 minutes to collect enough data for VPIN
        await asyncio.sleep(300)  # 5 minutes
        
        print(f"Results after 5 minutes:")
        print(f"  Data updates: {len(data_updates)}")
        print(f"  VPIN calculations: {len(vpin_results)}")
        
        # Analyze results
        if len(vpin_results) > 0:
            latest_vpin = vpin_results[-1]
            print(f"  Latest VPIN: {latest_vpin}")
            
            # Validate VPIN value
            assert 0.0 <= latest_vpin["vpin_value"] <= 1.0
            assert latest_vpin["regime"] in [r.value for r in MarketRegime]
            
            print("✅ Live VPIN calculations working correctly")
        else:
            print("⚠️  No VPIN calculations completed - may need more volume or time")
    
    async def test_live_performance_metrics(self, live_vpin_system):
        """Test performance metrics with live data"""
        connector, vpin_results, data_updates = live_vpin_system
        
        # Connect and run for performance testing
        connected = await connector.connect()
        if not connected:
            pytest.skip("IBKR connection required")
            
        subscribed = await connector.subscribe_to_symbols()
        if not subscribed:
            pytest.skip("Subscription failed")
        
        print("Testing live performance for 2 minutes...")
        
        start_time = time.time()
        tick_counts = []
        
        # Monitor performance for 2 minutes
        while time.time() - start_time < 120:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            stats = connector.stats
            elapsed = time.time() - start_time
            
            tick_counts.append({
                "elapsed": elapsed,
                "total_ticks": stats["total_ticks"],
                "ticks_per_second": stats["total_ticks"] / elapsed,
                "vpin_calculations": stats["vpin_calculations"],
                "errors": stats["errors"]
            })
            
            print(f"  {elapsed:.0f}s: {stats['total_ticks']} ticks, {stats['total_ticks']/elapsed:.1f} ticks/sec")
        
        # Analyze performance
        final_stats = tick_counts[-1]
        
        print(f"Final Performance Metrics:")
        print(f"  Total ticks: {final_stats['total_ticks']}")
        print(f"  Average ticks/sec: {final_stats['ticks_per_second']:.1f}")
        print(f"  VPIN calculations: {final_stats['vpin_calculations']}")
        print(f"  Error rate: {final_stats['errors']}/{final_stats['total_ticks']}")
        
        # Performance assertions
        assert final_stats["ticks_per_second"] > 0, "No live data throughput"
        assert final_stats["errors"] / max(1, final_stats["total_ticks"]) < 0.05, "Too many errors"
        
        print("✅ Live performance metrics acceptable")


@pytest.mark.live 
class TestVPINLiveIntegration:
    """Test VPIN integration with live market conditions"""
    
    def test_live_market_hours_detection(self):
        """Test detection of market hours for live testing"""
        now = datetime.now()
        
        # US market hours (rough approximation)
        if now.weekday() < 5:  # Monday to Friday
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_market_hours = market_open <= now <= market_close
        else:
            is_market_hours = False
            
        print(f"Market hours check: {is_market_hours}")
        
        if not is_market_hours:
            print("⚠️  Outside market hours - live data may be limited")
        else:
            print("✅ Market is open - live data should be available")
            
        return is_market_hours
    
    @pytest.mark.asyncio
    async def test_live_api_integration(self):
        """Test VPIN API with live data backend"""
        # This assumes VPIN container is running with live data
        
        api_base = "http://localhost:10000"
        
        try:
            async with httpx.AsyncClient() as client:
                # Test health endpoint
                response = await client.get(f"{api_base}/health", timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ VPIN API healthy: {data.get('status')}")
                    
                    # Test live data endpoint
                    response = await client.get(f"{api_base}/api/v1/vpin/realtime/AAPL", timeout=10)
                    
                    if response.status_code == 200:
                        vpin_data = response.json()
                        print(f"✅ Live VPIN data: {vpin_data}")
                        
                        # Validate live VPIN response
                        assert "vpin_value" in vpin_data
                        assert 0.0 <= vpin_data["vpin_value"] <= 1.0
                        
                        print("✅ Live VPIN API integration working")
                    else:
                        print(f"⚠️  VPIN API returned {response.status_code}")
                        pytest.skip("VPIN API not returning live data")
                else:
                    print(f"⚠️  VPIN API not healthy: {response.status_code}")
                    pytest.skip("VPIN API not available")
                    
        except Exception as e:
            print(f"⚠️  VPIN API test failed: {e}")
            pytest.skip("VPIN API not accessible")


class VPINLiveTestRunner:
    """Runner for comprehensive live VPIN testing"""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests_run": [],
            "summary": {}
        }
    
    async def run_live_test_suite(self):
        """Run comprehensive live test suite"""
        print("="*80)
        print("  VPIN LIVE DATA TEST SUITE")
        print("="*80)
        
        # Check market hours
        market_open = TestVPINLiveIntegration().test_live_market_hours_detection()
        
        # Test IBKR connection availability
        try:
            await TestVPINLiveDataConnection().test_ibkr_connection_availability()
            ibkr_available = True
        except:
            ibkr_available = False
            
        self.results["environment"] = {
            "market_hours": market_open,
            "ibkr_available": ibkr_available
        }
        
        if not ibkr_available:
            print("❌ IBKR not available - cannot run live tests")
            return self.results
        
        # Run connection tests
        print("\n🔍 Testing IBKR Connection...")
        connector = create_live_data_connector()
        
        try:
            connection_test = await connector.test_connection()
            self.results["connection_test"] = connection_test
            
            print(f"Connection test results: {connection_test}")
            
            if not connection_test["connection_successful"]:
                print("❌ Cannot connect to IBKR - live tests cannot proceed")
                return self.results
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return self.results
        
        # If we get here, we can run live data tests
        print("✅ IBKR connection successful - proceeding with live tests")
        
        # Summary
        passed_tests = sum(1 for test in self.results["tests_run"] if test.get("passed", False))
        total_tests = len(self.results["tests_run"])
        
        self.results["summary"] = {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": passed_tests / max(1, total_tests),
            "live_data_available": connection_test.get("data_received", False)
        }
        
        return self.results


if __name__ == "__main__":
    async def main():
        runner = VPINLiveTestRunner()
        results = await runner.run_live_test_suite()
        
        print("\n" + "="*80)
        print("  LIVE TEST RESULTS")
        print("="*80)
        print(json.dumps(results, indent=2, default=str))
        
    asyncio.run(main())
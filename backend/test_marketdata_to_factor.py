#!/usr/bin/env python3
"""
Test MarketData to Factor Communication
Test real communication between MarketData Engine and Factor Engine via dual messagebus.
"""

import asyncio
import logging
import json
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our working simple client
from simple_dual_messagebus_client import get_simple_dual_bus_client, EngineType

async def test_marketdata_to_factor_communication():
    """Test MarketData Engine → Factor Engine communication"""
    
    logger.info("🧪 Testing MarketData → Factor Engine Communication")
    
    try:
        # Create MarketData publisher (simulating the Enhanced IBKR MarketData Engine)
        logger.info("📡 Creating MarketData Engine publisher...")
        marketdata_client = await get_simple_dual_bus_client(EngineType.MARKETDATA, "marketdata-8800")
        
        # Wait for Factor Engine to be ready
        await asyncio.sleep(1.0)
        
        # Send realistic market data that Factor Engine expects
        test_market_data = [
            {
                "symbol": "AAPL",
                "price": 175.50,
                "volume": 2500000,
                "bid": 175.48,
                "ask": 175.52,
                "timestamp": datetime.now().isoformat(),
                "source": "IBKR_LIVE"
            },
            {
                "symbol": "GOOGL",
                "price": 2750.25,
                "volume": 1800000,
                "bid": 2750.20,
                "ask": 2750.30,
                "timestamp": datetime.now().isoformat(),
                "source": "IBKR_LIVE"
            },
            {
                "symbol": "TSLA",
                "price": 245.75,
                "volume": 3200000,
                "bid": 245.70,
                "ask": 245.80,
                "timestamp": datetime.now().isoformat(),
                "source": "IBKR_LIVE"
            }
        ]
        
        logger.info("📤 Publishing realistic market data to Factor Engine...")
        
        for data in test_market_data:
            logger.info(f"   📊 Publishing {data['symbol']}: ${data['price']}")
            
            # Publish to market_data channel (Factor Engine is subscribed to this)
            await marketdata_client.publish_to_marketdata("market_data", data)
            
            # Also publish price updates
            price_update = {
                "symbol": data["symbol"],
                "price": data["price"],
                "timestamp": data["timestamp"]
            }
            await marketdata_client.publish_to_marketdata("price_update", price_update)
            
            # Small delay between messages
            await asyncio.sleep(0.5)
        
        # Wait for Factor Engine to process all data
        logger.info("⏳ Waiting for Factor Engine processing...")
        await asyncio.sleep(3.0)
        
        # Check Factor Engine status and calculations
        logger.info("🔍 Checking Factor Engine calculations...")
        
        import requests
        response = requests.get("http://localhost:8300/status")
        if response.status_code == 200:
            status = response.json()
            calculations = status.get("calculations_performed", 0)
            symbols = status.get("symbols_being_tracked", [])
            
            logger.info(f"✅ Factor Engine processed {calculations} calculations")
            logger.info(f"✅ Tracking symbols: {symbols}")
            
            if calculations > 0 and symbols:
                logger.info("🎉 MarketData → Factor Communication SUCCESSFUL!")
                
                # Test factor retrieval
                for symbol in symbols:
                    try:
                        factor_response = requests.get(f"http://localhost:8300/factors/{symbol}")
                        if factor_response.status_code == 200:
                            factors = factor_response.json()
                            logger.info(f"   📊 {symbol} factors calculated: {len(factors.get('factors', {}))}")
                    except:
                        pass
                
                return True
            else:
                logger.error("❌ No calculations performed by Factor Engine")
                return False
        else:
            logger.error("❌ Cannot reach Factor Engine")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False
    
    finally:
        # Close connections
        try:
            await marketdata_client.close()
            logger.info("🔌 MarketData client connection closed")
        except:
            pass

if __name__ == "__main__":
    result = asyncio.run(test_marketdata_to_factor_communication())
    if result:
        print("\n✅ MARKETDATA → FACTOR COMMUNICATION TEST PASSED")
        print("   Dual messagebus communication is working!")
        print("   Ready to scale up to all engines")
    else:
        print("\n❌ MARKETDATA → FACTOR COMMUNICATION TEST FAILED")
        print("   Need to investigate communication issues")
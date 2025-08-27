#!/usr/bin/env python3
"""
Test Simple Dual MessageBus Communication
Quick test to verify the simplified dual messagebus client works properly.
"""

import asyncio
import logging
import json
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our new simple client
from simple_dual_messagebus_client import get_simple_dual_bus_client, EngineType

async def test_dual_messagebus_communication():
    """Test dual messagebus communication between two engines"""
    
    logger.info("🧪 Starting Simple Dual MessageBus Communication Test")
    
    try:
        # Create two clients - one as MarketData publisher, one as Factor subscriber
        logger.info("📡 Creating MarketData publisher client...")
        marketdata_client = await get_simple_dual_bus_client(EngineType.MARKETDATA, "test-publisher")
        
        logger.info("🧮 Creating Factor subscriber client...")
        factor_client = await get_simple_dual_bus_client(EngineType.FACTOR, "test-subscriber")
        
        # Track received messages
        received_messages = []
        
        async def handle_market_data(message):
            """Handle market data messages in Factor engine"""
            logger.info(f"✅ Factor Engine received market data: {message}")
            received_messages.append(message)
        
        # Subscribe Factor engine to market data
        await factor_client.subscribe_to_marketdata("market_data_test", handle_market_data)
        
        # Wait a moment for subscription to be ready
        await asyncio.sleep(0.5)
        
        # Publish test market data
        test_data = {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 1000,
            "timestamp": datetime.now().isoformat(),
            "test_id": "dual_messagebus_test_1"
        }
        
        logger.info(f"📤 Publishing test market data: {test_data}")
        await marketdata_client.publish_to_marketdata("market_data_test", test_data)
        
        # Wait for message processing
        logger.info("⏳ Waiting for message processing...")
        await asyncio.sleep(2.0)
        
        # Check results
        if received_messages:
            logger.info(f"✅ SUCCESS: Received {len(received_messages)} messages")
            for i, msg in enumerate(received_messages):
                logger.info(f"   Message {i+1}: {msg}")
            logger.info("🎉 Dual MessageBus Communication Test PASSED")
            return True
        else:
            logger.error("❌ FAILED: No messages received")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False
    
    finally:
        # Close connections
        try:
            await marketdata_client.close()
            await factor_client.close()
            logger.info("🔌 Connections closed")
        except:
            pass

if __name__ == "__main__":
    result = asyncio.run(test_dual_messagebus_communication())
    if result:
        print("\n✅ DUAL MESSAGEBUS COMMUNICATION TEST PASSED")
        print("   Ready to restart engines with simplified client")
    else:
        print("\n❌ DUAL MESSAGEBUS COMMUNICATION TEST FAILED")
        print("   Need to investigate Redis connection issues")
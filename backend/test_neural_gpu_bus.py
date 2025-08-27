#!/usr/bin/env python3
"""
Neural-GPU Bus Connectivity Test
Tests connection to the Neural-GPU Bus (Port 6382)
"""

import asyncio
import redis.asyncio as redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_neural_gpu_bus():
    """Test Neural-GPU Bus connectivity"""
    try:
        logger.info("🧠⚡ Testing Neural-GPU Bus connectivity (Port 6382)")
        
        # Create Redis client for Neural-GPU Bus
        client = redis.Redis(
            host='localhost', 
            port=6382, 
            db=0, 
            decode_responses=True,
            socket_timeout=5.0
        )
        
        # Test ping
        response = await client.ping()
        logger.info(f"✅ Neural-GPU Bus ping successful: {response}")
        
        # Test basic operations
        await client.set("neural_test_key", "neural_gpu_bus_operational")
        value = await client.get("neural_test_key")
        logger.info(f"✅ Neural-GPU Bus read/write test: {value}")
        
        # Test info
        info = await client.info("stats")
        logger.info(f"✅ Neural-GPU Bus stats: Connected clients: {info.get('connected_clients', 'N/A')}")
        
        # Cleanup
        await client.delete("neural_test_key")
        await client.aclose()
        
        logger.info("🎉 Neural-GPU Bus is FULLY OPERATIONAL and ready for triple-bus integration!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Neural-GPU Bus connectivity failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_neural_gpu_bus())
    exit(0 if success else 1)
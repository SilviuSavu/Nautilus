#!/usr/bin/env python3

import asyncio
from unittest.mock import patch, AsyncMock
from messagebus_client import MessageBusClient

async def test_simple_start_stop():
    """Simple test to verify start/stop works without hanging"""
    client = MessageBusClient(
        max_reconnect_attempts=1,
        reconnect_base_delay=0.1,
        connection_timeout=1.0,
        health_check_interval=1.0
    )
    
    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.xgroup_create = AsyncMock()
    mock_redis.xreadgroup.return_value = []
    mock_redis.xack = AsyncMock()
    
    with patch('messagebus_client.redis.Redis', return_value=mock_redis):
        print("Starting client...")
        await client.start()
        print(f"Client started. Running: {client._running}")
        
        await asyncio.sleep(0.1)
        print(f"After sleep. Connection state: {client.connection_status.state}")
        
        print("Stopping client...")
        await client.stop()
        print(f"Client stopped. Running: {client._running}")
        
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_simple_start_stop())
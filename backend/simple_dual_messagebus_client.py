#!/usr/bin/env python3
"""
Simple Dual MessageBus Client - Bypass UltraFastRedisPool
Uses standard Redis async connections for immediate restoration of dual messagebus functionality.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import redis.asyncio as redis

from universal_enhanced_messagebus_client import (
    MessageType, EngineType, MessagePriority
)

logger = logging.getLogger(__name__)

@dataclass
class SimpleDualBusConfig:
    """Simplified configuration for dual message bus"""
    engine_type: EngineType
    engine_instance_id: str
    
    # MarketData Bus (Port 6380)
    marketdata_redis_host: str = "localhost"
    marketdata_redis_port: int = 6380
    marketdata_redis_db: int = 0
    
    # Engine Logic Bus (Port 6381)
    engine_logic_redis_host: str = "localhost" 
    engine_logic_redis_port: int = 6381
    engine_logic_redis_db: int = 0

class SimpleDualMessageBusClient:
    """
    Simplified dual messagebus client using standard Redis async connections.
    This bypasses UltraFastRedisPool to restore immediate functionality.
    """
    
    def __init__(self, config: SimpleDualBusConfig):
        self.config = config
        self.marketdata_redis = None
        self.engine_logic_redis = None
        self.connected = False
        
    async def initialize(self):
        """Initialize Redis connections to both buses"""
        try:
            # MarketData Bus connection
            self.marketdata_redis = redis.Redis(
                host=self.config.marketdata_redis_host,
                port=self.config.marketdata_redis_port,
                db=self.config.marketdata_redis_db,
                decode_responses=True
            )
            
            # Engine Logic Bus connection
            self.engine_logic_redis = redis.Redis(
                host=self.config.engine_logic_redis_host,
                port=self.config.engine_logic_redis_port,
                db=self.config.engine_logic_redis_db,
                decode_responses=True
            )
            
            # Test connections
            marketdata_ping = await self.marketdata_redis.ping()
            engine_logic_ping = await self.engine_logic_redis.ping()
            
            if marketdata_ping and engine_logic_ping:
                self.connected = True
                logger.info(f"✅ SimpleDualMessageBusClient connected - MarketData Bus: {self.config.marketdata_redis_port}, Engine Logic Bus: {self.config.engine_logic_redis_port}")
                return True
            else:
                logger.error("❌ Failed to ping Redis buses")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize SimpleDualMessageBusClient: {e}")
            return False
    
    async def subscribe_to_marketdata(self, channel: str, callback: Callable):
        """Subscribe to market data messages"""
        if not self.connected:
            logger.warning("Not connected to messagebus")
            return
            
        try:
            pubsub = self.marketdata_redis.pubsub()
            await pubsub.subscribe(channel)
            
            async def message_handler():
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            await callback(data)
                        except Exception as e:
                            logger.error(f"Error processing market data message: {e}")
            
            # Start message handler in background
            asyncio.create_task(message_handler())
            logger.info(f"✅ Subscribed to MarketData channel: {channel}")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to market data channel {channel}: {e}")
    
    async def subscribe_to_engine_logic(self, channel: str, callback: Callable):
        """Subscribe to engine logic messages"""
        if not self.connected:
            logger.warning("Not connected to messagebus")
            return
            
        try:
            pubsub = self.engine_logic_redis.pubsub()
            await pubsub.subscribe(channel)
            
            async def message_handler():
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            await callback(data)
                        except Exception as e:
                            logger.error(f"Error processing engine logic message: {e}")
            
            # Start message handler in background
            asyncio.create_task(message_handler())
            logger.info(f"✅ Subscribed to Engine Logic channel: {channel}")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to engine logic channel {channel}: {e}")
    
    async def publish_to_marketdata(self, channel: str, data: Dict[str, Any]):
        """Publish to market data bus"""
        if not self.connected:
            logger.warning("Not connected to messagebus")
            return
            
        try:
            message = json.dumps(data)
            await self.marketdata_redis.publish(channel, message)
            logger.debug(f"📤 Published to MarketData channel {channel}")
        except Exception as e:
            logger.error(f"❌ Failed to publish to market data channel {channel}: {e}")
    
    async def publish_to_engine_logic(self, channel: str, data: Dict[str, Any]):
        """Publish to engine logic bus"""
        if not self.connected:
            logger.warning("Not connected to messagebus")
            return
            
        try:
            message = json.dumps(data)
            await self.engine_logic_redis.publish(channel, message)
            logger.debug(f"📤 Published to Engine Logic channel {channel}")
        except Exception as e:
            logger.error(f"❌ Failed to publish to engine logic channel {channel}: {e}")
    
    async def close(self):
        """Close all Redis connections"""
        try:
            if self.marketdata_redis:
                await self.marketdata_redis.aclose()
            if self.engine_logic_redis:
                await self.engine_logic_redis.aclose()
            self.connected = False
            logger.info("✅ SimpleDualMessageBusClient connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

# Factory function for backward compatibility
async def get_simple_dual_bus_client(engine_type: EngineType, instance_id: Optional[str] = None) -> SimpleDualMessageBusClient:
    """Create and initialize a simplified dual bus client"""
    
    if instance_id is None:
        instance_id = f"{engine_type.value}-{asyncio.current_task().get_name() if asyncio.current_task() else 'default'}"
    
    config = SimpleDualBusConfig(
        engine_type=engine_type,
        engine_instance_id=instance_id
    )
    
    client = SimpleDualMessageBusClient(config)
    
    if await client.initialize():
        return client
    else:
        raise Exception("Failed to initialize SimpleDualMessageBusClient")
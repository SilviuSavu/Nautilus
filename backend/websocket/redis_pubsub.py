"""
Redis Pub/Sub Implementation for WebSocket Message Distribution
Sprint 3 Priority 1: WebSocket Streaming Infrastructure

Handles scalable message distribution across multiple WebSocket connections
using Redis pub/sub for horizontal scaling and message routing
"""

import asyncio
import logging
import json
import redis.asyncio as redis
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    MARKET_DATA = "market_data"
    TRADE_UPDATE = "trade_update"
    ENGINE_STATUS = "engine_status" 
    RISK_ALERT = "risk_alert"
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE_UPDATE = "performance_update"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    HEARTBEAT = "heartbeat"

class ChannelType(Enum):
    """Redis channel types"""
    BROADCAST = "broadcast"  # All connections
    PORTFOLIO = "portfolio"   # Specific portfolio
    STRATEGY = "strategy"     # Specific strategy
    SYMBOL = "symbol"         # Specific symbol/instrument
    USER = "user"             # Specific user

@dataclass
class WebSocketMessage:
    """Standardized WebSocket message format"""
    message_id: str
    message_type: MessageType
    channel: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata or {},
            "ttl_seconds": self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from dictionary"""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            channel=data["channel"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            metadata=data.get("metadata"),
            ttl_seconds=data.get("ttl_seconds")
        )

@dataclass
class SubscriptionFilter:
    """WebSocket subscription filter"""
    channel_pattern: str
    message_types: Set[MessageType]
    metadata_filters: Dict[str, Any]
    user_id: Optional[str] = None
    portfolio_ids: Optional[Set[str]] = None
    strategy_ids: Optional[Set[str]] = None
    symbol_filters: Optional[Set[str]] = None

class RedisPubSubManager:
    """
    Redis-based pub/sub manager for scalable WebSocket message distribution
    """
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "nautilus_ws",
        max_message_size: int = 1024 * 1024,  # 1MB
        message_ttl: int = 300  # 5 minutes
    ):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.max_message_size = max_message_size
        self.message_ttl = message_ttl
        
        # Redis connections
        self.redis_pub: Optional[redis.Redis] = None
        self.redis_sub: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.subscription_filters: Dict[str, SubscriptionFilter] = {}
        
        # Statistics
        self.messages_published = 0
        self.messages_received = 0
        self.active_subscriptions = 0
        
        # Background tasks
        self._subscription_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """
        Initialize Redis connections and pub/sub
        """
        try:
            # Create Redis connections
            self.redis_pub = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_sub = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test connections
            await self.redis_pub.ping()
            await self.redis_sub.ping()
            
            # Create pub/sub instance
            self.pubsub = self.redis_sub.pubsub()
            
            # Start subscription listener
            self._subscription_task = asyncio.create_task(self._subscription_listener())
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_task_runner())
            
            self.logger.info("Redis pub/sub manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis pub/sub: {e}")
            raise
    
    async def shutdown(self) -> None:
        """
        Shutdown Redis connections and cleanup
        """
        try:
            # Cancel background tasks
            if self._subscription_task:
                self._subscription_task.cancel()
                try:
                    await self._subscription_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Close pub/sub
            if self.pubsub:
                await self.pubsub.unsubscribe()
                await self.pubsub.close()
            
            # Close Redis connections
            if self.redis_pub:
                await self.redis_pub.close()
            if self.redis_sub:
                await self.redis_sub.close()
            
            self.logger.info("Redis pub/sub manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def generate_channel_name(
        self, 
        channel_type: ChannelType, 
        identifier: Optional[str] = None
    ) -> str:
        """
        Generate standardized channel name
        """
        if identifier:
            return f"{self.channel_prefix}:{channel_type.value}:{identifier}"
        else:
            return f"{self.channel_prefix}:{channel_type.value}"
    
    async def publish_message(
        self,
        message: WebSocketMessage,
        channel_override: Optional[str] = None
    ) -> int:
        """
        Publish message to Redis channel
        """
        try:
            channel = channel_override or message.channel
            message_data = json.dumps(message.to_dict())
            
            # Check message size
            if len(message_data) > self.max_message_size:
                self.logger.warning(
                    f"Message size ({len(message_data)} bytes) exceeds limit "
                    f"({self.max_message_size} bytes). Message truncated."
                )
                # Truncate data field
                truncated_data = str(message.data)[:self.max_message_size//2] + "...[truncated]"
                message.data = {"truncated": True, "data": truncated_data}
                message_data = json.dumps(message.to_dict())
            
            # Publish to Redis
            channel_name = f"{self.channel_prefix}:{channel}"
            subscribers = await self.redis_pub.publish(channel_name, message_data)
            
            # Store message with TTL for potential replay
            if message.ttl_seconds:
                key = f"{self.channel_prefix}:messages:{message.message_id}"
                await self.redis_pub.setex(key, message.ttl_seconds, message_data)
            
            self.messages_published += 1
            
            self.logger.debug(
                f"Published message {message.message_id} to {channel_name}, "
                f"reached {subscribers} subscribers"
            )
            
            return subscribers
            
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            return 0
    
    async def subscribe_to_channel(
        self,
        channel: str,
        message_handler: Callable,
        subscription_filter: Optional[SubscriptionFilter] = None
    ) -> str:
        """
        Subscribe to Redis channel with optional filtering
        """
        try:
            subscription_id = str(uuid.uuid4())
            
            # Store handler and filter
            channel_name = f"{self.channel_prefix}:{channel}"
            if channel_name not in self.message_handlers:
                self.message_handlers[channel_name] = []
                # Subscribe to Redis channel
                await self.pubsub.subscribe(channel_name)
            
            self.message_handlers[channel_name].append(message_handler)
            
            if subscription_filter:
                self.subscription_filters[subscription_id] = subscription_filter
            
            self.active_subscriptions += 1
            
            self.logger.debug(f"Subscribed to {channel_name} with ID {subscription_id}")
            
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Error subscribing to channel: {e}")
            raise
    
    async def unsubscribe_from_channel(
        self,
        subscription_id: str,
        channel: str,
        message_handler: Callable
    ) -> None:
        """
        Unsubscribe from Redis channel
        """
        try:
            channel_name = f"{self.channel_prefix}:{channel}"
            
            if channel_name in self.message_handlers:
                if message_handler in self.message_handlers[channel_name]:
                    self.message_handlers[channel_name].remove(message_handler)
                
                # If no more handlers, unsubscribe from Redis
                if not self.message_handlers[channel_name]:
                    await self.pubsub.unsubscribe(channel_name)
                    del self.message_handlers[channel_name]
            
            # Remove subscription filter
            if subscription_id in self.subscription_filters:
                del self.subscription_filters[subscription_id]
            
            self.active_subscriptions = max(0, self.active_subscriptions - 1)
            
            self.logger.debug(f"Unsubscribed from {channel_name}")
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from channel: {e}")
    
    async def _subscription_listener(self) -> None:
        """
        Background task to listen for Redis pub/sub messages
        """
        try:
            while True:
                try:
                    message = await self.pubsub.get_message(timeout=1.0)
                    
                    if message and message['type'] == 'message':
                        channel = message['channel']
                        data = message['data']
                        
                        await self._handle_received_message(channel, data)
                        
                except asyncio.TimeoutError:
                    # Normal timeout, continue listening
                    continue
                except Exception as e:
                    self.logger.error(f"Error in subscription listener: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
                    
        except asyncio.CancelledError:
            self.logger.info("Subscription listener cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Fatal error in subscription listener: {e}")
    
    async def _handle_received_message(self, channel: str, data: str) -> None:
        """
        Handle received Redis pub/sub message
        """
        try:
            # Parse message
            message_dict = json.loads(data)
            message = WebSocketMessage.from_dict(message_dict)
            
            # Get handlers for this channel
            handlers = self.message_handlers.get(channel, [])
            
            # Process each handler
            for handler in handlers:
                try:
                    # Apply filters if needed
                    if await self._should_deliver_message(message, handler):
                        await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")
            
            self.messages_received += 1
            
        except Exception as e:
            self.logger.error(f"Error handling received message: {e}")
    
    async def _should_deliver_message(
        self, 
        message: WebSocketMessage, 
        handler: Callable
    ) -> bool:
        """
        Check if message should be delivered based on subscription filters
        """
        # For now, deliver all messages
        # In production, implement filtering logic based on subscription_filters
        return True
    
    async def _cleanup_task_runner(self) -> None:
        """
        Background task for periodic cleanup
        """
        try:
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_messages()
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in cleanup task: {e}")
    
    async def _cleanup_expired_messages(self) -> None:
        """
        Clean up expired messages from Redis
        """
        try:
            # Get all message keys
            pattern = f"{self.channel_prefix}:messages:*"
            keys = await self.redis_pub.keys(pattern)
            
            if keys:
                # Redis will automatically expire keys with TTL
                # This is just for statistics
                self.logger.debug(f"Found {len(keys)} stored messages")
                
        except Exception as e:
            self.logger.error(f"Error during message cleanup: {e}")
    
    # High-level message publishing methods
    
    async def publish_market_data(
        self,
        symbol: str,
        price_data: Dict[str, Any]
    ) -> int:
        """
        Publish market data update
        """
        channel = self.generate_channel_name(ChannelType.SYMBOL, symbol)
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            channel=channel,
            timestamp=datetime.utcnow(),
            data={
                "symbol": symbol,
                **price_data
            },
            ttl_seconds=60  # Market data expires quickly
        )
        
        return await self.publish_message(message)
    
    async def publish_trade_update(
        self,
        trade_data: Dict[str, Any],
        portfolio_id: Optional[str] = None
    ) -> int:
        """
        Publish trade execution update
        """
        if portfolio_id:
            channel = self.generate_channel_name(ChannelType.PORTFOLIO, portfolio_id)
        else:
            channel = self.generate_channel_name(ChannelType.BROADCAST)
        
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TRADE_UPDATE,
            channel=channel,
            timestamp=datetime.utcnow(),
            data=trade_data,
            ttl_seconds=300
        )
        
        return await self.publish_message(message)
    
    async def publish_risk_alert(
        self,
        alert_data: Dict[str, Any],
        portfolio_id: str
    ) -> int:
        """
        Publish risk alert
        """
        # Publish to both portfolio channel and broadcast
        portfolio_channel = self.generate_channel_name(ChannelType.PORTFOLIO, portfolio_id)
        broadcast_channel = self.generate_channel_name(ChannelType.BROADCAST)
        
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RISK_ALERT,
            channel=portfolio_channel,
            timestamp=datetime.utcnow(),
            data=alert_data,
            ttl_seconds=3600  # Alerts persist longer
        )
        
        subscribers = await self.publish_message(message, portfolio_channel)
        subscribers += await self.publish_message(message, broadcast_channel)
        
        return subscribers
    
    async def publish_engine_status(
        self,
        status_data: Dict[str, Any]
    ) -> int:
        """
        Publish engine status update
        """
        channel = self.generate_channel_name(ChannelType.BROADCAST)
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ENGINE_STATUS,
            channel=channel,
            timestamp=datetime.utcnow(),
            data=status_data,
            ttl_seconds=120
        )
        
        return await self.publish_message(message)
    
    async def publish_performance_update(
        self,
        performance_data: Dict[str, Any],
        portfolio_id: str
    ) -> int:
        """
        Publish performance metrics update
        """
        channel = self.generate_channel_name(ChannelType.PORTFOLIO, portfolio_id)
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PERFORMANCE_UPDATE,
            channel=channel,
            timestamp=datetime.utcnow(),
            data=performance_data,
            ttl_seconds=600
        )
        
        return await self.publish_message(message)
    
    # Statistics and monitoring
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get pub/sub statistics
        """
        return {
            "messages_published": self.messages_published,
            "messages_received": self.messages_received,
            "active_subscriptions": self.active_subscriptions,
            "active_channels": len(self.message_handlers),
            "redis_connected": self.redis_pub is not None and self.redis_sub is not None
        }

# Global instance
redis_pubsub_manager = None

def get_redis_pubsub_manager() -> RedisPubSubManager:
    """Get global Redis pub/sub manager instance"""
    global redis_pubsub_manager
    if redis_pubsub_manager is None:
        raise RuntimeError("Redis pub/sub manager not initialized. Call init_redis_pubsub() first.")
    return redis_pubsub_manager

def init_redis_pubsub(redis_url: str = "redis://localhost:6379") -> RedisPubSubManager:
    """Initialize global Redis pub/sub manager instance"""
    global redis_pubsub_manager
    redis_pubsub_manager = RedisPubSubManager(redis_url)
    return redis_pubsub_manager
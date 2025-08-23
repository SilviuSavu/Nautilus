"""
WebSocket Subscription Manager for Client Connection Management
Sprint 3 Priority 1: WebSocket Streaming Infrastructure

Manages WebSocket client subscriptions, filtering, and message routing
with Redis integration for horizontal scaling and session persistence
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import weakref
from collections import defaultdict, deque

from .websocket_manager import websocket_manager
from .event_dispatcher import EventType
from .redis_pubsub import (
    RedisPubSubManager, 
    WebSocketMessage, 
    MessageType, 
    ChannelType,
    SubscriptionFilter,
    get_redis_pubsub_manager
)

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class SubscriptionType(Enum):
    """Types of WebSocket subscriptions"""
    MARKET_DATA = "market_data"
    TRADE_UPDATES = "trade_updates"
    RISK_ALERTS = "risk_alerts"
    ENGINE_STATUS = "engine_status"
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE_UPDATES = "performance_updates"
    ORDER_UPDATES = "order_updates"
    POSITION_UPDATES = "position_updates"
    PORTFOLIO_UPDATES = "portfolio_updates"
    RISK_METRICS = "risk_metrics"
    STRATEGY_PERFORMANCE = "strategy_performance"
    ORDER_BOOK = "order_book"
    NEWS_FEED = "news_feed"
    ALERTS = "alerts"
    ALL = "all"


@dataclass
class ClientSubscription:
    """Individual client subscription configuration"""
    subscription_id: str
    client_id: str
    subscription_type: SubscriptionType
    filters: Dict[str, Any]
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    error_count: int = 0
    is_active: bool = True
    rate_limit: int = 10  # messages per second
    buffer_size: int = 100  # max queued messages
    priority: int = 1  # 1-5, higher = more important
    auto_unsubscribe: Optional[timedelta] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        self.message_count += 1

@dataclass
class WebSocketClient:
    """WebSocket client connection management"""
    client_id: str
    connection_id: str
    websocket: Any  # WebSocket connection object
    state: ConnectionState
    connected_at: datetime
    last_heartbeat: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    subscriptions: Dict[str, ClientSubscription] = field(default_factory=dict)
    message_queue: List[WebSocketMessage] = field(default_factory=list)
    max_queue_size: int = 1000
    
    def add_subscription(self, subscription: ClientSubscription):
        """Add a subscription to this client"""
        self.subscriptions[subscription.subscription_id] = subscription
    
    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove a subscription from this client"""
        return self.subscriptions.pop(subscription_id, None) is not None
    
    def get_active_subscriptions(self) -> List[ClientSubscription]:
        """Get all active subscriptions for this client"""
        return [sub for sub in self.subscriptions.values() if sub.is_active]
    
    def queue_message(self, message: WebSocketMessage) -> bool:
        """Queue message for delivery to client"""
        if len(self.message_queue) >= self.max_queue_size:
            # Remove oldest message
            self.message_queue.pop(0)
            logger.warning(f"Message queue full for client {self.client_id}, dropping oldest message")
        
        self.message_queue.append(message)
        return True
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    max_requests_per_second: int = 10
    max_requests_per_minute: int = 600
    max_subscriptions_per_connection: int = 50
    burst_allowance: int = 20


class WebSocketSubscriptionManager:
    """
    Manages WebSocket client subscriptions and message routing
    Integrates with Redis pub/sub for horizontal scaling
    """
    
    def __init__(
        self,
        redis_pubsub_manager: Optional[RedisPubSubManager] = None,
        heartbeat_interval: int = 30,
        subscription_timeout: int = 300,
        max_subscriptions_per_client: int = 50
    ):
        self.redis_pubsub = redis_pubsub_manager
        self.heartbeat_interval = heartbeat_interval
        self.subscription_timeout = subscription_timeout
        self.max_subscriptions_per_client = max_subscriptions_per_client
        
        # Client and subscription management
        self.clients: Dict[str, WebSocketClient] = {}
        self.subscriptions: Dict[str, ClientSubscription] = {}
        self.redis_subscriptions: Dict[str, str] = {}  # Redis subscription ID mapping
        
        # Message routing
        self.message_handlers: Dict[SubscriptionType, List[Callable]] = {}
        self.message_filters: Dict[str, Callable] = {}
        
        # Rate limiting (legacy support)
        self.rate_limits: Dict[str, RateLimitConfig] = {}
        self.rate_counters: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.default_rate_limit = RateLimitConfig()
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_connections = 0
        self.active_connections = 0
        self.total_subscriptions = 0
        self.messages_sent = 0
        self.messages_dropped = 0
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize the subscription manager"""
        try:
            # Initialize Redis pub/sub if not provided
            if self.redis_pubsub is None:
                self.redis_pubsub = get_redis_pubsub_manager()
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_subscriptions())
            
            self.logger.info("WebSocket subscription manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subscription manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the subscription manager"""
        try:
            # Cancel background tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Disconnect all clients
            for client in list(self.clients.values()):
                await self.disconnect_client(client.client_id)
            
            self.logger.info("WebSocket subscription manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def connect_client(
        self,
        websocket: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Connect a new WebSocket client"""
        try:
            client_id = str(uuid.uuid4())
            connection_id = str(uuid.uuid4())
            
            client = WebSocketClient(
                client_id=client_id,
                connection_id=connection_id,
                websocket=websocket,
                state=ConnectionState.CONNECTED,
                connected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            self.clients[client_id] = client
            self.total_connections += 1
            self.active_connections += 1
            
            self.logger.info(f"WebSocket client connected: {client_id}")
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SYSTEM_HEALTH,
                channel="system",
                timestamp=datetime.utcnow(),
                data={
                    "type": "connection_established",
                    "client_id": client_id,
                    "server_time": datetime.utcnow().isoformat()
                }
            )
            
            await self.send_message_to_client(client_id, welcome_message)
            
            return client_id
            
        except Exception as e:
            self.logger.error(f"Error connecting client: {e}")
            raise
    
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect a WebSocket client"""
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            # Unsubscribe from all subscriptions
            for subscription_id in list(client.subscriptions.keys()):
                await self.unsubscribe(client_id, subscription_id)
            
            # Update client state
            client.state = ConnectionState.DISCONNECTED
            
            # Remove from active clients
            del self.clients[client_id]
            self.active_connections = max(0, self.active_connections - 1)
            
            self.logger.info(f"WebSocket client disconnected: {client_id}")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting client: {e}")
            
    async def subscribe(
        self,
        client_id: str,
        subscription_type: SubscriptionType,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new subscription for a connection
        
        Args:
            connection_id: WebSocket connection identifier
            subscription_id: Unique subscription identifier
            config: Subscription configuration
            
        Returns:
            bool: True if subscription created successfully
        """
        try:
            # Validate connection exists
            if not self._validate_connection(connection_id):
                logger.warning(f"Invalid connection for subscription: {connection_id}")
                return False
                
            # Check subscription limits
            if not await self._check_subscription_limits(connection_id):
                logger.warning(f"Subscription limit exceeded for connection: {connection_id}")
                return False
                
            # Validate subscription config
            if not self._validate_subscription_config(config):
                logger.warning(f"Invalid subscription config: {subscription_id}")
                return False
                
            # Store subscription
            self.active_subscriptions[connection_id][subscription_id] = config
            
            # Initialize statistics
            self.subscription_stats[connection_id][subscription_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "messages_sent": 0,
                "messages_dropped": 0,
                "last_activity": datetime.utcnow().isoformat(),
                "rate_limit_hits": 0
            }
            
            # Create message queue if needed
            if connection_id not in self.message_queues:
                self.message_queues[connection_id] = asyncio.Queue(maxsize=1000)
                
            # Start processing task if needed
            if connection_id not in self.processing_tasks:
                self.processing_tasks[connection_id] = asyncio.create_task(
                    self._process_messages_for_connection(connection_id)
                )
                
            # Subscribe to relevant topics in WebSocket manager
            await self._subscribe_to_websocket_topics(connection_id, config)
            
            # Persist subscription
            await self._persist_subscription(connection_id, subscription_id, config)
            
            logger.info(f"Created subscription {subscription_id} for connection {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create subscription {subscription_id}: {e}")
            return False
            
    async def remove_subscription(
        self,
        connection_id: str,
        subscription_id: str
    ) -> bool:
        """
        Remove a subscription for a connection
        
        Args:
            connection_id: WebSocket connection identifier
            subscription_id: Subscription identifier to remove
            
        Returns:
            bool: True if subscription removed successfully
        """
        try:
            # Check if subscription exists
            if (connection_id not in self.active_subscriptions or
                subscription_id not in self.active_subscriptions[connection_id]):
                logger.warning(f"Subscription not found: {subscription_id}")
                return False
                
            # Get subscription config for cleanup
            config = self.active_subscriptions[connection_id][subscription_id]
            
            # Remove subscription
            del self.active_subscriptions[connection_id][subscription_id]
            
            # Remove statistics
            if connection_id in self.subscription_stats:
                self.subscription_stats[connection_id].pop(subscription_id, None)
                
            # Unsubscribe from WebSocket topics
            await self._unsubscribe_from_websocket_topics(connection_id, config)
            
            # Remove from persistence
            await self._remove_persisted_subscription(connection_id, subscription_id)
            
            logger.info(f"Removed subscription {subscription_id} for connection {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove subscription {subscription_id}: {e}")
            return False
            
    async def remove_all_subscriptions(self, connection_id: str) -> int:
        """
        Remove all subscriptions for a connection
        
        Args:
            connection_id: WebSocket connection identifier
            
        Returns:
            int: Number of subscriptions removed
        """
        try:
            if connection_id not in self.active_subscriptions:
                return 0
                
            subscription_ids = list(self.active_subscriptions[connection_id].keys())
            removed_count = 0
            
            for subscription_id in subscription_ids:
                if await self.remove_subscription(connection_id, subscription_id):
                    removed_count += 1
                    
            # Stop processing task
            if connection_id in self.processing_tasks:
                self.processing_tasks[connection_id].cancel()
                del self.processing_tasks[connection_id]
                
            # Clear message queue
            if connection_id in self.message_queues:
                del self.message_queues[connection_id]
                
            # Clear rate limiting data
            self.rate_counters.pop(connection_id, None)
            self.rate_limits.pop(connection_id, None)
            
            logger.info(f"Removed {removed_count} subscriptions for connection {connection_id}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to remove subscriptions for connection {connection_id}: {e}")
            return 0
            
    async def update_subscription(
        self,
        connection_id: str,
        subscription_id: str,
        new_config: SubscriptionConfig
    ) -> bool:
        """
        Update an existing subscription configuration
        
        Args:
            connection_id: WebSocket connection identifier
            subscription_id: Subscription identifier to update
            new_config: New subscription configuration
            
        Returns:
            bool: True if subscription updated successfully
        """
        try:
            # Check if subscription exists
            if (connection_id not in self.active_subscriptions or
                subscription_id not in self.active_subscriptions[connection_id]):
                logger.warning(f"Subscription not found for update: {subscription_id}")
                return False
                
            # Validate new config
            if not self._validate_subscription_config(new_config):
                logger.warning(f"Invalid subscription config for update: {subscription_id}")
                return False
                
            # Get old config for cleanup
            old_config = self.active_subscriptions[connection_id][subscription_id]
            
            # Update subscription
            self.active_subscriptions[connection_id][subscription_id] = new_config
            
            # Update WebSocket subscriptions if topic changed
            if old_config.subscription_type != new_config.subscription_type:
                await self._unsubscribe_from_websocket_topics(connection_id, old_config)
                await self._subscribe_to_websocket_topics(connection_id, new_config)
                
            # Update persistence
            await self._persist_subscription(connection_id, subscription_id, new_config)
            
            # Update statistics
            if connection_id in self.subscription_stats and subscription_id in self.subscription_stats[connection_id]:
                self.subscription_stats[connection_id][subscription_id]["updated_at"] = datetime.utcnow().isoformat()
                
            logger.info(f"Updated subscription {subscription_id} for connection {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update subscription {subscription_id}: {e}")
            return False
            
    async def send_message_to_subscription(
        self,
        connection_id: str,
        subscription_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send a message to a specific subscription
        
        Args:
            connection_id: WebSocket connection identifier
            subscription_id: Target subscription identifier
            message: Message to send
            
        Returns:
            bool: True if message queued successfully
        """
        try:
            # Check if subscription exists
            if (connection_id not in self.active_subscriptions or
                subscription_id not in self.active_subscriptions[connection_id]):
                return False
                
            # Get subscription config
            config = self.active_subscriptions[connection_id][subscription_id]
            
            # Check rate limits
            if not await self._check_rate_limits(connection_id, subscription_id):
                self._update_stats(connection_id, subscription_id, "rate_limit_hit")
                return False
                
            # Apply filters
            if not self._passes_filters(message, config.filters):
                return False
                
            # Add subscription metadata to message
            enriched_message = {
                **message,
                "subscription_id": subscription_id,
                "subscription_type": config.subscription_type.value,
                "priority": config.priority,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Queue message
            if connection_id in self.message_queues:
                try:
                    self.message_queues[connection_id].put_nowait(enriched_message)
                    self._update_stats(connection_id, subscription_id, "message_queued")
                    return True
                except asyncio.QueueFull:
                    self._update_stats(connection_id, subscription_id, "message_dropped")
                    logger.warning(f"Message queue full for connection {connection_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send message to subscription {subscription_id}: {e}")
            
        return False
        
    def get_subscription_info(self, connection_id: str) -> Dict[str, Any]:
        """
        Get subscription information for a connection
        
        Args:
            connection_id: WebSocket connection identifier
            
        Returns:
            Dict containing subscription information
        """
        if connection_id not in self.active_subscriptions:
            return {"subscriptions": [], "total": 0}
            
        subscriptions = []
        for sub_id, config in self.active_subscriptions[connection_id].items():
            stats = self.subscription_stats[connection_id].get(sub_id, {})
            subscriptions.append({
                "subscription_id": sub_id,
                "type": config.subscription_type.value,
                "parameters": config.parameters,
                "rate_limit": config.rate_limit,
                "priority": config.priority,
                "statistics": stats
            })
            
        return {
            "subscriptions": subscriptions,
            "total": len(subscriptions),
            "queue_size": self.message_queues.get(connection_id, asyncio.Queue()).qsize()
        }
        
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global subscription statistics"""
        total_connections = len(self.active_subscriptions)
        total_subscriptions = sum(len(subs) for subs in self.active_subscriptions.values())
        
        subscription_types = defaultdict(int)
        for subs in self.active_subscriptions.values():
            for config in subs.values():
                subscription_types[config.subscription_type.value] += 1
                
        return {
            "total_connections": total_connections,
            "total_subscriptions": total_subscriptions,
            "subscription_types": dict(subscription_types),
            "active_processing_tasks": len(self.processing_tasks),
            "total_message_queues": len(self.message_queues)
        }
        
    # Private methods
    
    def _validate_connection(self, connection_id: str) -> bool:
        """Validate that connection exists in WebSocket manager"""
        return connection_id in websocket_manager.active_connections
        
    async def _check_subscription_limits(self, connection_id: str) -> bool:
        """Check if connection can create more subscriptions"""
        current_count = len(self.active_subscriptions.get(connection_id, {}))
        rate_config = self.rate_limits.get(connection_id, self.default_rate_limit)
        return current_count < rate_config.max_subscriptions_per_connection
        
    def _validate_subscription_config(self, config: SubscriptionConfig) -> bool:
        """Validate subscription configuration"""
        if not isinstance(config.subscription_type, SubscriptionType):
            return False
        if config.rate_limit <= 0 or config.rate_limit > 1000:
            return False
        if config.priority < 1 or config.priority > 5:
            return False
        return True
        
    async def _check_rate_limits(self, connection_id: str, subscription_id: str) -> bool:
        """Check if message can be sent within rate limits"""
        now = datetime.utcnow()
        rate_config = self.rate_limits.get(connection_id, self.default_rate_limit)
        
        # Get rate counter for this subscription
        counter = self.rate_counters[connection_id][subscription_id]
        
        # Clean old entries (older than 1 second)
        while counter and (now - counter[0]).total_seconds() > 1.0:
            counter.popleft()
            
        # Check rate limit
        if len(counter) >= rate_config.max_requests_per_second:
            return False
            
        # Add current request
        counter.append(now)
        return True
        
    def _passes_filters(self, message: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if message passes subscription filters"""
        if not filters:
            return True
            
        for filter_key, filter_value in filters.items():
            if filter_key in message:
                if isinstance(filter_value, list):
                    if message[filter_key] not in filter_value:
                        return False
                elif message[filter_key] != filter_value:
                    return False
                    
        return True
        
    async def _subscribe_to_websocket_topics(self, connection_id: str, config: SubscriptionConfig) -> None:
        """Subscribe connection to relevant WebSocket topics"""
        topic = config.subscription_type.value
        websocket_manager.subscribe_to_topic(connection_id, topic)
        
    async def _unsubscribe_from_websocket_topics(self, connection_id: str, config: SubscriptionConfig) -> None:
        """Unsubscribe connection from WebSocket topics"""
        topic = config.subscription_type.value
        websocket_manager.unsubscribe_from_topic(connection_id, topic)
        
    async def _process_messages_for_connection(self, connection_id: str) -> None:
        """Process queued messages for a connection"""
        try:
            while True:
                if connection_id not in self.message_queues:
                    break
                    
                try:
                    # Get message from queue
                    message = await asyncio.wait_for(
                        self.message_queues[connection_id].get(),
                        timeout=5.0
                    )
                    
                    # Send message via WebSocket manager
                    success = await websocket_manager.send_personal_message(message, connection_id)
                    
                    # Update statistics
                    subscription_id = message.get("subscription_id")
                    if subscription_id:
                        if success:
                            self._update_stats(connection_id, subscription_id, "message_sent")
                        else:
                            self._update_stats(connection_id, subscription_id, "message_failed")
                            
                except asyncio.TimeoutError:
                    # No messages in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing message for {connection_id}: {e}")
                    
        except asyncio.CancelledError:
            logger.info(f"Message processor cancelled for connection {connection_id}")
        except Exception as e:
            logger.error(f"Error in message processor for {connection_id}: {e}")
            
    def _update_stats(self, connection_id: str, subscription_id: str, stat_type: str) -> None:
        """Update subscription statistics"""
        if connection_id in self.subscription_stats and subscription_id in self.subscription_stats[connection_id]:
            stats = self.subscription_stats[connection_id][subscription_id]
            
            if stat_type == "message_sent":
                stats["messages_sent"] += 1
                stats["last_activity"] = datetime.utcnow().isoformat()
            elif stat_type == "message_dropped":
                stats["messages_dropped"] += 1
            elif stat_type == "rate_limit_hit":
                stats["rate_limit_hits"] += 1
            elif stat_type in ["message_queued", "message_failed"]:
                stats["last_activity"] = datetime.utcnow().isoformat()
                
    async def _persist_subscription(self, connection_id: str, subscription_id: str, config: SubscriptionConfig) -> None:
        """Persist subscription to Redis"""
        if not self.redis_client:
            return
            
        try:
            key = f"subscriptions:{connection_id}:{subscription_id}"
            data = asdict(config)
            # Convert enum to string for JSON serialization
            data["subscription_type"] = config.subscription_type.value
            await self.redis_client.setex(key, 3600, json.dumps(data))  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to persist subscription: {e}")
            
    async def _remove_persisted_subscription(self, connection_id: str, subscription_id: str) -> None:
        """Remove persisted subscription from Redis"""
        if not self.redis_client:
            return
            
        try:
            key = f"subscriptions:{connection_id}:{subscription_id}"
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to remove persisted subscription: {e}")
            
    async def _load_persisted_subscriptions(self) -> None:
        """Load persisted subscriptions from Redis"""
        if not self.redis_client:
            return
            
        try:
            pattern = "subscriptions:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        config_data = json.loads(data)
                        config_data["subscription_type"] = SubscriptionType(config_data["subscription_type"])
                        config = SubscriptionConfig(**config_data)
                        
                        # Extract connection_id and subscription_id from key
                        key_parts = key.split(":")
                        if len(key_parts) >= 3:
                            connection_id = key_parts[1]
                            subscription_id = key_parts[2]
                            
                            # Only load if connection is still active
                            if self._validate_connection(connection_id):
                                self.active_subscriptions[connection_id][subscription_id] = config
                            else:
                                # Clean up stale subscription
                                await self.redis_client.delete(key)
                                
                except Exception as e:
                    logger.error(f"Error loading persisted subscription from {key}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load persisted subscriptions: {e}")
            
    def _create_subscription_templates(self) -> Dict[SubscriptionType, SubscriptionConfig]:
        """Create default subscription templates"""
        return {
            SubscriptionType.ENGINE_STATUS: SubscriptionConfig(
                subscription_type=SubscriptionType.ENGINE_STATUS,
                parameters={},
                rate_limit=5,
                priority=4
            ),
            SubscriptionType.MARKET_DATA: SubscriptionConfig(
                subscription_type=SubscriptionType.MARKET_DATA,
                parameters={"symbol": ""},
                rate_limit=100,
                priority=3
            ),
            SubscriptionType.TRADE_UPDATES: SubscriptionConfig(
                subscription_type=SubscriptionType.TRADE_UPDATES,
                parameters={},
                rate_limit=50,
                priority=5
            ),
            SubscriptionType.SYSTEM_HEALTH: SubscriptionConfig(
                subscription_type=SubscriptionType.SYSTEM_HEALTH,
                parameters={},
                rate_limit=2,
                priority=2
            )
        }
        
    async def _get_redis_client(self):
        """Get Redis client instance"""
        try:
            from ..redis_cache import redis_cache
            if not redis_cache._connected:
                await redis_cache.connect()
            return redis_cache._redis
        except Exception as e:
            logger.warning(f"Redis not available for subscription persistence: {e}")
            return None


# Global subscription manager instance
subscription_manager = SubscriptionManager()
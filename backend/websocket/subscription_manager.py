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
from dataclasses import dataclass, field
from enum import Enum
import weakref

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
        """Subscribe client to a message type with optional filters"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError(f"Client {client_id} not found")
            
            if len(client.subscriptions) >= self.max_subscriptions_per_client:
                raise ValueError(f"Maximum subscriptions ({self.max_subscriptions_per_client}) reached for client")
            
            subscription_id = str(uuid.uuid4())
            filters = filters or {}
            
            # Create client subscription
            subscription = ClientSubscription(
                subscription_id=subscription_id,
                client_id=client_id,
                subscription_type=subscription_type,
                filters=filters,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            
            # Add to client and global tracking
            client.add_subscription(subscription)
            self.subscriptions[subscription_id] = subscription
            self.total_subscriptions += 1
            
            # Create Redis subscription filter
            redis_filter = self._create_redis_filter(subscription_type, filters)
            
            # Subscribe to Redis channels based on subscription type
            redis_channels = self._get_redis_channels_for_subscription(subscription_type, filters)
            
            for channel in redis_channels:
                redis_sub_id = await self.redis_pubsub.subscribe_to_channel(
                    channel=channel,
                    message_handler=self._create_message_handler(client_id, subscription_id),
                    subscription_filter=redis_filter
                )
                self.redis_subscriptions[f"{subscription_id}_{channel}"] = redis_sub_id
            
            self.logger.info(
                f"Client {client_id} subscribed to {subscription_type.value} "
                f"with subscription {subscription_id}"
            )
            
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Error creating subscription: {e}")
            raise
    
    async def unsubscribe(self, client_id: str, subscription_id: str) -> bool:
        """Unsubscribe client from a specific subscription"""
        try:
            client = self.clients.get(client_id)
            subscription = self.subscriptions.get(subscription_id)
            
            if not client or not subscription:
                return False
            
            # Remove Redis subscriptions
            redis_channels = self._get_redis_channels_for_subscription(
                subscription.subscription_type, 
                subscription.filters
            )
            
            for channel in redis_channels:
                redis_key = f"{subscription_id}_{channel}"
                redis_sub_id = self.redis_subscriptions.get(redis_key)
                
                if redis_sub_id:
                    await self.redis_pubsub.unsubscribe_from_channel(
                        subscription_id=redis_sub_id,
                        channel=channel,
                        message_handler=self._create_message_handler(client_id, subscription_id)
                    )
                    del self.redis_subscriptions[redis_key]
            
            # Remove from client and global tracking
            client.remove_subscription(subscription_id)
            del self.subscriptions[subscription_id]
            
            self.logger.info(f"Unsubscribed {client_id} from {subscription_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing: {e}")
            return False
    
    async def send_message_to_client(
        self, 
        client_id: str, 
        message: WebSocketMessage
    ) -> bool:
        """Send message directly to a specific client"""
        try:
            client = self.clients.get(client_id)
            if not client or client.state != ConnectionState.CONNECTED:
                return False
            
            # Apply client-specific filters
            if not await self._should_send_to_client(client, message):
                return True  # Filtered out, but not an error
            
            # Send via WebSocket
            message_data = json.dumps(message.to_dict())
            await client.websocket.send_text(message_data)
            
            self.messages_sent += 1
            client.update_heartbeat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message to client {client_id}: {e}")
            # Mark client as error state
            if client_id in self.clients:
                self.clients[client_id].state = ConnectionState.ERROR
            self.messages_dropped += 1
            return False
    
    def _create_redis_filter(
        self, 
        subscription_type: SubscriptionType, 
        filters: Dict[str, Any]
    ) -> SubscriptionFilter:
        """Create Redis subscription filter from client subscription"""
        # Map subscription type to message types
        message_types = {
            SubscriptionType.MARKET_DATA: {MessageType.MARKET_DATA},
            SubscriptionType.TRADE_UPDATES: {MessageType.TRADE_UPDATE, MessageType.ORDER_UPDATE},
            SubscriptionType.RISK_ALERTS: {MessageType.RISK_ALERT},
            SubscriptionType.ENGINE_STATUS: {MessageType.ENGINE_STATUS},
            SubscriptionType.SYSTEM_HEALTH: {MessageType.SYSTEM_HEALTH},
            SubscriptionType.PERFORMANCE_UPDATES: {MessageType.PERFORMANCE_UPDATE},
            SubscriptionType.ORDER_UPDATES: {MessageType.ORDER_UPDATE},
            SubscriptionType.POSITION_UPDATES: {MessageType.POSITION_UPDATE},
            SubscriptionType.ALL: set(MessageType)
        }.get(subscription_type, set())
        
        return SubscriptionFilter(
            channel_pattern="*",
            message_types=message_types,
            metadata_filters=filters,
            user_id=filters.get("user_id"),
            portfolio_ids=set(filters.get("portfolio_ids", [])),
            strategy_ids=set(filters.get("strategy_ids", [])),
            symbol_filters=set(filters.get("symbols", []))
        )
    
    def _get_redis_channels_for_subscription(
        self, 
        subscription_type: SubscriptionType, 
        filters: Dict[str, Any]
    ) -> List[str]:
        """Get Redis channels to subscribe to based on subscription type and filters"""
        channels = []
        
        if subscription_type == SubscriptionType.MARKET_DATA:
            symbols = filters.get("symbols", [])
            if symbols:
                for symbol in symbols:
                    channels.append(self.redis_pubsub.generate_channel_name(ChannelType.SYMBOL, symbol))
            else:
                channels.append(self.redis_pubsub.generate_channel_name(ChannelType.BROADCAST))
        
        elif subscription_type in [SubscriptionType.TRADE_UPDATES, SubscriptionType.PERFORMANCE_UPDATES]:
            portfolio_ids = filters.get("portfolio_ids", [])
            if portfolio_ids:
                for portfolio_id in portfolio_ids:
                    channels.append(self.redis_pubsub.generate_channel_name(ChannelType.PORTFOLIO, portfolio_id))
            else:
                channels.append(self.redis_pubsub.generate_channel_name(ChannelType.BROADCAST))
        
        elif subscription_type == SubscriptionType.RISK_ALERTS:
            # Risk alerts go to both portfolio and broadcast channels
            portfolio_ids = filters.get("portfolio_ids", [])
            if portfolio_ids:
                for portfolio_id in portfolio_ids:
                    channels.append(self.redis_pubsub.generate_channel_name(ChannelType.PORTFOLIO, portfolio_id))
            channels.append(self.redis_pubsub.generate_channel_name(ChannelType.BROADCAST))
        
        else:
            # Default to broadcast channel
            channels.append(self.redis_pubsub.generate_channel_name(ChannelType.BROADCAST))
        
        return channels
    
    def _create_message_handler(self, client_id: str, subscription_id: str) -> Callable:
        """Create message handler for Redis subscription"""
        async def handle_message(message: WebSocketMessage):
            try:
                client = self.clients.get(client_id)
                subscription = self.subscriptions.get(subscription_id)
                
                if client and subscription and subscription.is_active:
                    # Update subscription activity
                    subscription.update_activity()
                    
                    # Send message to client
                    await self.send_message_to_client(client_id, message)
                
            except Exception as e:
                self.logger.error(f"Error in message handler for {client_id}/{subscription_id}: {e}")
        
        return handle_message
    
    async def _should_send_to_client(
        self, 
        client: WebSocketClient, 
        message: WebSocketMessage
    ) -> bool:
        """Check if message should be sent to specific client"""
        # Check connection state
        if client.state != ConnectionState.CONNECTED:
            return False
        
        # Check if client has any matching subscriptions
        for subscription in client.get_active_subscriptions():
            if self._subscription_matches_message(subscription, message):
                return True
        
        return False
    
    def _subscription_matches_message(
        self, 
        subscription: ClientSubscription, 
        message: WebSocketMessage
    ) -> bool:
        """Check if subscription should receive this message"""
        # Check subscription type matches
        type_mapping = {
            SubscriptionType.MARKET_DATA: [MessageType.MARKET_DATA],
            SubscriptionType.TRADE_UPDATES: [MessageType.TRADE_UPDATE, MessageType.ORDER_UPDATE],
            SubscriptionType.RISK_ALERTS: [MessageType.RISK_ALERT],
            SubscriptionType.ENGINE_STATUS: [MessageType.ENGINE_STATUS],
            SubscriptionType.SYSTEM_HEALTH: [MessageType.SYSTEM_HEALTH],
            SubscriptionType.PERFORMANCE_UPDATES: [MessageType.PERFORMANCE_UPDATE],
            SubscriptionType.ORDER_UPDATES: [MessageType.ORDER_UPDATE],
            SubscriptionType.POSITION_UPDATES: [MessageType.POSITION_UPDATE],
            SubscriptionType.ALL: list(MessageType)
        }
        
        allowed_types = type_mapping.get(subscription.subscription_type, [])
        if message.message_type not in allowed_types:
            return False
        
        # Apply filters
        filters = subscription.filters
        
        # Symbol filter
        if "symbols" in filters and message.data.get("symbol"):
            if message.data["symbol"] not in filters["symbols"]:
                return False
        
        # Portfolio filter
        if "portfolio_ids" in filters and message.data.get("portfolio_id"):
            if message.data["portfolio_id"] not in filters["portfolio_ids"]:
                return False
        
        # Strategy filter
        if "strategy_ids" in filters and message.data.get("strategy_id"):
            if message.data["strategy_id"] not in filters["strategy_ids"]:
                return False
        
        return True
    
    async def _heartbeat_monitor(self) -> None:
        """Background task to monitor client heartbeats"""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.heartbeat_interval * 2)
                
                # Check for timed out clients
                disconnected_clients = []
                
                for client_id, client in self.clients.items():
                    if client.last_heartbeat < timeout_threshold:
                        disconnected_clients.append(client_id)
                        client.state = ConnectionState.DISCONNECTED
                
                # Disconnect timed out clients
                for client_id in disconnected_clients:
                    await self.disconnect_client(client_id)
                    self.logger.warning(f"Client {client_id} disconnected due to heartbeat timeout")
                
        except asyncio.CancelledError:
            self.logger.info("Heartbeat monitor cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in heartbeat monitor: {e}")
    
    async def _cleanup_expired_subscriptions(self) -> None:
        """Background task to clean up expired subscriptions"""
        try:
            while True:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.subscription_timeout)
                
                # Find expired subscriptions
                expired_subscriptions = []
                
                for subscription_id, subscription in self.subscriptions.items():
                    if subscription.last_activity < timeout_threshold:
                        expired_subscriptions.append(subscription_id)
                
                # Remove expired subscriptions
                for subscription_id in expired_subscriptions:
                    subscription = self.subscriptions[subscription_id]
                    await self.unsubscribe(subscription.client_id, subscription_id)
                    self.logger.info(f"Removed expired subscription {subscription_id}")
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in cleanup task: {e}")
    
    async def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client information"""
        client = self.clients.get(client_id)
        if not client:
            return None
        
        return {
            "client_id": client.client_id,
            "connection_id": client.connection_id,
            "state": client.state.value,
            "connected_at": client.connected_at.isoformat(),
            "last_heartbeat": client.last_heartbeat.isoformat(),
            "user_id": client.user_id,
            "session_id": client.session_id,
            "metadata": client.metadata,
            "subscriptions": [
                {
                    "subscription_id": sub.subscription_id,
                    "type": sub.subscription_type.value,
                    "filters": sub.filters,
                    "message_count": sub.message_count,
                    "error_count": sub.error_count,
                    "created_at": sub.created_at.isoformat(),
                    "last_activity": sub.last_activity.isoformat()
                }
                for sub in client.get_active_subscriptions()
            ]
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get subscription manager statistics"""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "total_subscriptions": self.total_subscriptions,
            "active_subscriptions": len(self.subscriptions),
            "messages_sent": self.messages_sent,
            "messages_dropped": self.messages_dropped,
            "redis_subscriptions": len(self.redis_subscriptions),
            "subscription_types": {
                sub_type.value: len([
                    sub for sub in self.subscriptions.values() 
                    if sub.subscription_type == sub_type
                ])
                for sub_type in SubscriptionType
            }
        }

# Global instance
subscription_manager = None

def get_subscription_manager() -> WebSocketSubscriptionManager:
    """Get global subscription manager instance"""
    global subscription_manager
    if subscription_manager is None:
        raise RuntimeError("Subscription manager not initialized. Call init_subscription_manager() first.")
    return subscription_manager

def init_subscription_manager(
    redis_pubsub_manager: Optional[RedisPubSubManager] = None
) -> WebSocketSubscriptionManager:
    """Initialize global subscription manager instance"""
    global subscription_manager
    subscription_manager = WebSocketSubscriptionManager(redis_pubsub_manager)
    return subscription_manager
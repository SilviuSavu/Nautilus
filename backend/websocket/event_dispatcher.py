"""
Event Dispatcher - Sprint 3 Priority 1

Redis pub/sub integration and event routing system for:
- Real-time event distribution
- Message routing logic
- Event serialization/deserialization
- Cross-service communication
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Set
from enum import Enum
import redis.asyncio as redis

from .websocket_manager import websocket_manager
from .message_protocols import MessageProtocol, EventMessage
# Redis import removed for now - will add back when needed
# from ..redis_cache import get_redis

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for system-wide messaging"""
    ENGINE_STATUS_CHANGED = "engine.status.changed"
    MARKET_DATA_UPDATE = "market.data.update"
    TRADE_EXECUTED = "trade.executed"
    ORDER_STATUS_CHANGED = "order.status.changed"
    POSITION_UPDATED = "position.updated"
    SYSTEM_ALERT = "system.alert"
    USER_ACTION = "user.action"
    STRATEGY_EVENT = "strategy.event"
    RISK_EVENT = "risk.event"
    HEALTH_CHECK = "health.check"


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Event:
    """Event data structure"""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_id = f"{event_type.value}_{datetime.utcnow().timestamp()}"
        self.event_type = event_type
        self.data = data
        self.source = source
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.processed_by: Set[str] = set()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "processed_by": list(self.processed_by)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        event = cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            source=data["source"],
            priority=EventPriority(data["priority"]),
            metadata=data.get("metadata", {})
        )
        event.event_id = data["event_id"]
        event.timestamp = datetime.fromisoformat(data["timestamp"])
        event.processed_by = set(data.get("processed_by", []))
        return event


class EventDispatcher:
    """
    Core event dispatcher for real-time system-wide messaging
    
    Features:
    - Redis pub/sub for cross-service communication
    - Event routing and filtering
    - Priority-based event handling
    - WebSocket integration for real-time client updates
    """
    
    def __init__(self):
        self.redis_client = None
        self.pubsub = None
        self.message_protocol = MessageProtocol()
        
        # Event handlers and routing
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.connection_filters: Dict[str, Dict[str, Any]] = {}  # connection_id -> filters
        
        # Event processing
        self.event_queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.subscriber_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "events_published": 0,
            "events_received": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "last_event_time": None
        }
        
    async def initialize(self) -> bool:
        """
        Initialize the event dispatcher
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Get Redis client (disabled for now)
            self.redis_client = None  # await self._get_redis_client()
            # if not self.redis_client:
            #     logger.error("Failed to get Redis client")
            #     return False
                
            # Setup pub/sub (disabled for now)
            # self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to event channels
            # await self._subscribe_to_event_channels()
            
            # Start background tasks
            self.processing_task = asyncio.create_task(self._process_events())
            # self.subscriber_task = asyncio.create_task(self._handle_redis_messages())
            
            logger.info("Event dispatcher initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize event dispatcher: {e}")
            return False
            
    async def shutdown(self) -> None:
        """Shutdown the event dispatcher"""
        try:
            # Cancel background tasks
            if self.processing_task:
                self.processing_task.cancel()
            if self.subscriber_task:
                self.subscriber_task.cancel()
                
            # Close Redis connections
            if self.pubsub:
                await self.pubsub.unsubscribe()
                await self.pubsub.close()
            if self.redis_client:
                await self.redis_client.close()
                
            logger.info("Event dispatcher shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during event dispatcher shutdown: {e}")
            
    async def publish_event(self, event: Event) -> bool:
        """
        Publish an event to the system
        
        Args:
            event: Event to publish
            
        Returns:
            bool: True if event published successfully
        """
        try:
            # Validate event data
            if not self._validate_event(event):
                logger.warning(f"Invalid event: {event.event_id}")
                return False
                
            # Serialize event
            event_data = json.dumps(event.to_dict())
            
            # Determine Redis channel based on event type
            channel = self._get_event_channel(event.event_type)
            
            # Publish to Redis
            if self.redis_client:
                await self.redis_client.publish(channel, event_data)
                
            # Add to local processing queue for immediate WebSocket dispatch
            await self.event_queue.put(event)
            
            # Update statistics
            self.stats["events_published"] += 1
            self.stats["last_event_time"] = datetime.utcnow().isoformat()
            
            logger.debug(f"Published event: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False
            
    async def subscribe_connection(
        self,
        connection_id: str,
        event_types: List[EventType],
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Subscribe a WebSocket connection to specific event types
        
        Args:
            connection_id: WebSocket connection identifier
            event_types: List of event types to subscribe to
            filters: Optional filters for event data
            
        Returns:
            bool: True if subscription successful
        """
        try:
            # Store connection filters
            if filters:
                self.connection_filters[connection_id] = filters
                
            # Subscribe to topics
            for event_type in event_types:
                topic = event_type.value
                if topic not in self.topic_subscribers:
                    self.topic_subscribers[topic] = set()
                self.topic_subscribers[topic].add(connection_id)
                
            logger.info(f"Connection {connection_id} subscribed to {len(event_types)} event types")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe connection {connection_id}: {e}")
            return False
            
    async def unsubscribe_connection(self, connection_id: str) -> bool:
        """
        Unsubscribe a WebSocket connection from all events
        
        Args:
            connection_id: WebSocket connection identifier
            
        Returns:
            bool: True if unsubscription successful
        """
        try:
            # Remove from all topic subscribers
            for topic_subs in self.topic_subscribers.values():
                topic_subs.discard(connection_id)
                
            # Remove connection filters
            self.connection_filters.pop(connection_id, None)
            
            logger.info(f"Connection {connection_id} unsubscribed from all events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe connection {connection_id}: {e}")
            return False
            
    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register an event handler for specific event type
        
        Args:
            event_type: Type of event to handle
            handler: Async function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
        
    async def dispatch_to_websockets(self, event: Event) -> int:
        """
        Dispatch event to subscribed WebSocket connections
        
        Args:
            event: Event to dispatch
            
        Returns:
            int: Number of connections the event was sent to
        """
        sent_count = 0
        
        try:
            # Get subscribers for this event type
            topic = event.event_type.value
            subscribers = self.topic_subscribers.get(topic, set())
            
            # Filter subscribers based on connection filters
            filtered_subscribers = []
            for connection_id in subscribers:
                if self._passes_filters(event, connection_id):
                    filtered_subscribers.append(connection_id)
                    
            # Create WebSocket message
            ws_message = EventMessage(
                type="event",
                event_type=event.event_type.value,
                data=event.data,
                source=event.source,
                priority=event.priority.value,
                timestamp=event.timestamp,
                event_id=event.event_id
            )
            
            # Send to each subscribed connection
            for connection_id in filtered_subscribers:
                try:
                    success = await websocket_manager.send_personal_message(
                        ws_message.dict(),
                        connection_id
                    )
                    if success:
                        sent_count += 1
                        event.processed_by.add(connection_id)
                except Exception as e:
                    logger.error(f"Failed to send event to {connection_id}: {e}")
                    
            logger.debug(f"Dispatched event {event.event_id} to {sent_count} connections")
            
        except Exception as e:
            logger.error(f"Error dispatching event {event.event_id}: {e}")
            
        return sent_count
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get event dispatcher statistics"""
        return {
            **self.stats,
            "active_subscriptions": sum(len(subs) for subs in self.topic_subscribers.values()),
            "registered_handlers": sum(len(handlers) for handlers in self.event_handlers.values()),
            "queue_size": self.event_queue.qsize(),
            "connection_filters": len(self.connection_filters)
        }
        
    # Private methods
    
    async def _get_redis_client(self):
        """Get Redis client instance"""
        try:
            from ..redis_cache import redis_cache
            if not redis_cache._connected:
                await redis_cache.connect()
            return redis_cache._redis
        except Exception as e:
            logger.error(f"Failed to get Redis client: {e}")
            return None
            
    async def _subscribe_to_event_channels(self) -> None:
        """Subscribe to Redis event channels"""
        channels = [
            "events:engine",
            "events:market",
            "events:trading",
            "events:system",
            "events:user",
            "events:strategy",
            "events:risk"
        ]
        
        for channel in channels:
            await self.pubsub.subscribe(channel)
            
        logger.info(f"Subscribed to {len(channels)} event channels")
        
    async def _handle_redis_messages(self) -> None:
        """Handle incoming Redis pub/sub messages"""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Deserialize event
                        event_data = json.loads(message["data"])
                        event = Event.from_dict(event_data)
                        
                        # Add to processing queue
                        await self.event_queue.put(event)
                        
                        # Update statistics
                        self.stats["events_received"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        self.stats["events_dropped"] += 1
                        
        except asyncio.CancelledError:
            logger.info("Redis message handler cancelled")
        except Exception as e:
            logger.error(f"Error in Redis message handler: {e}")
            
    async def _process_events(self) -> None:
        """Process events from the queue"""
        try:
            while True:
                # Get event from queue
                event = await self.event_queue.get()
                
                try:
                    # Call registered handlers
                    await self._call_event_handlers(event)
                    
                    # Dispatch to WebSocket connections
                    await self.dispatch_to_websockets(event)
                    
                    # Update statistics
                    self.stats["events_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing event {event.event_id}: {e}")
                    self.stats["events_dropped"] += 1
                finally:
                    self.event_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Event processor cancelled")
        except Exception as e:
            logger.error(f"Error in event processor: {e}")
            
    async def _call_event_handlers(self, event: Event) -> None:
        """Call registered handlers for an event"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type.value}: {e}")
                
    def _validate_event(self, event: Event) -> bool:
        """Validate event data"""
        if not event.event_type or not event.data or not event.source:
            return False
        return True
        
    def _get_event_channel(self, event_type: EventType) -> str:
        """Get Redis channel for event type"""
        mapping = {
            EventType.ENGINE_STATUS_CHANGED: "events:engine",
            EventType.MARKET_DATA_UPDATE: "events:market",
            EventType.TRADE_EXECUTED: "events:trading",
            EventType.ORDER_STATUS_CHANGED: "events:trading",
            EventType.POSITION_UPDATED: "events:trading",
            EventType.SYSTEM_ALERT: "events:system",
            EventType.USER_ACTION: "events:user",
            EventType.STRATEGY_EVENT: "events:strategy",
            EventType.RISK_EVENT: "events:risk",
            EventType.HEALTH_CHECK: "events:system"
        }
        return mapping.get(event_type, "events:system")
        
    def _passes_filters(self, event: Event, connection_id: str) -> bool:
        """Check if event passes connection filters"""
        filters = self.connection_filters.get(connection_id, {})
        
        if not filters:
            return True
            
        # Apply filters
        for filter_key, filter_value in filters.items():
            if filter_key == "source" and event.source != filter_value:
                return False
            elif filter_key == "priority" and event.priority.value < filter_value:
                return False
            elif filter_key == "symbols" and "symbol" in event.data:
                if event.data["symbol"] not in filter_value:
                    return False
                    
        return True


# Global event dispatcher instance
event_dispatcher = EventDispatcher()


# Convenience functions for common events
async def publish_engine_status_event(status_data: Dict[str, Any], source: str = "engine_service") -> bool:
    """Publish engine status change event"""
    event = Event(
        event_type=EventType.ENGINE_STATUS_CHANGED,
        data=status_data,
        source=source,
        priority=EventPriority.HIGH
    )
    return await event_dispatcher.publish_event(event)


async def publish_market_data_event(market_data: Dict[str, Any], source: str = "market_service") -> bool:
    """Publish market data update event"""
    event = Event(
        event_type=EventType.MARKET_DATA_UPDATE,
        data=market_data,
        source=source,
        priority=EventPriority.NORMAL
    )
    return await event_dispatcher.publish_event(event)


async def publish_trade_event(trade_data: Dict[str, Any], source: str = "trading_service") -> bool:
    """Publish trade execution event"""
    event = Event(
        event_type=EventType.TRADE_EXECUTED,
        data=trade_data,
        source=source,
        priority=EventPriority.HIGH
    )
    return await event_dispatcher.publish_event(event)


async def publish_system_alert(alert_data: Dict[str, Any], source: str = "system_monitor") -> bool:
    """Publish system alert event"""
    event = Event(
        event_type=EventType.SYSTEM_ALERT,
        data=alert_data,
        source=source,
        priority=EventPriority.CRITICAL
    )
    return await event_dispatcher.publish_event(event)
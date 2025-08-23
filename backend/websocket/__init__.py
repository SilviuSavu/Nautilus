"""
WebSocket Infrastructure Package - Sprint 3 Priority 1

Real-time streaming infrastructure for the Nautilus trading platform.
"""

from .websocket_manager import WebSocketManager, websocket_manager
from .streaming_service import StreamingService
from .event_dispatcher import EventDispatcher, Event, EventType, event_dispatcher
from .subscription_manager import WebSocketSubscriptionManager, SubscriptionType, subscription_manager
from .message_protocols import (
    MessageProtocol, 
    BaseMessage, 
    EngineStatusMessage, 
    MarketDataMessage, 
    TradeUpdateMessage, 
    SystemHealthMessage,
    default_protocol
)

__all__ = [
    # Manager classes
    "WebSocketManager",
    "StreamingService", 
    "EventDispatcher",
    "WebSocketSubscriptionManager",
    "MessageProtocol",
    
    # Global instances
    "websocket_manager",
    "event_dispatcher", 
    "subscription_manager",
    "default_protocol",
    
    # Message types
    "BaseMessage",
    "EngineStatusMessage",
    "MarketDataMessage", 
    "TradeUpdateMessage",
    "SystemHealthMessage",
    
    # Enums and data types
    "Event",
    "EventType",
    "SubscriptionType"
]
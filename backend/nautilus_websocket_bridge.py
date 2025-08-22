"""
Nautilus Trader WebSocket Bridge
Translates Nautilus Trader message bus events to frontend WebSocket messages.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.events import OrderFilled, OrderUpdated, PositionChanged, PositionOpened, PositionClosed
from nautilus_trader.model.orders import Order
from nautilus_trader.model.position import Position
from nautilus_trader.common.component import MessageBus
from nautilus_trader.model.identifiers import InstrumentId


class NautilusWebSocketBridge:
    """
    Bridges Nautilus Trader message bus events to WebSocket messages for frontend.
    """
    
    def __init__(self, websocket_manager):
        self.websocket_manager = websocket_manager
        self._message_bus: Optional[MessageBus] = None
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._running = False
        
        # Message type mappings
        self.event_handlers = {
            # Market data events
            'QuoteTick': self._handle_quote_tick,
            'TradeTick': self._handle_trade_tick,
            'Bar': self._handle_bar,
            
            # Order events  
            'OrderFilled': self._handle_order_filled,
            'OrderUpdated': self._handle_order_updated,
            'OrderAccepted': self._handle_order_accepted,
            'OrderRejected': self._handle_order_rejected,
            'OrderCanceled': self._handle_order_canceled,
            
            # Position events
            'PositionOpened': self._handle_position_opened,
            'PositionChanged': self._handle_position_changed,
            'PositionClosed': self._handle_position_closed,
        }
        
        logging.info("Initialized Nautilus WebSocket Bridge")
    
    def set_message_bus(self, message_bus: MessageBus):
        """Set the Nautilus message bus to listen to."""
        self._message_bus = message_bus
        logging.info("Connected to Nautilus message bus")
    
    async def start(self):
        """Start listening to message bus events."""
        if self._message_bus is None:
            logging.error("Cannot start bridge without message bus")
            return False
            
        if self._running:
            logging.warning("Bridge already running")
            return True
        
        try:
            self._running = True
            
            # Subscribe to all relevant event types
            await self._subscribe_to_events()
            
            logging.info("✅ Nautilus WebSocket Bridge started")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start WebSocket bridge: {e}")
            self._running = False
            return False
    
    async def stop(self):
        """Stop listening to message bus events."""
        if not self._running:
            return True
            
        try:
            self._running = False
            # Unsubscribe from events would go here
            
            logging.info("✅ Nautilus WebSocket Bridge stopped")
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop WebSocket bridge: {e}")
            return False
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant Nautilus events."""
        # This would use Nautilus's subscription mechanism
        # For now, we'll set up a generic event handler
        pass
    
    def handle_event(self, event: Event):
        """Handle incoming Nautilus events and convert to WebSocket messages."""
        if not self._running:
            return
            
        try:
            event_type = event.__class__.__name__
            handler = self.event_handlers.get(event_type)
            
            if handler:
                handler(event)
            else:
                # Generic event handling
                self._handle_generic_event(event)
                
        except Exception as e:
            logging.error(f"Error handling event {event}: {e}")
    
    def _handle_quote_tick(self, tick: QuoteTick):
        """Handle quote tick events."""
        message = {
            "type": "market_data_update",
            "topic": "quotes",
            "payload": {
                "symbol": str(tick.instrument_id),
                "bid": float(tick.bid_price),
                "ask": float(tick.ask_price),
                "bid_size": float(tick.bid_size),
                "ask_size": float(tick.ask_size),
                "timestamp": tick.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_trade_tick(self, tick: TradeTick):
        """Handle trade tick events."""
        message = {
            "type": "market_data_update", 
            "topic": "trades",
            "payload": {
                "symbol": str(tick.instrument_id),
                "price": float(tick.price),
                "size": float(tick.size),
                "timestamp": tick.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_bar(self, bar: Bar):
        """Handle bar events."""
        message = {
            "type": "market_data_update",
            "topic": "bars", 
            "payload": {
                "symbol": str(bar.instrument_id),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "timestamp": bar.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_order_filled(self, event: OrderFilled):
        """Handle order filled events.""" 
        message = {
            "type": "order_update",
            "topic": "orders",
            "payload": {
                "order_id": str(event.order_id),
                "status": "FILLED",
                "fill_price": float(event.last_px) if event.last_px else None,
                "fill_qty": float(event.last_qty) if event.last_qty else None,
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_order_updated(self, event: OrderUpdated):
        """Handle order updated events."""
        message = {
            "type": "order_update",
            "topic": "orders",
            "payload": {
                "order_id": str(event.order_id),
                "status": str(event.order.status),
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_order_accepted(self, event):
        """Handle order accepted events."""
        message = {
            "type": "order_update",
            "topic": "orders",
            "payload": {
                "order_id": str(event.order_id),
                "status": "ACCEPTED",
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_order_rejected(self, event):
        """Handle order rejected events."""
        message = {
            "type": "order_update",
            "topic": "orders", 
            "payload": {
                "order_id": str(event.order_id),
                "status": "REJECTED",
                "reason": getattr(event, 'reason', None),
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_order_canceled(self, event):
        """Handle order canceled events."""
        message = {
            "type": "order_update",
            "topic": "orders",
            "payload": {
                "order_id": str(event.order_id),
                "status": "CANCELED", 
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_position_opened(self, event: PositionOpened):
        """Handle position opened events."""
        message = {
            "type": "position_update",
            "topic": "positions",
            "payload": {
                "position_id": str(event.position.id),
                "instrument_id": str(event.position.instrument_id),
                "side": str(event.position.side),
                "quantity": float(event.position.quantity),
                "avg_open_price": float(event.position.avg_px_open),
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_position_changed(self, event: PositionChanged):
        """Handle position changed events."""
        message = {
            "type": "position_update", 
            "topic": "positions",
            "payload": {
                "position_id": str(event.position.id),
                "instrument_id": str(event.position.instrument_id),
                "side": str(event.position.side),
                "quantity": float(event.position.quantity),
                "avg_open_price": float(event.position.avg_px_open),
                "unrealized_pnl": float(event.position.unrealized_pnl),
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_position_closed(self, event: PositionClosed):
        """Handle position closed events."""
        message = {
            "type": "position_update",
            "topic": "positions", 
            "payload": {
                "position_id": str(event.position.id),
                "instrument_id": str(event.position.instrument_id),
                "status": "CLOSED",
                "realized_pnl": float(event.position.realized_pnl),
                "timestamp": event.ts_event,
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _handle_generic_event(self, event: Event):
        """Handle generic events."""
        message = {
            "type": "nautilus_event",
            "topic": "events",
            "payload": {
                "event_type": event.__class__.__name__,
                "timestamp": getattr(event, 'ts_event', None),
            },
            "timestamp": int(datetime.utcnow().timestamp() * 1000)
        }
        
        self._broadcast_message(message)
    
    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients."""
        try:
            # Use existing WebSocket manager to broadcast
            asyncio.create_task(
                self.websocket_manager.broadcast_json(message)
            )
        except Exception as e:
            logging.error(f"Failed to broadcast message: {e}")


# Global bridge instance
_websocket_bridge: Optional[NautilusWebSocketBridge] = None


def get_websocket_bridge(websocket_manager) -> NautilusWebSocketBridge:
    """Get or create the global WebSocket bridge."""
    global _websocket_bridge
    
    if _websocket_bridge is None:
        _websocket_bridge = NautilusWebSocketBridge(websocket_manager)
    
    return _websocket_bridge


def reset_websocket_bridge():
    """Reset the global WebSocket bridge."""
    global _websocket_bridge
    _websocket_bridge = None
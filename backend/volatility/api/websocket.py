"""
WebSocket Streaming for Volatility Forecasting

This module provides real-time streaming of volatility forecasts and market data
through WebSocket connections.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, List, Optional, Any
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.websockets import WebSocketState

from ..engine.volatility_engine import get_engine, VolatilityEngine
from .models import WebSocketMessage, SubscriptionRequest, ErrorResponse

logger = logging.getLogger(__name__)

# Create WebSocket router
websocket_router = APIRouter(tags=["Volatility WebSocket"])


class ConnectionManager:
    """
    WebSocket connection manager for volatility streaming.
    
    Manages client connections, subscriptions, and message broadcasting.
    """
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions by symbol
        self.symbol_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Stream type subscriptions
        self.stream_subscribers: Dict[str, Dict[str, Set[str]]] = {
            'forecasts': defaultdict(set),
            'realtime': defaultdict(set),
            'all': defaultdict(set)
        }
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Message queues for buffering
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # Statistics
        self.connection_count = 0
        self.messages_sent = 0
        self.start_time = datetime.utcnow()
    
    def generate_connection_id(self) -> str:
        """Generate unique connection ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Connection ID
        """
        await websocket.accept()
        
        connection_id = self.generate_connection_id()
        self.active_connections[connection_id] = websocket
        self.message_queues[connection_id] = asyncio.Queue(maxsize=100)
        self.connection_metadata[connection_id] = {
            'connected_at': datetime.utcnow(),
            'client_ip': websocket.client.host if websocket.client else 'unknown',
            'subscriptions': set(),
            'messages_sent': 0
        }
        
        self.connection_count += 1
        logger.info(f"WebSocket connected: {connection_id} (total: {len(self.active_connections)})")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """
        Handle WebSocket disconnection.
        
        Args:
            connection_id: Connection ID to disconnect
        """
        try:
            # Remove from all subscriptions
            for symbol_subs in self.symbol_subscribers.values():
                symbol_subs.discard(connection_id)
            
            for stream_type in self.stream_subscribers:
                for symbol_subs in self.stream_subscribers[stream_type].values():
                    symbol_subs.discard(connection_id)
            
            # Clean up connection data
            self.active_connections.pop(connection_id, None)
            self.message_queues.pop(connection_id, None)
            self.connection_metadata.pop(connection_id, None)
            
            logger.info(f"WebSocket disconnected: {connection_id} (remaining: {len(self.active_connections)})")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def subscribe(self, connection_id: str, symbol: str, stream_type: str = "forecasts"):
        """
        Subscribe connection to symbol updates.
        
        Args:
            connection_id: Connection ID
            symbol: Trading symbol
            stream_type: Type of stream to subscribe to
        """
        try:
            symbol = symbol.upper()
            
            # Add to symbol subscribers
            self.symbol_subscribers[symbol].add(connection_id)
            
            # Add to stream type subscribers
            self.stream_subscribers[stream_type][symbol].add(connection_id)
            
            # Update metadata
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]['subscriptions'].add(f"{symbol}:{stream_type}")
            
            logger.debug(f"Subscribed {connection_id} to {symbol} ({stream_type})")
            
        except Exception as e:
            logger.error(f"Error subscribing {connection_id} to {symbol}: {e}")
    
    async def unsubscribe(self, connection_id: str, symbol: str, stream_type: str = "forecasts"):
        """
        Unsubscribe connection from symbol updates.
        
        Args:
            connection_id: Connection ID
            symbol: Trading symbol
            stream_type: Type of stream to unsubscribe from
        """
        try:
            symbol = symbol.upper()
            
            # Remove from subscribers
            self.symbol_subscribers[symbol].discard(connection_id)
            self.stream_subscribers[stream_type][symbol].discard(connection_id)
            
            # Update metadata
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]['subscriptions'].discard(f"{symbol}:{stream_type}")
            
            logger.debug(f"Unsubscribed {connection_id} from {symbol} ({stream_type})")
            
        except Exception as e:
            logger.error(f"Error unsubscribing {connection_id} from {symbol}: {e}")
    
    async def send_personal_message(self, connection_id: str, message: Dict[str, Any]):
        """
        Send message to specific connection.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
        """
        if connection_id not in self.active_connections:
            return
        
        websocket = self.active_connections[connection_id]
        
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message, default=str))
                
                # Update statistics
                self.messages_sent += 1
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]['messages_sent'] += 1
                    
        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def broadcast_to_symbol(self, symbol: str, message: Dict[str, Any], stream_type: str = "forecasts"):
        """
        Broadcast message to all subscribers of a symbol.
        
        Args:
            symbol: Trading symbol
            message: Message to broadcast
            stream_type: Stream type for filtering subscribers
        """
        try:
            symbol = symbol.upper()
            
            # Get subscribers for this symbol and stream type
            subscribers = self.stream_subscribers.get(stream_type, {}).get(symbol, set())
            
            # Also include 'all' stream subscribers
            all_subscribers = self.stream_subscribers.get('all', {}).get(symbol, set())
            subscribers = subscribers.union(all_subscribers)
            
            if not subscribers:
                return
            
            # Send to all subscribers
            tasks = []
            for connection_id in subscribers.copy():  # Copy to avoid modification during iteration
                task = asyncio.create_task(self.send_personal_message(connection_id, message))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.debug(f"Broadcasted {stream_type} message to {len(tasks)} subscribers of {symbol}")
                
        except Exception as e:
            logger.error(f"Error broadcasting to symbol {symbol}: {e}")
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return
        
        tasks = []
        for connection_id in list(self.active_connections.keys()):
            task = asyncio.create_task(self.send_personal_message(connection_id, message))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Broadcasted message to {len(tasks)} connections")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'total_connections': self.connection_count,
            'active_connections': len(self.active_connections),
            'messages_sent': self.messages_sent,
            'uptime_seconds': uptime,
            'active_subscriptions': sum(len(subs) for subs in self.symbol_subscribers.values()),
            'unique_symbols': len([s for s, subs in self.symbol_subscribers.items() if subs])
        }


# Global connection manager
connection_manager = ConnectionManager()


@websocket_router.websocket("/ws/volatility")
async def volatility_websocket(
    websocket: WebSocket,
    engine: VolatilityEngine = Depends(get_engine)
):
    """
    Main WebSocket endpoint for volatility streaming.
    
    Handles:
    - Real-time volatility forecasts
    - Market data updates
    - Model status changes
    - Subscription management
    """
    connection_id = None
    
    try:
        # Accept connection
        connection_id = await connection_manager.connect(websocket)
        
        # Send welcome message
        welcome_message = {
            'type': 'welcome',
            'connection_id': connection_id,
            'message': 'Connected to Nautilus Volatility Forecasting Engine',
            'timestamp': datetime.utcnow().isoformat(),
            'available_streams': ['forecasts', 'realtime', 'all']
        }
        await connection_manager.send_personal_message(connection_id, welcome_message)
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    error_msg = {
                        'type': 'error',
                        'error': 'Invalid JSON format',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    await connection_manager.send_personal_message(connection_id, error_msg)
                    continue
                
                # Process message
                await process_websocket_message(connection_id, message, engine)
                
            except WebSocketDisconnect:
                break
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message from {connection_id}: {e}")
                error_msg = {
                    'type': 'error',
                    'error': f'Processing error: {str(e)}',
                    'timestamp': datetime.utcnow().isoformat()
                }
                await connection_manager.send_personal_message(connection_id, error_msg)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {connection_id} disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    
    finally:
        if connection_id:
            await connection_manager.disconnect(connection_id)


async def process_websocket_message(
    connection_id: str,
    message: Dict[str, Any],
    engine: VolatilityEngine
):
    """
    Process incoming WebSocket message from client.
    
    Args:
        connection_id: Client connection ID
        message: Parsed message from client
        engine: Volatility engine instance
    """
    try:
        message_type = message.get('type', '').lower()
        
        if message_type == 'subscribe':
            await handle_subscription(connection_id, message, True)
        
        elif message_type == 'unsubscribe':
            await handle_subscription(connection_id, message, False)
        
        elif message_type == 'get_forecast':
            await handle_forecast_request(connection_id, message, engine)
        
        elif message_type == 'get_status':
            await handle_status_request(connection_id, message, engine)
        
        elif message_type == 'ping':
            await handle_ping(connection_id, message)
        
        else:
            error_msg = {
                'type': 'error',
                'error': f'Unknown message type: {message_type}',
                'timestamp': datetime.utcnow().isoformat()
            }
            await connection_manager.send_personal_message(connection_id, error_msg)
    
    except Exception as e:
        logger.error(f"Error processing message from {connection_id}: {e}")
        error_msg = {
            'type': 'error',
            'error': f'Message processing error: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }
        await connection_manager.send_personal_message(connection_id, error_msg)


async def handle_subscription(connection_id: str, message: Dict[str, Any], subscribe: bool):
    """Handle subscription/unsubscription requests"""
    try:
        symbol = message.get('symbol', '').upper()
        stream_type = message.get('stream_type', 'forecasts')
        
        if not symbol:
            raise ValueError("Symbol is required for subscription")
        
        if subscribe:
            await connection_manager.subscribe(connection_id, symbol, stream_type)
            response_msg = {
                'type': 'subscription_confirmed',
                'action': 'subscribed',
                'symbol': symbol,
                'stream_type': stream_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            await connection_manager.unsubscribe(connection_id, symbol, stream_type)
            response_msg = {
                'type': 'subscription_confirmed',
                'action': 'unsubscribed',
                'symbol': symbol,
                'stream_type': stream_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        await connection_manager.send_personal_message(connection_id, response_msg)
        
    except Exception as e:
        error_msg = {
            'type': 'subscription_error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
        await connection_manager.send_personal_message(connection_id, error_msg)


async def handle_forecast_request(connection_id: str, message: Dict[str, Any], engine: VolatilityEngine):
    """Handle forecast request"""
    try:
        symbol = message.get('symbol', '').upper()
        if not symbol:
            raise ValueError("Symbol is required for forecast request")
        
        # Get latest forecast
        forecast_result = await engine.get_latest_forecast(symbol)
        
        if forecast_result:
            response_msg = {
                'type': 'forecast_response',
                'symbol': symbol,
                'data': forecast_result['forecast'],
                'source': forecast_result.get('source', 'unknown'),
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            response_msg = {
                'type': 'forecast_response',
                'symbol': symbol,
                'error': 'No forecast available',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        await connection_manager.send_personal_message(connection_id, response_msg)
        
    except Exception as e:
        error_msg = {
            'type': 'forecast_error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
        await connection_manager.send_personal_message(connection_id, error_msg)


async def handle_status_request(connection_id: str, message: Dict[str, Any], engine: VolatilityEngine):
    """Handle status request"""
    try:
        status = await engine.get_engine_status()
        
        response_msg = {
            'type': 'status_response',
            'data': status,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await connection_manager.send_personal_message(connection_id, response_msg)
        
    except Exception as e:
        error_msg = {
            'type': 'status_error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
        await connection_manager.send_personal_message(connection_id, error_msg)


async def handle_ping(connection_id: str, message: Dict[str, Any]):
    """Handle ping request"""
    response_msg = {
        'type': 'pong',
        'timestamp': datetime.utcnow().isoformat()
    }
    await connection_manager.send_personal_message(connection_id, response_msg)


# Utility functions for broadcasting from the engine

async def broadcast_forecast_update(symbol: str, forecast: Dict[str, Any]):
    """
    Broadcast forecast update to WebSocket subscribers.
    
    Args:
        symbol: Trading symbol
        forecast: Forecast data
    """
    message = {
        'type': 'forecast_update',
        'symbol': symbol,
        'data': forecast,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    await connection_manager.broadcast_to_symbol(symbol, message, 'forecasts')


async def broadcast_realtime_update(symbol: str, market_data: Dict[str, Any]):
    """
    Broadcast real-time market data update.
    
    Args:
        symbol: Trading symbol
        market_data: Market data
    """
    message = {
        'type': 'realtime_update',
        'symbol': symbol,
        'data': market_data,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    await connection_manager.broadcast_to_symbol(symbol, message, 'realtime')


async def broadcast_model_status_change(symbol: str, model_name: str, status: str):
    """
    Broadcast model status change.
    
    Args:
        symbol: Trading symbol
        model_name: Model name
        status: New status
    """
    message = {
        'type': 'model_status_change',
        'symbol': symbol,
        'model_name': model_name,
        'status': status,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    await connection_manager.broadcast_to_symbol(symbol, message, 'all')


@websocket_router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return connection_manager.get_connection_stats()
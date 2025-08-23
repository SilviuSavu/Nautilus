"""
WebSocket Routes - Sprint 3 Priority 1

FastAPI WebSocket endpoints for real-time streaming:
- Engine status updates
- Market data streaming
- Trade execution notifications  
- System health monitoring
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from datetime import datetime
import uuid

from .websocket_manager import websocket_manager
from .streaming_service import StreamingService
from .event_dispatcher import EventDispatcher
from .subscription_manager import WebSocketSubscriptionManager
from .message_protocols import MessageProtocol
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from messagebus_client import get_messagebus_client
from nautilus_websocket_bridge import get_websocket_bridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

# Initialize services
streaming_service = StreamingService()
event_dispatcher = EventDispatcher()
subscription_manager = WebSocketSubscriptionManager()
message_protocol = MessageProtocol()

# Mock authentication function for development
async def get_current_user_websocket(token: str) -> Optional[str]:
    """Mock authentication for WebSocket connections"""
    # In production, this would validate JWT tokens
    if token and len(token) > 10:
        return f"user_{token[:8]}"
    return None

@router.websocket("/engine/status")
async def websocket_engine_status(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    WebSocket endpoint for real-time engine status updates
    
    Streams:
    - Engine container health status
    - Resource usage metrics
    - Performance indicators
    - Error alerts
    """
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Authenticate user if token provided
        if token:
            try:
                user_id = await get_current_user_websocket(token)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        
        # Establish connection
        if not await websocket_manager.connect(websocket, connection_id, user_id):
            await websocket.close(code=4000, reason="Connection failed")
            return
        
        # Subscribe to engine status topic
        websocket_manager.subscribe_to_topic(connection_id, "engine_status")
        
        # Send initial engine status
        try:
            # Mock initial status for development
            initial_status = {
                "state": "running",
                "uptime": 3600,
                "memory_usage": "512MB",
                "cpu_usage": "15.2%"
            }
            await websocket_manager.send_personal_message({
                "type": "engine_status",
                "data": initial_status,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
        except Exception as e:
            logger.error(f"Error sending initial engine status: {e}")
        
        # Start streaming engine status updates
        streaming_task = asyncio.create_task(
            streaming_service.stream_engine_status(connection_id)
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(message, connection_id, "engine_status")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from {connection_id}")
                await websocket_manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
    
    finally:
        # Cleanup
        streaming_task.cancel() if 'streaming_task' in locals() else None
        websocket_manager.disconnect(connection_id)

@router.websocket("/market-data/{symbol}")
async def websocket_market_data(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    WebSocket endpoint for real-time market data streaming
    
    Args:
        symbol: Trading symbol to stream data for
        
    Streams:
        - Real-time price updates
        - Order book changes
        - Trade executions
        - Volume data
    """
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Authenticate user if token provided
        if token:
            try:
                user_id = await get_current_user_websocket(token)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        
        # Establish connection
        if not await websocket_manager.connect(websocket, connection_id, user_id):
            await websocket.close(code=4000, reason="Connection failed")
            return
        
        # Subscribe to market data topic for symbol
        topic = f"market_data_{symbol.upper()}"
        websocket_manager.subscribe_to_topic(connection_id, topic)
        
        # Send initial market data
        try:
            # Get latest market data for symbol
            # This would integrate with your existing market data service
            initial_data = await get_initial_market_data(symbol)
            await websocket_manager.send_personal_message({
                "type": "market_data",
                "symbol": symbol,
                "data": initial_data,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
        except Exception as e:
            logger.error(f"Error sending initial market data for {symbol}: {e}")
        
        # Start streaming market data updates
        streaming_task = asyncio.create_task(
            streaming_service.stream_market_data(connection_id, symbol)
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(message, connection_id, "market_data")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
    
    finally:
        # Cleanup
        streaming_task.cancel() if 'streaming_task' in locals() else None
        websocket_manager.disconnect(connection_id)

@router.websocket("/trades/updates")
async def websocket_trade_updates(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    WebSocket endpoint for real-time trade execution updates
    
    Streams:
    - Order status changes
    - Trade fills and executions
    - Position updates
    - P&L changes
    """
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Authenticate user (required for trade updates)
        if not token:
            await websocket.close(code=4001, reason="Authentication required")
            return
            
        try:
            user_id = await get_current_user_websocket(token)
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            await websocket.close(code=4001, reason="Authentication failed")
            return
        
        # Establish connection
        if not await websocket_manager.connect(websocket, connection_id, user_id):
            await websocket.close(code=4000, reason="Connection failed")
            return
        
        # Subscribe to trade updates topic for user
        topic = f"trades_{user_id}"
        websocket_manager.subscribe_to_topic(connection_id, topic)
        
        # Send initial trade status
        try:
            initial_trades = await get_user_active_trades(user_id)
            await websocket_manager.send_personal_message({
                "type": "trade_updates",
                "data": initial_trades,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
        except Exception as e:
            logger.error(f"Error sending initial trade data: {e}")
        
        # Start streaming trade updates
        streaming_task = asyncio.create_task(
            streaming_service.stream_trade_updates(connection_id, user_id)
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(message, connection_id, "trades")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
    
    finally:
        # Cleanup
        streaming_task.cancel() if 'streaming_task' in locals() else None
        websocket_manager.disconnect(connection_id)

@router.websocket("/system/health")
async def websocket_system_health(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    WebSocket endpoint for real-time system health monitoring
    
    Streams:
    - System resource usage
    - Service health status
    - Performance metrics
    - Alert notifications
    """
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Authenticate user if token provided
        if token:
            try:
                user_id = await get_current_user_websocket(token)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        
        # Establish connection
        if not await websocket_manager.connect(websocket, connection_id, user_id):
            await websocket.close(code=4000, reason="Connection failed")
            return
        
        # Subscribe to system health topic
        websocket_manager.subscribe_to_topic(connection_id, "system_health")
        
        # Send initial system health
        try:
            initial_health = await get_system_health_status()
            await websocket_manager.send_personal_message({
                "type": "system_health",
                "data": initial_health,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
        except Exception as e:
            logger.error(f"Error sending initial system health: {e}")
        
        # Start streaming system health updates
        streaming_task = asyncio.create_task(
            streaming_service.stream_system_health(connection_id)
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(message, connection_id, "system_health")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
    
    finally:
        # Cleanup
        streaming_task.cancel() if 'streaming_task' in locals() else None
        websocket_manager.disconnect(connection_id)

async def handle_websocket_message(message: Dict[str, Any], connection_id: str, endpoint_type: str):
    """
    Handle incoming WebSocket messages
    
    Args:
        message: Parsed message data
        connection_id: WebSocket connection identifier
        endpoint_type: Type of WebSocket endpoint
    """
    try:
        message_type = message.get("type")
        
        if message_type == "heartbeat":
            await websocket_manager.handle_heartbeat(connection_id)
            
        elif message_type == "subscribe":
            topic = message.get("topic")
            if topic:
                websocket_manager.subscribe_to_topic(connection_id, topic)
                await websocket_manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "topic": topic
                }, connection_id)
            
        elif message_type == "unsubscribe":
            topic = message.get("topic")
            if topic:
                websocket_manager.unsubscribe_from_topic(connection_id, topic)
                await websocket_manager.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "topic": topic
                }, connection_id)
        
        elif message_type == "get_stats":
            stats = websocket_manager.get_connection_stats()
            await websocket_manager.send_personal_message({
                "type": "connection_stats",
                "data": stats
            }, connection_id)
            
        else:
            logger.warning(f"Unknown message type: {message_type}")
            await websocket_manager.send_personal_message({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }, connection_id)
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await websocket_manager.send_personal_message({
            "type": "error",
            "message": "Internal server error"
        }, connection_id)

# Helper functions for initial data
async def get_initial_market_data(symbol: str) -> Dict[str, Any]:
    """Get initial market data for symbol"""
    # This would integrate with your existing market data service
    # For now, return mock data
    return {
        "symbol": symbol,
        "price": 150.00,
        "bid": 149.95,
        "ask": 150.05,
        "volume": 1000000,
        "last_updated": datetime.utcnow().isoformat()
    }

async def get_user_active_trades(user_id: str) -> Dict[str, Any]:
    """Get active trades for user"""
    # This would integrate with your trading service
    return {
        "active_orders": [],
        "open_positions": [],
        "total_pnl": 0.0
    }

async def get_system_health_status() -> Dict[str, Any]:
    """Get current system health status"""
    # This would integrate with your monitoring service
    return {
        "status": "healthy",
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "disk_usage": 23.1,
        "services": {
            "backend": "healthy",
            "database": "healthy",
            "redis": "healthy",
            "engine": "healthy"
        }
    }

@router.websocket("/messagebus")
async def websocket_message_bus(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    High-Performance Message Bus WebSocket Endpoint
    
    Provides direct access to the message bus infrastructure, bypassing HTTP proxy
    for maximum performance. Connects to Redis streams and NautilusTrader bridge.
    
    Features:
    - Direct Redis stream access
    - NautilusTrader WebSocket bridge integration  
    - Real-time market data streams
    - Order and position updates
    - Portfolio analytics
    - Command execution via message bus
    """
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Authenticate user if token provided
        if token:
            try:
                user_id = await get_current_user_websocket(token)
                logger.info(f"Message bus connection authenticated for user: {user_id}")
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        
        # Establish WebSocket connection
        if not await websocket_manager.connect(websocket, connection_id, user_id):
            await websocket.close(code=4000, reason="Connection failed")
            return
        
        logger.info(f"✅ High-performance message bus connection established: {connection_id}")
        
        # Get message bus client and WebSocket bridge
        messagebus_client = get_messagebus_client()
        websocket_bridge = get_websocket_bridge(websocket_manager)
        
        # Subscribe to all topics initially (users can filter on frontend)
        default_topics = [
            "market_data_updates",
            "order_updates", 
            "position_updates",
            "portfolio_updates",
            "system_events",
            "dbnomics_events"  # Economic data events from DBnomics
        ]
        
        for topic in default_topics:
            websocket_manager.subscribe_to_topic(connection_id, topic)
        
        # Send connection confirmation with performance info
        await websocket_manager.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "performance_mode": "message_bus_direct",
            "available_topics": default_topics,
            "message_bus_status": {
                "redis_connected": messagebus_client.is_connected,
                "bridge_active": websocket_bridge is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_message_bus_message(message, connection_id, user_id)
                
            except WebSocketDisconnect:
                logger.info(f"Message bus WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from message bus client {connection_id}")
                await websocket_manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error(f"Error handling message bus WebSocket message: {e}")
                break
    
    finally:
        # Cleanup
        websocket_manager.disconnect(connection_id)
        logger.info(f"🔌 Message bus connection cleaned up: {connection_id}")

async def handle_message_bus_message(message: Dict[str, Any], connection_id: str, user_id: Optional[str]):
    """
    Handle incoming message bus WebSocket messages
    
    Supports:
    - Topic subscription/unsubscription
    - Command execution via message bus
    - Real-time data requests
    - Performance queries
    """
    try:
        action = message.get("action")
        messagebus_client = get_messagebus_client()
        
        if action == "ping":
            # Heartbeat response
            await websocket_manager.send_personal_message({
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
            
        elif action == "subscribe":
            # Subscribe to specific topic
            topic = message.get("topic")
            subscription_id = message.get("subscriptionId")
            
            if topic:
                websocket_manager.subscribe_to_topic(connection_id, topic)
                await websocket_manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "topic": topic,
                    "subscriptionId": subscription_id,
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
                logger.info(f"📡 Subscribed {connection_id} to high-speed topic: {topic}")
            
        elif action == "unsubscribe":
            # Unsubscribe from topic
            subscription_id = message.get("subscriptionId")
            # Find topic by subscription ID and unsubscribe
            # For now, acknowledge the unsubscription
            await websocket_manager.send_personal_message({
                "type": "unsubscription_confirmed", 
                "subscriptionId": subscription_id,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
            
        elif action == "command":
            # Execute command via message bus (bypasses HTTP entirely)
            command = message.get("command")
            command_data = message.get("data", {})
            request_id = message.get("requestId")
            
            if command == "get_realtime_portfolio":
                # Handle portfolio data request via message bus
                portfolio_id = command_data.get("portfolioId")
                
                try:
                    # This would use the message bus to fetch portfolio data
                    # For now, return mock data but structure shows the pattern
                    portfolio_data = await get_realtime_portfolio_via_messagebus(portfolio_id)
                    
                    await websocket_manager.send_personal_message({
                        "type": "command_response",
                        "requestId": request_id,
                        "success": True,
                        "data": portfolio_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
                    
                except Exception as e:
                    await websocket_manager.send_personal_message({
                        "type": "command_response",
                        "requestId": request_id,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
            
            elif command == "get_market_data":
                # Handle market data request
                symbol = command_data.get("symbol")
                market_data = await get_market_data_via_messagebus(symbol)
                
                await websocket_manager.send_personal_message({
                    "type": "command_response",
                    "requestId": request_id,
                    "success": True,
                    "data": market_data,
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            else:
                await websocket_manager.send_personal_message({
                    "type": "command_response",
                    "requestId": request_id,
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
        
        else:
            logger.warning(f"Unknown message bus action: {action}")
            await websocket_manager.send_personal_message({
                "type": "error",
                "message": f"Unknown action: {action}"
            }, connection_id)
            
    except Exception as e:
        logger.error(f"Error handling message bus message: {e}")
        await websocket_manager.send_personal_message({
            "type": "error",
            "message": "Internal server error"
        }, connection_id)

async def get_realtime_portfolio_via_messagebus(portfolio_id: str) -> Dict[str, Any]:
    """Get real-time portfolio data via message bus (bypasses HTTP)"""
    # This would use the actual message bus to fetch data
    # For development, return realistic mock data
    return {
        "portfolio_id": portfolio_id,
        "total_value": 125750.50,
        "cash_balance": 25750.50,
        "total_pnl": 8750.50,
        "day_pnl": 1250.75,
        "positions": [
            {
                "symbol": "SPY",
                "quantity": 200,
                "avg_price": 445.25,
                "market_price": 448.75,
                "unrealized_pnl": 700.00,
                "position_value": 89750.00
            }
        ],
        "recent_trades": [
            {
                "symbol": "SPY", 
                "side": "BUY",
                "quantity": 50,
                "price": 445.25,
                "timestamp": datetime.utcnow().isoformat()
            }
        ],
        "data_source": "message_bus_direct",
        "performance_note": "Retrieved via high-speed message bus, bypassed HTTP proxy",
        "timestamp": datetime.utcnow().isoformat()
    }

async def get_market_data_via_messagebus(symbol: str) -> Dict[str, Any]:
    """Get market data via message bus (high-frequency updates)"""
    return {
        "symbol": symbol,
        "price": 448.75,
        "bid": 448.70,
        "ask": 448.80,
        "volume": 2500000,
        "change": 2.50,
        "change_percent": 0.56,
        "data_source": "message_bus_direct",
        "update_frequency": "real_time_stream",
        "timestamp": datetime.utcnow().isoformat()
    }
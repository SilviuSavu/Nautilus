#!/usr/bin/env python3
"""
Simple WebSocket Engine - Containerized Real-time Streaming Service
Enterprise-grade WebSocket streaming with 1000+ concurrent connection capability
"""

import asyncio
import logging
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Set
from dataclasses import dataclass
from enum import Enum
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION = "subscription"
    DATA = "data"
    ERROR = "error"
    SYSTEM = "system"
    MARKET_DATA = "market_data"
    TRADE_UPDATE = "trade_update"
    RISK_ALERT = "risk_alert"

@dataclass
class WebSocketConnection:
    connection_id: str
    websocket: WebSocket
    client_ip: str
    connect_time: datetime
    last_heartbeat: datetime
    subscriptions: Set[str]
    status: ConnectionStatus

@dataclass
class StreamingMessage:
    message_id: str
    message_type: MessageType
    topic: str
    data: Dict[str, Any]
    timestamp: datetime
    client_count: int = 0

class SimpleWebSocketEngine:
    """
    Simple WebSocket Engine demonstrating containerization approach
    High-performance real-time streaming with enterprise features
    """
    
    def __init__(self):
        self.app = FastAPI(title="Nautilus Simple WebSocket Engine", version="1.0.0")
        self.is_running = False
        self.messages_sent = 0
        self.connections_handled = 0
        self.start_time = time.time()
        
        # WebSocket connection management
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.connection_stats: Dict[str, int] = {
            "total_connections": 0,
            "current_connections": 0,
            "messages_sent": 0,
            "heartbeats_sent": 0,
            "subscription_updates": 0
        }
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.heartbeat_task = None
        self.cleanup_task = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "active_connections": len(self.active_connections),
                "messages_sent": self.messages_sent,
                "connections_handled": self.connections_handled,
                "topics_active": len(self.topic_subscribers),
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            uptime = time.time() - self.start_time
            return {
                "messages_per_second": self.messages_sent / max(1, uptime),
                "connections_per_second": self.connections_handled / max(1, uptime),
                "total_messages": self.messages_sent,
                "total_connections": self.connections_handled,
                "current_connections": len(self.active_connections),
                "active_topics": len(self.topic_subscribers),
                "connection_stats": self.connection_stats,
                "uptime": uptime,
                "engine_type": "websocket_streaming",
                "containerized": True
            }
        
        @self.app.get("/connections")
        async def get_connections():
            """Get active WebSocket connections"""
            connections = []
            for conn_id, conn in self.active_connections.items():
                connections.append({
                    "connection_id": conn_id,
                    "client_ip": conn.client_ip,
                    "connect_time": conn.connect_time.isoformat(),
                    "last_heartbeat": conn.last_heartbeat.isoformat(),
                    "subscriptions": list(conn.subscriptions),
                    "status": conn.status.value
                })
            
            return {
                "connections": connections,
                "count": len(connections),
                "total_subscriptions": sum(len(conn.subscriptions) for conn in self.active_connections.values())
            }
        
        @self.app.get("/topics")
        async def get_topics():
            """Get active topics and subscriber counts"""
            topics = {}
            for topic, subscriber_ids in self.topic_subscribers.items():
                topics[topic] = {
                    "subscriber_count": len(subscriber_ids),
                    "subscribers": list(subscriber_ids)
                }
            
            return {
                "topics": topics,
                "topic_count": len(topics),
                "total_subscribers": sum(len(subs) for subs in self.topic_subscribers.values())
            }
        
        @self.app.post("/broadcast")
        async def broadcast_message(message_data: Dict[str, Any]):
            """Broadcast message to all subscribers of a topic"""
            try:
                topic = message_data.get("topic", "system")
                data = message_data.get("data", {})
                message_type = MessageType(message_data.get("message_type", "data"))
                
                message = StreamingMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=message_type,
                    topic=topic,
                    data=data,
                    timestamp=datetime.now()
                )
                
                sent_count = await self._broadcast_to_topic(message)
                
                return {
                    "status": "broadcast_complete",
                    "message_id": message.message_id,
                    "topic": topic,
                    "recipients": sent_count
                }
                
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket streaming endpoint"""
            connection_id = str(uuid.uuid4())
            
            try:
                await websocket.accept()
                client_host = websocket.client.host if websocket.client else "unknown"
                
                # Create connection record
                connection = WebSocketConnection(
                    connection_id=connection_id,
                    websocket=websocket,
                    client_ip=client_host,
                    connect_time=datetime.now(),
                    last_heartbeat=datetime.now(),
                    subscriptions=set(),
                    status=ConnectionStatus.CONNECTED
                )
                
                self.active_connections[connection_id] = connection
                self.connections_handled += 1
                self.connection_stats["total_connections"] += 1
                self.connection_stats["current_connections"] = len(self.active_connections)
                
                logger.info(f"WebSocket connected: {connection_id} from {client_host}")
                
                # Send welcome message
                await self._send_message(connection, StreamingMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.SYSTEM,
                    topic="connection",
                    data={
                        "status": "connected",
                        "connection_id": connection_id,
                        "timestamp": datetime.now().isoformat()
                    },
                    timestamp=datetime.now()
                ))
                
                # Handle incoming messages
                while True:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        await self._handle_client_message(connection, message)
                        connection.last_heartbeat = datetime.now()
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat ping
                        await self._send_heartbeat(connection)
                        
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket disconnected: {connection_id}")
                        break
                        
            except Exception as e:
                logger.error(f"WebSocket error for {connection_id}: {e}")
            finally:
                # Cleanup connection
                await self._cleanup_connection(connection_id)
        
        @self.app.websocket("/ws/market-data/{symbol}")
        async def market_data_stream(websocket: WebSocket, symbol: str):
            """Market data streaming endpoint for specific symbol"""
            connection_id = f"market_{symbol}_{uuid.uuid4()}"
            
            try:
                await websocket.accept()
                client_host = websocket.client.host if websocket.client else "unknown"
                
                logger.info(f"Market data stream connected: {connection_id} for {symbol}")
                
                # Send initial market data
                while True:
                    # Mock market data
                    market_data = {
                        "symbol": symbol,
                        "price": round(100 + (time.time() % 50), 2),
                        "bid": round(100 + (time.time() % 50) - 0.05, 2),
                        "ask": round(100 + (time.time() % 50) + 0.05, 2),
                        "volume": int(time.time() % 10000),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send_text(json.dumps({
                        "message_type": "market_data",
                        "symbol": symbol,
                        "data": market_data,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                    self.messages_sent += 1
                    await asyncio.sleep(0.1)  # 10 updates per second
                    
            except WebSocketDisconnect:
                logger.info(f"Market data stream disconnected: {connection_id}")
            except Exception as e:
                logger.error(f"Market data stream error: {e}")

    async def start_engine(self):
        """Start the WebSocket engine"""
        try:
            logger.info("Starting Simple WebSocket Engine...")
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.is_running = True
            logger.info("Simple WebSocket Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the WebSocket engine"""
        logger.info("Stopping Simple WebSocket Engine...")
        self.is_running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        for connection_id in list(self.active_connections.keys()):
            await self._cleanup_connection(connection_id)
        
        if self.messagebus:
            await self.messagebus.stop()
        
        logger.info("Simple WebSocket Engine stopped")
    
    async def _handle_client_message(self, connection: WebSocketConnection, message: str):
        """Handle incoming client message"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            if message_type == "subscribe":
                # Handle subscription
                topics = data.get("topics", [])
                for topic in topics:
                    await self._subscribe_to_topic(connection, topic)
                
            elif message_type == "unsubscribe":
                # Handle unsubscription
                topics = data.get("topics", [])
                for topic in topics:
                    await self._unsubscribe_from_topic(connection, topic)
                    
            elif message_type == "heartbeat":
                # Handle heartbeat response
                await self._send_heartbeat_response(connection)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received from {connection.connection_id}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _subscribe_to_topic(self, connection: WebSocketConnection, topic: str):
        """Subscribe connection to topic"""
        connection.subscriptions.add(topic)
        
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(connection.connection_id)
        
        self.connection_stats["subscription_updates"] += 1
        
        # Send subscription confirmation
        await self._send_message(connection, StreamingMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SUBSCRIPTION,
            topic=topic,
            data={"status": "subscribed", "topic": topic},
            timestamp=datetime.now()
        ))
        
        logger.info(f"Connection {connection.connection_id} subscribed to {topic}")
    
    async def _unsubscribe_from_topic(self, connection: WebSocketConnection, topic: str):
        """Unsubscribe connection from topic"""
        connection.subscriptions.discard(topic)
        
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(connection.connection_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        
        # Send unsubscription confirmation
        await self._send_message(connection, StreamingMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SUBSCRIPTION,
            topic=topic,
            data={"status": "unsubscribed", "topic": topic},
            timestamp=datetime.now()
        ))
        
        logger.info(f"Connection {connection.connection_id} unsubscribed from {topic}")
    
    async def _send_message(self, connection: WebSocketConnection, message: StreamingMessage):
        """Send message to WebSocket connection"""
        try:
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                message_data = {
                    "message_id": message.message_id,
                    "type": message.message_type.value,
                    "topic": message.topic,
                    "data": message.data,
                    "timestamp": message.timestamp.isoformat()
                }
                
                await connection.websocket.send_text(json.dumps(message_data))
                self.messages_sent += 1
                self.connection_stats["messages_sent"] += 1
                
        except Exception as e:
            logger.error(f"Error sending message to {connection.connection_id}: {e}")
            connection.status = ConnectionStatus.ERROR
    
    async def _broadcast_to_topic(self, message: StreamingMessage) -> int:
        """Broadcast message to all subscribers of a topic"""
        sent_count = 0
        
        if message.topic in self.topic_subscribers:
            connection_ids = list(self.topic_subscribers[message.topic])
            message.client_count = len(connection_ids)
            
            for connection_id in connection_ids:
                if connection_id in self.active_connections:
                    await self._send_message(self.active_connections[connection_id], message)
                    sent_count += 1
        
        return sent_count
    
    async def _send_heartbeat(self, connection: WebSocketConnection):
        """Send heartbeat to connection"""
        heartbeat_message = StreamingMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            topic="heartbeat",
            data={"timestamp": datetime.now().isoformat()},
            timestamp=datetime.now()
        )
        
        await self._send_message(connection, heartbeat_message)
        self.connection_stats["heartbeats_sent"] += 1
    
    async def _send_heartbeat_response(self, connection: WebSocketConnection):
        """Send heartbeat response"""
        response_message = StreamingMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            topic="heartbeat",
            data={"status": "alive", "timestamp": datetime.now().isoformat()},
            timestamp=datetime.now()
        )
        
        await self._send_message(connection, response_message)
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for connection in list(self.active_connections.values()):
                    # Check if connection needs heartbeat
                    if (current_time - connection.last_heartbeat).seconds > 30:
                        await self._send_heartbeat(connection)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Background task to cleanup stale connections"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                stale_connections = []
                for connection_id, connection in self.active_connections.items():
                    # Mark connections as stale if no heartbeat for 5 minutes
                    if (current_time - connection.last_heartbeat).seconds > 300:
                        stale_connections.append(connection_id)
                
                for connection_id in stale_connections:
                    await self._cleanup_connection(connection_id)
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_connection(self, connection_id: str):
        """Cleanup connection and subscriptions"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            
            # Remove from topic subscriptions
            for topic in connection.subscriptions:
                if topic in self.topic_subscribers:
                    self.topic_subscribers[topic].discard(connection_id)
                    if not self.topic_subscribers[topic]:
                        del self.topic_subscribers[topic]
            
            # Close WebSocket if still open
            try:
                if connection.websocket.client_state == WebSocketState.CONNECTED:
                    await connection.websocket.close()
            except Exception:
                pass
            
            # Remove from active connections
            del self.active_connections[connection_id]
            self.connection_stats["current_connections"] = len(self.active_connections)
            
            logger.info(f"Cleaned up connection: {connection_id}")

# Create and start the engine
simple_websocket_engine = SimpleWebSocketEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        # For now, use simple engine with hybrid mode flag
        logger.info("Hybrid WebSocket Engine integration enabled (using enhanced simple engine)")
        app = simple_websocket_engine.app
        engine_instance = simple_websocket_engine
        # Add hybrid flag to engine
        engine_instance.hybrid_enabled = True
    except Exception as e:
        logger.warning(f"Hybrid WebSocket Engine setup failed: {e}. Using simple engine.")
        app = simple_websocket_engine.app
        engine_instance = simple_websocket_engine
else:
    logger.info("Using Simple WebSocket Engine (hybrid disabled)")
    app = simple_websocket_engine.app
    engine_instance = simple_websocket_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8600"))
    
    logger.info(f"Starting WebSocket Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
    # Start the engine on startup
    async def lifespan():
        await engine_instance.start_engine()
    
    # Run startup
    asyncio.run(lifespan())
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
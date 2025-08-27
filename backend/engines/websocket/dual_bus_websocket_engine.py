#!/usr/bin/env python3
"""
Dual Bus WebSocket Engine - Real-time Streaming with Dual MessageBus
Provides WebSocket endpoints for real-time data streaming via specialized message buses
"""

import asyncio
import logging
import sys
import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# Add backend to path
sys.path.append('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# Import dual messagebus client
try:
    from dual_messagebus_client import get_dual_bus_client, EngineType, MessageType
    DUAL_MESSAGEBUS_AVAILABLE = True
except ImportError:
    print("❌ dual_messagebus_client not available - using fallback mode")
    DUAL_MESSAGEBUS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualBusWebSocketEngine:
    def __init__(self):
        self.engine_id = "websocket-8600"
        self.active_connections: List[WebSocket] = []
        self.messagebus_client = None
        self.subscription_tasks = []
        self.message_count = 0
        self.start_time = time.time()
        
    async def initialize_messagebus(self):
        """Initialize dual messagebus client"""
        if DUAL_MESSAGEBUS_AVAILABLE:
            try:
                self.messagebus_client = await get_dual_bus_client(EngineType.WEBSOCKET)
                
                # Subscribe to market data from MarketData Bus
                await self.messagebus_client.subscribe_to_marketdata(
                    "market_data", self.handle_market_data_message
                )
                await self.messagebus_client.subscribe_to_marketdata(
                    "price_update", self.handle_price_update
                )
                
                # Subscribe to engine logic from Engine Logic Bus  
                await self.messagebus_client.subscribe_to_engine_logic(
                    "analytics_result", self.handle_analytics_result
                )
                await self.messagebus_client.subscribe_to_engine_logic(
                    "risk_alert", self.handle_risk_alert
                )
                
                logger.info("✅ Dual MessageBus connected - WebSocket Engine ready for real-time streaming")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize dual messagebus: {e}")
                return False
        return False
    
    async def handle_market_data_message(self, message: Dict[str, Any]):
        """Handle market data from MarketData Bus and broadcast via WebSocket"""
        self.message_count += 1
        websocket_message = {
            "type": "market_data",
            "timestamp": datetime.now().isoformat(),
            "data": message,
            "source": "marketdata_bus_6380"
        }
        await self.broadcast_to_websockets(websocket_message)
    
    async def handle_price_update(self, message: Dict[str, Any]):
        """Handle price updates and stream to WebSocket clients"""
        self.message_count += 1
        websocket_message = {
            "type": "price_update", 
            "timestamp": datetime.now().isoformat(),
            "data": message,
            "source": "marketdata_bus_6380"
        }
        await self.broadcast_to_websockets(websocket_message)
    
    async def handle_analytics_result(self, message: Dict[str, Any]):
        """Handle analytics results from Engine Logic Bus"""
        self.message_count += 1
        websocket_message = {
            "type": "analytics_result",
            "timestamp": datetime.now().isoformat(), 
            "data": message,
            "source": "engine_logic_bus_6381"
        }
        await self.broadcast_to_websockets(websocket_message)
    
    async def handle_risk_alert(self, message: Dict[str, Any]):
        """Handle risk alerts from Engine Logic Bus"""
        self.message_count += 1
        websocket_message = {
            "type": "risk_alert",
            "timestamp": datetime.now().isoformat(),
            "data": message, 
            "source": "engine_logic_bus_6381",
            "priority": "HIGH"
        }
        await self.broadcast_to_websockets(websocket_message)
    
    async def broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if not self.active_connections:
            return
            
        disconnected_clients = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket client: {e}")
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.active_connections:
                self.active_connections.remove(client)
    
    async def connect_websocket(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ WebSocket client connected. Total: {len(self.active_connections)}")
        
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to Dual Bus WebSocket Engine",
            "engine_info": {
                "engine_id": self.engine_id,
                "dual_messagebus": True,
                "marketdata_bus": 6380,
                "engine_logic_bus": 6381
            }
        }
        await websocket.send_json(welcome_message)
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"❌ WebSocket client disconnected. Total: {len(self.active_connections)}")

# Global engine instance
websocket_engine = DualBusWebSocketEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("🚀 Starting Dual Bus WebSocket Engine Server...")
    logger.info("   Architecture: DUAL REDIS BUSES")
    logger.info("   📡 MarketData Bus: localhost:6380 (Real-time market data streaming)")
    logger.info("   ⚙️ Engine Logic Bus: localhost:6381 (Analytics & risk alerts)")
    logger.info("   🌐 WebSocket Endpoints: Real-time data distribution")
    
    # Initialize dual messagebus
    messagebus_connected = await websocket_engine.initialize_messagebus()
    if messagebus_connected:
        logger.info("✅ Dual Bus WebSocket Engine started successfully")
        logger.info("   📡 MarketData Bus (Port 6380): Real-time streaming")
        logger.info("   ⚙️ Engine Logic Bus (Port 6381): Analytics & alerts")
    else:
        logger.warning("⚠️ Running without dual messagebus - limited functionality")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Dual Bus WebSocket Engine...")

app = FastAPI(
    title="Dual Bus WebSocket Engine",
    description="Real-time WebSocket streaming with dual messagebus architecture", 
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """WebSocket Engine health check with dual messagebus status"""
    uptime = time.time() - websocket_engine.start_time
    
    return {
        "status": "healthy",
        "engine": "websocket",
        "port": 8600,
        "architecture": "dual_bus",
        "marketdata_bus": "6380", 
        "engine_logic_bus": "6381",
        "timestamp": time.time(),
        "websocket_connections": len(websocket_engine.active_connections),
        "messages_streamed": websocket_engine.message_count,
        "uptime_seconds": uptime,
        "dual_messagebus_connected": websocket_engine.messagebus_client is not None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data streaming"""
    await websocket_engine.connect_websocket(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            message = await websocket.receive_text()
            try:
                client_message = json.loads(message)
                # Echo back client message with timestamp
                response = {
                    "type": "client_message_received",
                    "timestamp": datetime.now().isoformat(),
                    "received": client_message
                }
                await websocket.send_json(response)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
    except WebSocketDisconnect:
        websocket_engine.disconnect_websocket(websocket)

@app.websocket("/ws/market_data")
async def market_data_websocket(websocket: WebSocket):
    """Dedicated WebSocket for market data streaming only"""
    await websocket_engine.connect_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        websocket_engine.disconnect_websocket(websocket)

@app.websocket("/ws/alerts") 
async def alerts_websocket(websocket: WebSocket):
    """Dedicated WebSocket for risk alerts and notifications"""
    await websocket_engine.connect_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        websocket_engine.disconnect_websocket(websocket)

@app.get("/connections")
async def get_connections():
    """Get current WebSocket connection count"""
    return {
        "active_connections": len(websocket_engine.active_connections),
        "messages_streamed": websocket_engine.message_count,
        "uptime_seconds": time.time() - websocket_engine.start_time
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8600))
    logger.info(f"🚀 Starting Dual Bus WebSocket Engine on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
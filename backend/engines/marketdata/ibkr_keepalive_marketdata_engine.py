#!/usr/bin/env python3
"""
IBKR Keep-Alive MarketData Engine - Using Repository's Configured IB Connector
Maintains persistent IBKR connection with dual messagebus integration
"""

import asyncio
import logging
import sys
import os
import time
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum

# Add backend to path
sys.path.append('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import existing MarketData client (MANDATORY - as specified in repository)
from engines.analytics.marketdata_client import (
    MarketDataClient, 
    create_marketdata_client,
    DataSource, 
    DataType
)

# Import dual messagebus client
try:
    from dual_messagebus_client import get_dual_bus_client, EngineType, MessageType
    DUAL_MESSAGEBUS_AVAILABLE = True
except ImportError:
    print("❌ dual_messagebus_client not available - using fallback mode")
    DUAL_MESSAGEBUS_AVAILABLE = False

# Import universal messagebus (as required by existing MarketData client)
try:
    from universal_enhanced_messagebus_client import create_messagebus_client, EngineType as UniversalEngineType
    UNIVERSAL_MESSAGEBUS_AVAILABLE = True
except ImportError:
    print("⚠️ universal_enhanced_messagebus_client not available")
    UNIVERSAL_MESSAGEBUS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBKRConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class IBKRKeepAliveMarketDataEngine:
    def __init__(self):
        self.engine_id = "marketdata-8800"
        self.port = 8800
        
        # IBKR Connection Management
        self.ibkr_status = IBKRConnectionStatus.DISCONNECTED
        self.ibkr_client: Optional[MarketDataClient] = None
        self.connection_attempts = 0
        self.last_heartbeat = None
        self.connection_start_time = None
        
        # Market Data State
        self.active_symbols: Set[str] = set()
        self.market_data_cache: Dict[str, Dict] = {}
        self.subscription_callbacks: Dict[str, Any] = {}
        self.data_requests_processed = 0
        self.ibkr_messages_received = 0
        
        # Keep-Alive Configuration
        self.heartbeat_interval = 30  # Send heartbeat every 30 seconds
        self.connection_timeout = 60  # Consider connection dead after 60 seconds
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 10  # Wait 10 seconds between reconnect attempts
        
        # Dual MessageBus
        self.dual_messagebus_client = None
        self.start_time = time.time()
        
        # Background tasks
        self.background_tasks = []
        
    async def initialize(self) -> bool:
        """Initialize IBKR Keep-Alive MarketData Engine"""
        logger.info("🚀 Initializing IBKR Keep-Alive MarketData Engine...")
        logger.info("   📊 Using Repository's Configured IB Connector")
        logger.info("   🔄 Dual MessageBus Integration")
        logger.info("   ❤️ IBKR Connection Keep-Alive")
        
        # Initialize dual messagebus
        if DUAL_MESSAGEBUS_AVAILABLE:
            try:
                self.dual_messagebus_client = await get_dual_bus_client(EngineType.MARKETDATA)
                logger.info("✅ Dual MessageBus connected")
            except Exception as e:
                logger.warning(f"⚠️ Dual MessageBus connection failed: {e}")
        
        # Initialize IBKR MarketData Client (using repository's existing client)
        success = await self._initialize_ibkr_client()
        if success:
            logger.info("✅ IBKR MarketData Client initialized")
        else:
            logger.error("❌ IBKR MarketData Client initialization failed")
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Initialize default symbols for IBKR data
        await self._initialize_default_ibkr_symbols()
        
        logger.info("🎉 IBKR Keep-Alive MarketData Engine ready!")
        return True
    
    async def _initialize_ibkr_client(self) -> bool:
        """Initialize IBKR client using repository's existing connector"""
        try:
            self.ibkr_status = IBKRConnectionStatus.CONNECTING
            self.connection_attempts += 1
            
            # Create MarketData client specifically for MARKETDATA engine type
            if UNIVERSAL_MESSAGEBUS_AVAILABLE:
                # Use the repository's existing MarketData client
                self.ibkr_client = create_marketdata_client(
                    UniversalEngineType.MARKETDATA, 
                    self.port
                )
                
                logger.info("✅ Using repository's IBKR MarketData Client")
                self.ibkr_status = IBKRConnectionStatus.CONNECTED
                self.connection_start_time = datetime.now()
                self.last_heartbeat = time.time()
                return True
            else:
                logger.error("❌ Universal MessageBus not available - cannot initialize IBKR client")
                self.ibkr_status = IBKRConnectionStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"❌ IBKR client initialization failed: {e}")
            self.ibkr_status = IBKRConnectionStatus.ERROR
            return False
    
    async def _start_background_tasks(self):
        """Start background tasks for IBKR connection management"""
        # IBKR Connection Keep-Alive
        self.background_tasks.append(
            asyncio.create_task(self._ibkr_heartbeat_task())
        )
        
        # IBKR Data Collection
        self.background_tasks.append(
            asyncio.create_task(self._ibkr_data_collection_task())
        )
        
        # Connection Health Monitoring
        self.background_tasks.append(
            asyncio.create_task(self._connection_health_monitor())
        )
        
        # Dual MessageBus Data Distribution
        if self.dual_messagebus_client:
            self.background_tasks.append(
                asyncio.create_task(self._distribute_market_data())
            )
    
    async def _initialize_default_ibkr_symbols(self):
        """Initialize default symbols for IBKR data collection"""
        default_symbols = [
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", 
            "NVDA", "META", "NFLX", "AMD", "CRM"
        ]
        
        for symbol in default_symbols:
            await self._subscribe_to_ibkr_symbol(symbol)
            self.active_symbols.add(symbol)
            
        logger.info(f"✅ Subscribed to {len(default_symbols)} IBKR symbols")
    
    async def _subscribe_to_ibkr_symbol(self, symbol: str):
        """Subscribe to IBKR data for a specific symbol"""
        if not self.ibkr_client or self.ibkr_status != IBKRConnectionStatus.CONNECTED:
            logger.warning(f"⚠️ Cannot subscribe to {symbol} - IBKR not connected")
            return False
        
        try:
            # Define callback for real-time updates
            def handle_ibkr_update(data: Dict[str, Any]):
                self._process_ibkr_data(symbol, data)
            
            # Subscribe using repository's MarketData client
            subscription_id = await self.ibkr_client.subscribe(
                symbols=[symbol],
                data_types=[DataType.TICK, DataType.QUOTE, DataType.LEVEL2],
                callback=handle_ibkr_update
            )
            
            self.subscription_callbacks[symbol] = subscription_id
            logger.debug(f"✅ Subscribed to IBKR data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ IBKR subscription failed for {symbol}: {e}")
            return False
    
    def _process_ibkr_data(self, symbol: str, data: Dict[str, Any]):
        """Process incoming IBKR data"""
        try:
            self.ibkr_messages_received += 1
            
            # Update cache
            self.market_data_cache[symbol] = {
                **data,
                "symbol": symbol,
                "source": "IBKR",
                "timestamp": datetime.now().isoformat(),
                "engine_id": self.engine_id
            }
            
            # Update heartbeat (we're receiving data, so connection is alive)
            self.last_heartbeat = time.time()
            
            logger.debug(f"📊 IBKR data received for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error processing IBKR data for {symbol}: {e}")
    
    async def _ibkr_heartbeat_task(self):
        """IBKR connection keep-alive heartbeat task"""
        while True:
            try:
                if self.ibkr_status == IBKRConnectionStatus.CONNECTED:
                    # Check if we need to send heartbeat
                    current_time = time.time()
                    
                    if current_time - self.last_heartbeat > self.heartbeat_interval:
                        await self._send_ibkr_heartbeat()
                    
                    # Check for connection timeout
                    if current_time - self.last_heartbeat > self.connection_timeout:
                        logger.warning("⚠️ IBKR connection timeout detected")
                        await self._handle_connection_timeout()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ IBKR heartbeat task error: {e}")
                await asyncio.sleep(30)
    
    async def _send_ibkr_heartbeat(self):
        """Send heartbeat to IBKR to keep connection alive"""
        if not self.ibkr_client:
            return
        
        try:
            # Request data for a test symbol to keep connection active
            test_data = await self.ibkr_client.get_data(
                symbols=["AAPL"],
                data_types=[DataType.QUOTE],
                sources=[DataSource.IBKR],
                cache=True,
                timeout=5.0
            )
            
            if test_data:
                self.last_heartbeat = time.time()
                logger.debug("❤️ IBKR heartbeat sent successfully")
            else:
                logger.warning("⚠️ IBKR heartbeat failed - no data received")
                
        except Exception as e:
            logger.error(f"❌ IBKR heartbeat error: {e}")
            await self._handle_connection_error()
    
    async def _handle_connection_timeout(self):
        """Handle IBKR connection timeout"""
        logger.warning("🔄 IBKR connection timed out - initiating reconnection")
        self.ibkr_status = IBKRConnectionStatus.RECONNECTING
        await self._attempt_reconnection()
    
    async def _handle_connection_error(self):
        """Handle IBKR connection error"""
        logger.error("❌ IBKR connection error detected")
        self.ibkr_status = IBKRConnectionStatus.ERROR
        await self._attempt_reconnection()
    
    async def _attempt_reconnection(self):
        """Attempt to reconnect to IBKR"""
        if self.connection_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
            self.ibkr_status = IBKRConnectionStatus.ERROR
            return
        
        logger.info(f"🔄 Attempting IBKR reconnection ({self.connection_attempts + 1}/{self.max_reconnect_attempts})")
        
        # Wait before reconnecting
        await asyncio.sleep(self.reconnect_delay)
        
        # Re-initialize IBKR client
        success = await self._initialize_ibkr_client()
        if success:
            # Re-subscribe to all active symbols
            for symbol in list(self.active_symbols):
                await self._subscribe_to_ibkr_symbol(symbol)
            logger.info("✅ IBKR reconnection successful")
        else:
            logger.error("❌ IBKR reconnection failed")
    
    async def _ibkr_data_collection_task(self):
        """Continuously collect IBKR market data"""
        while True:
            try:
                if self.ibkr_status == IBKRConnectionStatus.CONNECTED and self.ibkr_client:
                    # Collect data for all active symbols
                    for symbol in list(self.active_symbols):
                        try:
                            data = await self.ibkr_client.get_data(
                                symbols=[symbol],
                                data_types=[DataType.QUOTE, DataType.TICK],
                                sources=[DataSource.IBKR],  # Specifically use IBKR
                                cache=True,
                                timeout=5.0
                            )
                            
                            if data:
                                self._process_ibkr_data(symbol, data)
                                self.data_requests_processed += 1
                                
                        except Exception as e:
                            logger.error(f"❌ Data collection failed for {symbol}: {e}")
                
                await asyncio.sleep(1)  # Collect data every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ IBKR data collection task error: {e}")
                await asyncio.sleep(10)
    
    async def _connection_health_monitor(self):
        """Monitor IBKR connection health"""
        while True:
            try:
                if self.ibkr_client:
                    metrics = self.ibkr_client.get_metrics()
                    
                    # Log connection health periodically
                    if metrics.get("messagebus_connected"):
                        logger.debug(f"📊 IBKR Health: {metrics['avg_latency_ms']}ms avg, "
                                   f"{metrics['messagebus_ratio']} MessageBus ratio")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Connection health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _distribute_market_data(self):
        """Distribute market data via dual messagebus"""
        while True:
            try:
                if self.dual_messagebus_client:
                    for symbol, data in self.market_data_cache.items():
                        # Publish to MarketData Bus for distribution to other engines
                        await self.dual_messagebus_client.publish_to_marketdata(
                            "market_data_update",
                            {
                                "symbol": symbol,
                                "data": data,
                                "source": "IBKR",
                                "timestamp": datetime.now().isoformat(),
                                "engine_id": self.engine_id
                            }
                        )
                
                await asyncio.sleep(0.5)  # Distribute every 500ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Market data distribution error: {e}")
                await asyncio.sleep(5)
    
    async def get_ibkr_status(self) -> Dict[str, Any]:
        """Get comprehensive IBKR connection status"""
        uptime = time.time() - self.start_time
        
        # Get client metrics if available
        client_metrics = {}
        if self.ibkr_client:
            try:
                client_metrics = self.ibkr_client.get_metrics()
            except Exception as e:
                logger.error(f"Error getting client metrics: {e}")
        
        return {
            "engine": "ibkr_keepalive_marketdata",
            "port": self.port,
            "architecture": "dual_bus_with_ibkr",
            "ibkr_connection": {
                "status": self.ibkr_status.value,
                "connected": self.ibkr_status == IBKRConnectionStatus.CONNECTED,
                "connection_attempts": self.connection_attempts,
                "last_heartbeat": self.last_heartbeat,
                "connection_duration": str(datetime.now() - self.connection_start_time) if self.connection_start_time else None
            },
            "market_data": {
                "active_symbols": len(self.active_symbols),
                "symbols": list(self.active_symbols),
                "data_requests_processed": self.data_requests_processed,
                "ibkr_messages_received": self.ibkr_messages_received,
                "cache_size": len(self.market_data_cache)
            },
            "client_metrics": client_metrics,
            "dual_messagebus_connected": self.dual_messagebus_client is not None,
            "uptime_seconds": uptime,
            "timestamp": time.time()
        }
    
    async def add_symbol(self, symbol: str) -> bool:
        """Add new symbol for IBKR data collection"""
        if symbol not in self.active_symbols:
            success = await self._subscribe_to_ibkr_symbol(symbol)
            if success:
                self.active_symbols.add(symbol)
                logger.info(f"✅ Added IBKR symbol: {symbol}")
                return True
            else:
                logger.error(f"❌ Failed to add IBKR symbol: {symbol}")
                return False
        return True
    
    async def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from IBKR data collection"""
        if symbol in self.active_symbols:
            try:
                if symbol in self.subscription_callbacks and self.ibkr_client:
                    await self.ibkr_client.unsubscribe(self.subscription_callbacks[symbol])
                    del self.subscription_callbacks[symbol]
                
                self.active_symbols.remove(symbol)
                if symbol in self.market_data_cache:
                    del self.market_data_cache[symbol]
                
                logger.info(f"✅ Removed IBKR symbol: {symbol}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to remove IBKR symbol {symbol}: {e}")
                return False
        return True
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down IBKR Keep-Alive MarketData Engine...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Unsubscribe from all symbols
        for symbol in list(self.active_symbols):
            await self.remove_symbol(symbol)
        
        logger.info("✅ IBKR Keep-Alive MarketData Engine shutdown complete")

# Global engine instance
marketdata_engine = IBKRKeepAliveMarketDataEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("🚀 Starting IBKR Keep-Alive MarketData Engine...")
    
    # Initialize engine
    success = await marketdata_engine.initialize()
    if success:
        logger.info("✅ IBKR Keep-Alive MarketData Engine started successfully")
    else:
        logger.error("❌ IBKR Keep-Alive MarketData Engine initialization failed")
    
    yield
    
    # Cleanup on shutdown
    await marketdata_engine.shutdown()

app = FastAPI(
    title="IBKR Keep-Alive MarketData Engine",
    description="MarketData Engine with IBKR connection keep-alive and dual messagebus", 
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """IBKR MarketData Engine health check"""
    try:
        status = await marketdata_engine.get_ibkr_status()
        return {
            "status": "healthy",
            **status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ibkr/status")
async def get_ibkr_status():
    """Get detailed IBKR connection status"""
    return await marketdata_engine.get_ibkr_status()

@app.post("/ibkr/reconnect")
async def force_ibkr_reconnect():
    """Force IBKR reconnection"""
    try:
        await marketdata_engine._attempt_reconnection()
        return {"success": True, "message": "IBKR reconnection initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/symbols/add/{symbol}")
async def add_symbol(symbol: str):
    """Add symbol for IBKR data collection"""
    try:
        success = await marketdata_engine.add_symbol(symbol.upper())
        if success:
            return {"success": True, "message": f"Symbol {symbol} added", "symbol": symbol.upper()}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to add symbol {symbol}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/symbols/remove/{symbol}")
async def remove_symbol(symbol: str):
    """Remove symbol from IBKR data collection"""
    try:
        success = await marketdata_engine.remove_symbol(symbol.upper())
        if success:
            return {"success": True, "message": f"Symbol {symbol} removed", "symbol": symbol.upper()}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to remove symbol {symbol}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{symbol}")
async def get_market_data(symbol: str):
    """Get latest market data for symbol"""
    try:
        symbol = symbol.upper()
        if symbol in marketdata_engine.market_data_cache:
            return {
                "symbol": symbol,
                "data": marketdata_engine.market_data_cache[symbol],
                "source": "IBKR",
                "cached": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_active_symbols():
    """Get list of active IBKR symbols"""
    return {
        "symbols": list(marketdata_engine.active_symbols),
        "count": len(marketdata_engine.active_symbols),
        "ibkr_status": marketdata_engine.ibkr_status.value
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8800))
    logger.info(f"🚀 Starting IBKR Keep-Alive MarketData Engine on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
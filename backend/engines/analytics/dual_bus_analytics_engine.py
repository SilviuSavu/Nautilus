#!/usr/bin/env python3
"""
Dual Bus Analytics Engine - CORRECT FUCKING IMPLEMENTATION
Uses TWO separate Redis instances as intended:

1. MarketData Bus (Port 6380): ONLY for market data from MarketData Hub
2. Engine Logic Bus (Port 6381): ONLY for engine-to-engine business logic

This eliminates the single Redis bottleneck and provides proper message separation.
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import dual bus client
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dual_messagebus_client import (
    DualMessageBusClient, get_dual_bus_client, MessageBusType
)
from universal_enhanced_messagebus_client import (
    MessageType, EngineType, MessagePriority
)

logger = logging.getLogger(__name__)


class DualBusAnalyticsEngine:
    """
    Dual Bus Analytics Engine with CORRECT architecture.
    
    Communication Paths:
    1. Market data: MarketData Hub → MarketData Bus (6380) → Analytics Engine
    2. Business logic: Analytics Engine → Engine Logic Bus (6381) → Other Engines
    """
    
    def __init__(self):
        self.engine_name = "analytics"
        self.engine_type = EngineType.ANALYTICS
        self.port = 8100
        self.dual_bus_client: Optional[DualMessageBusClient] = None
        self.analytics_cache: Dict[str, Any] = {}
        self._initialized = False
        self._running = False
        
    async def initialize(self):
        """Initialize dual message bus client"""
        if self._initialized:
            return
        
        # Initialize dual bus client
        self.dual_bus_client = await get_dual_bus_client(
            engine_type=self.engine_type,
            instance_id=f"{self.engine_name}-{self.port}"
        )
        
        # Subscribe to market data (MarketData Bus - Port 6380)
        await self._subscribe_to_market_data()
        
        # Subscribe to engine logic messages (Engine Logic Bus - Port 6381)
        await self._subscribe_to_engine_logic()
        
        self._initialized = True
        logger.info(f"✅ DualBusAnalyticsEngine initialized")
    
    async def _subscribe_to_market_data(self):
        """Subscribe to market data from MarketData Hub (MarketData Bus - Port 6380)"""
        if not self.dual_bus_client:
            return
        
        # Subscribe ONLY to market data types on MarketData Bus
        market_data_types = [
            MessageType.MARKET_DATA,
            MessageType.PRICE_UPDATE,
            MessageType.TRADE_EXECUTION
        ]
        
        await self.dual_bus_client.subscribe_to_marketdata(
            message_types=market_data_types,
            handler=self._handle_market_data
        )
        
        logger.info("📡 Subscribed to MarketData Bus (Port 6380)")
    
    async def _subscribe_to_engine_logic(self):
        """Subscribe to engine logic messages (Engine Logic Bus - Port 6381)"""
        if not self.dual_bus_client:
            return
        
        # Subscribe ONLY to relevant engine logic types for analytics
        engine_logic_types = [
            MessageType.STRATEGY_SIGNAL,  # From strategy engine
            MessageType.RISK_METRIC,      # From risk engine
            MessageType.ML_PREDICTION     # From ML engine
        ]
        
        await self.dual_bus_client.subscribe_to_engine_logic(
            message_types=engine_logic_types,
            handler=self._handle_engine_logic
        )
        
        logger.info("⚙️ Subscribed to Engine Logic Bus (Port 6381)")
    
    async def _handle_market_data(self, message: Dict[str, Any]):
        """Handle incoming market data from MarketData Hub"""
        try:
            message_type = message.get("message_type")
            payload_str = message.get("payload", "{}")
            
            # Parse payload
            import json
            try:
                payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            except json.JSONDecodeError:
                payload = {}
            
            if message_type == "price_update":
                symbol = payload.get("symbol")
                price = payload.get("price")
                
                if symbol and price:
                    await self._update_analytics_for_symbol(symbol, float(price))
                    logger.debug(f"Processed price update: {symbol} = {price}")
                    
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_engine_logic(self, message: Dict[str, Any]):
        """Handle incoming engine logic messages from other engines"""
        try:
            message_type = message.get("message_type")
            source_engine = message.get("source_engine")
            
            logger.info(f"Received engine logic message: {message_type} from {source_engine}")
            
            # Process different engine logic message types
            if message_type == "strategy_signal":
                await self._process_strategy_signal(message)
            elif message_type == "risk_metric":
                await self._process_risk_metric(message)
            elif message_type == "ml_prediction":
                await self._process_ml_prediction(message)
                
        except Exception as e:
            logger.error(f"Error handling engine logic: {e}")
    
    async def _process_strategy_signal(self, message: Dict[str, Any]):
        """Process strategy signal from strategy engine"""
        # Implementation for processing strategy signals
        pass
    
    async def _process_risk_metric(self, message: Dict[str, Any]):
        """Process risk metric from risk engine"""
        # Implementation for processing risk metrics
        pass
    
    async def _process_ml_prediction(self, message: Dict[str, Any]):
        """Process ML prediction from ML engine"""
        # Implementation for processing ML predictions
        pass
    
    async def _update_analytics_for_symbol(self, symbol: str, price: float):
        """Update analytics calculations for symbol"""
        try:
            current_time = time.time()
            
            if symbol not in self.analytics_cache:
                self.analytics_cache[symbol] = {
                    "prices": [],
                    "timestamps": [],
                    "last_analysis": 0
                }
            
            cache = self.analytics_cache[symbol]
            cache["prices"].append(price)
            cache["timestamps"].append(current_time)
            
            # Keep only last 100 data points
            if len(cache["prices"]) > 100:
                cache["prices"] = cache["prices"][-100:]
                cache["timestamps"] = cache["timestamps"][-100:]
            
            # Perform analytics if enough data and time elapsed
            if (len(cache["prices"]) >= 10 and 
                current_time - cache["last_analysis"] > 30):  # 30 seconds
                
                await self._perform_analytics(symbol, cache)
                cache["last_analysis"] = current_time
                
        except Exception as e:
            logger.error(f"Error updating analytics for {symbol}: {e}")
    
    async def _perform_analytics(self, symbol: str, cache: Dict[str, Any]):
        """Perform analytics calculations and send results via Engine Logic Bus"""
        try:
            prices = np.array(cache["prices"])
            
            if len(prices) < 10:
                return
            
            # Calculate analytics metrics
            volatility = float(np.std(prices) / np.mean(prices))
            trend = float((prices[-1] - prices[0]) / prices[0])
            momentum = float((prices[-5:].mean() - prices[-10:-5].mean()) / prices[-10:-5].mean())
            
            # Create analytics result
            analytics_result = {
                "symbol": symbol,
                "volatility": volatility,
                "trend": trend,
                "momentum": momentum,
                "price_current": float(prices[-1]),
                "timestamp": time.time(),
                "confidence": 0.85
            }
            
            # Send to Engine Logic Bus (Port 6381) for other engines
            if self.dual_bus_client:
                success = await self.dual_bus_client.publish_message(
                    message_type=MessageType.ANALYTICS_RESULT,
                    payload=analytics_result,
                    priority=MessagePriority.NORMAL
                )
                
                if success:
                    logger.debug(f"Analytics result sent to Engine Logic Bus for {symbol}")
                else:
                    logger.warning(f"Failed to send analytics result for {symbol}")
            
        except Exception as e:
            logger.error(f"Error performing analytics for {symbol}: {e}")
    
    async def send_trading_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Send trading signal via Engine Logic Bus"""
        if not self.dual_bus_client:
            return False
        
        return await self.dual_bus_client.publish_message(
            message_type=MessageType.STRATEGY_SIGNAL,
            payload=signal_data,
            priority=MessagePriority.HIGH
        )
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        symbols_analyzed = len(self.analytics_cache)
        total_calculations = sum(
            len(cache["prices"]) for cache in self.analytics_cache.values()
        )
        
        # Get dual bus stats
        bus_stats = {}
        if self.dual_bus_client:
            bus_stats = await self.dual_bus_client.get_stats()
        
        return {
            "engine": "analytics",
            "engine_type": "dual_bus",
            "status": "running" if self._running else "stopped",
            "symbols_analyzed": symbols_analyzed,
            "total_calculations": total_calculations,
            "cache_size": len(self.analytics_cache),
            "dual_bus_stats": bus_stats,
            "timestamp": time.time()
        }
    
    async def start(self):
        """Start analytics engine"""
        self._running = True
        logger.info("🚀 DualBusAnalyticsEngine started")
    
    async def stop(self):
        """Stop analytics engine"""
        self._running = False
        if self.dual_bus_client:
            await self.dual_bus_client.close()
        logger.info("🛑 DualBusAnalyticsEngine stopped")


# Global engine instance
dual_bus_analytics_engine: Optional[DualBusAnalyticsEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global dual_bus_analytics_engine
    
    try:
        logger.info("🚀 Starting Dual Bus Analytics Engine...")
        
        dual_bus_analytics_engine = DualBusAnalyticsEngine()
        await dual_bus_analytics_engine.initialize()
        await dual_bus_analytics_engine.start()
        
        app.state.analytics_engine = dual_bus_analytics_engine
        
        logger.info("✅ Dual Bus Analytics Engine started successfully")
        logger.info("   📡 MarketData Bus (Port 6380): Market data subscription")
        logger.info("   ⚙️ Engine Logic Bus (Port 6381): Business logic communication")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Dual Bus Analytics Engine: {e}")
        raise
    finally:
        logger.info("🔄 Stopping Dual Bus Analytics Engine...")
        if dual_bus_analytics_engine:
            await dual_bus_analytics_engine.stop()


# Create FastAPI app
app = FastAPI(
    title="Dual Bus Analytics Engine",
    description="Analytics Engine with Dual Redis Bus Architecture",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# HTTP API endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "analytics",
        "port": 8100,
        "architecture": "dual_bus",
        "marketdata_bus": "6380",
        "engine_logic_bus": "6381",
        "timestamp": time.time()
    }


@app.get("/api/v1/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary"""
    if not dual_bus_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    
    return await dual_bus_analytics_engine.get_analytics_summary()


@app.post("/api/v1/analytics/signal")
async def send_trading_signal(signal_data: Dict[str, Any]):
    """Send trading signal to other engines via Engine Logic Bus"""
    if not dual_bus_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    
    success = await dual_bus_analytics_engine.send_trading_signal(signal_data)
    
    return {
        "success": success,
        "message": "Trading signal sent via Engine Logic Bus" if success else "Failed to send trading signal",
        "bus": "engine_logic_bus_6381"
    }


@app.get("/api/v1/analytics/bus-stats")
async def get_bus_statistics():
    """Get dual bus statistics"""
    if not dual_bus_analytics_engine or not dual_bus_analytics_engine.dual_bus_client:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    
    stats = await dual_bus_analytics_engine.dual_bus_client.get_stats()
    
    return {
        "dual_bus_stats": stats,
        "architecture": "dual_redis_buses",
        "marketdata_bus": "localhost:6380",
        "engine_logic_bus": "localhost:6381"
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("🚀 Starting Dual Bus Analytics Engine Server...")
    logger.info("   Architecture: DUAL REDIS BUSES")
    logger.info("   📡 MarketData Bus: localhost:6380 (Market data from MarketData Hub)")
    logger.info("   ⚙️ Engine Logic Bus: localhost:6381 (Engine-to-engine business logic)")
    logger.info("   🎯 This eliminates the single Redis bottleneck!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )
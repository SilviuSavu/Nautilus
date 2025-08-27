#!/usr/bin/env python3
"""
Dual Bus Risk Engine - CORRECT FUCKING IMPLEMENTATION
Uses TWO separate Redis instances:

1. MarketData Bus (Port 6380): ONLY for market data from MarketData Hub
2. Engine Logic Bus (Port 6381): ONLY for engine-to-engine business logic

This eliminates the single Redis bottleneck and provides proper message separation.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
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


class DualBusRiskEngine:
    """
    Dual Bus Risk Engine with CORRECT architecture.
    
    Communication Paths:
    1. Market data: MarketData Hub → MarketData Bus (6380) → Risk Engine
    2. Business logic: Risk Engine → Engine Logic Bus (6381) → Other Engines
    """
    
    def __init__(self):
        self.engine_name = "risk"
        self.engine_type = EngineType.RISK
        self.port = 8200
        self.dual_bus_client: Optional[DualMessageBusClient] = None
        self.risk_metrics: Dict[str, Any] = {}
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
        logger.info(f"✅ DualBusRiskEngine initialized")
    
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
        
        # Subscribe ONLY to relevant engine logic types for risk
        engine_logic_types = [
            MessageType.ANALYTICS_RESULT,  # From analytics engine
            MessageType.PORTFOLIO_UPDATE,  # From portfolio engine
            MessageType.STRATEGY_SIGNAL    # From strategy engine
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
                    await self._update_risk_for_symbol(symbol, float(price))
                    logger.debug(f"Risk: Processed price update: {symbol} = {price}")
                    
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_engine_logic(self, message: Dict[str, Any]):
        """Handle incoming engine logic messages from other engines"""
        try:
            message_type = message.get("message_type")
            source_engine = message.get("source_engine")
            
            logger.info(f"Risk: Received engine logic message: {message_type} from {source_engine}")
            
            # Process different engine logic message types
            if message_type == "analytics_result":
                await self._process_analytics_result(message)
            elif message_type == "portfolio_update":
                await self._process_portfolio_update(message)
            elif message_type == "strategy_signal":
                await self._process_strategy_signal(message)
                
        except Exception as e:
            logger.error(f"Error handling engine logic: {e}")
    
    async def _process_analytics_result(self, message: Dict[str, Any]):
        """Process analytics result from analytics engine"""
        # Implementation for processing analytics results
        pass
    
    async def _process_portfolio_update(self, message: Dict[str, Any]):
        """Process portfolio update from portfolio engine"""
        # Implementation for processing portfolio updates
        pass
    
    async def _process_strategy_signal(self, message: Dict[str, Any]):
        """Process strategy signal from strategy engine"""
        # Implementation for processing strategy signals
        pass
    
    async def _update_risk_for_symbol(self, symbol: str, price: float):
        """Update risk calculations for symbol"""
        try:
            current_time = time.time()
            
            if symbol not in self.risk_metrics:
                self.risk_metrics[symbol] = {
                    "prices": [],
                    "var": 0.0,
                    "beta": 1.0,
                    "last_risk_calc": 0
                }
            
            metric = self.risk_metrics[symbol]
            metric["prices"].append(price)
            
            # Keep only last 50 data points for risk calculation
            if len(metric["prices"]) > 50:
                metric["prices"] = metric["prices"][-50:]
            
            # Calculate risk metrics if enough data and time elapsed
            if (len(metric["prices"]) >= 10 and 
                current_time - metric["last_risk_calc"] > 60):  # 60 seconds
                
                await self._calculate_risk_metrics(symbol, metric)
                metric["last_risk_calc"] = current_time
                
        except Exception as e:
            logger.error(f"Error updating risk for {symbol}: {e}")
    
    async def _calculate_risk_metrics(self, symbol: str, metric: Dict[str, Any]):
        """Calculate risk metrics and send alerts via Engine Logic Bus"""
        try:
            import numpy as np
            prices = np.array(metric["prices"])
            
            if len(prices) < 10:
                return
            
            # Calculate risk metrics
            returns = np.diff(np.log(prices))
            volatility = float(np.std(returns) * np.sqrt(252))  # Annualized
            var_95 = float(np.percentile(returns, 5))  # Value at Risk
            
            metric["var"] = var_95
            
            # Create risk alert if high volatility
            if volatility > 0.3:  # 30% volatility threshold
                risk_alert = {
                    "symbol": symbol,
                    "alert_type": "high_volatility",
                    "volatility": volatility,
                    "var_95": var_95,
                    "price_current": float(prices[-1]),
                    "timestamp": time.time(),
                    "severity": "high" if volatility > 0.5 else "medium"
                }
                
                # Send to Engine Logic Bus (Port 6381) for other engines
                if self.dual_bus_client:
                    success = await self.dual_bus_client.publish_message(
                        message_type=MessageType.RISK_METRIC,
                        payload=risk_alert,
                        priority=MessagePriority.HIGH if volatility > 0.5 else MessagePriority.NORMAL
                    )
                    
                    if success:
                        logger.info(f"Risk alert sent to Engine Logic Bus for {symbol} (vol: {volatility:.2%})")
                    else:
                        logger.warning(f"Failed to send risk alert for {symbol}")
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary"""
        symbols_monitored = len(self.risk_metrics)
        high_risk_symbols = [
            symbol for symbol, metric in self.risk_metrics.items()
            if abs(metric.get("var", 0)) > 0.05  # 5% daily VaR threshold
        ]
        
        # Get dual bus stats
        bus_stats = {}
        if self.dual_bus_client:
            bus_stats = await self.dual_bus_client.get_stats()
        
        return {
            "engine": "risk",
            "engine_type": "dual_bus",
            "status": "running" if self._running else "stopped",
            "symbols_monitored": symbols_monitored,
            "high_risk_symbols": len(high_risk_symbols),
            "high_risk_list": high_risk_symbols,
            "dual_bus_stats": bus_stats,
            "timestamp": time.time()
        }
    
    async def start(self):
        """Start risk engine"""
        self._running = True
        logger.info("🚀 DualBusRiskEngine started")
    
    async def stop(self):
        """Stop risk engine"""
        self._running = False
        if self.dual_bus_client:
            await self.dual_bus_client.close()
        logger.info("🛑 DualBusRiskEngine stopped")


# Global engine instance
dual_bus_risk_engine: Optional[DualBusRiskEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global dual_bus_risk_engine
    
    try:
        logger.info("🚀 Starting Dual Bus Risk Engine...")
        
        dual_bus_risk_engine = DualBusRiskEngine()
        await dual_bus_risk_engine.initialize()
        await dual_bus_risk_engine.start()
        
        app.state.risk_engine = dual_bus_risk_engine
        
        logger.info("✅ Dual Bus Risk Engine started successfully")
        logger.info("   📡 MarketData Bus (Port 6380): Market data subscription")
        logger.info("   ⚙️ Engine Logic Bus (Port 6381): Business logic communication")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Dual Bus Risk Engine: {e}")
        raise
    finally:
        logger.info("🔄 Stopping Dual Bus Risk Engine...")
        if dual_bus_risk_engine:
            await dual_bus_risk_engine.stop()


# Create FastAPI app
app = FastAPI(
    title="Dual Bus Risk Engine",
    description="Risk Engine with Dual Redis Bus Architecture",
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
        "engine": "risk",
        "port": 8200,
        "architecture": "dual_bus",
        "marketdata_bus": "6380",
        "engine_logic_bus": "6381",
        "timestamp": time.time()
    }


@app.get("/api/v1/risk/summary")
async def get_risk_summary():
    """Get risk summary"""
    if not dual_bus_risk_engine:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    return await dual_bus_risk_engine.get_risk_summary()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("🚀 Starting Dual Bus Risk Engine Server...")
    logger.info("   Architecture: DUAL REDIS BUSES")
    logger.info("   📡 MarketData Bus: localhost:6380 (Market data from MarketData Hub)")
    logger.info("   ⚙️ Engine Logic Bus: localhost:6381 (Engine-to-engine business logic)")
    logger.info("   🎯 This eliminates the single Redis bottleneck!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8200,
        log_level="info"
    )
#!/usr/bin/env python3
"""
Dual Bus Strategy Engine - Trading Signal Coordination
Uses TWO separate Redis instances:

1. MarketData Bus (Port 6380): ONLY for market data from MarketData Hub
2. Engine Logic Bus (Port 6381): ONLY for engine-to-engine business logic

This eliminates the single Redis bottleneck and provides proper message separation.
Strategy Engine coordinates trading signals between Risk, ML, Analytics, and Portfolio engines.
"""

import asyncio
import logging
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict

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


@dataclass
class TradingSignal:
    """Trading signal structure"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    quantity: int
    timestamp: float
    strategy_name: str
    pattern_detected: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    avg_confidence: float = 0.0
    strategies_active: int = 0
    last_signal_time: Optional[float] = None


class DualBusStrategyEngine:
    """
    Dual Bus Strategy Engine with CORRECT architecture.
    
    Communication Paths:
    1. Market data: MarketData Hub → MarketData Bus (6380) → Strategy Engine
    2. Business logic: Strategy Engine → Engine Logic Bus (6381) → Other Engines
    3. Coordination: Strategy ↔ Risk ↔ ML ↔ Analytics ↔ Portfolio (all via Engine Logic Bus)
    """
    
    def __init__(self):
        self.engine_name = "strategy"
        self.engine_type = EngineType.STRATEGY
        self.port = 8700
        self.dual_bus_client: Optional[DualMessageBusClient] = None
        self.strategy_metrics = StrategyMetrics()
        self.active_positions: Dict[str, Dict] = {}
        self.market_data_cache: Dict[str, Any] = {}
        self.risk_limits: Dict[str, float] = {}
        self._initialized = False
        self._running = False
        
        # Strategy configurations
        self.strategies = {
            "momentum": {"enabled": True, "confidence_threshold": 0.75},
            "mean_reversion": {"enabled": True, "confidence_threshold": 0.70},
            "breakout": {"enabled": True, "confidence_threshold": 0.80},
            "pattern_recognition": {"enabled": True, "confidence_threshold": 0.85},
            "multi_timeframe": {"enabled": True, "confidence_threshold": 0.90}
        }
        
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
        logger.info(f"✅ DualBusStrategyEngine initialized")
    
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
        
        # Subscribe to business logic message types on Engine Logic Bus
        engine_logic_types = [
            MessageType.RISK_METRIC,
            MessageType.ML_PREDICTION, 
            MessageType.ANALYTICS_RESULT,
            MessageType.PORTFOLIO_UPDATE,
            MessageType.STRATEGY_SIGNAL,
            MessageType.TOXICITY_ALERT
        ]
        
        await self.dual_bus_client.subscribe_to_engine_logic(
            message_types=engine_logic_types,
            handler=self._handle_engine_logic
        )
        
        logger.info("🔗 Subscribed to Engine Logic Bus (Port 6381)")
    
    async def _handle_market_data(self, message_type: MessageType, data: Dict[str, Any]):
        """Handle market data messages from MarketData Bus"""
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Cache market data
            self.market_data_cache[symbol] = {
                'price': data.get('price', 0.0),
                'timestamp': time.time(),
                'volume': data.get('volume', 0),
                'data': data
            }
            
            # Generate trading signals based on market data
            await self._analyze_market_data_for_signals(symbol, data)
            
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_engine_logic(self, message_type: MessageType, data: Dict[str, Any]):
        """Handle engine logic messages from other engines"""
        try:
            if message_type == MessageType.RISK_METRIC:
                await self._handle_risk_alert(data)
            elif message_type == MessageType.TOXICITY_ALERT:
                await self._handle_risk_alert(data)
            elif message_type == MessageType.ML_PREDICTION:
                await self._handle_ml_prediction(data)
            elif message_type == MessageType.ANALYTICS_RESULT:
                await self._handle_analytics_result(data)
            elif message_type == MessageType.PORTFOLIO_UPDATE:
                await self._handle_portfolio_update(data)
            
        except Exception as e:
            logger.error(f"Error handling engine logic message: {e}")
    
    async def _analyze_market_data_for_signals(self, symbol: str, data: Dict[str, Any]):
        """Analyze market data and generate trading signals"""
        try:
            current_price = data.get('price', 0.0)
            if current_price <= 0:
                return
            
            signals = []
            
            # Momentum Strategy
            if self.strategies["momentum"]["enabled"]:
                momentum_signal = await self._momentum_strategy(symbol, data)
                if momentum_signal:
                    signals.append(momentum_signal)
            
            # Mean Reversion Strategy
            if self.strategies["mean_reversion"]["enabled"]:
                mean_reversion_signal = await self._mean_reversion_strategy(symbol, data)
                if mean_reversion_signal:
                    signals.append(mean_reversion_signal)
            
            # Breakout Strategy
            if self.strategies["breakout"]["enabled"]:
                breakout_signal = await self._breakout_strategy(symbol, data)
                if breakout_signal:
                    signals.append(breakout_signal)
            
            # Send signals to other engines via Engine Logic Bus
            for signal in signals:
                await self._publish_trading_signal(signal)
                
        except Exception as e:
            logger.error(f"Error analyzing market data for signals: {e}")
    
    async def _momentum_strategy(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Momentum-based trading strategy"""
        try:
            current_price = data.get('price', 0.0)
            volume = data.get('volume', 0)
            
            # Simple momentum calculation (in real implementation would use more sophisticated analysis)
            # Simulate momentum based on price and volume
            momentum_score = (current_price % 10) / 10.0  # Simplified for demo
            volume_factor = min(volume / 100000, 2.0) if volume > 0 else 1.0
            
            confidence = momentum_score * volume_factor * 0.8
            threshold = self.strategies["momentum"]["confidence_threshold"]
            
            if confidence > threshold:
                signal_type = "BUY" if momentum_score > 0.6 else "SELL"
                
                return TradingSignal(
                    signal_id=f"momentum_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    quantity=100,
                    timestamp=time.time(),
                    strategy_name="momentum",
                    pattern_detected="momentum_trend"
                )
                
        except Exception as e:
            logger.error(f"Momentum strategy error: {e}")
        
        return None
    
    async def _mean_reversion_strategy(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Mean reversion trading strategy"""
        try:
            current_price = data.get('price', 0.0)
            
            # Simulate mean reversion analysis
            price_deviation = abs(current_price - 100) / 100  # Assuming 100 as mean
            confidence = min(price_deviation * 2, 0.9)
            threshold = self.strategies["mean_reversion"]["confidence_threshold"]
            
            if confidence > threshold:
                # Mean reversion: buy when below mean, sell when above mean
                signal_type = "BUY" if current_price < 100 else "SELL"
                
                return TradingSignal(
                    signal_id=f"mean_rev_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    quantity=150,
                    timestamp=time.time(),
                    strategy_name="mean_reversion",
                    pattern_detected="mean_reversion_opportunity"
                )
                
        except Exception as e:
            logger.error(f"Mean reversion strategy error: {e}")
        
        return None
    
    async def _breakout_strategy(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Breakout detection strategy"""
        try:
            current_price = data.get('price', 0.0)
            volume = data.get('volume', 0)
            
            # Simulate breakout detection
            breakout_threshold = 105  # Resistance level
            support_level = 95
            
            volume_surge = volume > 150000  # High volume breakout
            
            if current_price > breakout_threshold and volume_surge:
                return TradingSignal(
                    signal_id=f"breakout_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type="BUY",
                    confidence=0.85,
                    price=current_price,
                    quantity=200,
                    timestamp=time.time(),
                    strategy_name="breakout",
                    pattern_detected="resistance_breakout"
                )
            elif current_price < support_level and volume_surge:
                return TradingSignal(
                    signal_id=f"breakdown_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type="SELL",
                    confidence=0.85,
                    price=current_price,
                    quantity=200,
                    timestamp=time.time(),
                    strategy_name="breakout",
                    pattern_detected="support_breakdown"
                )
                
        except Exception as e:
            logger.error(f"Breakout strategy error: {e}")
        
        return None
    
    async def _publish_trading_signal(self, signal: TradingSignal):
        """Publish trading signal to Engine Logic Bus"""
        try:
            if not self.dual_bus_client:
                return
            
            # Update strategy metrics
            self.strategy_metrics.total_signals += 1
            if signal.signal_type == "BUY":
                self.strategy_metrics.buy_signals += 1
            elif signal.signal_type == "SELL":
                self.strategy_metrics.sell_signals += 1
            else:
                self.strategy_metrics.hold_signals += 1
            
            self.strategy_metrics.last_signal_time = signal.timestamp
            self.strategy_metrics.avg_confidence = (
                (self.strategy_metrics.avg_confidence * (self.strategy_metrics.total_signals - 1) + 
                 signal.confidence) / self.strategy_metrics.total_signals
            )
            
            # Publish to Engine Logic Bus
            await self.dual_bus_client.publish_to_engine_logic(
                message_type=MessageType.STRATEGY_SIGNAL,
                data=asdict(signal),
                priority=MessagePriority.HIGH
            )
            
            logger.info(f"📤 Published {signal.signal_type} signal for {signal.symbol} "
                       f"(confidence: {signal.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error publishing trading signal: {e}")
    
    async def _handle_risk_alert(self, data: Dict[str, Any]):
        """Handle risk alerts from Risk Engine"""
        try:
            risk_level = data.get('risk_level', 'LOW')
            symbol = data.get('symbol', 'UNKNOWN')
            
            if risk_level in ['HIGH', 'CRITICAL']:
                # Disable trading for this symbol temporarily
                logger.warning(f"🚨 Risk alert for {symbol}: {risk_level}")
                # Could implement risk-based position sizing or strategy disabling
                
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_ml_prediction(self, data: Dict[str, Any]):
        """Handle ML predictions from ML Engine"""
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            prediction = data.get('prediction', 0.0)
            confidence = data.get('confidence', 0.0)
            
            # Use ML predictions to enhance strategy decisions
            if confidence > 0.8:
                logger.info(f"🤖 High-confidence ML prediction for {symbol}: {prediction}")
                # Could enhance existing strategies with ML insights
                
        except Exception as e:
            logger.error(f"Error handling ML prediction: {e}")
    
    async def _handle_analytics_result(self, data: Dict[str, Any]):
        """Handle analytics results from Analytics Engine"""
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            analytics_score = data.get('score', 0.0)
            
            # Use analytics results for strategy refinement
            logger.info(f"📊 Analytics score for {symbol}: {analytics_score}")
            
        except Exception as e:
            logger.error(f"Error handling analytics result: {e}")
    
    async def _handle_portfolio_update(self, data: Dict[str, Any]):
        """Handle portfolio updates from Portfolio Engine"""
        try:
            # Update position tracking
            symbol = data.get('symbol', 'UNKNOWN')
            position = data.get('position', {})
            
            if position:
                self.active_positions[symbol] = position
                logger.info(f"📈 Updated position for {symbol}")
                
        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")
    
    def get_health(self) -> Dict[str, Any]:
        """Get engine health status"""
        return {
            "status": "healthy" if self._initialized and self.dual_bus_client else "initializing",
            "engine": self.engine_name,
            "port": self.port,
            "dual_bus_connected": self.dual_bus_client is not None if self.dual_bus_client else False,
            "strategies_enabled": sum(1 for s in self.strategies.values() if s["enabled"]),
            "active_positions": len(self.active_positions),
            "cached_symbols": len(self.market_data_cache),
            "uptime_seconds": time.time() - getattr(self, 'start_time', time.time()),
            "total_signals": self.strategy_metrics.total_signals,
            "last_signal_time": self.strategy_metrics.last_signal_time
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics"""
        return {
            "strategy_metrics": asdict(self.strategy_metrics),
            "strategies_config": self.strategies,
            "active_positions": self.active_positions,
            "market_data_symbols": list(self.market_data_cache.keys()),
            "performance": {
                "signals_per_minute": self.strategy_metrics.total_signals / max((time.time() - getattr(self, 'start_time', time.time())) / 60, 1),
                "avg_confidence": self.strategy_metrics.avg_confidence,
                "signal_distribution": {
                    "buy": self.strategy_metrics.buy_signals,
                    "sell": self.strategy_metrics.sell_signals,
                    "hold": self.strategy_metrics.hold_signals
                }
            }
        }


# Global engine instance
strategy_engine = DualBusStrategyEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    strategy_engine.start_time = time.time()
    await strategy_engine.initialize()
    logger.info(f"✅ Strategy Engine started on port {strategy_engine.port}")
    yield
    logger.info("Strategy Engine shutting down")


# Create FastAPI app
app = FastAPI(
    title="Dual Bus Strategy Engine",
    description="Trading signal coordination with dual messagebus architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return strategy_engine.get_health()


@app.get("/metrics")
async def get_metrics():
    """Get strategy metrics"""
    return strategy_engine.get_metrics()


@app.get("/strategies")
async def get_strategies():
    """Get strategy configurations"""
    return {
        "strategies": strategy_engine.strategies,
        "active_count": sum(1 for s in strategy_engine.strategies.values() if s["enabled"])
    }


@app.post("/strategies/{strategy_name}/toggle")
async def toggle_strategy(strategy_name: str):
    """Toggle strategy on/off"""
    if strategy_name not in strategy_engine.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    current_status = strategy_engine.strategies[strategy_name]["enabled"]
    strategy_engine.strategies[strategy_name]["enabled"] = not current_status
    
    return {
        "strategy": strategy_name,
        "enabled": strategy_engine.strategies[strategy_name]["enabled"],
        "message": f"Strategy {strategy_name} {'enabled' if not current_status else 'disabled'}"
    }


@app.get("/positions")
async def get_positions():
    """Get active positions"""
    return {
        "active_positions": strategy_engine.active_positions,
        "position_count": len(strategy_engine.active_positions)
    }


@app.get("/market-data")
async def get_market_data():
    """Get cached market data"""
    return {
        "cached_symbols": list(strategy_engine.market_data_cache.keys()),
        "cache_size": len(strategy_engine.market_data_cache),
        "recent_data": {
            symbol: {
                "price": data["price"],
                "timestamp": data["timestamp"],
                "age_seconds": time.time() - data["timestamp"]
            }
            for symbol, data in strategy_engine.market_data_cache.items()
        }
    }


@app.post("/test-signal")
async def generate_test_signal():
    """Generate a test trading signal"""
    test_signal = TradingSignal(
        signal_id=f"test_{int(time.time())}",
        symbol="TEST",
        signal_type="BUY",
        confidence=0.85,
        price=100.0,
        quantity=100,
        timestamp=time.time(),
        strategy_name="test_strategy",
        pattern_detected="test_pattern"
    )
    
    await strategy_engine._publish_trading_signal(test_signal)
    
    return {
        "message": "Test signal generated",
        "signal": asdict(test_signal)
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "dual_bus_strategy_engine:app",
        host="0.0.0.0",
        port=8700,
        reload=False
    )
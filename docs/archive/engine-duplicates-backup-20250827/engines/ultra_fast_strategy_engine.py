#!/usr/bin/env python3
"""
Ultra-Fast Strategy Engine - FastAPI Server with Enhanced MessageBus Integration
Sub-5ms strategy signal generation with M4 Max Neural Engine acceleration.

Features:
- FastAPI REST endpoints for backward compatibility
- Enhanced MessageBus background tasks for ultra-fast communication  
- Neural Engine hardware acceleration for strategy optimization
- Deterministic clock for consistent backtesting
- Real-time strategy performance monitoring
"""

import asyncio
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Import enhanced strategy engine with MessageBus
from enhanced_strategy_messagebus_integration import (
    EnhancedStrategyEngineMessageBus,
    StrategyDefinition,
    StrategySignal,
    StrategyStatus,
    StrategyType,
    SignalStrength,
    StrategyPerformanceMetrics
)

# Import clock for deterministic testing
from clock import LiveClock, TestClock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== PYDANTIC MODELS ====================

class StrategyCreateRequest(BaseModel):
    strategy_name: str = Field(..., description="Name of the strategy")
    strategy_type: str = Field(..., description="Type of strategy (mean_reversion, momentum, etc.)")
    description: str = Field("", description="Strategy description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    risk_limits: Dict[str, Any] = Field(default_factory=dict, description="Risk limits")


class SignalGenerationRequest(BaseModel):
    strategy_id: str = Field(..., description="Strategy ID")
    symbol: str = Field(..., description="Trading symbol")
    market_data: Dict[str, Any] = Field(..., description="Market data for signal generation")
    priority: str = Field("high", description="Signal priority (low, normal, high, urgent, critical)")


class StrategyOptimizationRequest(BaseModel):
    strategy_id: str = Field(..., description="Strategy ID to optimize")
    optimization_config: Dict[str, Any] = Field(default_factory=dict, description="Optimization configuration")


class PerformanceRequest(BaseModel):
    strategy_id: Optional[str] = Field(None, description="Strategy ID (optional, returns all if not provided)")
    time_range_hours: int = Field(24, description="Time range for performance analysis")


# ==================== ULTRA-FAST STRATEGY ENGINE ====================

class UltraFastStrategyEngine:
    """
    Ultra-Fast Strategy Engine with FastAPI and Enhanced MessageBus Integration
    
    Provides:
    - REST API endpoints for backward compatibility
    - MessageBus background processing for ultra-low latency
    - Neural Engine acceleration for <5ms strategy signals
    - Real-time performance monitoring and optimization
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Nautilus Ultra-Fast Strategy Engine",
            description="Neural Engine accelerated strategy engine with MessageBus integration",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Enhanced MessageBus engine
        self.enhanced_engine = EnhancedStrategyEngineMessageBus()
        
        # Performance metrics
        self.start_time = time.time()
        self.requests_processed = 0
        self.signals_generated = 0
        self.strategies_optimized = 0
        self.average_response_time_ms = 0.0
        
        # Background tasks tracking
        self.background_tasks_active = 0
        self.messagebus_messages_processed = 0
        
        # Setup FastAPI routes
        self.setup_routes()
        
        logger.info("🚀 Ultra-Fast Strategy Engine with FastAPI initialized")
    
    def setup_routes(self):
        """Setup FastAPI routes with enhanced functionality"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize the enhanced strategy engine on startup"""
            try:
                await self.enhanced_engine.initialize()
                logger.info("✅ Ultra-Fast Strategy Engine started successfully")
            except Exception as e:
                logger.error(f"❌ Failed to start enhanced engine: {e}")
                raise
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            await self.enhanced_engine.stop()
            logger.info("✅ Ultra-Fast Strategy Engine stopped")
        
        # ==================== HEALTH AND STATUS ENDPOINTS ====================
        
        @self.app.get("/health")
        async def health_check():
            """Enhanced health check with MessageBus status"""
            uptime_seconds = time.time() - self.start_time
            
            # Get enhanced engine performance
            engine_performance = await self.enhanced_engine.get_performance_summary()
            
            return {
                "status": "healthy",
                "uptime_seconds": uptime_seconds,
                "version": "2.0.0",
                "engine_type": "ultra_fast_strategy",
                
                # FastAPI metrics
                "api_metrics": {
                    "requests_processed": self.requests_processed,
                    "signals_generated": self.signals_generated,
                    "strategies_optimized": self.strategies_optimized,
                    "average_response_time_ms": self.average_response_time_ms,
                    "background_tasks_active": self.background_tasks_active
                },
                
                # Enhanced engine metrics
                "enhanced_engine": engine_performance,
                
                # Hardware status
                "hardware_acceleration": {
                    "neural_engine_available": self.enhanced_engine.neural_engine_available,
                    "m4_max_detected": self.enhanced_engine.m4_max_detected,
                    "messagebus_connected": self.enhanced_engine.messagebus_client is not None
                }
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get comprehensive performance metrics"""
            uptime = time.time() - self.start_time
            
            # Get detailed performance from enhanced engine
            enhanced_metrics = await self.enhanced_engine.get_performance_summary()
            
            return {
                "performance_metrics": {
                    "requests_per_second": (self.requests_processed / max(1, uptime)),
                    "signals_per_second": (self.signals_generated / max(1, uptime)),
                    "total_requests": self.requests_processed,
                    "total_signals": self.signals_generated,
                    "total_optimizations": self.strategies_optimized,
                    "average_response_time_ms": self.average_response_time_ms,
                    "uptime_seconds": uptime
                },
                
                "messagebus_metrics": {
                    "messages_processed": self.messagebus_messages_processed,
                    "background_tasks_active": self.background_tasks_active,
                    "connected": self.enhanced_engine.messagebus_client is not None
                },
                
                "enhanced_engine_metrics": enhanced_metrics,
                
                "target_performance": {
                    "signal_generation_target_ms": 5.0,
                    "requests_per_second_target": 1000,
                    "neural_engine_usage_target_pct": 80,
                    "performance_grade": self._calculate_performance_grade(enhanced_metrics)
                }
            }
        
        # ==================== STRATEGY MANAGEMENT ENDPOINTS ====================
        
        @self.app.get("/strategies")
        async def get_strategies():
            """Get all strategies with enhanced performance data"""
            start_time_req = time.time()
            
            try:
                strategies_data = []
                
                for strategy_id, strategy in self.enhanced_engine.strategies.items():
                    # Get recent signals for this strategy
                    recent_signals = self.enhanced_engine.strategy_signals.get(strategy_id, [])[-10:]
                    
                    strategy_data = {
                        "strategy_id": strategy.strategy_id,
                        "strategy_name": strategy.strategy_name,
                        "strategy_type": strategy.strategy_type.value,
                        "version": strategy.version,
                        "status": strategy.status.value,
                        "description": strategy.description,
                        "created_at": strategy.created_at.isoformat(),
                        "updated_at": strategy.updated_at.isoformat(),
                        
                        # Performance metrics
                        "performance": {
                            "total_signals_generated": strategy.total_signals_generated,
                            "successful_signals": strategy.successful_signals,
                            "average_signal_accuracy": strategy.average_signal_accuracy,
                            "sharpe_ratio": strategy.sharpe_ratio,
                            "max_drawdown": strategy.max_drawdown,
                            "total_return": strategy.total_return,
                            "average_processing_time_ms": strategy.average_processing_time_ms
                        },
                        
                        # Hardware acceleration status
                        "hardware_acceleration": {
                            "neural_engine_optimized": strategy.neural_engine_optimized,
                            "hardware_acceleration_enabled": strategy.hardware_acceleration_enabled
                        },
                        
                        # Recent signals summary
                        "recent_activity": {
                            "recent_signals_count": len(recent_signals),
                            "last_signal_time": recent_signals[-1].timestamp if recent_signals else None,
                            "recent_average_confidence": sum(s.confidence for s in recent_signals) / len(recent_signals) if recent_signals else 0.0
                        }
                    }
                    
                    strategies_data.append(strategy_data)
                
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                return {
                    "strategies": strategies_data,
                    "count": len(strategies_data),
                    "response_time_ms": response_time_ms,
                    "active_strategies": len(self.enhanced_engine.active_strategies)
                }
                
            except Exception as e:
                logger.error(f"Get strategies error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies")
        async def create_strategy(request: StrategyCreateRequest):
            """Create new strategy with enhanced features"""
            start_time_req = time.time()
            
            try:
                # Map strategy type string to enum
                strategy_type_map = {
                    "mean_reversion": StrategyType.MEAN_REVERSION,
                    "momentum": StrategyType.MOMENTUM,
                    "arbitrage": StrategyType.ARBITRAGE,
                    "market_making": StrategyType.MARKET_MAKING,
                    "pairs_trading": StrategyType.PAIRS_TRADING,
                    "momentum_crossover": StrategyType.MOMENTUM_CROSSOVER,
                    "rsi_contrarian": StrategyType.RSI_CONTRARIAN,
                    "volatility_breakout": StrategyType.VOLATILITY_BREAKOUT,
                    "ml_driven": StrategyType.ML_DRIVEN
                }
                
                strategy_type = strategy_type_map.get(request.strategy_type.lower(), StrategyType.MEAN_REVERSION)
                
                # Create strategy definition
                strategy = StrategyDefinition(
                    strategy_id=f"strategy_{int(time.time())}_{len(self.enhanced_engine.strategies)}",
                    strategy_name=request.strategy_name,
                    strategy_type=strategy_type,
                    version="1.0.0",
                    description=request.description,
                    parameters=request.parameters,
                    risk_limits=request.risk_limits,
                    status=StrategyStatus.DRAFT,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    neural_engine_optimized=self.enhanced_engine.neural_engine_available,
                    hardware_acceleration_enabled=True
                )
                
                # Add to enhanced engine
                self.enhanced_engine.strategies[strategy.strategy_id] = strategy
                
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                return {
                    "status": "created",
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.strategy_name,
                    "strategy_type": strategy.strategy_type.value,
                    "version": strategy.version,
                    "neural_engine_optimized": strategy.neural_engine_optimized,
                    "response_time_ms": response_time_ms
                }
                
            except Exception as e:
                logger.error(f"Strategy creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== STRATEGY SIGNAL ENDPOINTS ====================
        
        @self.app.post("/strategies/{strategy_id}/signals/generate")
        async def generate_strategy_signal(strategy_id: str, request: SignalGenerationRequest, 
                                         background_tasks: BackgroundTasks):
            """Generate strategy signal with Neural Engine acceleration"""
            start_time_req = time.time()
            
            try:
                # Map priority string to MessagePriority
                priority_map = {
                    "low": "LOW",
                    "normal": "NORMAL", 
                    "high": "HIGH",
                    "urgent": "URGENT",
                    "critical": "CRITICAL"
                }
                
                # Generate signal using enhanced engine
                from universal_enhanced_messagebus_client import MessagePriority
                priority = MessagePriority(priority_map.get(request.priority, "HIGH"))
                
                strategy_signal = await self.enhanced_engine.generate_strategy_signal(
                    strategy_id=strategy_id,
                    symbol=request.symbol,
                    market_data=request.market_data,
                    priority=priority
                )
                
                # Update metrics
                self.signals_generated += 1
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                # Schedule background MessageBus notification
                background_tasks.add_task(self._notify_signal_generated, strategy_signal)
                
                return {
                    "status": "signal_generated",
                    "signal_id": strategy_signal.signal_id,
                    "strategy_id": strategy_signal.strategy_id,
                    "symbol": strategy_signal.symbol,
                    "signal_type": strategy_signal.signal_type,
                    "signal_strength": strategy_signal.signal_strength.value,
                    "confidence": strategy_signal.confidence,
                    "target_price": strategy_signal.target_price,
                    "stop_loss": strategy_signal.stop_loss,
                    "take_profit": strategy_signal.take_profit,
                    "position_size": strategy_signal.position_size,
                    "reasoning": strategy_signal.reasoning,
                    "processing_time_ms": strategy_signal.processing_time_ms,
                    "hardware_used": strategy_signal.hardware_used,
                    "response_time_ms": response_time_ms,
                    "timestamp": strategy_signal.timestamp
                }
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Signal generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/{strategy_id}/signals/rebalance")
        async def generate_rebalancing_signal(strategy_id: str, portfolio_data: Dict[str, Any],
                                            background_tasks: BackgroundTasks):
            """Generate portfolio rebalancing signal"""
            start_time_req = time.time()
            
            try:
                rebalancing_signal = await self.enhanced_engine.generate_rebalancing_signal(
                    strategy_id=strategy_id,
                    portfolio_data=portfolio_data
                )
                
                self.signals_generated += 1
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                # Background notification
                background_tasks.add_task(self._notify_signal_generated, rebalancing_signal)
                
                return {
                    "status": "rebalancing_signal_generated",
                    "signal_id": rebalancing_signal.signal_id,
                    "strategy_id": rebalancing_signal.strategy_id,
                    "signal_type": rebalancing_signal.signal_type,
                    "confidence": rebalancing_signal.confidence,
                    "reasoning": rebalancing_signal.reasoning,
                    "processing_time_ms": rebalancing_signal.processing_time_ms,
                    "hardware_used": rebalancing_signal.hardware_used,
                    "response_time_ms": response_time_ms
                }
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Rebalancing signal error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/{strategy_id}/signals")
        async def get_strategy_signals(strategy_id: str, limit: int = 50):
            """Get recent signals for a strategy"""
            start_time_req = time.time()
            
            try:
                if strategy_id not in self.enhanced_engine.strategies:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                recent_signals = self.enhanced_engine.strategy_signals.get(strategy_id, [])[-limit:]
                
                signals_data = []
                for signal in recent_signals:
                    signal_data = {
                        "signal_id": signal.signal_id,
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "signal_strength": signal.signal_strength.value,
                        "confidence": signal.confidence,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "position_size": signal.position_size,
                        "reasoning": signal.reasoning,
                        "processing_time_ms": signal.processing_time_ms,
                        "hardware_used": signal.hardware_used,
                        "timestamp": signal.timestamp
                    }
                    signals_data.append(signal_data)
                
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                return {
                    "signals": signals_data,
                    "count": len(signals_data),
                    "strategy_id": strategy_id,
                    "response_time_ms": response_time_ms
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get signals error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== STRATEGY OPTIMIZATION ENDPOINTS ====================
        
        @self.app.post("/strategies/{strategy_id}/optimize")
        async def optimize_strategy(strategy_id: str, request: StrategyOptimizationRequest,
                                  background_tasks: BackgroundTasks):
            """Optimize strategy parameters with Neural Engine acceleration"""
            start_time_req = time.time()
            
            try:
                optimization_result = await self.enhanced_engine.optimize_strategy_parameters(
                    strategy_id=strategy_id,
                    optimization_data=request.optimization_config
                )
                
                self.strategies_optimized += 1
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                # Background notification
                background_tasks.add_task(self._notify_optimization_completed, strategy_id, optimization_result)
                
                return {
                    "status": optimization_result.get("status", "completed"),
                    "strategy_id": strategy_id,
                    "optimization_time_ms": optimization_result.get("optimization_time_ms", 0),
                    "optimized_parameters": optimization_result.get("optimized_parameters", {}),
                    "expected_improvement": optimization_result.get("expected_performance_improvement", 0),
                    "response_time_ms": response_time_ms,
                    "timestamp": optimization_result.get("timestamp")
                }
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Strategy optimization error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== PERFORMANCE AND MONITORING ENDPOINTS ====================
        
        @self.app.get("/strategies/{strategy_id}/performance")
        async def get_strategy_performance(strategy_id: str):
            """Get detailed strategy performance metrics"""
            start_time_req = time.time()
            
            try:
                performance_data = await self.enhanced_engine._get_strategy_performance_data(strategy_id)
                
                if "error" in performance_data:
                    raise HTTPException(status_code=404, detail=performance_data["error"])
                
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                performance_data["response_time_ms"] = response_time_ms
                return performance_data
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Performance data error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/performance/summary")
        async def get_performance_summary():
            """Get comprehensive performance summary"""
            start_time_req = time.time()
            
            try:
                enhanced_performance = await self.enhanced_engine.get_performance_summary()
                
                response_time_ms = (time.time() - start_time_req) * 1000
                
                return {
                    "ultra_fast_engine_performance": {
                        "requests_processed": self.requests_processed,
                        "signals_generated": self.signals_generated,
                        "strategies_optimized": self.strategies_optimized,
                        "average_response_time_ms": self.average_response_time_ms,
                        "background_tasks_active": self.background_tasks_active
                    },
                    "enhanced_engine_performance": enhanced_performance,
                    "response_time_ms": response_time_ms,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Performance summary error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== HARDWARE ACCELERATION ENDPOINTS ====================
        
        @self.app.get("/hardware/status")
        async def get_hardware_status():
            """Get hardware acceleration status"""
            start_time_req = time.time()
            
            try:
                routing_stats = {}
                if self.enhanced_engine.hardware_router:
                    routing_stats = self.enhanced_engine.hardware_router.get_routing_statistics()
                
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                return {
                    "hardware_status": {
                        "neural_engine_available": self.enhanced_engine.neural_engine_available,
                        "metal_gpu_available": getattr(self.enhanced_engine.hardware_router, 'metal_gpu_available', False),
                        "m4_max_detected": self.enhanced_engine.m4_max_detected,
                        "hardware_router_active": self.enhanced_engine.hardware_router is not None
                    },
                    "performance_metrics": {
                        "neural_engine_signals": self.enhanced_engine.neural_engine_signals,
                        "cpu_fallback_signals": self.enhanced_engine.cpu_fallback_signals,
                        "hardware_acceleration_ratio": self.enhanced_engine.hardware_acceleration_ratio
                    },
                    "routing_statistics": routing_stats,
                    "response_time_ms": response_time_ms
                }
                
            except Exception as e:
                logger.error(f"Hardware status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== MESSAGEBUS ENDPOINTS ====================
        
        @self.app.get("/messagebus/status")
        async def get_messagebus_status():
            """Get MessageBus connection and performance status"""
            start_time_req = time.time()
            
            try:
                messagebus_metrics = {}
                if self.enhanced_engine.messagebus_client:
                    messagebus_metrics = await self.enhanced_engine.messagebus_client.get_performance_metrics()
                
                response_time_ms = (time.time() - start_time_req) * 1000
                self._update_request_metrics(response_time_ms)
                
                return {
                    "messagebus_status": {
                        "connected": self.enhanced_engine.messagebus_client is not None,
                        "subscribed_topics": len(self.enhanced_engine.subscribed_topics),
                        "messages_processed": self.messagebus_messages_processed
                    },
                    "messagebus_metrics": messagebus_metrics,
                    "response_time_ms": response_time_ms
                }
                
            except Exception as e:
                logger.error(f"MessageBus status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== BACKGROUND TASKS ====================
    
    async def _notify_signal_generated(self, strategy_signal: StrategySignal):
        """Background task to notify via MessageBus about signal generation"""
        try:
            self.background_tasks_active += 1
            
            # Additional MessageBus notifications can be sent here
            if self.enhanced_engine.messagebus_client:
                # Signal already published in enhanced engine, this could send additional notifications
                logger.debug(f"Background notification for signal: {strategy_signal.signal_id}")
            
            self.messagebus_messages_processed += 1
        except Exception as e:
            logger.error(f"Background signal notification error: {e}")
        finally:
            self.background_tasks_active = max(0, self.background_tasks_active - 1)
    
    async def _notify_optimization_completed(self, strategy_id: str, optimization_result: Dict[str, Any]):
        """Background task to notify optimization completion"""
        try:
            self.background_tasks_active += 1
            
            if self.enhanced_engine.messagebus_client:
                # Optimization result already published in enhanced engine
                logger.debug(f"Background notification for optimization: {strategy_id}")
            
            self.messagebus_messages_processed += 1
        except Exception as e:
            logger.error(f"Background optimization notification error: {e}")
        finally:
            self.background_tasks_active = max(0, self.background_tasks_active - 1)
    
    # ==================== HELPER METHODS ====================
    
    def _update_request_metrics(self, response_time_ms: float):
        """Update request processing metrics"""
        self.requests_processed += 1
        
        # Update average response time
        self.average_response_time_ms = (
            (self.average_response_time_ms * (self.requests_processed - 1) + response_time_ms) / 
            self.requests_processed
        )
    
    def _calculate_performance_grade(self, enhanced_metrics: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        try:
            avg_signal_time = enhanced_metrics.get("strategy_engine_performance", {}).get("average_signal_time_ms", 10)
            
            if avg_signal_time < 2.0:
                return "A++"
            elif avg_signal_time < 5.0:
                return "A+"
            elif avg_signal_time < 10.0:
                return "A"
            elif avg_signal_time < 20.0:
                return "B+"
            else:
                return "B"
        except:
            return "B"


# ==================== APPLICATION FACTORY ====================

def create_ultra_fast_strategy_app() -> FastAPI:
    """Factory function to create Ultra-Fast Strategy Engine application"""
    
    ultra_fast_engine = UltraFastStrategyEngine()
    return ultra_fast_engine.app


# ==================== MAIN APPLICATION ====================

# Create the FastAPI application
app = create_ultra_fast_strategy_app()

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8700"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"🚀 Starting Ultra-Fast Strategy Engine on {host}:{port}")
    logger.info(f"🧠 Neural Engine acceleration: {'enabled' if os.getenv('NEURAL_ENGINE_ENABLED', '1') == '1' else 'disabled'}")
    logger.info(f"📡 MessageBus integration: enabled")
    logger.info(f"🔧 Debug mode: {debug}")
    
    # Run the FastAPI application
    uvicorn.run(
        "ultra_fast_strategy_engine:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
        access_log=True,
        workers=1  # Single worker for MessageBus state consistency
    )
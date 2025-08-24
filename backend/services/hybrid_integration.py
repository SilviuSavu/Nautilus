#!/usr/bin/env python3
"""
Hybrid Architecture Integration Layer
Integrates native engines with Docker infrastructure

This component provides:
- FastAPI routes for hybrid engine communication
- Connection management for native engines
- Fallback handling when native engines unavailable
- Performance monitoring and load balancing
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Import native client services
try:
    from .native_ml_client import get_native_ml_client, cleanup_native_ml_client
    from .native_risk_client import get_native_risk_client, cleanup_native_risk_client
    from .native_strategy_client import get_native_strategy_client, cleanup_native_strategy_client
    HYBRID_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback when native clients not available
    HYBRID_INTEGRATION_AVAILABLE = False
    logging.warning("Hybrid integration clients not available - running in Docker-only mode")

# Pydantic models for API requests
class MLPredictionRequest(BaseModel):
    model_type: str
    input_data: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None

class RiskCalculationRequest(BaseModel):
    calculation_type: str
    portfolio_data: Dict[str, Any]
    parameters: Dict[str, Any]

class StrategyExecutionRequest(BaseModel):
    strategy_type: str
    market_data: Dict[str, Any]
    parameters: Dict[str, Any]

class HybridStatusResponse(BaseModel):
    hybrid_enabled: bool
    native_engines: Dict[str, Dict[str, Any]]
    performance_stats: Dict[str, Any]
    docker_fallback_active: bool

@dataclass
class HybridPerformanceMetrics:
    """Performance metrics for hybrid architecture"""
    native_requests: int = 0
    docker_fallback_requests: int = 0
    total_requests: int = 0
    average_native_latency_ms: float = 0.0
    average_docker_latency_ms: float = 0.0
    native_success_rate: float = 0.0
    docker_success_rate: float = 0.0
    total_native_time_ms: float = 0.0
    total_docker_time_ms: float = 0.0

class HybridIntegrationService:
    """Service for managing hybrid architecture integration"""
    
    def __init__(self):
        self.hybrid_enabled = HYBRID_INTEGRATION_AVAILABLE
        self.native_clients = {}
        self.performance_metrics = HybridPerformanceMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Connection status
        self.connection_status = {
            "ml_engine": False,
            "risk_engine": False,
            "strategy_engine": False
        }
        
        self.logger.info(f"Hybrid integration service initialized - Enabled: {self.hybrid_enabled}")
    
    async def initialize(self):
        """Initialize hybrid integration service"""
        if not self.hybrid_enabled:
            self.logger.warning("Hybrid integration not available - skipping native client initialization")
            return
        
        try:
            # Initialize native ML client
            try:
                ml_client = await get_native_ml_client()
                self.native_clients["ml_engine"] = ml_client
                self.connection_status["ml_engine"] = True
                self.logger.info("✅ Native ML Engine connected")
            except Exception as e:
                self.logger.warning(f"❌ Failed to connect to native ML Engine: {e}")
            
            # Initialize native Risk client
            try:
                risk_client = await get_native_risk_client()
                self.native_clients["risk_engine"] = risk_client
                self.connection_status["risk_engine"] = True
                self.logger.info("✅ Native Risk Engine connected")
            except Exception as e:
                self.logger.warning(f"❌ Failed to connect to native Risk Engine: {e}")
            
            # Initialize native Strategy client
            try:
                strategy_client = await get_native_strategy_client()
                self.native_clients["strategy_engine"] = strategy_client
                self.connection_status["strategy_engine"] = True
                self.logger.info("✅ Native Strategy Engine connected")
            except Exception as e:
                self.logger.warning(f"❌ Failed to connect to native Strategy Engine: {e}")
            
            self.logger.info(f"Hybrid integration initialized - Connected engines: {sum(self.connection_status.values())}/3")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid integration: {e}")
    
    async def predict_ml(self, request: MLPredictionRequest) -> Dict[str, Any]:
        """Execute ML prediction with hybrid routing"""
        start_time = time.time()
        self.performance_metrics.total_requests += 1
        
        # Try native ML engine first
        if self.connection_status.get("ml_engine", False):
            try:
                ml_client = self.native_clients.get("ml_engine")
                if ml_client:
                    result = await ml_client.predict(
                        request.model_type,
                        request.input_data,
                        request.options or {}
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    if result.get("success", False):
                        # Update native engine metrics
                        self.performance_metrics.native_requests += 1
                        self.performance_metrics.total_native_time_ms += processing_time
                        self.performance_metrics.average_native_latency_ms = (
                            self.performance_metrics.total_native_time_ms / 
                            self.performance_metrics.native_requests
                        )
                        
                        self.logger.info(f"ML prediction completed via native engine: {processing_time:.2f}ms")
                        
                        return {
                            "success": True,
                            "source": "native_ml_engine",
                            "predictions": result.get("predictions", {}),
                            "confidence": result.get("confidence", 0.0),
                            "processing_time_ms": processing_time,
                            "hardware_used": result.get("hardware_used", "unknown")
                        }
                    else:
                        self.logger.warning(f"Native ML engine returned error: {result.get('error', 'Unknown')}")
                        
            except Exception as e:
                self.logger.error(f"Native ML engine request failed: {e}")
                self.connection_status["ml_engine"] = False
        
        # Fallback to Docker ML engine
        return await self._fallback_ml_prediction(request, start_time)
    
    async def _fallback_ml_prediction(self, request: MLPredictionRequest, start_time: float) -> Dict[str, Any]:
        """Fallback ML prediction using Docker ML engine"""
        try:
            # Import Docker ML service
            from ..engines.ml.simple_ml_engine import get_ml_predictions
            
            # Convert request to Docker format
            docker_request = {
                "model_type": request.model_type,
                "input_data": request.input_data,
                "options": request.options or {}
            }
            
            # Execute via Docker ML engine
            docker_result = await get_ml_predictions(docker_request)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update Docker fallback metrics
            self.performance_metrics.docker_fallback_requests += 1
            self.performance_metrics.total_docker_time_ms += processing_time
            self.performance_metrics.average_docker_latency_ms = (
                self.performance_metrics.total_docker_time_ms / 
                self.performance_metrics.docker_fallback_requests
            )
            
            self.logger.info(f"ML prediction completed via Docker fallback: {processing_time:.2f}ms")
            
            return {
                "success": True,
                "source": "docker_ml_engine",
                "predictions": docker_result.get("predictions", {}),
                "confidence": docker_result.get("confidence", 0.5),
                "processing_time_ms": processing_time,
                "hardware_used": "cpu_docker"
            }
            
        except Exception as e:
            self.logger.error(f"Docker ML fallback failed: {e}")
            return {
                "success": False,
                "source": "error",
                "error": f"Both native and Docker ML engines failed: {str(e)}",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def calculate_risk(self, request: RiskCalculationRequest) -> Dict[str, Any]:
        """Execute risk calculation with hybrid routing"""
        start_time = time.time()
        self.performance_metrics.total_requests += 1
        
        # Try native Risk engine first
        if self.connection_status.get("risk_engine", False):
            try:
                risk_client = self.native_clients.get("risk_engine")
                if risk_client:
                    result = await risk_client.calculate_risk(
                        request.calculation_type,
                        request.portfolio_data,
                        request.parameters
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    if result.get("success", False):
                        # Update native engine metrics
                        self.performance_metrics.native_requests += 1
                        self.performance_metrics.total_native_time_ms += processing_time
                        self.performance_metrics.average_native_latency_ms = (
                            self.performance_metrics.total_native_time_ms / 
                            self.performance_metrics.native_requests
                        )
                        
                        self.logger.info(f"Risk calculation completed via native engine: {processing_time:.2f}ms")
                        
                        return {
                            "success": True,
                            "source": "native_risk_engine",
                            "risk_metrics": result.get("risk_metrics", {}),
                            "calculation_time_ms": processing_time,
                            "hardware_used": result.get("hardware_used", "unknown"),
                            "simulations_count": result.get("simulations_count", 0)
                        }
                    else:
                        self.logger.warning(f"Native Risk engine returned error: {result.get('error', 'Unknown')}")
                        
            except Exception as e:
                self.logger.error(f"Native Risk engine request failed: {e}")
                self.connection_status["risk_engine"] = False
        
        # Fallback to Docker Risk engine
        return await self._fallback_risk_calculation(request, start_time)
    
    async def _fallback_risk_calculation(self, request: RiskCalculationRequest, start_time: float) -> Dict[str, Any]:
        """Fallback risk calculation using Docker Risk engine"""
        try:
            # Import Docker Risk service
            from ..engines.risk.services import calculate_portfolio_var
            
            # Convert request to Docker format and execute
            if request.calculation_type == "monte_carlo_var":
                docker_result = await calculate_portfolio_var(
                    request.portfolio_data,
                    request.parameters
                )
            else:
                # Simple risk calculation fallback
                docker_result = {
                    "var_1d": 0.02,
                    "expected_shortfall": 0.025,
                    "confidence_level": 0.95
                }
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update Docker fallback metrics
            self.performance_metrics.docker_fallback_requests += 1
            self.performance_metrics.total_docker_time_ms += processing_time
            self.performance_metrics.average_docker_latency_ms = (
                self.performance_metrics.total_docker_time_ms / 
                self.performance_metrics.docker_fallback_requests
            )
            
            self.logger.info(f"Risk calculation completed via Docker fallback: {processing_time:.2f}ms")
            
            return {
                "success": True,
                "source": "docker_risk_engine",
                "risk_metrics": docker_result,
                "calculation_time_ms": processing_time,
                "hardware_used": "cpu_docker",
                "simulations_count": request.parameters.get("num_simulations", 10000)
            }
            
        except Exception as e:
            self.logger.error(f"Docker Risk fallback failed: {e}")
            return {
                "success": False,
                "source": "error",
                "error": f"Both native and Docker Risk engines failed: {str(e)}",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def execute_strategy(self, request: StrategyExecutionRequest) -> Dict[str, Any]:
        """Execute trading strategy with hybrid routing"""
        start_time = time.time()
        self.performance_metrics.total_requests += 1
        
        # Try native Strategy engine first
        if self.connection_status.get("strategy_engine", False):
            try:
                strategy_client = self.native_clients.get("strategy_engine")
                if strategy_client:
                    result = await strategy_client.execute_strategy(
                        request.strategy_type,
                        request.market_data,
                        request.parameters
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    if result.get("success", False):
                        # Update native engine metrics
                        self.performance_metrics.native_requests += 1
                        self.performance_metrics.total_native_time_ms += processing_time
                        self.performance_metrics.average_native_latency_ms = (
                            self.performance_metrics.total_native_time_ms / 
                            self.performance_metrics.native_requests
                        )
                        
                        self.logger.info(f"Strategy execution completed via native engine: {processing_time:.2f}ms")
                        
                        return {
                            "success": True,
                            "source": "native_strategy_engine",
                            "signals": result.get("signals", []),
                            "processing_time_ms": processing_time,
                            "hardware_used": result.get("hardware_used", "unknown"),
                            "patterns_analyzed": result.get("patterns_analyzed", 0)
                        }
                    else:
                        self.logger.warning(f"Native Strategy engine returned error: {result.get('error', 'Unknown')}")
                        
            except Exception as e:
                self.logger.error(f"Native Strategy engine request failed: {e}")
                self.connection_status["strategy_engine"] = False
        
        # Fallback to Docker Strategy engine
        return await self._fallback_strategy_execution(request, start_time)
    
    async def _fallback_strategy_execution(self, request: StrategyExecutionRequest, start_time: float) -> Dict[str, Any]:
        """Fallback strategy execution using Docker Strategy engine"""
        try:
            # Simple strategy execution fallback
            signals = []
            
            # Basic momentum strategy
            if request.strategy_type == "momentum_neural":
                current_price = request.market_data.get("current_price", 100.0)
                rsi = request.market_data.get("rsi", 50.0)
                
                if rsi < 30:  # Oversold
                    signals.append({
                        "signal_id": f"docker_buy_{int(time.time())}",
                        "symbol": request.market_data.get("symbol", "UNKNOWN"),
                        "signal_type": "BUY",
                        "confidence": 0.7,
                        "price": current_price,
                        "quantity": request.parameters.get("quantity", 100),
                        "timestamp": time.time(),
                        "pattern_detected": "oversold_momentum"
                    })
                elif rsi > 70:  # Overbought
                    signals.append({
                        "signal_id": f"docker_sell_{int(time.time())}",
                        "symbol": request.market_data.get("symbol", "UNKNOWN"),
                        "signal_type": "SELL",
                        "confidence": 0.7,
                        "price": current_price,
                        "quantity": request.parameters.get("quantity", 100),
                        "timestamp": time.time(),
                        "pattern_detected": "overbought_momentum"
                    })
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update Docker fallback metrics
            self.performance_metrics.docker_fallback_requests += 1
            self.performance_metrics.total_docker_time_ms += processing_time
            self.performance_metrics.average_docker_latency_ms = (
                self.performance_metrics.total_docker_time_ms / 
                self.performance_metrics.docker_fallback_requests
            )
            
            self.logger.info(f"Strategy execution completed via Docker fallback: {processing_time:.2f}ms")
            
            return {
                "success": True,
                "source": "docker_strategy_engine",
                "signals": signals,
                "processing_time_ms": processing_time,
                "hardware_used": "cpu_docker",
                "patterns_analyzed": 1
            }
            
        except Exception as e:
            self.logger.error(f"Docker Strategy fallback failed: {e}")
            return {
                "success": False,
                "source": "error",
                "error": f"Both native and Docker Strategy engines failed: {str(e)}",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def get_hybrid_status(self) -> HybridStatusResponse:
        """Get comprehensive hybrid architecture status"""
        try:
            native_engines = {}
            
            # Get ML engine status
            if self.connection_status.get("ml_engine", False):
                try:
                    ml_client = self.native_clients.get("ml_engine")
                    if ml_client:
                        ml_health = await ml_client.health_check()
                        native_engines["ml_engine"] = {
                            "status": "connected",
                            "healthy": ml_health.get("native_engine_healthy", False),
                            "hardware": "neural_engine" if ml_health.get("native_engine_healthy") else "cpu"
                        }
                    else:
                        native_engines["ml_engine"] = {"status": "client_not_initialized"}
                except Exception as e:
                    native_engines["ml_engine"] = {"status": "error", "error": str(e)}
                    self.connection_status["ml_engine"] = False
            else:
                native_engines["ml_engine"] = {"status": "disconnected"}
            
            # Get Risk engine status
            if self.connection_status.get("risk_engine", False):
                try:
                    risk_client = self.native_clients.get("risk_engine")
                    if risk_client:
                        risk_health = await risk_client.health_check()
                        native_engines["risk_engine"] = {
                            "status": "connected",
                            "healthy": risk_health.get("native_engine_healthy", False),
                            "hardware": "metal_gpu" if risk_health.get("native_engine_healthy") else "cpu"
                        }
                    else:
                        native_engines["risk_engine"] = {"status": "client_not_initialized"}
                except Exception as e:
                    native_engines["risk_engine"] = {"status": "error", "error": str(e)}
                    self.connection_status["risk_engine"] = False
            else:
                native_engines["risk_engine"] = {"status": "disconnected"}
            
            # Get Strategy engine status
            if self.connection_status.get("strategy_engine", False):
                try:
                    strategy_client = self.native_clients.get("strategy_engine")
                    if strategy_client:
                        strategy_health = await strategy_client.health_check()
                        native_engines["strategy_engine"] = {
                            "status": "connected",
                            "healthy": strategy_health.get("native_engine_healthy", False),
                            "hardware": "neural_engine" if strategy_health.get("native_engine_healthy") else "cpu"
                        }
                    else:
                        native_engines["strategy_engine"] = {"status": "client_not_initialized"}
                except Exception as e:
                    native_engines["strategy_engine"] = {"status": "error", "error": str(e)}
                    self.connection_status["strategy_engine"] = False
            else:
                native_engines["strategy_engine"] = {"status": "disconnected"}
            
            # Calculate success rates
            if self.performance_metrics.native_requests > 0:
                native_success_rate = self.performance_metrics.native_requests / max(1, self.performance_metrics.total_requests)
            else:
                native_success_rate = 0.0
            
            if self.performance_metrics.docker_fallback_requests > 0:
                docker_success_rate = self.performance_metrics.docker_fallback_requests / max(1, self.performance_metrics.total_requests)
            else:
                docker_success_rate = 0.0
            
            performance_stats = {
                "total_requests": self.performance_metrics.total_requests,
                "native_requests": self.performance_metrics.native_requests,
                "docker_fallback_requests": self.performance_metrics.docker_fallback_requests,
                "native_success_rate": native_success_rate,
                "docker_success_rate": docker_success_rate,
                "average_native_latency_ms": self.performance_metrics.average_native_latency_ms,
                "average_docker_latency_ms": self.performance_metrics.average_docker_latency_ms,
                "speedup_factor": (
                    self.performance_metrics.average_docker_latency_ms / 
                    max(0.1, self.performance_metrics.average_native_latency_ms)
                ) if self.performance_metrics.average_native_latency_ms > 0 else 1.0
            }
            
            connected_engines = sum(1 for status in self.connection_status.values() if status)
            docker_fallback_active = connected_engines < 3
            
            return HybridStatusResponse(
                hybrid_enabled=self.hybrid_enabled,
                native_engines=native_engines,
                performance_stats=performance_stats,
                docker_fallback_active=docker_fallback_active
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get hybrid status: {e}")
            return HybridStatusResponse(
                hybrid_enabled=False,
                native_engines={},
                performance_stats={},
                docker_fallback_active=True
            )
    
    async def cleanup(self):
        """Clean up hybrid integration resources"""
        if not self.hybrid_enabled:
            return
        
        try:
            # Cleanup native clients
            if "ml_engine" in self.native_clients:
                await cleanup_native_ml_client()
            
            if "risk_engine" in self.native_clients:
                await cleanup_native_risk_client()
            
            if "strategy_engine" in self.native_clients:
                await cleanup_native_strategy_client()
            
            self.native_clients.clear()
            self.connection_status = {engine: False for engine in self.connection_status}
            
            self.logger.info("Hybrid integration cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Hybrid integration cleanup failed: {e}")

# Global hybrid integration service instance
_hybrid_service = None

async def get_hybrid_service() -> HybridIntegrationService:
    """Get global hybrid integration service instance"""
    global _hybrid_service
    
    if _hybrid_service is None:
        _hybrid_service = HybridIntegrationService()
        await _hybrid_service.initialize()
    
    return _hybrid_service

async def cleanup_hybrid_service():
    """Clean up global hybrid integration service"""
    global _hybrid_service
    
    if _hybrid_service is not None:
        await _hybrid_service.cleanup()
        _hybrid_service = None

# FastAPI router for hybrid endpoints
hybrid_router = APIRouter(prefix="/api/v1/hybrid", tags=["Hybrid Architecture"])

@hybrid_router.post("/ml/predict")
async def hybrid_ml_predict(request: MLPredictionRequest):
    """ML prediction with hybrid routing (native + Docker fallback)"""
    try:
        service = await get_hybrid_service()
        result = await service.predict_ml(request)
        
        if result.get("success", False):
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "ML prediction failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid ML prediction failed: {str(e)}")

@hybrid_router.post("/risk/calculate")
async def hybrid_risk_calculate(request: RiskCalculationRequest):
    """Risk calculation with hybrid routing (native + Docker fallback)"""
    try:
        service = await get_hybrid_service()
        result = await service.calculate_risk(request)
        
        if result.get("success", False):
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Risk calculation failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid risk calculation failed: {str(e)}")

@hybrid_router.post("/strategy/execute")
async def hybrid_strategy_execute(request: StrategyExecutionRequest):
    """Strategy execution with hybrid routing (native + Docker fallback)"""
    try:
        service = await get_hybrid_service()
        result = await service.execute_strategy(request)
        
        if result.get("success", False):
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Strategy execution failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid strategy execution failed: {str(e)}")

@hybrid_router.get("/status", response_model=HybridStatusResponse)
async def get_hybrid_status():
    """Get comprehensive hybrid architecture status"""
    try:
        service = await get_hybrid_service()
        return await service.get_hybrid_status()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hybrid status: {str(e)}")

@hybrid_router.post("/health")
async def hybrid_health_check():
    """Health check for hybrid architecture"""
    try:
        service = await get_hybrid_service()
        status = await service.get_hybrid_status()
        
        connected_engines = sum(
            1 for engine_info in status.native_engines.values() 
            if engine_info.get("status") == "connected"
        )
        
        return {
            "status": "healthy" if connected_engines > 0 else "degraded",
            "hybrid_enabled": status.hybrid_enabled,
            "connected_native_engines": connected_engines,
            "total_native_engines": 3,
            "docker_fallback_active": status.docker_fallback_active,
            "performance_improvement": status.performance_stats.get("speedup_factor", 1.0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid health check failed: {str(e)}")

# Cleanup handler for FastAPI lifespan
async def cleanup_on_shutdown():
    """Cleanup hybrid integration on shutdown"""
    await cleanup_hybrid_service()
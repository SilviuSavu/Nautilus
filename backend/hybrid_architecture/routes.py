"""
FastAPI Routes for Hybrid Architecture Management
Provides REST API endpoints for monitoring and controlling the hybrid routing system.
"""

import asyncio
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from .circuit_breaker import circuit_breaker_registry, ENGINE_CIRCUIT_CONFIGS, CircuitBreakerConfig
from .health_monitor import health_monitor
from .enhanced_gateway import enhanced_gateway, RequestPriority
from .hybrid_router import hybrid_router, RoutingStrategy, OperationCategory

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/hybrid", tags=["Hybrid Architecture"])


# Request/Response Models
class RoutingRequest(BaseModel):
    engine: str
    endpoint: str
    method: str = "GET"
    priority: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class RoutingResponse(BaseModel):
    strategy: str
    target_engine: str
    use_direct_access: bool
    use_gateway: bool
    enable_fallback: bool
    expected_latency_ms: int
    confidence: float
    reasoning: str
    alternatives: List[str] = []


class OutcomeRecord(BaseModel):
    engine: str
    endpoint: str
    strategy: str
    actual_latency_ms: float
    success: bool
    reasoning: str


class CircuitBreakerUpdate(BaseModel):
    failure_threshold: Optional[int] = None
    recovery_timeout: Optional[int] = None
    success_threshold: Optional[int] = None
    timeout: Optional[float] = None


class SystemConfiguration(BaseModel):
    hybrid_routing_enabled: bool = True
    default_strategy: str = "hybrid_intelligent"
    health_check_interval: int = 30
    performance_update_interval: int = 60


# Health and Status Endpoints

@router.get("/health", response_model=Dict[str, Any])
async def get_hybrid_system_health():
    """Get comprehensive hybrid system health status"""
    try:
        health_summary = await health_monitor.get_system_health_summary()
        gateway_status = await enhanced_gateway.get_gateway_status()
        routing_metrics = hybrid_router.get_routing_metrics()
        circuit_breaker_status = await circuit_breaker_registry.get_all_status()
        
        return {
            "system_status": "healthy" if health_summary["overall_status"] == "healthy" else "degraded",
            "engines": health_summary,
            "gateway": gateway_status,
            "routing": routing_metrics,
            "circuit_breakers": circuit_breaker_status,
            "last_updated": health_summary.get("last_check", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting hybrid system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    """Get detailed system status including performance metrics"""
    try:
        return {
            "hybrid_routing": {
                "enabled": hybrid_router.enabled,
                "metrics": hybrid_router.get_routing_metrics()
            },
            "gateway": await enhanced_gateway.get_gateway_status(),
            "engines": await health_monitor.get_system_health_summary(),
            "circuit_breakers": await circuit_breaker_registry.get_all_status()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Routing Decision Endpoints

@router.post("/route", response_model=RoutingResponse)
async def make_routing_decision(request: RoutingRequest):
    """Make intelligent routing decision for a request"""
    try:
        # Convert priority string to enum
        priority = None
        if request.priority:
            try:
                priority = RequestPriority(request.priority.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")
        
        # Make routing decision
        decision = await hybrid_router.make_routing_decision(
            engine=request.engine,
            endpoint=request.endpoint,
            method=request.method,
            priority=priority,
            context=request.context
        )
        
        return RoutingResponse(
            strategy=decision.strategy.value,
            target_engine=decision.target_engine,
            use_direct_access=decision.use_direct_access,
            use_gateway=decision.use_gateway,
            enable_fallback=decision.enable_fallback,
            expected_latency_ms=decision.expected_latency_ms,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            alternatives=decision.alternatives
        )
        
    except Exception as e:
        logger.error(f"Error making routing decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-outcome")
async def record_routing_outcome(outcome: OutcomeRecord):
    """Record the outcome of a routing decision for learning"""
    try:
        # Create a minimal routing decision for recording
        from .hybrid_router import RoutingDecision
        
        decision = RoutingDecision(
            strategy=RoutingStrategy(outcome.strategy),
            target_engine=outcome.engine,
            use_direct_access=outcome.strategy in ["direct_only", "hybrid_performance", "hybrid_intelligent"],
            use_gateway=not (outcome.strategy == "direct_only"),
            enable_fallback=True,
            expected_latency_ms=0,  # Not available in outcome
            confidence=1.0,
            reasoning=outcome.reasoning
        )
        
        hybrid_router.record_routing_outcome(
            decision=decision,
            actual_latency_ms=outcome.actual_latency_ms,
            success=outcome.success
        )
        
        return {"success": True, "message": "Outcome recorded"}
        
    except Exception as e:
        logger.error(f"Error recording outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration Endpoints

@router.get("/config", response_model=Dict[str, Any])
async def get_system_configuration():
    """Get current hybrid system configuration"""
    try:
        return {
            "hybrid_routing_enabled": hybrid_router.enabled,
            "default_strategy": hybrid_router.default_strategy.value,
            "routing_rules_count": len(hybrid_router.routing_rules),
            "performance_profiles_count": len(hybrid_router.performance_profiles),
            "circuit_breaker_configs": {
                name: {
                    "failure_threshold": config.failure_threshold,
                    "recovery_timeout": config.recovery_timeout,
                    "success_threshold": config.success_threshold,
                    "timeout": config.timeout
                }
                for name, config in ENGINE_CIRCUIT_CONFIGS.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_system_configuration(config: SystemConfiguration):
    """Update hybrid system configuration"""
    try:
        # Update hybrid router settings
        if config.hybrid_routing_enabled:
            hybrid_router.enable_routing()
        else:
            hybrid_router.disable_routing()
        
        # Update default strategy
        try:
            hybrid_router.default_strategy = RoutingStrategy(config.default_strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {config.default_strategy}")
        
        return {"success": True, "message": "Configuration updated"}
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Circuit Breaker Management

@router.get("/circuit-breakers", response_model=Dict[str, Any])
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    try:
        return await circuit_breaker_registry.get_all_status()
        
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breakers/{engine}/reset")
async def reset_circuit_breaker(engine: str):
    """Reset a specific circuit breaker"""
    try:
        await circuit_breaker_registry.reset_breaker(engine)
        return {"success": True, "message": f"Circuit breaker {engine} reset"}
        
    except Exception as e:
        logger.error(f"Error resetting circuit breaker {engine}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breakers/reset-all")
async def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    try:
        await circuit_breaker_registry.reset_all()
        return {"success": True, "message": "All circuit breakers reset"}
        
    except Exception as e:
        logger.error(f"Error resetting all circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Engine Health Management

@router.get("/engines", response_model=Dict[str, Any])
async def get_engine_status():
    """Get detailed status of all engines"""
    try:
        return await health_monitor.get_system_health_summary()
        
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engines/{engine}/health-check")
async def force_engine_health_check(engine: str):
    """Force health check for specific engine"""
    try:
        await health_monitor.force_health_check(engine)
        return {"success": True, "message": f"Health check completed for {engine}"}
        
    except Exception as e:
        logger.error(f"Error forcing health check for {engine}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engines/health-check-all")
async def force_all_engines_health_check():
    """Force health check for all engines"""
    try:
        await health_monitor.force_health_check()
        return {"success": True, "message": "Health check completed for all engines"}
        
    except Exception as e:
        logger.error(f"Error forcing health check for all engines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Metrics

@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        routing_metrics = hybrid_router.get_routing_metrics()
        gateway_status = await enhanced_gateway.get_gateway_status()
        
        return {
            "routing": routing_metrics,
            "gateway": gateway_status["gateway"],
            "timestamp": gateway_status.get("timestamp", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/routing", response_model=Dict[str, Any])
async def get_routing_metrics():
    """Get detailed routing metrics"""
    try:
        return hybrid_router.get_routing_metrics()
        
    except Exception as e:
        logger.error(f"Error getting routing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/gateway", response_model=Dict[str, Any])
async def get_gateway_metrics():
    """Get detailed gateway metrics"""
    try:
        return await enhanced_gateway.get_gateway_status()
        
    except Exception as e:
        logger.error(f"Error getting gateway metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Testing and Debugging

@router.post("/test/routing")
async def test_routing_scenarios():
    """Test routing decisions for common scenarios"""
    try:
        test_scenarios = [
            {"engine": "strategy", "endpoint": "/execute", "method": "POST", "priority": "critical"},
            {"engine": "risk", "endpoint": "/calculate-var", "method": "POST", "priority": "high"},
            {"engine": "analytics", "endpoint": "/real-time", "method": "GET", "priority": "normal"},
            {"engine": "ml", "endpoint": "/predict", "method": "POST", "priority": "high"},
            {"engine": "factor", "endpoint": "/calculate", "method": "GET", "priority": "normal"}
        ]
        
        results = []
        for scenario in test_scenarios:
            priority = RequestPriority(scenario["priority"]) if scenario.get("priority") else None
            
            decision = await hybrid_router.make_routing_decision(
                engine=scenario["engine"],
                endpoint=scenario["endpoint"],
                method=scenario["method"],
                priority=priority
            )
            
            results.append({
                "scenario": scenario,
                "decision": {
                    "strategy": decision.strategy.value,
                    "use_direct_access": decision.use_direct_access,
                    "use_gateway": decision.use_gateway,
                    "expected_latency_ms": decision.expected_latency_ms,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning
                }
            })
        
        return {"test_results": results}
        
    except Exception as e:
        logger.error(f"Error testing routing scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_hybrid_system(background_tasks: BackgroundTasks):
    """Initialize the hybrid architecture system"""
    try:
        # Initialize components in background
        async def init_components():
            await enhanced_gateway.initialize()
            await health_monitor.start_monitoring()
            await hybrid_router.update_performance_profiles()
        
        background_tasks.add_task(init_components)
        
        return {"success": True, "message": "Hybrid system initialization started"}
        
    except Exception as e:
        logger.error(f"Error initializing hybrid system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shutdown")
async def shutdown_hybrid_system(background_tasks: BackgroundTasks):
    """Shutdown the hybrid architecture system"""
    try:
        # Shutdown components in background
        async def shutdown_components():
            await enhanced_gateway.shutdown()
            await health_monitor.stop_monitoring()
        
        background_tasks.add_task(shutdown_components)
        
        return {"success": True, "message": "Hybrid system shutdown started"}
        
    except Exception as e:
        logger.error(f"Error shutting down hybrid system: {e}")
        raise HTTPException(status_code=500, detail=str(e))
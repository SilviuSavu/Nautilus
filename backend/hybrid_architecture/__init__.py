"""
Nautilus Hybrid Architecture Module
Provides intelligent routing between direct engine access and enhanced gateway
for optimal performance in trading operations.
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    circuit_breaker_registry,
    circuit_breaker,
    ENGINE_CIRCUIT_CONFIGS
)

from .health_monitor import (
    EngineHealthChecker,
    EngineStatus,
    EngineMetrics,
    health_monitor,
    get_engine_health_summary,
    is_engine_healthy
)

from .enhanced_gateway import (
    EnhancedAPIGateway,
    RequestPriority,
    CacheStrategy,
    enhanced_gateway,
    gateway_request,
    get_gateway_health
)

from .hybrid_router import (
    IntelligentHybridRouter,
    RoutingStrategy,
    OperationCategory,
    RoutingDecision,
    hybrid_router,
    route_request,
    record_outcome,
    get_routing_status
)

__version__ = "1.0.0"
__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig", 
    "CircuitBreakerOpenException",
    "circuit_breaker_registry",
    "circuit_breaker",
    "ENGINE_CIRCUIT_CONFIGS",
    
    # Health Monitor
    "EngineHealthChecker",
    "EngineStatus",
    "EngineMetrics", 
    "health_monitor",
    "get_engine_health_summary",
    "is_engine_healthy",
    
    # Enhanced Gateway
    "EnhancedAPIGateway",
    "RequestPriority",
    "CacheStrategy",
    "enhanced_gateway",
    "gateway_request",
    "get_gateway_health",
    
    # Hybrid Router
    "IntelligentHybridRouter",
    "RoutingStrategy",
    "OperationCategory", 
    "RoutingDecision",
    "hybrid_router",
    "route_request",
    "record_outcome",
    "get_routing_status"
]
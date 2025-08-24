"""
Enhanced API Gateway for Nautilus Hybrid Architecture
Provides intelligent routing, connection pooling, caching, and circuit breaker integration.
"""

import asyncio
import httpx
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from contextlib import asynccontextmanager

from .circuit_breaker import circuit_breaker_registry, ENGINE_CIRCUIT_CONFIGS, CircuitBreakerOpenException
from .health_monitor import health_monitor, EngineStatus

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = "critical"      # Trading operations - <50ms target
    HIGH = "high"             # Risk calculations - <100ms target
    NORMAL = "normal"         # Analytics - <500ms target
    LOW = "low"               # Background processing - <2000ms target


class CacheStrategy(Enum):
    """Cache strategies for different operation types"""
    NONE = "none"             # No caching (trading operations)
    SHORT = "short"           # 30 second cache
    MEDIUM = "medium"         # 5 minute cache
    LONG = "long"             # 1 hour cache


@dataclass
class RoutingConfig:
    """Configuration for engine routing"""
    engine_name: str
    priority: RequestPriority
    timeout_ms: int
    cache_strategy: CacheStrategy
    retry_count: int = 2
    enable_circuit_breaker: bool = True
    
    @property
    def timeout_seconds(self) -> float:
        return self.timeout_ms / 1000.0


class EnhancedGatewayCache:
    """
    Intelligent caching system with TTL and cache strategies.
    Uses Redis for distributed caching across gateway instances.
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry)
        
        # Cache TTL configurations
        self.ttl_configs = {
            CacheStrategy.NONE: 0,
            CacheStrategy.SHORT: 30,      # 30 seconds
            CacheStrategy.MEDIUM: 300,    # 5 minutes
            CacheStrategy.LONG: 3600      # 1 hour
        }
    
    def _generate_cache_key(self, engine: str, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request"""
        key_data = f"{engine}:{endpoint}"
        if params:
            # Sort params for consistent key generation
            sorted_params = json.dumps(params, sort_keys=True)
            key_data += f":{sorted_params}"
        
        # Hash for consistent key length
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, engine: str, endpoint: str, params: Dict = None) -> Optional[Any]:
        """Get cached response"""
        cache_key = self._generate_cache_key(engine, endpoint, params)
        
        # Check local cache first
        if cache_key in self.local_cache:
            value, expiry = self.local_cache[cache_key]
            if time.time() < expiry:
                logger.debug(f"📋 Cache HIT (local): {engine}:{endpoint}")
                return value
            else:
                # Expired - remove from local cache
                del self.local_cache[cache_key]
        
        # Check Redis cache if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"gateway_cache:{cache_key}")
                if cached_data:
                    logger.debug(f"📋 Cache HIT (redis): {engine}:{endpoint}")
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"📋 Redis cache error: {e}")
        
        logger.debug(f"📋 Cache MISS: {engine}:{endpoint}")
        return None
    
    async def set(self, engine: str, endpoint: str, value: Any, 
                  cache_strategy: CacheStrategy, params: Dict = None):
        """Set cached response"""
        if cache_strategy == CacheStrategy.NONE:
            return
        
        cache_key = self._generate_cache_key(engine, endpoint, params)
        ttl = self.ttl_configs[cache_strategy]
        expiry = time.time() + ttl
        
        # Store in local cache
        self.local_cache[cache_key] = (value, expiry)
        
        # Store in Redis cache if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"gateway_cache:{cache_key}",
                    ttl,
                    json.dumps(value, default=str)
                )
                logger.debug(f"📋 Cached response: {engine}:{endpoint} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"📋 Redis cache set error: {e}")
        
        # Clean up expired local cache entries periodically
        if len(self.local_cache) > 1000:
            await self._cleanup_local_cache()
    
    async def _cleanup_local_cache(self):
        """Clean up expired local cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.local_cache.items()
            if current_time >= expiry
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
        
        logger.debug(f"📋 Cleaned up {len(expired_keys)} expired cache entries")


class ConnectionPoolManager:
    """
    Manages HTTP connection pools for all engines with M4 Max optimization.
    Provides connection pooling, keep-alive, and HTTP/2 support.
    """
    
    def __init__(self):
        self.pools: Dict[str, httpx.AsyncClient] = {}
        self.pool_configs = self._get_pool_configurations()
        
    def _get_pool_configurations(self) -> Dict[str, Dict]:
        """Get connection pool configurations for each engine"""
        return {
            # Critical trading engines - higher connection limits
            "strategy": {
                "base_url": "http://localhost:8700",
                "pool_size": 25,
                "keepalive": 20,
                "timeout": 5.0
            },
            "risk": {
                "base_url": "http://localhost:8200", 
                "pool_size": 20,
                "keepalive": 15,
                "timeout": 10.0
            },
            # Real-time engines - medium connection limits
            "analytics": {
                "base_url": "http://localhost:8100",
                "pool_size": 15,
                "keepalive": 10,
                "timeout": 15.0
            },
            "ml": {
                "base_url": "http://localhost:8400",
                "pool_size": 15,
                "keepalive": 10,
                "timeout": 30.0  # ML inference can take longer
            },
            "websocket": {
                "base_url": "http://localhost:8600",
                "pool_size": 20,
                "keepalive": 15,
                "timeout": 5.0
            },
            # Background engines - standard connection limits
            "factor": {
                "base_url": "http://localhost:8300",
                "pool_size": 10,
                "keepalive": 5,
                "timeout": 30.0
            },
            "features": {
                "base_url": "http://localhost:8500",
                "pool_size": 10,
                "keepalive": 5,
                "timeout": 30.0
            },
            "marketdata": {
                "base_url": "http://localhost:8800",
                "pool_size": 12,
                "keepalive": 8,
                "timeout": 15.0
            },
            "portfolio": {
                "base_url": "http://localhost:8900",
                "pool_size": 10,
                "keepalive": 5,
                "timeout": 20.0
            }
        }
    
    async def initialize(self):
        """Initialize connection pools for all engines"""
        for engine_name, config in self.pool_configs.items():
            try:
                self.pools[engine_name] = httpx.AsyncClient(
                    base_url=config["base_url"],
                    limits=httpx.Limits(
                        max_connections=config["pool_size"],
                        max_keepalive_connections=config["keepalive"],
                        keepalive_expiry=30.0  # Keep connections alive for 30s
                    ),
                    timeout=httpx.Timeout(
                        connect=2.0,
                        read=config["timeout"],
                        write=5.0,
                        pool=1.0
                    ),
                    http2=True,  # Enable HTTP/2 for multiplexing
                    headers={
                        "User-Agent": "Nautilus-Gateway/1.0",
                        "Connection": "keep-alive"
                    }
                )
                logger.info(f"🔗 Initialized connection pool for {engine_name} "
                           f"(pool: {config['pool_size']}, keepalive: {config['keepalive']})")
                
            except Exception as e:
                logger.error(f"🔗 Failed to initialize pool for {engine_name}: {e}")
        
        logger.info(f"🔗 Connection pool manager initialized with {len(self.pools)} pools")
    
    async def get_client(self, engine_name: str) -> Optional[httpx.AsyncClient]:
        """Get HTTP client for specific engine"""
        return self.pools.get(engine_name)
    
    async def close_all(self):
        """Close all connection pools"""
        for engine_name, client in self.pools.items():
            try:
                await client.aclose()
                logger.debug(f"🔗 Closed connection pool for {engine_name}")
            except Exception as e:
                logger.warning(f"🔗 Error closing pool for {engine_name}: {e}")
        
        self.pools.clear()
        logger.info("🔗 All connection pools closed")
    
    def get_pool_stats(self) -> Dict[str, Dict]:
        """Get connection pool statistics"""
        stats = {}
        for engine_name, client in self.pools.items():
            try:
                # Note: httpx doesn't expose detailed pool stats by default
                # This would need to be extended with custom metrics collection
                stats[engine_name] = {
                    "pool_size": self.pool_configs[engine_name]["pool_size"],
                    "keepalive": self.pool_configs[engine_name]["keepalive"],
                    "is_closed": client.is_closed,
                    "base_url": str(client.base_url)
                }
            except Exception as e:
                stats[engine_name] = {"error": str(e)}
        
        return stats


class EnhancedAPIGateway:
    """
    Enhanced API Gateway with intelligent routing, connection pooling, 
    caching, and circuit breaker integration for Nautilus trading platform.
    """
    
    def __init__(self, redis_client=None):
        self.connection_manager = ConnectionPoolManager()
        self.cache = EnhancedGatewayCache(redis_client)
        self.routing_configs = self._initialize_routing_configs()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "circuit_breaker_opens": 0,
            "fallback_used": 0,
            "average_response_time": 0.0,
            "response_times": []
        }
    
    def _initialize_routing_configs(self) -> Dict[str, RoutingConfig]:
        """Initialize routing configurations for all engines"""
        return {
            # Critical trading operations
            "strategy": RoutingConfig(
                engine_name="strategy",
                priority=RequestPriority.CRITICAL,
                timeout_ms=5000,  # 5 seconds max
                cache_strategy=CacheStrategy.NONE,  # No caching for trading
                retry_count=1  # Minimal retries for speed
            ),
            "risk": RoutingConfig(
                engine_name="risk", 
                priority=RequestPriority.HIGH,
                timeout_ms=10000,  # 10 seconds max
                cache_strategy=CacheStrategy.SHORT,  # Brief caching for risk metrics
                retry_count=2
            ),
            # Real-time analysis engines
            "analytics": RoutingConfig(
                engine_name="analytics",
                priority=RequestPriority.NORMAL,
                timeout_ms=15000,
                cache_strategy=CacheStrategy.MEDIUM,
                retry_count=2
            ),
            "ml": RoutingConfig(
                engine_name="ml",
                priority=RequestPriority.HIGH,
                timeout_ms=30000,  # ML inference can take longer
                cache_strategy=CacheStrategy.SHORT,
                retry_count=1  # ML predictions change frequently
            ),
            "websocket": RoutingConfig(
                engine_name="websocket",
                priority=RequestPriority.HIGH,
                timeout_ms=5000,
                cache_strategy=CacheStrategy.NONE,  # Real-time data
                retry_count=2
            ),
            # Background processing engines
            "factor": RoutingConfig(
                engine_name="factor",
                priority=RequestPriority.NORMAL,
                timeout_ms=30000,
                cache_strategy=CacheStrategy.LONG,  # Factor data changes slowly
                retry_count=3
            ),
            "features": RoutingConfig(
                engine_name="features", 
                priority=RequestPriority.NORMAL,
                timeout_ms=30000,
                cache_strategy=CacheStrategy.MEDIUM,
                retry_count=3
            ),
            "marketdata": RoutingConfig(
                engine_name="marketdata",
                priority=RequestPriority.NORMAL,
                timeout_ms=15000,
                cache_strategy=CacheStrategy.SHORT,
                retry_count=2
            ),
            "portfolio": RoutingConfig(
                engine_name="portfolio",
                priority=RequestPriority.NORMAL,
                timeout_ms=20000,
                cache_strategy=CacheStrategy.MEDIUM,
                retry_count=2
            )
        }
    
    async def initialize(self):
        """Initialize the enhanced gateway"""
        await self.connection_manager.initialize()
        await health_monitor.start_monitoring()
        logger.info("🚀 Enhanced API Gateway initialized")
    
    async def shutdown(self):
        """Shutdown the enhanced gateway"""
        await self.connection_manager.close_all()
        await health_monitor.stop_monitoring()
        logger.info("🚀 Enhanced API Gateway shutdown complete")
    
    async def route_request(
        self,
        engine: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        priority: Optional[RequestPriority] = None
    ) -> Dict[str, Any]:
        """
        Route request to appropriate engine with intelligent handling
        
        Args:
            engine: Target engine name
            endpoint: API endpoint path
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Request body data
            params: Query parameters
            priority: Request priority override
            
        Returns:
            Dict containing response data and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        # Get routing configuration
        config = self.routing_configs.get(engine)
        if not config:
            return self._error_response(f"Unknown engine: {engine}", 400)
        
        # Override priority if specified
        if priority:
            config.priority = priority
        
        try:
            # Check cache for GET requests (non-POST operations)
            cached_response = None
            if method.upper() == "GET" and config.cache_strategy != CacheStrategy.NONE:
                cached_response = await self.cache.get(engine, endpoint, params)
                if cached_response:
                    self.metrics["cache_hits"] += 1
                    return self._success_response(
                        cached_response, 
                        time.time() - start_time, 
                        cached=True
                    )
                else:
                    self.metrics["cache_misses"] += 1
            
            # Execute request with circuit breaker protection
            response_data = await self._execute_with_circuit_breaker(
                engine, endpoint, method, data, params, config
            )
            
            # Cache successful GET responses
            if (method.upper() == "GET" and 
                config.cache_strategy != CacheStrategy.NONE and
                response_data.get("success", False)):
                await self.cache.set(
                    engine, endpoint, response_data["data"], 
                    config.cache_strategy, params
                )
            
            # Record success metrics
            response_time = time.time() - start_time
            self._record_success_metrics(response_time)
            
            return self._success_response(response_data["data"], response_time)
            
        except CircuitBreakerOpenException as e:
            # Circuit breaker is open - try fallback
            self.metrics["circuit_breaker_opens"] += 1
            logger.warning(f"🔄 Circuit breaker open for {engine}: {e}")
            
            return await self._handle_fallback(engine, endpoint, method, data, params, str(e))
            
        except Exception as e:
            # Other errors
            response_time = time.time() - start_time
            self._record_failure_metrics(response_time)
            
            logger.error(f"🚨 Gateway error for {engine}{endpoint}: {e}")
            return self._error_response(str(e), 500, response_time)
    
    async def _execute_with_circuit_breaker(
        self,
        engine: str,
        endpoint: str,
        method: str,
        data: Optional[Dict],
        params: Optional[Dict],
        config: RoutingConfig
    ) -> Dict[str, Any]:
        """Execute request with circuit breaker protection"""
        
        if not config.enable_circuit_breaker:
            # Direct execution without circuit breaker
            return await self._execute_request(engine, endpoint, method, data, params, config)
        
        # Get or create circuit breaker for this engine
        circuit_config = ENGINE_CIRCUIT_CONFIGS.get(engine, ENGINE_CIRCUIT_CONFIGS["default"])
        breaker = await circuit_breaker_registry.get_or_create(engine, circuit_config)
        
        # Execute with circuit breaker protection
        return await breaker.call(
            self._execute_request, 
            engine, endpoint, method, data, params, config
        )
    
    async def _execute_request(
        self,
        engine: str,
        endpoint: str,
        method: str,
        data: Optional[Dict],
        params: Optional[Dict],
        config: RoutingConfig
    ) -> Dict[str, Any]:
        """Execute the actual HTTP request"""
        
        # Get HTTP client from connection pool
        client = await self.connection_manager.get_client(engine)
        if not client:
            raise Exception(f"No connection pool available for {engine}")
        
        # Check engine health
        engine_health = await health_monitor.get_engine_health(engine)
        if engine_health and engine_health.status == EngineStatus.UNHEALTHY:
            raise Exception(f"Engine {engine} is unhealthy")
        
        # Prepare request
        request_kwargs = {
            "timeout": config.timeout_seconds
        }
        
        if params:
            request_kwargs["params"] = params
        
        if data:
            request_kwargs["json"] = data
        
        # Execute request with retries
        last_exception = None
        for attempt in range(config.retry_count + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(endpoint, **request_kwargs)
                elif method.upper() == "POST":
                    response = await client.post(endpoint, **request_kwargs)
                elif method.upper() == "PUT":
                    response = await client.put(endpoint, **request_kwargs)
                elif method.upper() == "DELETE":
                    response = await client.delete(endpoint, **request_kwargs)
                else:
                    raise Exception(f"Unsupported HTTP method: {method}")
                
                # Check response status
                if response.status_code >= 400:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                # Parse response
                try:
                    response_data = response.json()
                except:
                    response_data = {"message": response.text}
                
                return {"success": True, "data": response_data}
                
            except Exception as e:
                last_exception = e
                if attempt < config.retry_count:
                    # Brief pause before retry
                    await asyncio.sleep(0.1 * (attempt + 1))
                    logger.debug(f"🔄 Retrying {engine}{endpoint} (attempt {attempt + 2})")
                else:
                    logger.warning(f"🔄 All retries failed for {engine}{endpoint}")
        
        # All attempts failed
        raise last_exception
    
    async def _handle_fallback(
        self,
        engine: str,
        endpoint: str,
        method: str,
        data: Optional[Dict],
        params: Optional[Dict],
        error: str
    ) -> Dict[str, Any]:
        """Handle fallback when primary engine fails"""
        
        self.metrics["fallback_used"] += 1
        
        # Specific fallback strategies
        fallback_data = None
        
        if engine == "risk" and "var" in endpoint.lower():
            # Risk calculation fallback
            fallback_data = {
                "var_95": 0.05,  # Conservative fallback
                "var_99": 0.10,
                "fallback_reason": "risk_engine_unavailable",
                "confidence": 0.3
            }
        
        elif engine == "ml" and "predict" in endpoint.lower():
            # ML prediction fallback
            fallback_data = {
                "prediction": "hold",
                "confidence": 0.1,
                "fallback_reason": "ml_engine_unavailable"
            }
        
        elif engine == "analytics":
            # Analytics fallback
            fallback_data = {
                "trend": "neutral",
                "indicators": {},
                "fallback_reason": "analytics_engine_unavailable"
            }
        
        if fallback_data:
            logger.info(f"🔄 Using fallback data for {engine}{endpoint}")
            return self._success_response(fallback_data, 0, fallback=True)
        
        # No specific fallback available
        return self._error_response(
            f"Service temporarily unavailable: {error}",
            503,
            0,
            {"retry_after": 30, "fallback_attempted": True}
        )
    
    def _success_response(
        self, 
        data: Any, 
        response_time: float, 
        cached: bool = False, 
        fallback: bool = False
    ) -> Dict[str, Any]:
        """Create success response with metadata"""
        return {
            "success": True,
            "data": data,
            "metadata": {
                "response_time_ms": round(response_time * 1000, 2),
                "cached": cached,
                "fallback": fallback,
                "timestamp": time.time()
            }
        }
    
    def _error_response(
        self, 
        error: str, 
        status_code: int, 
        response_time: float = 0, 
        extra_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create error response with metadata"""
        response = {
            "success": False,
            "error": error,
            "status_code": status_code,
            "metadata": {
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": time.time()
            }
        }
        
        if extra_data:
            response["metadata"].update(extra_data)
        
        return response
    
    def _record_success_metrics(self, response_time: float):
        """Record successful request metrics"""
        self.metrics["successful_requests"] += 1
        self.metrics["response_times"].append(response_time)
        
        # Keep only last 1000 response times
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
        
        # Update average response time
        if self.metrics["response_times"]:
            self.metrics["average_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
    
    def _record_failure_metrics(self, response_time: float):
        """Record failed request metrics"""
        self.metrics["failed_requests"] += 1
        self.metrics["response_times"].append(response_time)
    
    async def get_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive gateway status"""
        pool_stats = self.connection_manager.get_pool_stats()
        health_summary = await health_monitor.get_system_health_summary()
        circuit_status = await circuit_breaker_registry.get_all_status()
        
        success_rate = 0
        if self.metrics["total_requests"] > 0:
            success_rate = (self.metrics["successful_requests"] / self.metrics["total_requests"]) * 100
        
        return {
            "gateway": {
                "total_requests": self.metrics["total_requests"],
                "success_rate": round(success_rate, 2),
                "average_response_time_ms": round(self.metrics["average_response_time"] * 1000, 2),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "circuit_breaker_opens": self.metrics["circuit_breaker_opens"],
                "fallback_usage": self.metrics["fallback_used"]
            },
            "connection_pools": pool_stats,
            "engine_health": health_summary,
            "circuit_breakers": circuit_status
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache_requests == 0:
            return 0.0
        return (self.metrics["cache_hits"] / total_cache_requests) * 100


# Global enhanced gateway instance
enhanced_gateway = EnhancedAPIGateway()


# Convenience functions for easy integration
async def gateway_request(
    engine: str,
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    priority: Optional[RequestPriority] = None
) -> Dict[str, Any]:
    """Convenience function for gateway requests"""
    return await enhanced_gateway.route_request(engine, endpoint, method, data, params, priority)


async def get_gateway_health() -> Dict[str, Any]:
    """Get gateway health status"""
    return await enhanced_gateway.get_gateway_status()
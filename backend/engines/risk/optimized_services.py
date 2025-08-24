#!/usr/bin/env python3
"""
Optimized Risk Engine Services - Enhanced performance implementation
"""

import asyncio
import logging
import time
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from cachetools import TTLCache, LRUCache

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available - falling back to standard NumPy")

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using memory cache only")

from models import RiskLimit, RiskBreach, RiskLimitType, BreachSeverity
from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority


logger = logging.getLogger(__name__)


class AsyncBatchProcessor:
    """Batch processor for grouping async requests"""
    
    def __init__(self, batch_size: int = 100, max_wait_ms: int = 50):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.pending_futures = []
        self.last_process_time = time.time()
        self._lock = asyncio.Lock()
        
    async def add_request(self, func, *args) -> Any:
        """Add request to batch and return future result"""
        
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_requests.append((func, args))
            self.pending_futures.append(future)
            
            current_time = time.time()
            should_process = (
                len(self.pending_requests) >= self.batch_size or
                (current_time - self.last_process_time) * 1000 >= self.max_wait_ms
            )
            
            if should_process:
                await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated batch of requests"""
        if not self.pending_requests:
            return
            
        requests = self.pending_requests.copy()
        futures = self.pending_futures.copy()
        
        self.pending_requests.clear()
        self.pending_futures.clear()
        self.last_process_time = time.time()
        
        try:
            # Group by function type
            func_groups = {}
            for i, (func, args) in enumerate(requests):
                if func not in func_groups:
                    func_groups[func] = []
                func_groups[func].append((i, args))
            
            # Process each function group
            results = [None] * len(requests)
            
            for func, grouped_requests in func_groups.items():
                indices, args_list = zip(*grouped_requests)
                
                # Execute batch function
                if hasattr(func, '_batch_execute'):
                    batch_results = await func._batch_execute(args_list)
                else:
                    # Fallback to individual execution
                    tasks = [func(*args) for args in args_list]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Map results back to original indices
                for idx, result in zip(indices, batch_results):
                    results[idx] = result
            
            # Set future results
            for future, result in zip(futures, results):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)


class RiskCalculationCache:
    """Advanced caching system for risk calculations"""
    
    def __init__(self):
        # Multi-tier caching
        self.memory_cache = TTLCache(maxsize=1000, ttl=30)  # 30 seconds
        self.redis_cache = None
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def initialize(self):
        """Initialize Redis connection if available"""
        if REDIS_AVAILABLE:
            try:
                self.redis_cache = redis.Redis(
                    host='redis', port=6379, db=2,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    decode_responses=False
                )
                # Test connection
                await self.redis_cache.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
                self.redis_cache = None
        
    def get_cache_key(self, portfolio_id: str, position_data: Dict, calculation_type: str) -> str:
        """Generate deterministic cache key"""
        data_str = f"{portfolio_id}:{str(sorted(position_data.items()))}:{calculation_type}"
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"risk:{calculation_type}:{data_hash}"
    
    async def get_cached_result(self, cache_key: str):
        """Get result from multi-tier cache"""
        
        # Check memory cache first (fastest)
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        # Check Redis cache if available
        if self.redis_cache:
            try:
                redis_result = await self.redis_cache.get(cache_key)
                if redis_result:
                    result = pickle.loads(redis_result)
                    # Promote to memory cache
                    self.memory_cache[cache_key] = result
                    self.cache_hits += 1
                    return result
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
        
        self.cache_misses += 1
        return None
    
    async def cache_result(self, cache_key: str, result: Any, ttl_seconds: int = 300):
        """Store result in multi-tier cache"""
        
        # Store in memory cache
        self.memory_cache[cache_key] = result
        
        # Store in Redis cache if available
        if self.redis_cache:
            try:
                await self.redis_cache.setex(
                    cache_key, 
                    ttl_seconds, 
                    pickle.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Redis cache store error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_ratio = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio_percent": round(hit_ratio, 2),
            "memory_cache_size": len(self.memory_cache),
            "redis_available": self.redis_cache is not None
        }


class HighPerformanceRiskCalculator:
    """High-performance risk calculation engine"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="risk-calc")
        
    def calculate_var_fast(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Fast VaR calculation"""
        if NUMBA_AVAILABLE:
            return self._calculate_var_numba(returns, confidence)
        else:
            return self._calculate_var_numpy(returns, confidence)
    
    @staticmethod
    def _calculate_var_numpy(returns: np.ndarray, confidence: float) -> float:
        """NumPy-based VaR calculation"""
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return -sorted_returns[index] if index < len(sorted_returns) else 0.0
    
    if NUMBA_AVAILABLE:
        @staticmethod
        @numba.jit(nopython=True, parallel=True)
        def _calculate_var_numba(returns: np.ndarray, confidence: float) -> float:
            """Numba-optimized VaR calculation"""
            sorted_returns = np.sort(returns)
            index = int((1 - confidence) * len(sorted_returns))
            return -sorted_returns[index] if index < len(sorted_returns) else 0.0
    
    def calculate_portfolio_metrics_fast(
        self, 
        positions: np.ndarray, 
        prices: np.ndarray,
        correlations: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Fast portfolio risk metrics calculation"""
        
        # Market values
        market_values = positions * prices
        total_value = np.sum(market_values)
        
        if total_value == 0:
            return self._zero_portfolio_metrics()
        
        weights = market_values / total_value
        
        # Basic metrics
        metrics = {
            'portfolio_value': total_value,
            'concentration_risk': np.max(np.abs(weights)),
            'gross_exposure': np.sum(np.abs(market_values)),
            'net_exposure': total_value,
            'leverage': np.sum(np.abs(market_values)) / max(np.sum(market_values[market_values > 0]), 1.0)
        }
        
        # Add volatility if correlation matrix provided
        if correlations is not None and len(correlations) == len(weights):
            try:
                portfolio_variance = np.dot(weights, np.dot(correlations, weights))
                metrics['portfolio_volatility'] = np.sqrt(max(portfolio_variance, 0))
            except Exception as e:
                logger.warning(f"Volatility calculation error: {e}")
                metrics['portfolio_volatility'] = 0.0
        
        return metrics
    
    @staticmethod
    def _zero_portfolio_metrics() -> Dict[str, float]:
        """Default metrics for empty portfolio"""
        return {
            'portfolio_value': 0.0,
            'concentration_risk': 0.0,
            'gross_exposure': 0.0,
            'net_exposure': 0.0,
            'leverage': 1.0,
            'portfolio_volatility': 0.0
        }


class OptimizedRiskCalculationService:
    """Optimized risk calculation service with async processing and caching"""
    
    def __init__(self):
        # Original service state
        self.active_limits: Dict[str, RiskLimit] = {}
        self.active_breaches: Dict[str, RiskBreach] = {}
        self.risk_checks_processed = 0
        self.breaches_detected = 0
        
        # Performance optimization components
        self.cache = RiskCalculationCache()
        self.calculator = HighPerformanceRiskCalculator()
        self.batch_processor = AsyncBatchProcessor(batch_size=50, max_wait_ms=100)
        
        # Performance metrics
        self.performance_metrics = {
            "total_calculations": 0,
            "cache_hit_ratio": 0.0,
            "avg_calculation_time_ms": 0.0,
            "batch_size_avg": 0.0,
            "parallel_calculations": 0
        }
        
    async def initialize(self):
        """Initialize the optimized service"""
        await self.cache.initialize()
        logger.info("Optimized Risk Calculation Service initialized")
    
    def add_limit(self, limit: RiskLimit):
        """Add a new risk limit"""
        self.active_limits[limit.limit_id] = limit
        
    def remove_limit(self, limit_id: str):
        """Remove a risk limit"""
        if limit_id in self.active_limits:
            del self.active_limits[limit_id]
    
    async def check_position_risk_async(self, portfolio_id: str, position_data: Dict[str, Any]) -> List[RiskBreach]:
        """Async risk check with caching and optimization"""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self.cache.get_cache_key(portfolio_id, position_data, "risk_check")
        cached_result = await self.cache.get_cached_result(cache_key)
        
        if cached_result is not None:
            self._update_performance_metrics(time.time() - start_time, cached=True)
            return cached_result
        
        # Cache miss - perform calculation
        result = await self._calculate_risk_with_optimization(portfolio_id, position_data)
        
        # Cache the result (shorter TTL for risk checks)
        await self.cache.cache_result(cache_key, result, ttl_seconds=30)
        
        self._update_performance_metrics(time.time() - start_time, cached=False)
        return result
    
    async def _calculate_risk_with_optimization(self, portfolio_id: str, position_data: Dict[str, Any]) -> List[RiskBreach]:
        """Optimized risk calculation with parallel processing"""
        
        breaches = []
        self.risk_checks_processed += 1
        
        # Get applicable limits
        applicable_limits = [
            limit for limit in self.active_limits.values()
            if (limit.enabled and 
                (not limit.portfolio_id or limit.portfolio_id == portfolio_id))
        ]
        
        if not applicable_limits:
            return breaches
        
        # Process limits in parallel
        limit_tasks = []
        for limit in applicable_limits:
            task = asyncio.create_task(
                self._check_single_limit_async(limit, position_data)
            )
            limit_tasks.append(task)
        
        # Wait for all limit checks
        limit_results = await asyncio.gather(*limit_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(limit_results):
            if isinstance(result, Exception):
                logger.error(f"Limit check error for {applicable_limits[i].limit_id}: {result}")
                continue
            
            if result:  # Breach detected
                breaches.append(result)
                self.active_breaches[result.breach_id] = result
                self.breaches_detected += 1
        
        return breaches
    
    async def _check_single_limit_async(self, limit: RiskLimit, position_data: Dict[str, Any]) -> Optional[RiskBreach]:
        """Async check of single risk limit"""
        
        try:
            # Calculate current value based on limit type
            current_value = await self._calculate_limit_value_async(limit, position_data)
            
            # Check for breach
            if current_value > limit.limit_value * limit.threshold_breach:
                return self._create_breach(limit, current_value)
                
        except Exception as e:
            logger.error(f"Single limit check error: {e}")
            
        return None
    
    async def _calculate_limit_value_async(self, limit: RiskLimit, position_data: Dict[str, Any]) -> float:
        """Async calculation of limit value"""
        
        if limit.limit_type == RiskLimitType.POSITION_SIZE:
            return abs(position_data.get("quantity", 0))
            
        elif limit.limit_type == RiskLimitType.PORTFOLIO_VALUE:
            return position_data.get("market_value", 0)
            
        elif limit.limit_type == RiskLimitType.DAILY_LOSS:
            return abs(position_data.get("unrealized_pnl", 0))
            
        elif limit.limit_type == RiskLimitType.LEVERAGE:
            # Use optimized calculation for complex metrics
            positions = np.array(position_data.get("positions", [1.0]))
            prices = np.array(position_data.get("prices", [position_data.get("price", 100.0)]))
            
            if len(positions) == len(prices):
                metrics = self.calculator.calculate_portfolio_metrics_fast(positions, prices)
                return metrics['leverage']
            else:
                return position_data.get("leverage", 1.0)
                
        else:
            return 0.0
    
    def _create_breach(self, limit: RiskLimit, current_value: float) -> RiskBreach:
        """Create a breach record"""
        breach_percentage = (current_value / limit.limit_value) * 100
        
        # Determine severity
        if breach_percentage >= 150:
            severity = BreachSeverity.CRITICAL
        elif breach_percentage >= 120:
            severity = BreachSeverity.HIGH
        elif breach_percentage >= 100:
            severity = BreachSeverity.MEDIUM
        else:
            severity = BreachSeverity.LOW
        
        return RiskBreach(
            breach_id=f"{limit.limit_id}_{int(time.time())}",
            limit_id=limit.limit_id,
            portfolio_id=limit.portfolio_id,
            limit_type=limit.limit_type,
            limit_value=limit.limit_value,
            current_value=current_value,
            breach_percentage=breach_percentage,
            severity=severity,
            breach_time=datetime.now(),
            resolved=False
        )
    
    def _update_performance_metrics(self, calculation_time: float, cached: bool):
        """Update performance tracking metrics"""
        
        self.performance_metrics["total_calculations"] += 1
        
        # Update cache hit ratio
        cache_stats = self.cache.get_cache_stats()
        self.performance_metrics["cache_hit_ratio"] = cache_stats["hit_ratio_percent"]
        
        # Update average calculation time (exponential moving average)
        calc_time_ms = calculation_time * 1000
        current_avg = self.performance_metrics["avg_calculation_time_ms"]
        self.performance_metrics["avg_calculation_time_ms"] = (
            0.9 * current_avg + 0.1 * calc_time_ms
        )
        
        if not cached:
            self.performance_metrics["parallel_calculations"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        cache_stats = self.cache.get_cache_stats()
        
        return {
            "calculation_performance": self.performance_metrics,
            "cache_performance": cache_stats,
            "service_metrics": {
                "risk_checks_processed": self.risk_checks_processed,
                "breaches_detected": self.breaches_detected,
                "active_limits": len(self.active_limits),
                "active_breaches": len(self.active_breaches)
            }
        }
    
    async def resolve_breach(self, breach_id: str, resolution_data: Dict[str, Any]):
        """Resolve an active breach"""
        if breach_id in self.active_breaches:
            breach = self.active_breaches[breach_id]
            breach.resolved = True
            breach.resolved_time = datetime.now()
            breach.action_taken = resolution_data.get("action_taken", "Manual resolution")
            del self.active_breaches[breach_id]


class OptimizedRiskMonitoringService:
    """Optimized monitoring service with improved performance"""
    
    def __init__(self, calculation_service: OptimizedRiskCalculationService, messagebus: BufferedMessageBusClient):
        self.calculation_service = calculation_service
        self.messagebus = messagebus
        self.monitoring_active = False
        self.monitor_task = None
        self.alert_batches = []
        
    async def start_monitoring(self):
        """Start optimized continuous risk monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_task = asyncio.create_task(self._optimized_monitoring_loop())
            
    async def stop_monitoring(self):
        """Stop continuous risk monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _optimized_monitoring_loop(self):
        """Optimized monitoring loop with batched alerts"""
        
        batch_size = 10
        batch_timeout = 5.0  # seconds
        last_batch_time = time.time()
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check for critical breaches
                critical_breaches = [
                    breach for breach in self.calculation_service.active_breaches.values()
                    if breach.severity == BreachSeverity.CRITICAL and not breach.resolved
                ]
                
                # Add to batch
                self.alert_batches.extend(critical_breaches)
                
                # Process batch if full or timeout reached
                if (len(self.alert_batches) >= batch_size or
                    (self.alert_batches and current_time - last_batch_time > batch_timeout)):
                    
                    await self._process_alert_batch()
                    self.alert_batches.clear()
                    last_batch_time = current_time
                
                # Sleep for shorter interval for better responsiveness
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Optimized monitoring error: {e}")
                await asyncio.sleep(2)  # Back off on error
    
    async def _process_alert_batch(self):
        """Process batch of alerts efficiently"""
        
        if not self.alert_batches:
            return
        
        # Group alerts by severity for batch publishing
        severity_groups = {}
        for breach in self.alert_batches:
            severity = breach.severity
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(breach)
        
        # Publish grouped alerts
        publish_tasks = []
        
        for severity, breaches in severity_groups.items():
            priority = MessagePriority.URGENT if severity == BreachSeverity.CRITICAL else MessagePriority.HIGH
            
            # Create batch message
            batch_message = {
                "type": "batch_risk_alert",
                "severity": severity.value,
                "breach_count": len(breaches),
                "breaches": [asdict(breach) for breach in breaches],
                "timestamp": datetime.now().isoformat()
            }
            
            task = self.messagebus.publish(
                f"risk.breach.batch.{severity.value}",
                batch_message,
                priority=priority
            )
            publish_tasks.append(task)
        
        # Wait for all publishes to complete
        if publish_tasks:
            await asyncio.gather(*publish_tasks, return_exceptions=True)


# Export optimized services
__all__ = [
    'OptimizedRiskCalculationService',
    'OptimizedRiskMonitoringService', 
    'RiskCalculationCache',
    'HighPerformanceRiskCalculator'
]
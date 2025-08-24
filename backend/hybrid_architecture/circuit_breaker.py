"""
Enhanced Circuit Breaker Implementation for Nautilus Hybrid Architecture
Provides fault tolerance and automatic failover for engine communications.
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests are failing
    HALF_OPEN = "half_open"  # Testing if service is recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout: int = 60          # Seconds before attempting recovery
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0               # Request timeout in seconds
    monitor_window: int = 300           # Monitoring window in seconds


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    state_changes: int = 0
    average_response_time: float = 0
    response_times: list = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage"""
        return 100.0 - self.success_rate


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Enhanced circuit breaker with M4 Max optimization support.
    Provides automatic fault tolerance and failover for engine communications.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.lock = asyncio.Lock()
        
        logger.info(f"🔧 Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._check_circuit_state()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with result recording"""
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure(str(exc_val) if exc_val else "Unknown error")
        return False
    
    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            TimeoutError: If function times out
        """
        async with self.lock:
            await self._check_circuit_state()
        
        start_time = time.time()
        
        try:
            # Execute function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record successful execution
            execution_time = time.time() - start_time
            async with self.lock:
                await self._record_success(execution_time)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            async with self.lock:
                await self._record_failure(f"Timeout after {execution_time:.2f}s")
            raise TimeoutError(f"Circuit breaker timeout for {self.name}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            async with self.lock:
                await self._record_failure(f"{type(e).__name__}: {str(e)}")
            raise
    
    async def _check_circuit_state(self):
        """Check and update circuit breaker state"""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if current_time - self.metrics.last_failure_time >= self.config.recovery_timeout:
                await self._transition_to_half_open()
            else:
                remaining_time = self.config.recovery_timeout - (current_time - self.metrics.last_failure_time)
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Retry in {remaining_time:.1f} seconds"
                )
        
        elif self.state == CircuitState.HALF_OPEN:
            # In half-open state, allow limited requests through
            logger.debug(f"🔧 Circuit breaker '{self.name}' in half-open state, allowing request")
    
    async def _record_success(self, execution_time: float = 0):
        """Record successful execution"""
        current_time = time.time()
        
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = current_time
        
        # Update response time metrics
        if execution_time > 0:
            self.metrics.response_times.append(execution_time)
            # Keep only recent response times (last 100)
            if len(self.metrics.response_times) > 100:
                self.metrics.response_times = self.metrics.response_times[-100:]
            self.metrics.average_response_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
        
        # Check if we should close the circuit
        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                await self._transition_to_closed()
        
        logger.debug(f"✅ Success recorded for '{self.name}' - {self.metrics.consecutive_successes} consecutive")
    
    async def _record_failure(self, error: str):
        """Record failed execution"""
        current_time = time.time()
        
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = current_time
        
        logger.warning(f"❌ Failure recorded for '{self.name}': {error} - {self.metrics.consecutive_failures} consecutive")
        
        # Check if we should open the circuit
        if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
            self.metrics.consecutive_failures >= self.config.failure_threshold):
            await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition circuit to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.metrics.state_changes += 1
        
        logger.error(f"🚨 Circuit breaker '{self.name}' opened: {self.metrics.consecutive_failures} consecutive failures")
        logger.info(f"🔧 Circuit breaker '{self.name}': {old_state.value} → {self.state.value}")
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.metrics.state_changes += 1
        
        logger.info(f"🟡 Circuit breaker '{self.name}' half-opened: Testing recovery")
        logger.info(f"🔧 Circuit breaker '{self.name}': {old_state.value} → {self.state.value}")
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.metrics.state_changes += 1
        
        logger.info(f"✅ Circuit breaker '{self.name}' closed: Service recovered")
        logger.info(f"🔧 Circuit breaker '{self.name}': {old_state.value} → {self.state.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": round(self.metrics.success_rate, 2),
                "failure_rate": round(self.metrics.failure_rate, 2),
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "average_response_time_ms": round(self.metrics.average_response_time * 1000, 2),
                "state_changes": self.metrics.state_changes,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers across engines.
    Provides centralized circuit breaker management and monitoring.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.lock = asyncio.Lock()
    
    async def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        async with self.lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
                logger.info(f"🔧 Created new circuit breaker: {name}")
            return self._breakers[name]
    
    async def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        async with self.lock:
            return {name: breaker.get_status() for name, breaker in self._breakers.items()}
    
    async def reset_breaker(self, name: str):
        """Reset a specific circuit breaker"""
        async with self.lock:
            if name in self._breakers:
                self._breakers[name].state = CircuitState.CLOSED
                self._breakers[name].metrics = CircuitBreakerMetrics()
                logger.info(f"🔄 Reset circuit breaker: {name}")
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        async with self.lock:
            for name, breaker in self._breakers.items():
                breaker.state = CircuitState.CLOSED
                breaker.metrics = CircuitBreakerMetrics()
            logger.info("🔄 Reset all circuit breakers")


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


@asynccontextmanager
async def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """
    Async context manager for circuit breaker usage
    
    Usage:
        async with circuit_breaker("strategy-engine") as breaker:
            result = await some_engine_call()
    """
    breaker = await circuit_breaker_registry.get_or_create(name, config)
    async with breaker:
        yield breaker


# Engine-specific circuit breaker configurations
ENGINE_CIRCUIT_CONFIGS = {
    "strategy": CircuitBreakerConfig(
        failure_threshold=3,    # Critical path - fail fast
        recovery_timeout=10,    # Quick recovery attempts
        success_threshold=2,    # Quick to restore
        timeout=5.0            # 5 second timeout for trading
    ),
    "risk": CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=15,
        success_threshold=2,
        timeout=10.0           # Risk calculations may take longer
    ),
    "analytics": CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30,
        success_threshold=3,
        timeout=15.0
    ),
    "ml": CircuitBreakerConfig(
        failure_threshold=4,
        recovery_timeout=20,
        success_threshold=2,
        timeout=30.0           # ML inference may take longer
    ),
    "default": CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60,
        success_threshold=3,
        timeout=30.0
    )
}
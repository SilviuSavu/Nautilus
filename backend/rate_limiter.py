"""
Rate Limiting and Throttling Service
Provides advanced rate limiting with circuit breaker patterns and backpressure handling.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

from enums import Venue


class ThrottleStrategy(Enum):
    """Throttling strategies"""
    DROP = "drop"  # Drop excess messages
    QUEUE = "queue"  # Queue messages with backpressure
    SAMPLE = "sample"  # Sample messages at regular intervals


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, rejecting all requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_second: int
    burst_size: int
    window_size_seconds: int = 1
    throttle_strategy: ThrottleStrategy = ThrottleStrategy.QUEUE
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 30


@dataclass
class ThrottleMetrics:
    """Throttling metrics"""
    total_requests: int = 0
    allowed_requests: int = 0
    throttled_requests: int = 0
    dropped_requests: int = 0
    queued_requests: int = 0
    circuit_trips: int = 0
    last_reset: datetime = None


class TokenBucket:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_update = time.time()
        
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        now = time.time()
        
        # Add tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
            
        return False
        
    def available_tokens(self) -> int:
        """Get number of available tokens"""
        now = time.time()
        elapsed = now - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return True
            
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
        
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class MessageQueue:
    """Async message queue with backpressure"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = asyncio.Queue(maxsize=max_size)
        self.overflow_count = 0
        
    async def put(self, item, block: bool = True) -> bool:
        """Put item in queue"""
        try:
            if block:
                await self.queue.put(item)
                return True
            else:
                self.queue.put_nowait(item)
                return True
        except asyncio.QueueFull:
            self.overflow_count += 1
            return False
            
    async def get(self):
        """Get item from queue"""
        return await self.queue.get()
        
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
        
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
        
    def full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()


class RateLimiter:
    """
    Advanced rate limiter with multiple throttling strategies,
    circuit breaker pattern, and comprehensive metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._venue_limiters: Dict[Venue, TokenBucket] = {}
        self._global_limiter: TokenBucket = None
        self._circuit_breakers: Dict[Venue, CircuitBreaker] = {}
        self._message_queues: Dict[Venue, MessageQueue] = {}
        self._metrics: Dict[Venue, ThrottleMetrics] = {}
        self._configs: Dict[Venue, RateLimitConfig] = {}
        self._sliding_windows: Dict[Venue, deque] = {}
        self._setup_default_configs()
        
    def _setup_default_configs(self):
        """Setup default rate limit configurations for venues"""
        default_configs = {
            Venue.BINANCE: RateLimitConfig(
                requests_per_second=100,
                burst_size=200,
                throttle_strategy=ThrottleStrategy.QUEUE
            ),
            Venue.COINBASE: RateLimitConfig(
                requests_per_second=50,
                burst_size=100,
                throttle_strategy=ThrottleStrategy.QUEUE
            ),
            Venue.KRAKEN: RateLimitConfig(
                requests_per_second=30,
                burst_size=60,
                throttle_strategy=ThrottleStrategy.DROP
            ),
            Venue.BYBIT: RateLimitConfig(
                requests_per_second=80,
                burst_size=160,
                throttle_strategy=ThrottleStrategy.QUEUE
            ),
            Venue.OKX: RateLimitConfig(
                requests_per_second=60,
                burst_size=120,
                throttle_strategy=ThrottleStrategy.SAMPLE
            ),
        }
        
        for venue, config in default_configs.items():
            self.configure_venue(venue, config)
            
        # Global rate limiter
        self._global_limiter = TokenBucket(rate=500, capacity=1000)
        
    def configure_venue(self, venue: Venue, config: RateLimitConfig):
        """Configure rate limiting for a specific venue"""
        self._configs[venue] = config
        self._venue_limiters[venue] = TokenBucket(
            rate=config.requests_per_second,
            capacity=config.burst_size
        )
        
        if config.circuit_breaker_enabled:
            self._circuit_breakers[venue] = CircuitBreaker(
                failure_threshold=config.circuit_failure_threshold,
                recovery_timeout=config.circuit_recovery_timeout
            )
            
        if config.throttle_strategy == ThrottleStrategy.QUEUE:
            self._message_queues[venue] = MessageQueue(max_size=config.burst_size * 2)
            
        self._metrics[venue] = ThrottleMetrics(last_reset=datetime.now())
        self._sliding_windows[venue] = deque()
        
    async def should_allow_request(self, venue: Venue, message_data: dict = None) -> Tuple[bool, str]:
        """
        Check if request should be allowed based on rate limits
        Returns (allowed, reason)
        """
        if venue not in self._configs:
            return True, "No rate limit configured"
            
        config = self._configs[venue]
        metrics = self._metrics[venue]
        metrics.total_requests += 1
        
        # Check circuit breaker first
        if venue in self._circuit_breakers:
            circuit = self._circuit_breakers[venue]
            if circuit.state == CircuitState.OPEN:
                return False, "Circuit breaker OPEN"
                
        # Check global rate limit
        if not self._global_limiter.consume():
            metrics.throttled_requests += 1
            return False, "Global rate limit exceeded"
            
        # Check venue-specific rate limit
        venue_limiter = self._venue_limiters[venue]
        if not venue_limiter.consume():
            metrics.throttled_requests += 1
            
            # Apply throttling strategy
            if config.throttle_strategy == ThrottleStrategy.DROP:
                metrics.dropped_requests += 1
                return False, "Rate limit exceeded - message dropped"
                
            elif config.throttle_strategy == ThrottleStrategy.QUEUE:
                # Try to queue the message
                if venue in self._message_queues:
                    queue = self._message_queues[venue]
                    if await queue.put(message_data, block=False):
                        metrics.queued_requests += 1
                        return False, "Rate limit exceeded - message queued"
                    else:
                        metrics.dropped_requests += 1
                        return False, "Rate limit exceeded - queue full"
                        
            elif config.throttle_strategy == ThrottleStrategy.SAMPLE:
                # Use sliding window sampling
                now = time.time()
                window = self._sliding_windows[venue]
                
                # Remove old entries
                while window and window[0] < now - config.window_size_seconds:
                    window.popleft()
                    
                # Check if we should sample this message
                if len(window) < config.requests_per_second:
                    window.append(now)
                    metrics.allowed_requests += 1
                    return True, "Sampled message allowed"
                else:
                    metrics.dropped_requests += 1
                    return False, "Rate limit exceeded - message not sampled"
                    
        metrics.allowed_requests += 1
        return True, "Request allowed"
        
    async def process_queued_messages(self, venue: Venue, processor_func) -> int:
        """Process queued messages for a venue"""
        if venue not in self._message_queues:
            return 0
            
        queue = self._message_queues[venue]
        processed_count = 0
        
        # Process messages while respecting rate limits
        while not queue.empty():
            venue_limiter = self._venue_limiters[venue]
            if not venue_limiter.consume():
                break  # Rate limit reached
                
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.1)
                await processor_func(message)
                processed_count += 1
            except asyncio.TimeoutError:
                break
            except Exception as e:
                self.logger.error(f"Error processing queued message: {e}")
                
        return processed_count
        
    def record_failure(self, venue: Venue, error: Exception = None):
        """Record a failure for circuit breaker"""
        if venue in self._circuit_breakers:
            circuit = self._circuit_breakers[venue]
            circuit._on_failure()
            
            if circuit.state == CircuitState.OPEN:
                self._metrics[venue].circuit_trips += 1
                self.logger.warning(f"Circuit breaker opened for venue {venue.value}")
                
    def record_success(self, venue: Venue):
        """Record a success for circuit breaker"""
        if venue in self._circuit_breakers:
            self._circuit_breakers[venue]._on_success()
            
    def get_venue_metrics(self, venue: Venue) -> Optional[ThrottleMetrics]:
        """Get metrics for a specific venue"""
        return self._metrics.get(venue)
        
    def get_all_metrics(self) -> Dict[str, ThrottleMetrics]:
        """Get metrics for all venues"""
        return {venue.value: metrics for venue, metrics in self._metrics.items()}
        
    def get_venue_status(self, venue: Venue) -> Dict[str, any]:
        """Get comprehensive status for a venue"""
        if venue not in self._configs:
            return {"error": "Venue not configured"}
            
        config = self._configs[venue]
        metrics = self._metrics[venue]
        limiter = self._venue_limiters[venue]
        
        status = {
            "venue": venue.value,
            "config": {
                "requests_per_second": config.requests_per_second,
                "burst_size": config.burst_size,
                "throttle_strategy": config.throttle_strategy.value,
            },
            "rate_limiter": {
                "available_tokens": int(limiter.available_tokens()),
                "capacity": limiter.capacity,
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "allowed_requests": metrics.allowed_requests,
                "throttled_requests": metrics.throttled_requests,
                "dropped_requests": metrics.dropped_requests,
                "queued_requests": metrics.queued_requests,
                "success_rate": metrics.allowed_requests / max(1, metrics.total_requests),
            }
        }
        
        # Add circuit breaker status
        if venue in self._circuit_breakers:
            circuit = self._circuit_breakers[venue]
            status["circuit_breaker"] = {
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "trips": metrics.circuit_trips,
            }
            
        # Add queue status
        if venue in self._message_queues:
            queue = self._message_queues[venue]
            status["queue"] = {
                "size": queue.qsize(),
                "max_size": queue.max_size,
                "overflow_count": queue.overflow_count,
                "utilization": queue.qsize() / queue.max_size,
            }
            
        return status
        
    def reset_metrics(self, venue: Optional[Venue] = None):
        """Reset metrics for venue or all venues"""
        if venue:
            if venue in self._metrics:
                self._metrics[venue] = ThrottleMetrics(last_reset=datetime.now())
        else:
            for v in self._metrics:
                self._metrics[v] = ThrottleMetrics(last_reset=datetime.now())
                
    async def health_check(self) -> Dict[str, any]:
        """Perform health check on rate limiting system"""
        total_venues = len(self._configs)
        healthy_venues = 0
        open_circuits = 0
        
        for venue in self._configs:
            if venue in self._circuit_breakers:
                circuit = self._circuit_breakers[venue]
                if circuit.state == CircuitState.OPEN:
                    open_circuits += 1
                else:
                    healthy_venues += 1
            else:
                healthy_venues += 1
                
        global_tokens = int(self._global_limiter.available_tokens())
        
        return {
            "status": "healthy" if open_circuits == 0 else "degraded",
            "total_venues": total_venues,
            "healthy_venues": healthy_venues,
            "open_circuits": open_circuits,
            "global_rate_limiter": {
                "available_tokens": global_tokens,
                "capacity": self._global_limiter.capacity,
                "utilization": 1.0 - (global_tokens / self._global_limiter.capacity),
            }
        }


# Global rate limiter instance
rate_limiter = RateLimiter()
"""
Advanced API Rate Limiting System
=================================

Sophisticated rate limiting with multiple strategies and intelligent throttling.
"""

import time
import asyncio
import hashlib
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
from enum import Enum
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"          # Fixed time windows
    SLIDING_WINDOW = "sliding_window"      # Sliding time windows  
    TOKEN_BUCKET = "token_bucket"          # Token bucket algorithm
    ADAPTIVE = "adaptive"                  # Adaptive based on load


class UserTier(str, Enum):
    """User tiers with different rate limits."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 0  # Additional burst capacity
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    current_usage: int = 0


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies and user tiers."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Rate limits by user tier
        self.tier_limits = {
            UserTier.FREE: RateLimit(
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=2000,
                burst_limit=10
            ),
            UserTier.BASIC: RateLimit(
                requests_per_minute=100,
                requests_per_hour=2000,
                requests_per_day=10000,
                burst_limit=25
            ),
            UserTier.PREMIUM: RateLimit(
                requests_per_minute=300,
                requests_per_hour=8000,
                requests_per_day=50000,
                burst_limit=50
            ),
            UserTier.ENTERPRISE: RateLimit(
                requests_per_minute=1000,
                requests_per_hour=25000,
                requests_per_day=200000,
                burst_limit=100
            )
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/historical/backfill": RateLimit(
                requests_per_minute=5,
                requests_per_hour=20,
                requests_per_day=100
            ),
            "/api/v1/ib/orders": RateLimit(
                requests_per_minute=60,
                requests_per_hour=500,
                requests_per_day=2000
            ),
            "/health": RateLimit(
                requests_per_minute=120,
                requests_per_hour=1000,
                requests_per_day=10000
            ),
            "/api/v1/nautilus/strategies/live": RateLimit(
                requests_per_minute=200,
                requests_per_hour=2000,
                requests_per_day=20000
            ),
            "/api/v1/nautilus/data/quality/refresh": RateLimit(
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=5000
            ),
            "/api/v1/performance/aggregate": RateLimit(
                requests_per_minute=150,
                requests_per_hour=1500,
                requests_per_day=10000
            )
        }
        
        # System-wide emergency limits
        self.emergency_limits = RateLimit(
            requests_per_minute=50,
            requests_per_hour=500,
            requests_per_day=2000
        )
        
        # Performance metrics
        self.total_requests = 0
        self.blocked_requests = 0
        self.emergency_activations = 0
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Advanced rate limiter connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for rate limiting: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Advanced rate limiter disconnected from Redis")
    
    def _get_rate_limit_key(self, identifier: str, window: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{identifier}:{window}"
    
    async def _check_sliding_window(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        if not self.redis_client:
            return RateLimitResult(allowed=True, remaining=limit, reset_time=0)
        
        current_time = time.time()
        window_start = current_time - window_seconds
        key = self._get_rate_limit_key(identifier, f"sliding_{window_seconds}")
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        current_count = results[1] + 1  # +1 for the current request
        
        allowed = current_count <= limit
        remaining = max(0, limit - current_count)
        reset_time = current_time + window_seconds
        
        if not allowed:
            # Remove the request we just added since it's not allowed
            await self.redis_client.zrem(key, str(current_time))
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            current_usage=current_count,
            retry_after=int(window_seconds) if not allowed else None
        )
    
    async def _check_token_bucket(
        self, 
        identifier: str, 
        rate: int, 
        capacity: int,
        window_seconds: int = 60
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        if not self.redis_client:
            return RateLimitResult(allowed=True, remaining=capacity, reset_time=0)
        
        current_time = time.time()
        key = self._get_rate_limit_key(identifier, "token_bucket")
        
        # Get current bucket state
        bucket_data = await self.redis_client.hmget(key, "tokens", "last_refill")
        
        current_tokens = float(bucket_data[0] or capacity)
        last_refill = float(bucket_data[1] or current_time)
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = (time_elapsed / window_seconds) * rate
        current_tokens = min(capacity, current_tokens + tokens_to_add)
        
        # Check if request is allowed
        allowed = current_tokens >= 1
        
        if allowed:
            current_tokens -= 1
        
        # Update bucket state
        await self.redis_client.hmset(key, {
            "tokens": current_tokens,
            "last_refill": current_time
        })
        await self.redis_client.expire(key, window_seconds * 2)
        
        return RateLimitResult(
            allowed=allowed,
            remaining=int(current_tokens),
            reset_time=current_time + (1 - current_tokens) / rate * window_seconds,
            current_usage=capacity - int(current_tokens),
            retry_after=int((1 - current_tokens) / rate * window_seconds) if not allowed else None
        )
    
    async def check_rate_limit(
        self, 
        request: Request,
        user_tier: UserTier = UserTier.FREE,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """Check if request is within rate limits."""
        self.total_requests += 1
        
        # Get client identifier
        if user_id:
            identifier = f"user:{user_id}"
        else:
            # Use IP address as fallback
            client_ip = request.client.host if request.client else "unknown"
            identifier = f"ip:{client_ip}"
        
        # Get applicable rate limits
        user_limits = self.tier_limits[user_tier]
        endpoint = request.url.path
        endpoint_limits = self.endpoint_limits.get(endpoint)
        
        # Check multiple time windows
        checks = [
            ("minute", user_limits.requests_per_minute, 60),
            ("hour", user_limits.requests_per_hour, 3600),
            ("day", user_limits.requests_per_day, 86400)
        ]
        
        # Add endpoint-specific checks if applicable
        if endpoint_limits:
            checks.extend([
                ("endpoint_minute", endpoint_limits.requests_per_minute, 60),
                ("endpoint_hour", endpoint_limits.requests_per_hour, 3600),
                ("endpoint_day", endpoint_limits.requests_per_day, 86400)
            ])
        
        # Check each time window
        for window_name, limit, window_seconds in checks:
            if user_limits.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = await self._check_token_bucket(
                    f"{identifier}:{window_name}",
                    limit,
                    limit + user_limits.burst_limit,
                    window_seconds
                )
            else:
                result = await self._check_sliding_window(
                    f"{identifier}:{window_name}",
                    limit,
                    window_seconds
                )
            
            if not result.allowed:
                self.blocked_requests += 1
                return result
        
        # All checks passed
        return RateLimitResult(allowed=True, remaining=user_limits.requests_per_minute, reset_time=0)
    
    async def check_system_load(self) -> bool:
        """Check if system is under high load and emergency limits should apply."""
        if not self.redis_client:
            return False
        
        try:
            # Check recent request rate
            current_time = time.time()
            window_start = current_time - 60  # Last minute
            
            # Count requests in last minute across all clients
            pattern = "rate_limit:*:sliding_60"
            keys = await self.redis_client.keys(pattern)
            
            total_requests = 0
            for key in keys[:50]:  # Limit to avoid performance issues
                count = await self.redis_client.zcard(key)
                total_requests += count
            
            # Emergency threshold: more than 5000 requests per minute system-wide
            if total_requests > 5000:
                self.emergency_activations += 1
                logger.warning(f"Emergency rate limiting activated: {total_requests} RPM")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking system load: {e}")
            return False
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        blocked_percentage = (
            (self.blocked_requests / self.total_requests * 100) 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "blocked_percentage": round(blocked_percentage, 2),
            "emergency_activations": self.emergency_activations,
            "tier_configurations": {
                tier.value: {
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_hour": limits.requests_per_hour,
                    "requests_per_day": limits.requests_per_day,
                    "burst_limit": limits.burst_limit,
                    "strategy": limits.strategy.value
                }
                for tier, limits in self.tier_limits.items()
            },
            "endpoint_limits": {
                endpoint: {
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_hour": limits.requests_per_hour,
                    "requests_per_day": limits.requests_per_day
                }
                for endpoint, limits in self.endpoint_limits.items()
            }
        }


# Dependency for FastAPI endpoints
async def rate_limit_dependency(
    request: Request,
    user_tier: UserTier = UserTier.FREE,
    user_id: Optional[str] = None
):
    """FastAPI dependency for rate limiting."""
    result = await advanced_rate_limiter.check_rate_limit(request, user_tier, user_id)
    
    if not result.allowed:
        headers = {
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_time)),
        }
        
        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "remaining": result.remaining,
                "reset_time": result.reset_time,
                "retry_after": result.retry_after
            },
            headers=headers
        )
    
    return result


# Rate limiting middleware
async def rate_limit_middleware(request: Request, call_next):
    """Middleware for automatic rate limiting."""
    
    # Skip rate limiting for health checks and static files
    if request.url.path.startswith(("/health", "/docs", "/openapi.json")):
        return await call_next(request)
    
    # Check system load for emergency limiting
    emergency_mode = await advanced_rate_limiter.check_system_load()
    user_tier = UserTier.FREE if emergency_mode else UserTier.BASIC
    
    try:
        result = await advanced_rate_limiter.check_rate_limit(request, user_tier)
        
        if not result.allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "remaining": result.remaining,
                    "reset_time": result.reset_time,
                    "retry_after": result.retry_after
                },
                headers={
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(int(result.reset_time)),
                    "Retry-After": str(result.retry_after) if result.retry_after else "60"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_time))
        
        return response
        
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        # Continue without rate limiting if there's an error
        return await call_next(request)


# Global rate limiter instance
advanced_rate_limiter = AdvancedRateLimiter()
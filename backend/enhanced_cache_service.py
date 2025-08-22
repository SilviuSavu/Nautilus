"""
Enhanced Redis Caching Service
==============================

High-performance caching layer for hot data and frequently accessed endpoints.
"""

import json
import asyncio
import hashlib
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies for different data types."""
    HOT_DATA = "hot"          # Frequently accessed data (5 min TTL)
    WARM_DATA = "warm"        # Moderately accessed data (15 min TTL)
    COLD_DATA = "cold"        # Rarely accessed data (1 hour TTL)
    SESSION_DATA = "session"  # Session data (30 min TTL)
    STATIC_DATA = "static"    # Static reference data (24 hour TTL)


@dataclass
class CacheConfig:
    """Cache configuration for different strategies."""
    ttl_seconds: int
    max_size_mb: int
    compression: bool = False


class EnhancedCacheService:
    """Enhanced Redis caching service with intelligent caching strategies."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Cache configurations by strategy
        self.cache_configs = {
            CacheStrategy.HOT_DATA: CacheConfig(ttl_seconds=300, max_size_mb=10),      # 5 min
            CacheStrategy.WARM_DATA: CacheConfig(ttl_seconds=900, max_size_mb=20),     # 15 min
            CacheStrategy.COLD_DATA: CacheConfig(ttl_seconds=3600, max_size_mb=50),    # 1 hour
            CacheStrategy.SESSION_DATA: CacheConfig(ttl_seconds=1800, max_size_mb=5),  # 30 min
            CacheStrategy.STATIC_DATA: CacheConfig(ttl_seconds=86400, max_size_mb=100) # 24 hours
        }
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Enhanced cache service connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Enhanced cache service disconnected from Redis")
    
    def _generate_cache_key(self, prefix: str, identifier: str, params: Dict = None) -> str:
        """Generate cache key with optional parameters hash."""
        if params:
            params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
            return f"cache:{prefix}:{identifier}:{params_hash}"
        return f"cache:{prefix}:{identifier}"
    
    async def get(self, key: str, strategy: CacheStrategy = CacheStrategy.WARM_DATA) -> Optional[Any]:
        """Get data from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(key)
            if cached_data:
                self.cache_hits += 1
                return json.loads(cached_data)
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            self.cache_errors += 1
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, data: Any, strategy: CacheStrategy = CacheStrategy.WARM_DATA) -> bool:
        """Set data in cache with TTL based on strategy."""
        if not self.redis_client:
            return False
        
        try:
            config = self.cache_configs[strategy]
            serialized_data = json.dumps(data, default=str)
            
            # Check data size (rough estimate)
            data_size_mb = len(serialized_data.encode()) / (1024 * 1024)
            if data_size_mb > config.max_size_mb:
                logger.warning(f"Data size {data_size_mb:.2f}MB exceeds limit {config.max_size_mb}MB for key {key}")
                return False
            
            await self.redis_client.setex(key, config.ttl_seconds, serialized_data)
            return True
            
        except Exception as e:
            self.cache_errors += 1
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete data from cache."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate pattern error for {pattern}: {e}")
            return 0
    
    async def get_or_set(
        self, 
        key: str, 
        data_fetcher: Callable,
        strategy: CacheStrategy = CacheStrategy.WARM_DATA,
        *args, 
        **kwargs
    ) -> Any:
        """Get data from cache or fetch and cache if not found."""
        cached_data = await self.get(key, strategy)
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        try:
            fresh_data = await data_fetcher(*args, **kwargs) if asyncio.iscoroutinefunction(data_fetcher) else data_fetcher(*args, **kwargs)
            if fresh_data is not None:
                await self.set(key, fresh_data, strategy)
            return fresh_data
        except Exception as e:
            logger.error(f"Data fetcher error for key {key}: {e}")
            return None
    
    async def warm_cache(self, keys_and_fetchers: List[tuple]) -> Dict[str, bool]:
        """Warm cache with multiple keys concurrently."""
        results = {}
        
        async def warm_single(key: str, fetcher: Callable, strategy: CacheStrategy):
            try:
                data = await fetcher() if asyncio.iscoroutinefunction(fetcher) else fetcher()
                success = await self.set(key, data, strategy)
                results[key] = success
            except Exception as e:
                logger.error(f"Cache warm error for key {key}: {e}")
                results[key] = False
        
        tasks = [
            warm_single(key, fetcher, strategy)
            for key, fetcher, strategy in keys_and_fetchers
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        info = {}
        if self.redis_client:
            try:
                info = await self.redis_client.info()
            except:
                pass
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "hit_rate_percentage": round(hit_rate, 2),
            "total_requests": total_requests,
            "redis_info": {
                "memory_used": info.get("used_memory_human", "N/A"),
                "connections": info.get("connected_clients", 0),
                "version": info.get("redis_version", "N/A"),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            },
            "strategies": {
                strategy.value: {
                    "ttl_seconds": config.ttl_seconds,
                    "max_size_mb": config.max_size_mb
                }
                for strategy, config in self.cache_configs.items()
            }
        }


# Decorator for caching function results
def cache_result(
    key_prefix: str, 
    strategy: CacheStrategy = CacheStrategy.WARM_DATA,
    key_generator: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Simple key generation based on function name and args
                args_hash = hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()[:8]
                cache_key = f"cache:{key_prefix}:{func.__name__}:{args_hash}"
            
            return await enhanced_cache.get_or_set(
                cache_key, 
                func,
                strategy,
                *args, 
                **kwargs
            )
        return wrapper
    return decorator


# Global enhanced cache instance
enhanced_cache = EnhancedCacheService()
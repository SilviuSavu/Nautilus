# Redis Optimization Layer for Performance Enhancement
# Implements intelligent caching strategies for query results and engine responses
# Part of the 3-4x performance improvement initiative

import redis.asyncio as redis
import json
import msgpack
import lz4.frame
import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategy options for different data types"""
    NO_CACHE = "no_cache"
    SHORT_TERM = "short_term"      # 30 seconds
    MEDIUM_TERM = "medium_term"    # 5 minutes
    LONG_TERM = "long_term"        # 30 minutes
    PERSISTENT = "persistent"      # 24 hours
    HYBRID = "hybrid"              # Intelligent TTL based on data characteristics

class CompressionType(Enum):
    """Compression options for cached data"""
    NONE = "none"
    LZ4 = "lz4"
    MSGPACK = "msgpack"
    MSGPACK_LZ4 = "msgpack_lz4"

@dataclass
class CacheMetrics:
    """Cache operation metrics"""
    cache_key: str
    operation: str  # get, set, delete
    hit: bool
    size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    operation_time_ms: Optional[float] = None
    ttl_seconds: Optional[int] = None

@dataclass
class CacheConfig:
    """Redis cache configuration"""
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    max_connections: int = 20
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = None
    default_ttl: int = 300  # 5 minutes
    max_memory_usage: str = "2gb"
    eviction_policy: str = "allkeys-lru"
    compression_threshold: int = 1024  # Compress data > 1KB
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 5   # TCP_KEEPCNT
            }

class RedisOptimizationLayer:
    """
    Advanced Redis caching layer optimized for trading platform performance
    
    Features:
    - Intelligent cache strategies based on data characteristics
    - Binary compression for large datasets
    - Connection pooling for high throughput
    - Automatic cache warming and invalidation
    - Real-time performance metrics
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._redis_client: Optional[redis.Redis] = None
        self._metrics: List[CacheMetrics] = []
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_operations = 0
        
        # Cache strategy TTL mapping
        self._strategy_ttl_map = {
            CacheStrategy.SHORT_TERM: 30,
            CacheStrategy.MEDIUM_TERM: 300,
            CacheStrategy.LONG_TERM: 1800,
            CacheStrategy.PERSISTENT: 86400,
            CacheStrategy.HYBRID: self._calculate_hybrid_ttl
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool"""
        try:
            self._redis_pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                max_connections=self.config.max_connections,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                decode_responses=False  # Handle binary data
            )
            
            self._redis_client = redis.Redis(
                connection_pool=self._redis_pool,
                decode_responses=False
            )
            
            # Configure Redis memory settings
            await self._configure_redis()
            
            logger.info(f"Redis optimization layer initialized: {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _configure_redis(self) -> None:
        """Configure Redis for optimal performance"""
        try:
            # Set memory usage limit
            await self._redis_client.config_set('maxmemory', self.config.max_memory_usage)
            
            # Set eviction policy
            await self._redis_client.config_set('maxmemory-policy', self.config.eviction_policy)
            
            # Optimize for performance
            await self._redis_client.config_set('tcp-keepalive', '60')
            await self._redis_client.config_set('timeout', '0')
            
            logger.info("Redis configuration optimized for trading platform")
            
        except Exception as e:
            logger.warning(f"Failed to configure Redis: {e}")
    
    def _generate_cache_key(self, namespace: str, identifier: str, params: Dict[str, Any] = None) -> str:
        """Generate consistent cache key"""
        key_parts = [namespace, identifier]
        
        if params:
            # Create deterministic hash from parameters
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(param_hash)
        
        return ":".join(key_parts)
    
    def _calculate_hybrid_ttl(self, data: Any, data_size: int) -> int:
        """Calculate intelligent TTL based on data characteristics"""
        base_ttl = self.config.default_ttl
        
        # Adjust TTL based on data size
        if data_size < 1024:  # Small data (< 1KB)
            return base_ttl // 2  # 2.5 minutes
        elif data_size < 10240:  # Medium data (< 10KB)
            return base_ttl  # 5 minutes
        else:  # Large data (> 10KB)
            return base_ttl * 2  # 10 minutes
        
        # Could add more intelligence based on:
        # - Data volatility (market data vs reference data)
        # - Update frequency
        # - Access patterns
        # - Time of day (trading hours vs off-hours)
    
    def _compress_data(self, data: Any, compression: CompressionType) -> Tuple[bytes, float]:
        """Compress data with specified method"""
        if compression == CompressionType.NONE:
            serialized = json.dumps(data).encode()
            return serialized, 1.0
        
        elif compression == CompressionType.LZ4:
            serialized = json.dumps(data).encode()
            compressed = lz4.frame.compress(serialized)
            ratio = len(serialized) / len(compressed)
            return compressed, ratio
        
        elif compression == CompressionType.MSGPACK:
            serialized = msgpack.packb(data)
            ratio = len(json.dumps(data).encode()) / len(serialized)
            return serialized, ratio
        
        elif compression == CompressionType.MSGPACK_LZ4:
            serialized = msgpack.packb(data)
            compressed = lz4.frame.compress(serialized)
            original_size = len(json.dumps(data).encode())
            ratio = original_size / len(compressed)
            return compressed, ratio
        
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def _decompress_data(self, data: bytes, compression: CompressionType) -> Any:
        """Decompress data with specified method"""
        if compression == CompressionType.NONE:
            return json.loads(data.decode())
        
        elif compression == CompressionType.LZ4:
            decompressed = lz4.frame.decompress(data)
            return json.loads(decompressed.decode())
        
        elif compression == CompressionType.MSGPACK:
            return msgpack.unpackb(data, raw=False)
        
        elif compression == CompressionType.MSGPACK_LZ4:
            decompressed = lz4.frame.decompress(data)
            return msgpack.unpackb(decompressed, raw=False)
        
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def _select_compression(self, data_size: int) -> CompressionType:
        """Select optimal compression based on data size"""
        if data_size < self.config.compression_threshold:
            return CompressionType.NONE
        elif data_size < 10240:  # < 10KB
            return CompressionType.MSGPACK
        else:  # >= 10KB
            return CompressionType.MSGPACK_LZ4
    
    async def get(
        self,
        namespace: str,
        identifier: str,
        params: Dict[str, Any] = None
    ) -> Tuple[Optional[Any], CacheMetrics]:
        """
        Get data from cache with metrics
        
        Args:
            namespace: Cache namespace (e.g., "market_data", "risk_calc")
            identifier: Data identifier (e.g., symbol, calculation_id)
            params: Additional parameters for cache key generation
        
        Returns:
            Tuple of (cached_data, cache_metrics)
        """
        start_time = asyncio.get_event_loop().time()
        cache_key = self._generate_cache_key(namespace, identifier, params)
        
        try:
            # Get data and metadata from Redis
            pipe = self._redis_client.pipeline()
            pipe.get(cache_key)
            pipe.get(f"{cache_key}:meta")
            
            cached_data, meta_data = await pipe.execute()
            
            if cached_data is None:
                # Cache miss
                self._cache_misses += 1
                self._total_operations += 1
                
                metrics = CacheMetrics(
                    cache_key=cache_key,
                    operation="get",
                    hit=False,
                    operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
                )
                self._metrics.append(metrics)
                
                return None, metrics
            
            # Cache hit - deserialize data
            meta = json.loads(meta_data.decode()) if meta_data else {}
            compression = CompressionType(meta.get('compression', 'none'))
            
            deserialized_data = self._decompress_data(cached_data, compression)
            
            self._cache_hits += 1
            self._total_operations += 1
            
            metrics = CacheMetrics(
                cache_key=cache_key,
                operation="get",
                hit=True,
                size_bytes=len(cached_data),
                compression_ratio=meta.get('compression_ratio'),
                operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                ttl_seconds=await self._redis_client.ttl(cache_key)
            )
            self._metrics.append(metrics)
            
            return deserialized_data, metrics
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            self._cache_misses += 1
            self._total_operations += 1
            
            metrics = CacheMetrics(
                cache_key=cache_key,
                operation="get",
                hit=False,
                operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
            
            return None, metrics
    
    async def set(
        self,
        namespace: str,
        identifier: str,
        data: Any,
        strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM,
        params: Dict[str, Any] = None
    ) -> CacheMetrics:
        """
        Set data in cache with intelligent compression and TTL
        
        Args:
            namespace: Cache namespace
            identifier: Data identifier
            data: Data to cache
            strategy: Caching strategy for TTL determination
            params: Additional parameters for cache key generation
        
        Returns:
            Cache operation metrics
        """
        start_time = asyncio.get_event_loop().time()
        cache_key = self._generate_cache_key(namespace, identifier, params)
        
        try:
            # Serialize data to determine size
            json_data = json.dumps(data).encode()
            data_size = len(json_data)
            
            # Select optimal compression
            compression = self._select_compression(data_size)
            
            # Compress data
            compressed_data, compression_ratio = self._compress_data(data, compression)
            
            # Calculate TTL
            if strategy == CacheStrategy.NO_CACHE:
                return CacheMetrics(
                    cache_key=cache_key,
                    operation="set",
                    hit=False,
                    operation_time_ms=0
                )
            
            if strategy == CacheStrategy.HYBRID:
                ttl = self._calculate_hybrid_ttl(data, data_size)
            else:
                ttl = self._strategy_ttl_map[strategy]
            
            # Store data and metadata
            meta = {
                'compression': compression.value,
                'compression_ratio': compression_ratio,
                'original_size': data_size,
                'compressed_size': len(compressed_data),
                'cached_at': datetime.utcnow().isoformat()
            }
            
            pipe = self._redis_client.pipeline()
            pipe.setex(cache_key, ttl, compressed_data)
            pipe.setex(f"{cache_key}:meta", ttl, json.dumps(meta))
            
            await pipe.execute()
            
            self._total_operations += 1
            
            metrics = CacheMetrics(
                cache_key=cache_key,
                operation="set",
                hit=True,
                size_bytes=len(compressed_data),
                compression_ratio=compression_ratio,
                operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                ttl_seconds=ttl
            )
            self._metrics.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            self._total_operations += 1
            
            return CacheMetrics(
                cache_key=cache_key,
                operation="set",
                hit=False,
                operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def delete(self, namespace: str, identifier: str, params: Dict[str, Any] = None) -> CacheMetrics:
        """Delete data from cache"""
        start_time = asyncio.get_event_loop().time()
        cache_key = self._generate_cache_key(namespace, identifier, params)
        
        try:
            pipe = self._redis_client.pipeline()
            pipe.delete(cache_key)
            pipe.delete(f"{cache_key}:meta")
            
            deleted_count = sum(await pipe.execute())
            
            metrics = CacheMetrics(
                cache_key=cache_key,
                operation="delete",
                hit=deleted_count > 0,
                operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
            self._metrics.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {e}")
            
            return CacheMetrics(
                cache_key=cache_key,
                operation="delete",
                hit=False,
                operation_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def warm_cache(
        self,
        warm_data: List[Tuple[str, str, Any, CacheStrategy]]
    ) -> List[CacheMetrics]:
        """
        Warm cache with frequently accessed data
        
        Args:
            warm_data: List of (namespace, identifier, data, strategy) tuples
        
        Returns:
            List of cache metrics for each operation
        """
        metrics = []
        
        for namespace, identifier, data, strategy in warm_data:
            metric = await self.set(namespace, identifier, data, strategy)
            metrics.append(metric)
        
        logger.info(f"Cache warmed with {len(warm_data)} entries")
        return metrics
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            redis_info = await self._redis_client.info("memory")
            
            hit_rate = (self._cache_hits / self._total_operations * 100) if self._total_operations > 0 else 0
            
            # Calculate recent operation metrics
            recent_metrics = self._metrics[-100:] if len(self._metrics) > 100 else self._metrics
            avg_operation_time = sum(m.operation_time_ms or 0 for m in recent_metrics) / max(len(recent_metrics), 1)
            
            # Calculate compression efficiency
            compressed_metrics = [m for m in recent_metrics if m.compression_ratio]
            avg_compression_ratio = sum(m.compression_ratio for m in compressed_metrics) / max(len(compressed_metrics), 1)
            
            return {
                'cache_performance': {
                    'hit_rate_percent': round(hit_rate, 2),
                    'total_hits': self._cache_hits,
                    'total_misses': self._cache_misses,
                    'total_operations': self._total_operations,
                    'avg_operation_time_ms': round(avg_operation_time, 2)
                },
                'compression_stats': {
                    'avg_compression_ratio': round(avg_compression_ratio, 2),
                    'compressed_operations': len(compressed_metrics)
                },
                'redis_memory': {
                    'used_memory_human': redis_info.get('used_memory_human'),
                    'used_memory_peak_human': redis_info.get('used_memory_peak_human'),
                    'memory_fragmentation_ratio': redis_info.get('mem_fragmentation_ratio')
                },
                'connection_info': {
                    'host': self.config.redis_host,
                    'port': self.config.redis_port,
                    'max_connections': self.config.max_connections,
                    'eviction_policy': self.config.eviction_policy
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                'error': str(e),
                'basic_stats': {
                    'hit_rate_percent': round((self._cache_hits / max(self._total_operations, 1)) * 100, 2),
                    'total_operations': self._total_operations
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health"""
        try:
            start_time = asyncio.get_event_loop().time()
            await self._redis_client.ping()
            ping_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'ping_time_ms': round(ping_time, 2),
                'connected': True,
                'pool_created_connections': getattr(self._redis_pool, 'created_connections', 0),
                'pool_available_connections': getattr(self._redis_pool, 'available_connections', 0)
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        pattern = f"{namespace}:*"
        
        try:
            keys = await self._redis_client.keys(pattern)
            
            if keys:
                deleted = await self._redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys from namespace '{namespace}'")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear namespace {namespace}: {e}")
            return 0
    
    async def close(self) -> None:
        """Close Redis connections"""
        try:
            if self._redis_client:
                await self._redis_client.close()
            
            if self._redis_pool:
                await self._redis_pool.disconnect()
            
            logger.info("Redis optimization layer closed")
        
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

# Global Redis optimization layer instance
_redis_cache: Optional[RedisOptimizationLayer] = None

async def get_redis_cache() -> RedisOptimizationLayer:
    """Get or create Redis optimization layer instance"""
    global _redis_cache
    
    if _redis_cache is None:
        config = CacheConfig()
        _redis_cache = RedisOptimizationLayer(config)
        await _redis_cache.initialize()
    
    return _redis_cache

async def close_redis_cache() -> None:
    """Close global Redis cache instance"""
    global _redis_cache
    
    if _redis_cache:
        await _redis_cache.close()
        _redis_cache = None
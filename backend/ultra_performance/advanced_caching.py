"""
Advanced Caching Strategies for Ultra-Performance Trading

Provides:
- Intelligent cache warming strategies
- Distributed caching across nodes
- Predictive cache preloading
- Cache coherency management
- Multi-level cache hierarchies
- Adaptive cache replacement policies
"""

import asyncio
import logging
import pickle
import time
import threading
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict
from enum import Enum
import weakref
import gc
import struct

# Redis and distributed caching
try:
    import redis
    import redis.sentinel
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Advanced serialization
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None
    MSGPACK_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    lz4 = None
    LZ4_AVAILABLE = False

# Machine learning for prediction
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    np = None
    LinearRegression = StandardScaler = None
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_CPU = "l1_cpu"           # CPU L1 cache-friendly
    L2_MEMORY = "l2_memory"     # In-memory cache
    L3_REDIS = "l3_redis"       # Redis distributed cache
    L4_DISK = "l4_disk"         # Persistent disk cache

class CachePolicy(Enum):
    """Cache replacement policies"""
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used
    ARC = "arc"                 # Adaptive Replacement Cache
    CLOCK = "clock"             # Clock algorithm
    PREDICTIVE = "predictive"   # ML-based predictive eviction

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_requests: int = 0
    avg_access_time_us: float = 0.0
    hit_ratio: float = 0.0
    memory_usage_bytes: int = 0
    prediction_accuracy: float = 0.0

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    access_count: int = 0
    last_access_time: float = 0.0
    creation_time: float = 0.0
    expiry_time: Optional[float] = None
    size_bytes: int = 0
    prediction_score: float = 0.0
    access_pattern: List[float] = None

    def __post_init__(self):
        if self.access_pattern is None:
            self.access_pattern = []

class IntelligentCacheWarmer:
    """
    Intelligent cache warming based on historical access patterns and predictions
    """
    
    def __init__(self):
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.warming_patterns: Dict[str, Dict] = {}
        self.ml_predictors: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    def record_access(self, key: str, timestamp: Optional[float] = None):
        """Record cache key access for pattern learning"""
        timestamp = timestamp or time.time()
        
        with self._lock:
            self.access_history[key].append(timestamp)
            
            # Keep only recent history (last 1000 accesses)
            if len(self.access_history[key]) > 1000:
                self.access_history[key] = self.access_history[key][-1000:]
                
    def analyze_access_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze historical access patterns to identify warming opportunities"""
        patterns = {}
        
        with self._lock:
            for key, timestamps in self.access_history.items():
                if len(timestamps) < 10:  # Need minimum data
                    continue
                    
                # Calculate access frequency
                time_span = timestamps[-1] - timestamps[0]
                frequency = len(timestamps) / max(time_span, 1.0)
                
                # Calculate time intervals between accesses
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                avg_interval = sum(intervals) / len(intervals) if intervals else float('inf')
                
                # Detect periodic patterns
                is_periodic = self._detect_periodicity(intervals)
                
                # Calculate access regularity score
                interval_variance = np.var(intervals) if ML_AVAILABLE and intervals else 0
                regularity_score = 1.0 / (1.0 + interval_variance) if interval_variance > 0 else 1.0
                
                patterns[key] = {
                    "frequency": frequency,
                    "avg_interval": avg_interval,
                    "is_periodic": is_periodic,
                    "regularity_score": regularity_score,
                    "total_accesses": len(timestamps),
                    "warming_priority": frequency * regularity_score
                }
                
        return patterns
        
    def _detect_periodicity(self, intervals: List[float]) -> bool:
        """Detect if access pattern is periodic"""
        if len(intervals) < 5 or not ML_AVAILABLE:
            return False
            
        # Simple periodicity detection using autocorrelation
        intervals_arr = np.array(intervals)
        mean_interval = np.mean(intervals_arr)
        
        # Check if intervals cluster around certain values
        unique_intervals = np.unique(np.round(intervals_arr))
        return len(unique_intervals) <= len(intervals) * 0.3
        
    async def warm_cache_intelligent(
        self,
        cache_manager: 'DistributedCacheManager',
        warm_function: Callable[[str], Any],
        max_warming_keys: int = 100
    ) -> Dict[str, Any]:
        """Intelligently warm cache based on access patterns"""
        patterns = self.analyze_access_patterns()
        
        # Sort keys by warming priority
        warming_candidates = sorted(
            patterns.items(),
            key=lambda x: x[1]["warming_priority"],
            reverse=True
        )[:max_warming_keys]
        
        warming_results = {
            "total_candidates": len(patterns),
            "warmed_keys": 0,
            "failed_keys": 0,
            "warming_time_ms": 0
        }
        
        start_time = time.time()
        
        # Warm cache with highest priority keys
        for key, pattern_info in warming_candidates:
            try:
                # Check if key is already cached
                if not await cache_manager.exists(key):
                    value = await warm_function(key) if asyncio.iscoroutinefunction(warm_function) else warm_function(key)
                    await cache_manager.set(key, value, ttl=3600)  # 1 hour TTL
                    warming_results["warmed_keys"] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to warm cache for key {key}: {e}")
                warming_results["failed_keys"] += 1
                
        warming_results["warming_time_ms"] = (time.time() - start_time) * 1000
        return warming_results
        
    def train_predictive_model(self, cache_key_pattern: str) -> bool:
        """Train ML model to predict cache access patterns"""
        if not ML_AVAILABLE:
            return False
            
        try:
            # Collect training data from access history
            training_data = []
            for key, timestamps in self.access_history.items():
                if cache_key_pattern in key and len(timestamps) >= 20:
                    training_data.extend(timestamps)
                    
            if len(training_data) < 50:
                return False
                
            # Prepare features (time-based patterns)
            X, y = self._prepare_prediction_features(training_data)
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            self.ml_predictors[cache_key_pattern] = {
                "model": model,
                "scaler": scaler,
                "trained_at": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train predictive model for {cache_key_pattern}: {e}")
            return False
            
    def _prepare_prediction_features(self, timestamps: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model training"""
        features = []
        targets = []
        
        for i in range(5, len(timestamps)):  # Use 5 previous accesses as features
            # Features: time intervals, frequency, time of day patterns
            recent_intervals = [timestamps[j] - timestamps[j-1] for j in range(i-4, i+1)]
            hour_of_day = time.localtime(timestamps[i]).tm_hour
            day_of_week = time.localtime(timestamps[i]).tm_wday
            
            feature_vector = recent_intervals + [hour_of_day, day_of_week]
            features.append(feature_vector)
            
            # Target: time until next access
            if i < len(timestamps) - 1:
                targets.append(timestamps[i+1] - timestamps[i])
            else:
                targets.append(0)  # No next access
                
        return np.array(features), np.array(targets)

class DistributedCacheManager:
    """
    Distributed caching manager with multi-level cache hierarchy
    """
    
    def __init__(self, redis_config: Optional[Dict] = None):
        self.redis_config = redis_config or {}
        self.redis_client: Optional[redis.Redis] = None
        self.local_caches: Dict[CacheLevel, Any] = {}
        self.cache_metrics: Dict[CacheLevel, CacheMetrics] = {}
        self.serialization_method = "msgpack" if MSGPACK_AVAILABLE else "pickle"
        self.compression_enabled = LZ4_AVAILABLE
        self._lock = threading.RLock()
        
        self._initialize_caches()
        
    def _initialize_caches(self):
        """Initialize multi-level cache hierarchy"""
        # L1: CPU cache-friendly (small, ultra-fast)
        self.local_caches[CacheLevel.L1_CPU] = LRUCache(maxsize=1000)
        
        # L2: In-memory cache (larger, fast)
        self.local_caches[CacheLevel.L2_MEMORY] = AdaptiveCache(maxsize=10000)
        
        # L3: Redis distributed cache
        if REDIS_AVAILABLE and self.redis_config:
            try:
                self.redis_client = redis.Redis(**self.redis_config)
                self.redis_client.ping()  # Test connection
                logger.info("Connected to Redis for L3 caching")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
                
        # Initialize metrics
        for level in CacheLevel:
            self.cache_metrics[level] = CacheMetrics()
            
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from multi-level cache hierarchy"""
        start_time = time.perf_counter()
        
        # L1 CPU cache check
        value = self._get_from_level(CacheLevel.L1_CPU, key)
        if value is not None:
            self._update_metrics(CacheLevel.L1_CPU, hit=True, access_time=time.perf_counter() - start_time)
            return value
            
        # L2 Memory cache check
        value = self._get_from_level(CacheLevel.L2_MEMORY, key)
        if value is not None:
            # Promote to L1
            await self._set_to_level(CacheLevel.L1_CPU, key, value)
            self._update_metrics(CacheLevel.L2_MEMORY, hit=True, access_time=time.perf_counter() - start_time)
            return value
            
        # L3 Redis cache check
        if self.redis_client:
            value = await self._get_from_redis(key)
            if value is not None:
                # Promote to upper levels
                await self._set_to_level(CacheLevel.L2_MEMORY, key, value)
                await self._set_to_level(CacheLevel.L1_CPU, key, value)
                self._update_metrics(CacheLevel.L3_REDIS, hit=True, access_time=time.perf_counter() - start_time)
                return value
                
        # Cache miss at all levels
        for level in [CacheLevel.L1_CPU, CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS]:
            self._update_metrics(level, hit=False, access_time=time.perf_counter() - start_time)
            
        return default
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in multi-level cache hierarchy"""
        try:
            # Set in all cache levels
            await self._set_to_level(CacheLevel.L1_CPU, key, value, ttl)
            await self._set_to_level(CacheLevel.L2_MEMORY, key, value, ttl)
            
            if self.redis_client:
                await self._set_to_redis(key, value, ttl)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        success = True
        
        # Delete from all levels
        for level in CacheLevel:
            try:
                if level == CacheLevel.L3_REDIS and self.redis_client:
                    await self.redis_client.delete(key)
                else:
                    cache = self.local_caches.get(level)
                    if cache:
                        cache.delete(key)
            except Exception as e:
                logger.warning(f"Failed to delete {key} from {level}: {e}")
                success = False
                
        return success
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level"""
        # Check L1 first (fastest)
        if self._get_from_level(CacheLevel.L1_CPU, key) is not None:
            return True
            
        # Check L2
        if self._get_from_level(CacheLevel.L2_MEMORY, key) is not None:
            return True
            
        # Check L3 Redis
        if self.redis_client:
            try:
                return bool(await self.redis_client.exists(key))
            except Exception:
                pass
                
        return False
        
    def _get_from_level(self, level: CacheLevel, key: str) -> Any:
        """Get value from specific cache level"""
        try:
            cache = self.local_caches.get(level)
            return cache.get(key) if cache else None
        except Exception:
            return None
            
    async def _set_to_level(self, level: CacheLevel, key: str, value: Any, ttl: Optional[int] = None):
        """Set value to specific cache level"""
        try:
            cache = self.local_caches.get(level)
            if cache:
                cache.set(key, value, ttl=ttl)
        except Exception as e:
            logger.warning(f"Failed to set {key} in {level}: {e}")
            
    async def _get_from_redis(self, key: str) -> Any:
        """Get value from Redis with deserialization"""
        try:
            data = await self.redis_client.get(key)
            if data:
                return self._deserialize(data)
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
        return None
        
    async def _set_to_redis(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value to Redis with serialization"""
        try:
            serialized_data = self._serialize(value)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_data)
            else:
                await self.redis_client.set(key, serialized_data)
        except Exception as e:
            logger.warning(f"Redis set failed for {key}: {e}")
            
    def _serialize(self, value: Any) -> bytes:
        """Serialize value with optional compression"""
        if self.serialization_method == "msgpack" and MSGPACK_AVAILABLE:
            data = msgpack.packb(value)
        else:
            data = pickle.dumps(value)
            
        if self.compression_enabled and LZ4_AVAILABLE:
            data = lz4.frame.compress(data)
            
        return data
        
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value with optional decompression"""
        if self.compression_enabled and LZ4_AVAILABLE:
            try:
                data = lz4.frame.decompress(data)
            except Exception:
                pass  # Data might not be compressed
                
        if self.serialization_method == "msgpack" and MSGPACK_AVAILABLE:
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception:
                pass  # Fallback to pickle
                
        return pickle.loads(data)
        
    def _update_metrics(self, level: CacheLevel, hit: bool, access_time: float):
        """Update cache metrics"""
        metrics = self.cache_metrics[level]
        metrics.total_requests += 1
        
        if hit:
            metrics.hit_count += 1
        else:
            metrics.miss_count += 1
            
        # Update hit ratio
        metrics.hit_ratio = metrics.hit_count / metrics.total_requests
        
        # Update average access time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        metrics.avg_access_time_us = (
            (1 - alpha) * metrics.avg_access_time_us + 
            alpha * (access_time * 1_000_000)  # Convert to microseconds
        )
        
    def get_metrics(self) -> Dict[str, CacheMetrics]:
        """Get comprehensive cache metrics"""
        return {level.value: asdict(metrics) for level, metrics in self.cache_metrics.items()}

class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Any:
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used item
                self.cache.popitem(last=False)
                
            self.cache[key] = value
            
    def delete(self, key: str):
        with self._lock:
            self.cache.pop(key, None)

class AdaptiveCache:
    """Adaptive Replacement Cache (ARC) implementation"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache = {}
        self.frequency_counter = defaultdict(int)
        self.access_time = defaultdict(float)
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Any:
        with self._lock:
            if key in self.cache:
                self.frequency_counter[key] += 1
                self.access_time[key] = time.time()
                return self.cache[key]
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self._evict_adaptive()
                
            self.cache[key] = value
            self.frequency_counter[key] += 1
            self.access_time[key] = time.time()
            
    def _evict_adaptive(self):
        """Adaptive eviction based on frequency and recency"""
        current_time = time.time()
        
        # Calculate adaptive score for each item
        scores = {}
        for key in self.cache:
            frequency = self.frequency_counter[key]
            recency = current_time - self.access_time[key]
            # Lower score = higher priority for eviction
            scores[key] = frequency / (1 + recency)
            
        # Evict item with lowest score
        victim_key = min(scores, key=scores.get)
        del self.cache[victim_key]
        del self.frequency_counter[victim_key]
        del self.access_time[victim_key]
        
    def delete(self, key: str):
        with self._lock:
            self.cache.pop(key, None)
            self.frequency_counter.pop(key, None)
            self.access_time.pop(key, None)

class PredictiveCacheLoader:
    """
    Predictive cache loading based on usage patterns and market conditions
    """
    
    def __init__(self, cache_manager: DistributedCacheManager):
        self.cache_manager = cache_manager
        self.prediction_models: Dict[str, Any] = {}
        self.loading_queue = asyncio.Queue()
        self.is_loading = False
        
    async def start_predictive_loading(self):
        """Start background predictive cache loading"""
        self.is_loading = True
        asyncio.create_task(self._loading_loop())
        
    async def stop_predictive_loading(self):
        """Stop predictive cache loading"""
        self.is_loading = False
        
    async def _loading_loop(self):
        """Background loop for predictive cache loading"""
        while self.is_loading:
            try:
                # Get next prediction
                prediction = await asyncio.wait_for(
                    self.loading_queue.get(), timeout=1.0
                )
                
                # Load predicted cache entries
                await self._load_predicted_entries(prediction)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Predictive loading error: {e}")
                
    async def _load_predicted_entries(self, prediction: Dict[str, Any]):
        """Load cache entries based on predictions"""
        # Implementation would depend on specific prediction format
        # and data loading functions
        pass
        
    def predict_cache_needs(self, context: Dict[str, Any]) -> List[str]:
        """Predict which cache keys will be needed"""
        # Simple heuristic-based prediction
        # In production, this would use ML models
        
        predicted_keys = []
        
        # Market hours prediction
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 16:  # Market hours
            predicted_keys.extend([
                "market_data:SPY",
                "market_data:QQQ", 
                "market_data:VIX"
            ])
            
        # Volatility-based prediction
        if context.get("market_volatility", 0) > 0.02:
            predicted_keys.extend([
                "risk_metrics:var",
                "risk_metrics:expected_shortfall"
            ])
            
        return predicted_keys

class CacheCoherencyManager:
    """
    Manage cache coherency across distributed nodes
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.subscriptions: Set[str] = set()
        self.invalidation_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.redis_client: Optional[redis.Redis] = None
        
    async def subscribe_to_invalidations(self, cache_key_pattern: str, callback: Callable):
        """Subscribe to cache invalidation notifications"""
        self.subscriptions.add(cache_key_pattern)
        self.invalidation_callbacks[cache_key_pattern].append(callback)
        
    async def invalidate_key(self, key: str, broadcast: bool = True):
        """Invalidate cache key across all nodes"""
        if broadcast and self.redis_client:
            # Broadcast invalidation to other nodes
            invalidation_message = {
                "type": "invalidate",
                "key": key,
                "node_id": self.node_id,
                "timestamp": time.time()
            }
            
            await self.redis_client.publish(
                "cache_invalidations",
                json.dumps(invalidation_message)
            )
            
        # Execute local callbacks
        for pattern in self.subscriptions:
            if pattern in key:
                for callback in self.invalidation_callbacks[pattern]:
                    try:
                        await callback(key) if asyncio.iscoroutinefunction(callback) else callback(key)
                    except Exception as e:
                        logger.error(f"Invalidation callback failed: {e}")

class CacheOptimizer:
    """
    Cache optimization and auto-tuning based on performance metrics
    """
    
    def __init__(self, cache_manager: DistributedCacheManager):
        self.cache_manager = cache_manager
        self.optimization_history: List[Dict] = []
        
    async def optimize_cache_sizes(self) -> Dict[str, Any]:
        """Optimize cache sizes based on hit ratios and performance"""
        metrics = self.cache_manager.get_metrics()
        
        optimization_results = {
            "recommendations": {},
            "current_performance": metrics
        }
        
        for level_name, level_metrics in metrics.items():
            hit_ratio = level_metrics.get("hit_ratio", 0)
            avg_access_time = level_metrics.get("avg_access_time_us", 0)
            
            if hit_ratio < 0.8 and level_name in ["l1_cpu", "l2_memory"]:
                # Recommend increasing cache size
                optimization_results["recommendations"][level_name] = {
                    "action": "increase_size",
                    "current_hit_ratio": hit_ratio,
                    "target_hit_ratio": 0.9
                }
            elif hit_ratio > 0.95 and avg_access_time > 100:
                # Cache might be too large, reducing efficiency
                optimization_results["recommendations"][level_name] = {
                    "action": "decrease_size",
                    "current_hit_ratio": hit_ratio,
                    "avg_access_time_us": avg_access_time
                }
                
        return optimization_results

# Global instances
intelligent_cache_warmer = IntelligentCacheWarmer()
distributed_cache_manager = DistributedCacheManager()
predictive_cache_loader = PredictiveCacheLoader(distributed_cache_manager)
cache_coherency_manager = CacheCoherencyManager("node_1")
cache_optimizer = CacheOptimizer(distributed_cache_manager)

# Convenience functions
async def get_cached(key: str, default: Any = None) -> Any:
    """Convenience function for cache retrieval"""
    return await distributed_cache_manager.get(key, default)

async def set_cached(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Convenience function for cache storage"""
    return await distributed_cache_manager.set(key, value, ttl)

def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{key_prefix}{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = await distributed_cache_manager.get(cache_key)
            if result is not None:
                return result
                
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await distributed_cache_manager.set(cache_key, result, ttl)
            
            return result
            
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, run in event loop
            return asyncio.run(async_wrapper(*args, **kwargs))
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
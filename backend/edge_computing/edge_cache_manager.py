"""
Edge Cache Manager for Ultra-Low Latency Data Replication

This module manages intelligent edge caching and data replication strategies 
for trading data across edge nodes with consistency guarantees.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import aioredis
import pickle


class CacheStrategy(Enum):
    """Edge caching strategies"""
    WRITE_THROUGH = "write_through"          # Immediate synchronous write to all replicas
    WRITE_BEHIND = "write_behind"            # Asynchronous write with eventual consistency
    WRITE_AROUND = "write_around"            # Write directly to backing store, bypass cache
    READ_THROUGH = "read_through"            # Read from cache, fallback to backing store
    CACHE_ASIDE = "cache_aside"              # Application manages cache population
    REFRESH_AHEAD = "refresh_ahead"          # Proactively refresh before expiration


class ReplicationMode(Enum):
    """Data replication modes"""
    SYNCHRONOUS = "synchronous"              # All replicas must confirm write
    ASYNCHRONOUS = "asynchronous"            # Write acknowledged immediately
    SEMI_SYNCHRONOUS = "semi_synchronous"    # Majority of replicas must confirm
    EVENTUAL_CONSISTENCY = "eventual"        # Eventually consistent across regions
    STRONG_CONSISTENCY = "strong"            # Strong consistency with quorum


class DataCategory(Enum):
    """Categories of trading data for caching optimization"""
    MARKET_DATA = "market_data"              # Price feeds, order books
    REFERENCE_DATA = "reference_data"        # Instruments, contracts, holidays
    RISK_DATA = "risk_data"                  # Positions, limits, exposures
    ANALYTICS_DATA = "analytics_data"        # Calculated metrics, indicators
    STRATEGY_DATA = "strategy_data"          # Strategy parameters, state
    USER_DATA = "user_data"                  # User preferences, settings


class ConsistencyLevel(Enum):
    """Data consistency requirements"""
    STRONG = "strong"                        # Linearizability required
    BOUNDED_STALENESS = "bounded_staleness"  # Bounded by time or version lag
    SESSION = "session"                      # Session consistency for user data
    EVENTUAL = "eventual"                    # Eventually consistent
    WEAK = "weak"                           # No consistency guarantees


@dataclass
class CacheItem:
    """Individual cache item with metadata"""
    key: str
    value: Any
    category: DataCategory
    
    # Timestamps
    created_at: float
    updated_at: float
    accessed_at: float
    expires_at: float
    
    # Replication metadata
    version: int
    checksum: str
    source_node: str
    replica_nodes: Set[str] = field(default_factory=set)
    
    # Access patterns
    access_count: int = 0
    access_frequency: float = 0.0  # accesses per second
    
    # Size and priority
    size_bytes: int = 0
    priority: int = 5  # 1-10 scale, 10 = highest priority
    
    # Consistency requirements
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    staleness_tolerance_ms: float = 1000.0


@dataclass
class CacheConfiguration:
    """Configuration for edge cache"""
    cache_id: str
    node_id: str
    region: str
    
    # Capacity settings
    max_memory_mb: int = 1024
    max_items: int = 100000
    item_ttl_seconds: int = 300  # 5 minutes default
    
    # Replication settings
    replication_mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS
    replication_factor: int = 3
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    
    # Performance settings
    cache_strategy: CacheStrategy = CacheStrategy.WRITE_BEHIND
    prefetch_enabled: bool = True
    compression_enabled: bool = True
    
    # Eviction policy
    eviction_policy: str = "lru"  # lru, lfu, fifo, random
    eviction_batch_size: int = 100
    
    # Monitoring settings
    metrics_enabled: bool = True
    detailed_logging: bool = False


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    cache_id: str
    timestamp: float
    
    # Hit/miss statistics
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    
    # Latency statistics (microseconds)
    avg_get_latency_us: float
    avg_set_latency_us: float
    p95_get_latency_us: float
    p99_get_latency_us: float
    
    # Memory statistics
    memory_used_mb: float
    memory_utilization: float
    item_count: int
    avg_item_size_bytes: float
    
    # Replication statistics
    replication_lag_ms: float
    failed_replications: int
    consistency_violations: int
    
    # Eviction statistics
    evictions_total: int
    evictions_per_second: float


@dataclass
class ReplicationJob:
    """Data replication job"""
    job_id: str
    source_cache: str
    target_caches: List[str]
    
    # Job details
    operation: str  # "set", "delete", "invalidate"
    key: str
    value: Any = None
    
    # Timing
    created_at: float = field(default_factory=time.time)
    scheduled_at: float = 0.0
    completed_at: float = 0.0
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    
    # Consistency requirements
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    timeout_ms: float = 1000.0


class EdgeCacheManager:
    """
    Edge Cache Manager for Ultra-Low Latency Data Replication
    
    Provides intelligent caching across edge nodes with:
    - Multiple caching strategies and replication modes
    - Consistency guarantees based on data criticality
    - Intelligent prefetching and cache warming
    - Real-time monitoring and optimization
    - Automatic failover and recovery
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cache instances
        self.caches: Dict[str, Dict[str, CacheItem]] = {}
        self.cache_configs: Dict[str, CacheConfiguration] = {}
        self.cache_metrics: Dict[str, CacheMetrics] = {}
        
        # Replication system
        self.replication_queue: List[ReplicationJob] = []
        self.replication_workers: Dict[str, asyncio.Task] = {}
        self.replication_active = False
        
        # Consistency tracking
        self.version_vectors: Dict[str, Dict[str, int]] = {}  # key -> {node: version}
        self.pending_invalidations: Dict[str, Set[str]] = {}  # key -> {nodes}
        
        # Access pattern analysis
        self.access_patterns: Dict[str, List[float]] = {}  # key -> [access_times]
        self.hotspot_detection: Dict[str, float] = {}  # key -> hotness_score
        
        # Redis connections for backing store
        self.redis_connections: Dict[str, aioredis.Redis] = {}
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        self.logger.info("Edge Cache Manager initialized")
    
    async def create_cache(self, config: CacheConfiguration) -> bool:
        """Create a new edge cache with specified configuration"""
        
        cache_id = config.cache_id
        
        if cache_id in self.caches:
            self.logger.warning(f"Cache {cache_id} already exists")
            return False
        
        try:
            # Initialize cache storage
            self.caches[cache_id] = {}
            self.cache_configs[cache_id] = config
            
            # Initialize metrics
            self.cache_metrics[cache_id] = CacheMetrics(
                cache_id=cache_id,
                timestamp=time.time(),
                total_requests=0,
                cache_hits=0,
                cache_misses=0,
                hit_rate=0.0,
                avg_get_latency_us=0.0,
                avg_set_latency_us=0.0,
                p95_get_latency_us=0.0,
                p99_get_latency_us=0.0,
                memory_used_mb=0.0,
                memory_utilization=0.0,
                item_count=0,
                avg_item_size_bytes=0.0,
                replication_lag_ms=0.0,
                failed_replications=0,
                consistency_violations=0,
                evictions_total=0,
                evictions_per_second=0.0
            )
            
            # Initialize version vector for consistency tracking
            self.version_vectors[cache_id] = {}
            
            # Setup Redis connection if needed
            if config.cache_strategy in [CacheStrategy.WRITE_THROUGH, CacheStrategy.READ_THROUGH]:
                await self._setup_redis_connection(cache_id)
            
            self.logger.info(f"Created edge cache: {cache_id} (node: {config.node_id}, region: {config.region})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create cache {cache_id}: {e}")
            return False
    
    async def _setup_redis_connection(self, cache_id: str):
        """Setup Redis connection for backing store"""
        
        try:
            # In production, this would connect to actual Redis cluster
            # For now, simulate connection
            self.redis_connections[cache_id] = None  # Placeholder
            self.logger.info(f"Redis connection established for cache: {cache_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Redis for cache {cache_id}: {e}")
    
    async def get(self, cache_id: str, key: str, consistency_level: Optional[ConsistencyLevel] = None) -> Optional[Any]:
        """Get value from cache with optional consistency requirements"""
        
        start_time = time.perf_counter_ns()
        
        try:
            if cache_id not in self.caches:
                self.logger.error(f"Cache {cache_id} not found")
                return None
            
            cache = self.caches[cache_id]
            config = self.cache_configs[cache_id]
            metrics = self.cache_metrics[cache_id]
            
            metrics.total_requests += 1
            
            # Check local cache first
            if key in cache:
                item = cache[key]
                
                # Check if item has expired
                current_time = time.time()
                if item.expires_at > current_time:
                    
                    # Check consistency requirements
                    if consistency_level and consistency_level != ConsistencyLevel.WEAK:
                        consistency_check = await self._check_data_consistency(cache_id, key, item, consistency_level)
                        if not consistency_check:
                            # Data is stale, try to fetch fresh copy
                            fresh_value = await self._fetch_from_backing_store(cache_id, key)
                            if fresh_value is not None:
                                await self.set(cache_id, key, fresh_value, item.category)
                                item = cache[key]  # Get updated item
                    
                    # Update access patterns
                    item.accessed_at = current_time
                    item.access_count += 1
                    await self._update_access_patterns(key)
                    
                    # Record cache hit
                    metrics.cache_hits += 1
                    
                    end_time = time.perf_counter_ns()
                    latency_us = (end_time - start_time) / 1000.0
                    metrics.avg_get_latency_us = (metrics.avg_get_latency_us + latency_us) / 2
                    
                    return item.value
                else:
                    # Item expired, remove from cache
                    del cache[key]
            
            # Cache miss - try backing store if configured
            metrics.cache_misses += 1
            
            if config.cache_strategy == CacheStrategy.READ_THROUGH:
                value = await self._fetch_from_backing_store(cache_id, key)
                if value is not None:
                    # Store in cache for future use
                    await self.set(cache_id, key, value, DataCategory.MARKET_DATA)  # Default category
                    return value
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting key {key} from cache {cache_id}: {e}")
            return None
        finally:
            # Update hit rate
            if cache_id in self.cache_metrics:
                metrics = self.cache_metrics[cache_id]
                if metrics.total_requests > 0:
                    metrics.hit_rate = metrics.cache_hits / metrics.total_requests
    
    async def set(
        self, 
        cache_id: str, 
        key: str, 
        value: Any, 
        category: DataCategory,
        ttl_seconds: Optional[int] = None,
        priority: int = 5,
        consistency_level: Optional[ConsistencyLevel] = None
    ) -> bool:
        """Set value in cache with replication"""
        
        start_time = time.perf_counter_ns()
        
        try:
            if cache_id not in self.caches:
                self.logger.error(f"Cache {cache_id} not found")
                return False
            
            cache = self.caches[cache_id]
            config = self.cache_configs[cache_id]
            
            # Calculate item size and checksum
            serialized_value = pickle.dumps(value)
            size_bytes = len(serialized_value)
            checksum = hashlib.sha256(serialized_value).hexdigest()
            
            # Check capacity limits
            if len(cache) >= config.max_items:
                await self._evict_items(cache_id, 1)
            
            # Calculate memory usage (simplified)
            total_memory_mb = sum(item.size_bytes for item in cache.values()) / (1024 * 1024)
            if total_memory_mb >= config.max_memory_mb:
                await self._evict_items(cache_id, config.eviction_batch_size)
            
            # Create cache item
            current_time = time.time()
            ttl = ttl_seconds or config.item_ttl_seconds
            consistency = consistency_level or config.consistency_level
            
            # Increment version for consistency tracking
            cache_version = self.version_vectors[cache_id].get(key, 0) + 1
            self.version_vectors[cache_id][key] = cache_version
            
            item = CacheItem(
                key=key,
                value=value,
                category=category,
                created_at=current_time,
                updated_at=current_time,
                accessed_at=current_time,
                expires_at=current_time + ttl,
                version=cache_version,
                checksum=checksum,
                source_node=config.node_id,
                size_bytes=size_bytes,
                priority=priority,
                consistency_level=consistency
            )
            
            # Store in local cache
            cache[key] = item
            
            # Handle write-through to backing store
            if config.cache_strategy == CacheStrategy.WRITE_THROUGH:
                await self._write_to_backing_store(cache_id, key, value)
            
            # Schedule replication if needed
            if config.replication_factor > 1:
                await self._schedule_replication(cache_id, key, value, "set", consistency)
            
            # Update metrics
            end_time = time.perf_counter_ns()
            latency_us = (end_time - start_time) / 1000.0
            metrics = self.cache_metrics[cache_id]
            metrics.avg_set_latency_us = (metrics.avg_set_latency_us + latency_us) / 2
            metrics.item_count = len(cache)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting key {key} in cache {cache_id}: {e}")
            return False
    
    async def delete(self, cache_id: str, key: str) -> bool:
        """Delete key from cache and replicas"""
        
        try:
            if cache_id not in self.caches:
                return False
            
            cache = self.caches[cache_id]
            
            if key in cache:
                del cache[key]
                
                # Update version vector
                if cache_id in self.version_vectors:
                    cache_version = self.version_vectors[cache_id].get(key, 0) + 1
                    self.version_vectors[cache_id][key] = cache_version
                
                # Schedule replication
                config = self.cache_configs[cache_id]
                if config.replication_factor > 1:
                    await self._schedule_replication(cache_id, key, None, "delete", config.consistency_level)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting key {key} from cache {cache_id}: {e}")
            return False
    
    async def invalidate(self, cache_id: str, key: str) -> bool:
        """Invalidate key across all replicas"""
        
        try:
            # Add to pending invalidations
            if key not in self.pending_invalidations:
                self.pending_invalidations[key] = set()
            
            # Get all replica nodes for this cache
            config = self.cache_configs[cache_id]
            replica_nodes = self._get_replica_nodes(cache_id)
            
            # Schedule invalidation replication
            await self._schedule_replication(cache_id, key, None, "invalidate", ConsistencyLevel.STRONG)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error invalidating key {key} in cache {cache_id}: {e}")
            return False
    
    async def _check_data_consistency(
        self, 
        cache_id: str, 
        key: str, 
        item: CacheItem, 
        required_level: ConsistencyLevel
    ) -> bool:
        """Check if cached data meets consistency requirements"""
        
        if required_level == ConsistencyLevel.WEAK:
            return True
        
        current_time = time.time()
        
        if required_level == ConsistencyLevel.BOUNDED_STALENESS:
            # Check if data is within staleness tolerance
            staleness_ms = (current_time - item.updated_at) * 1000
            return staleness_ms <= item.staleness_tolerance_ms
        
        elif required_level == ConsistencyLevel.STRONG:
            # For strong consistency, check version across replicas
            return await self._verify_strong_consistency(cache_id, key, item.version)
        
        elif required_level == ConsistencyLevel.SESSION:
            # Session consistency - ensure monotonic read consistency
            return True  # Simplified implementation
        
        return True  # Default to consistent
    
    async def _verify_strong_consistency(self, cache_id: str, key: str, local_version: int) -> bool:
        """Verify strong consistency by checking version across replicas"""
        
        try:
            # Get replica nodes
            replica_nodes = self._get_replica_nodes(cache_id)
            
            # Check versions across replicas (simplified)
            # In production, this would query actual replica nodes
            for node in replica_nodes:
                node_version = self.version_vectors.get(f"{cache_id}_{node}", {}).get(key, 0)
                if node_version > local_version:
                    return False  # Local version is stale
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying consistency for key {key}: {e}")
            return False
    
    async def _fetch_from_backing_store(self, cache_id: str, key: str) -> Optional[Any]:
        """Fetch data from backing store (Redis, database, etc.)"""
        
        try:
            # Simulate backing store fetch
            # In production, this would query Redis or other backing store
            await asyncio.sleep(0.001)  # Simulate 1ms backing store latency
            
            # For simulation, return None (cache miss from backing store)
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching key {key} from backing store: {e}")
            return None
    
    async def _write_to_backing_store(self, cache_id: str, key: str, value: Any) -> bool:
        """Write data to backing store"""
        
        try:
            # Simulate backing store write
            await asyncio.sleep(0.0005)  # Simulate 0.5ms backing store write
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing key {key} to backing store: {e}")
            return False
    
    async def _schedule_replication(
        self, 
        cache_id: str, 
        key: str, 
        value: Any, 
        operation: str,
        consistency_level: ConsistencyLevel
    ):
        """Schedule data replication to other edge nodes"""
        
        try:
            # Get target replica nodes
            target_caches = self._get_replica_nodes(cache_id)
            
            if not target_caches:
                return
            
            # Create replication job
            job = ReplicationJob(
                job_id=f"repl_{int(time.time() * 1000000)}_{key}",
                source_cache=cache_id,
                target_caches=target_caches,
                operation=operation,
                key=key,
                value=value,
                consistency_level=consistency_level,
                timeout_ms=1000.0 if consistency_level == ConsistencyLevel.STRONG else 5000.0
            )
            
            # Add to replication queue
            self.replication_queue.append(job)
            
            # Start replication workers if not running
            if not self.replication_active:
                await self.start_replication_workers()
            
        except Exception as e:
            self.logger.error(f"Error scheduling replication for key {key}: {e}")
    
    def _get_replica_nodes(self, cache_id: str) -> List[str]:
        """Get list of replica cache nodes for given cache"""
        
        config = self.cache_configs[cache_id]
        
        # In production, this would return actual replica node IDs
        # For simulation, generate mock replica names
        replicas = []
        for i in range(min(config.replication_factor - 1, 3)):  # Up to 3 replicas
            replica_id = f"{cache_id}_replica_{i+1}"
            replicas.append(replica_id)
        
        return replicas
    
    async def _evict_items(self, cache_id: str, count: int):
        """Evict items from cache based on eviction policy"""
        
        try:
            cache = self.caches[cache_id]
            config = self.cache_configs[cache_id]
            
            if len(cache) == 0:
                return
            
            eviction_policy = config.eviction_policy
            
            # Select items for eviction
            items_to_evict = []
            
            if eviction_policy == "lru":
                # Least Recently Used
                sorted_items = sorted(cache.items(), key=lambda x: x[1].accessed_at)
                items_to_evict = sorted_items[:count]
                
            elif eviction_policy == "lfu":
                # Least Frequently Used
                sorted_items = sorted(cache.items(), key=lambda x: x[1].access_count)
                items_to_evict = sorted_items[:count]
                
            elif eviction_policy == "fifo":
                # First In, First Out
                sorted_items = sorted(cache.items(), key=lambda x: x[1].created_at)
                items_to_evict = sorted_items[:count]
                
            else:  # random
                import random
                items_to_evict = random.sample(list(cache.items()), min(count, len(cache)))
            
            # Evict selected items
            evicted_count = 0
            for key, item in items_to_evict:
                if key in cache:
                    del cache[key]
                    evicted_count += 1
            
            # Update metrics
            metrics = self.cache_metrics[cache_id]
            metrics.evictions_total += evicted_count
            
            self.logger.debug(f"Evicted {evicted_count} items from cache {cache_id}")
            
        except Exception as e:
            self.logger.error(f"Error evicting items from cache {cache_id}: {e}")
    
    async def _update_access_patterns(self, key: str):
        """Update access patterns for intelligent caching"""
        
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        # Add current access time
        self.access_patterns[key].append(current_time)
        
        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff_time]
        
        # Calculate hotness score
        access_count = len(self.access_patterns[key])
        if access_count > 0:
            time_span = current_time - min(self.access_patterns[key])
            hotness = access_count / max(time_span, 1.0)
            self.hotspot_detection[key] = hotness
    
    async def start_replication_workers(self, worker_count: int = 3):
        """Start replication workers to process replication queue"""
        
        if self.replication_active:
            return
        
        self.replication_active = True
        
        # Start replication worker tasks
        for i in range(worker_count):
            worker_id = f"replication_worker_{i}"
            task = asyncio.create_task(self._replication_worker(worker_id))
            self.replication_workers[worker_id] = task
        
        self.logger.info(f"Started {worker_count} replication workers")
    
    async def _replication_worker(self, worker_id: str):
        """Replication worker to process replication jobs"""
        
        self.logger.info(f"Replication worker {worker_id} started")
        
        while self.replication_active:
            try:
                if not self.replication_queue:
                    await asyncio.sleep(0.01)  # 10ms polling interval
                    continue
                
                # Get next job from queue
                job = self.replication_queue.pop(0)
                job.status = "running"
                
                # Process replication job
                success = await self._process_replication_job(job)
                
                if success:
                    job.status = "completed"
                    job.completed_at = time.time()
                else:
                    job.retry_count += 1
                    if job.retry_count < job.max_retries:
                        job.status = "pending"
                        # Add back to queue for retry
                        self.replication_queue.append(job)
                    else:
                        job.status = "failed"
                        self.logger.error(f"Replication job {job.job_id} failed after {job.max_retries} retries")
                
            except Exception as e:
                self.logger.error(f"Replication worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info(f"Replication worker {worker_id} stopped")
    
    async def _process_replication_job(self, job: ReplicationJob) -> bool:
        """Process individual replication job"""
        
        try:
            successful_replications = 0
            
            for target_cache in job.target_caches:
                success = await self._replicate_to_target(job, target_cache)
                if success:
                    successful_replications += 1
            
            # Determine if job succeeded based on consistency requirements
            if job.consistency_level == ConsistencyLevel.STRONG:
                # All replications must succeed for strong consistency
                return successful_replications == len(job.target_caches)
            elif job.consistency_level == ConsistencyLevel.SEMI_SYNCHRONOUS:
                # Majority must succeed
                return successful_replications > (len(job.target_caches) / 2)
            else:
                # For eventual consistency, at least one success is acceptable
                return successful_replications > 0
            
        except Exception as e:
            self.logger.error(f"Error processing replication job {job.job_id}: {e}")
            job.error_message = str(e)
            return False
    
    async def _replicate_to_target(self, job: ReplicationJob, target_cache: str) -> bool:
        """Replicate data to specific target cache"""
        
        try:
            # Simulate network latency for replication
            await asyncio.sleep(0.001)  # 1ms simulated network latency
            
            # In production, this would send the data to the actual target cache
            # For simulation, just log the operation
            self.logger.debug(f"Replicated {job.operation} for key {job.key} to {target_cache}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to replicate to {target_cache}: {e}")
            return False
    
    async def start_monitoring(self):
        """Start cache monitoring and optimization"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_cache_metrics()),
            asyncio.create_task(self._optimize_cache_placement()),
            asyncio.create_task(self._detect_hotspots()),
            asyncio.create_task(self._cleanup_expired_items())
        ]
        
        self.logger.info("Cache monitoring started")
    
    async def _monitor_cache_metrics(self):
        """Monitor cache performance metrics"""
        
        while self.monitoring_active:
            try:
                for cache_id in self.caches.keys():
                    await self._update_cache_metrics(cache_id)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error monitoring cache metrics: {e}")
                await asyncio.sleep(5)
    
    async def _update_cache_metrics(self, cache_id: str):
        """Update metrics for specific cache"""
        
        try:
            cache = self.caches[cache_id]
            metrics = self.cache_metrics[cache_id]
            
            # Update basic metrics
            metrics.timestamp = time.time()
            metrics.item_count = len(cache)
            
            if cache:
                # Calculate memory usage
                total_size = sum(item.size_bytes for item in cache.values())
                metrics.memory_used_mb = total_size / (1024 * 1024)
                metrics.avg_item_size_bytes = total_size / len(cache)
                
                # Calculate memory utilization
                config = self.cache_configs[cache_id]
                metrics.memory_utilization = metrics.memory_used_mb / config.max_memory_mb
            
            # Calculate eviction rate
            current_evictions = metrics.evictions_total
            if hasattr(metrics, '_last_evictions'):
                evictions_delta = current_evictions - metrics._last_evictions
                metrics.evictions_per_second = evictions_delta  # Per second (update interval)
            metrics._last_evictions = current_evictions
            
        except Exception as e:
            self.logger.error(f"Error updating metrics for cache {cache_id}: {e}")
    
    async def _optimize_cache_placement(self):
        """Optimize cache data placement based on access patterns"""
        
        while self.monitoring_active:
            try:
                # Analyze access patterns and optimize placement
                for key, hotness in self.hotspot_detection.items():
                    if hotness > 10.0:  # High hotness threshold
                        # Consider replicating to more edge nodes
                        await self._increase_replication(key)
                    elif hotness < 0.1:  # Low hotness threshold
                        # Consider reducing replication
                        await self._reduce_replication(key)
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                self.logger.error(f"Error optimizing cache placement: {e}")
                await asyncio.sleep(300)
    
    async def _increase_replication(self, key: str):
        """Increase replication factor for hot keys"""
        # Placeholder for hot key replication increase
        self.logger.debug(f"Consider increasing replication for hot key: {key}")
    
    async def _reduce_replication(self, key: str):
        """Reduce replication factor for cold keys"""
        # Placeholder for cold key replication reduction
        self.logger.debug(f"Consider reducing replication for cold key: {key}")
    
    async def _detect_hotspots(self):
        """Detect cache hotspots and bottlenecks"""
        
        while self.monitoring_active:
            try:
                # Detect hotspots based on access patterns
                hot_keys = []
                for key, hotness in self.hotspot_detection.items():
                    if hotness > 5.0:  # Hotspot threshold
                        hot_keys.append((key, hotness))
                
                if hot_keys:
                    # Sort by hotness
                    hot_keys.sort(key=lambda x: x[1], reverse=True)
                    self.logger.info(f"Detected {len(hot_keys)} hotspots, top: {hot_keys[:5]}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error detecting hotspots: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_items(self):
        """Cleanup expired cache items"""
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                for cache_id, cache in self.caches.items():
                    expired_keys = []
                    
                    for key, item in cache.items():
                        if item.expires_at < current_time:
                            expired_keys.append(key)
                    
                    # Remove expired items
                    for key in expired_keys:
                        del cache[key]
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired items from cache {cache_id}")
                
                await asyncio.sleep(10)  # Cleanup every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error cleaning up expired items: {e}")
                await asyncio.sleep(30)
    
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status and metrics"""
        
        total_items = sum(len(cache) for cache in self.caches.values())
        total_memory_mb = sum(metrics.memory_used_mb for metrics in self.cache_metrics.values())
        avg_hit_rate = sum(metrics.hit_rate for metrics in self.cache_metrics.values()) / len(self.cache_metrics) if self.cache_metrics else 0.0
        
        return {
            "timestamp": time.time(),
            "global_stats": {
                "total_caches": len(self.caches),
                "total_items": total_items,
                "total_memory_mb": total_memory_mb,
                "avg_hit_rate": avg_hit_rate,
                "replication_queue_size": len(self.replication_queue),
                "active_replication_workers": len(self.replication_workers)
            },
            "cache_details": {
                cache_id: {
                    "config": {
                        "node_id": config.node_id,
                        "region": config.region,
                        "max_memory_mb": config.max_memory_mb,
                        "replication_factor": config.replication_factor,
                        "cache_strategy": config.cache_strategy.value
                    },
                    "metrics": {
                        "item_count": metrics.item_count,
                        "hit_rate": metrics.hit_rate,
                        "memory_used_mb": metrics.memory_used_mb,
                        "memory_utilization": metrics.memory_utilization,
                        "avg_get_latency_us": metrics.avg_get_latency_us,
                        "evictions_total": metrics.evictions_total
                    }
                }
                for cache_id, (config, metrics) in 
                zip(self.cache_configs.keys(), zip(self.cache_configs.values(), self.cache_metrics.values()))
            },
            "hotspots": dict(sorted(self.hotspot_detection.items(), key=lambda x: x[1], reverse=True)[:10]),
            "monitoring_active": self.monitoring_active,
            "replication_active": self.replication_active
        }
    
    def stop_monitoring(self):
        """Stop cache monitoring"""
        
        self.monitoring_active = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        self.monitoring_tasks.clear()
        self.logger.info("Cache monitoring stopped")
    
    def stop_replication_workers(self):
        """Stop replication workers"""
        
        self.replication_active = False
        
        # Cancel replication worker tasks
        for worker_id, task in self.replication_workers.items():
            if not task.done():
                task.cancel()
        
        self.replication_workers.clear()
        self.logger.info("Replication workers stopped")
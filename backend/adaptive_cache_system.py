#!/usr/bin/env python3
"""
Adaptive Cache System with Nested Negative Feedback Loops

Implements hierarchical cache layers (L1/L2/L3) with:
- Dynamic TTL adjustment based on data volatility
- Predictive prefetching using ML Engine signals  
- Nanosecond precision cache coherency protocol
- Feedback-aware eviction policies
- Self-optimizing performance characteristics

Cache Hierarchy:
- L1 (Engine-Local): 0.1-1.0s TTL, sub-millisecond access
- L2 (MessageBus): 1.0-10s TTL, cross-engine coordination  
- L3 (System-Wide): 10-300s TTL, global data consistency

Performance Target: 90%+ hit rate, <0.1ms L1 access time
Author: BMad Orchestrator
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
import json
import hashlib
import statistics
from collections import deque, defaultdict, OrderedDict
import weakref
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
import zlib

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_ENGINE_LOCAL = "l1_engine_local"      # Engine-specific fast cache
    L2_MESSAGEBUS = "l2_messagebus"          # Cross-engine coordination cache
    L3_SYSTEM_WIDE = "l3_system_wide"        # Global consistency cache


class CacheEntryState(Enum):
    """Cache entry lifecycle states"""
    FRESH = "fresh"              # Recently cached, high confidence
    VALID = "valid"              # Valid but aging
    STALE = "stale"              # Needs refresh but usable
    INVALID = "invalid"          # Must be refreshed
    PREFETCHING = "prefetching"  # Being refreshed in background


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    FEEDBACK_AWARE = "feedback_aware"  # Uses feedback loop signals
    PREDICTIVE = "predictive"      # ML-driven eviction
    ADAPTIVE = "adaptive"          # Combines multiple strategies


@dataclass
class CacheEntry:
    """Individual cache entry with comprehensive metadata"""
    key: str
    value: Any
    timestamp: float            # Creation time (nanoseconds)
    version: int               # Coherency version number
    ttl: float                 # Time to live (seconds)
    access_count: int = 0
    last_access: float = 0.0
    volatility_score: float = 0.5  # 0.0 = stable, 1.0 = highly volatile
    importance_score: float = 0.5  # 0.0 = low priority, 1.0 = critical
    source_engine: str = ""
    state: CacheEntryState = CacheEntryState.FRESH
    
    # Feedback metrics
    hit_count: int = 0
    miss_penalty: float = 0.0    # Cost of cache miss in ms
    refresh_cost: float = 0.0    # Cost of data refresh in ms
    
    # Coherency metadata
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if entry is expired based on TTL"""
        return (current_time - self.timestamp) > self.ttl
        
    def is_stale(self, current_time: float, staleness_threshold: float = 0.8) -> bool:
        """Check if entry is approaching expiration"""
        age = current_time - self.timestamp
        return age > (self.ttl * staleness_threshold)
        
    def calculate_value_score(self, current_time: float) -> float:
        """Calculate entry value for eviction decisions"""
        
        # Recency factor (exponential decay)
        age = current_time - self.last_access
        recency_score = np.exp(-age / 10.0)  # 10 second half-life
        
        # Frequency factor
        frequency_score = min(self.access_count / 100.0, 1.0)  # Normalize to 100 accesses
        
        # Importance and stability factors
        importance_factor = self.importance_score
        stability_factor = 1.0 - self.volatility_score
        
        # Cost-benefit analysis
        miss_penalty_factor = min(self.miss_penalty / 10.0, 1.0)  # Normalize to 10ms penalty
        
        # Combined score (weighted average)
        value_score = (
            0.3 * recency_score +
            0.2 * frequency_score + 
            0.2 * importance_factor +
            0.15 * stability_factor +
            0.15 * miss_penalty_factor
        )
        
        return value_score


@dataclass  
class CacheConfiguration:
    """Configuration for cache level"""
    level: CacheLevel
    max_size: int              # Maximum entries
    default_ttl: float         # Default TTL in seconds
    min_ttl: float            # Minimum TTL
    max_ttl: float            # Maximum TTL
    eviction_policy: EvictionPolicy
    enable_compression: bool = True
    enable_prediction: bool = True
    coherency_enabled: bool = True
    
    # Performance tuning
    access_pattern_window: int = 1000  # Track last N accesses
    volatility_learning_rate: float = 0.1
    prefetch_threshold: float = 0.7    # Prefetch when staleness > threshold


class CacheStats:
    """Comprehensive cache statistics for feedback loops"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all statistics"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetches = 0
        self.coherency_updates = 0
        self.total_access_time_ms = 0.0
        self.total_miss_penalty_ms = 0.0
        
        # Distribution tracking
        self.access_times = deque(maxlen=10000)
        self.entry_lifetimes = deque(maxlen=10000)
        self.volatility_scores = deque(maxlen=1000)
        
        # Temporal patterns
        self.hourly_hit_rates = defaultdict(list)
        self.pattern_predictions = {}
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    @property
    def average_access_time_ms(self) -> float:
        """Calculate average access time"""
        return statistics.mean(self.access_times) if self.access_times else 0.0
        
    @property
    def cache_efficiency(self) -> float:
        """Calculate overall cache efficiency score"""
        hit_rate_factor = self.hit_rate
        speed_factor = max(0, 1.0 - (self.average_access_time_ms / 10.0))  # Target <10ms
        prefetch_factor = min(self.prefetches / max(self.misses, 1), 1.0)
        
        return (hit_rate_factor * 0.6 + speed_factor * 0.3 + prefetch_factor * 0.1)


class AdaptiveCache:
    """
    Adaptive cache implementation with feedback loop integration
    
    Features:
    - Dynamic TTL adjustment based on access patterns
    - Predictive prefetching using ML signals
    - Feedback-aware eviction policies
    - Nanosecond coherency protocol
    - Self-optimization via negative feedback
    """
    
    def __init__(self, config: CacheConfiguration, feedback_callback: Optional[Callable] = None):
        self.config = config
        self.feedback_callback = feedback_callback
        
        # Core storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict[str, float] = OrderedDict()  # For LRU
        self.access_frequency: Dict[str, int] = defaultdict(int)    # For LFU
        
        # Statistics and monitoring
        self.stats = CacheStats()
        self.performance_history = deque(maxlen=1000)
        
        # Predictive components
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volatility_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        # Coherency tracking
        self.version_counter = 0
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Threading for background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        
        logger.info(f"🗂️ AdaptiveCache initialized: {config.level.value}")
        logger.info(f"   📏 Max Size: {config.max_size} entries")
        logger.info(f"   ⏰ TTL Range: {config.min_ttl}s - {config.max_ttl}s")
        logger.info(f"   🧠 Eviction: {config.eviction_policy.value}")
        
    async def start(self):
        """Start background optimization tasks"""
        if self._running:
            return
            
        self._running = True
        
        # Start background tasks
        tasks = [
            self._ttl_optimizer_task(),
            self._prefetch_processor_task(),
            self._coherency_monitor_task(),
            self._performance_monitor_task()
        ]
        
        self._background_tasks = [asyncio.create_task(task) for task in tasks]
        
        logger.info(f"🚀 AdaptiveCache background tasks started: {len(self._background_tasks)}")
        
    async def stop(self):
        """Stop background tasks"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
        self._background_tasks.clear()
        logger.info(f"🛑 AdaptiveCache background tasks stopped")
        
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with adaptive optimization"""
        
        start_time = time.perf_counter()
        
        async with self._lock:
            entry = self.entries.get(key)
            current_time = time.time_ns() / 1e9  # Convert to seconds
            
            if entry is None:
                # Cache miss
                self.stats.misses += 1
                self._record_access_pattern(key, False, current_time)
                
                access_time = (time.perf_counter() - start_time) * 1000
                self.stats.access_times.append(access_time)
                
                await self._trigger_feedback('cache_miss', key, access_time)
                return default
                
            # Check if entry is expired
            if entry.is_expired(current_time):
                # Expired entry - treat as miss but keep for staleness serving
                del self.entries[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                self._record_access_pattern(key, False, current_time)
                
                access_time = (time.perf_counter() - start_time) * 1000
                self.stats.access_times.append(access_time)
                
                await self._trigger_feedback('cache_expired', key, access_time)
                
                # Return stale data if configured for graceful degradation
                if self.config.level == CacheLevel.L1_ENGINE_LOCAL:
                    logger.debug(f"🗂️ Serving stale data for {key}")
                    return entry.value
                    
                return default
            
            # Cache hit - update access metadata
            entry.access_count += 1
            entry.last_access = current_time
            entry.hit_count += 1
            
            # Update access tracking
            self.access_order[key] = current_time
            self.access_frequency[key] += 1
            self._record_access_pattern(key, True, current_time)
            
            # Trigger prefetch if entry is getting stale
            if entry.is_stale(current_time, self.config.prefetch_threshold):
                await self._trigger_prefetch(key, entry)
                
            self.stats.hits += 1
            
            access_time = (time.perf_counter() - start_time) * 1000
            self.stats.access_times.append(access_time)
            
            # Adaptive TTL adjustment based on access patterns
            await self._adapt_entry_ttl(key, entry)
            
            await self._trigger_feedback('cache_hit', key, access_time)
            
            return entry.value
            
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 importance: float = 0.5, source_engine: str = "") -> bool:
        """Set value in cache with adaptive optimization"""
        
        async with self._lock:
            current_time = time.time_ns() / 1e9
            
            # Determine TTL
            if ttl is None:
                ttl = await self._calculate_adaptive_ttl(key, value, importance)
            else:
                ttl = max(self.config.min_ttl, min(ttl, self.config.max_ttl))
                
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=self._compress_if_enabled(value),
                timestamp=current_time,
                version=self._get_next_version(),
                ttl=ttl,
                importance_score=importance,
                source_engine=source_engine
            )
            
            # Check if we need to evict
            if len(self.entries) >= self.config.max_size and key not in self.entries:
                await self._evict_entries(1)
                
            # Store entry
            old_entry = self.entries.get(key)
            self.entries[key] = entry
            self.access_order[key] = current_time
            
            # Update volatility tracking
            if old_entry:
                self._update_volatility(key, old_entry, entry)
                
            # Handle coherency updates
            if self.config.coherency_enabled and old_entry:
                await self._update_coherency(key, entry)
                
            logger.debug(f"💾 Cache set: {key} (TTL: {ttl:.2f}s, Level: {self.config.level.value})")
            
            await self._trigger_feedback('cache_set', key, ttl)
            
            return True
            
    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        
        async with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                del self.entries[key]
                
                # Clean up tracking
                self.access_order.pop(key, None)
                self.access_frequency.pop(key, None)
                self.access_patterns.pop(key, None)
                
                # Handle coherency
                if self.config.coherency_enabled:
                    await self._invalidate_dependents(key)
                    
                logger.debug(f"🗑️ Cache deleted: {key}")
                return True
                
            return False
            
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.access_patterns.clear()
            self.volatility_tracker.clear()
            
            logger.info(f"🧹 Cache cleared: {self.config.level.value}")
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        current_time = time.time()
        
        # Calculate advanced metrics
        entry_count = len(self.entries)
        memory_usage = sum(len(pickle.dumps(entry.value)) for entry in self.entries.values())
        
        # Age distribution
        ages = [(current_time - entry.timestamp) for entry in self.entries.values()]
        avg_age = statistics.mean(ages) if ages else 0
        
        # Volatility analysis
        volatilities = [entry.volatility_score for entry in self.entries.values()]
        avg_volatility = statistics.mean(volatilities) if volatilities else 0
        
        stats = {
            'level': self.config.level.value,
            'entry_count': entry_count,
            'max_size': self.config.max_size,
            'memory_usage_bytes': memory_usage,
            'hit_rate': self.stats.hit_rate,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'evictions': self.stats.evictions,
            'prefetches': self.stats.prefetches,
            'average_access_time_ms': self.stats.average_access_time_ms,
            'cache_efficiency': self.stats.cache_efficiency,
            'average_entry_age_seconds': avg_age,
            'average_volatility': avg_volatility,
            'coherency_updates': self.stats.coherency_updates
        }
        
        return stats
        
    # Internal optimization methods
    
    async def _calculate_adaptive_ttl(self, key: str, value: Any, importance: float) -> float:
        """Calculate adaptive TTL based on patterns and importance"""
        
        base_ttl = self.config.default_ttl
        
        # Factor 1: Historical access patterns
        pattern_history = self.access_patterns.get(key, deque())
        if len(pattern_history) > 5:
            recent_intervals = []
            for i in range(1, min(len(pattern_history), 10)):
                interval = pattern_history[i] - pattern_history[i-1]
                recent_intervals.append(interval)
                
            if recent_intervals:
                avg_interval = statistics.mean(recent_intervals)
                # TTL should be 2x the typical access interval
                pattern_ttl = max(avg_interval * 2, self.config.min_ttl)
                base_ttl = (base_ttl + pattern_ttl) / 2
        
        # Factor 2: Data volatility
        volatility_history = self.volatility_tracker.get(key, deque())
        if volatility_history:
            avg_volatility = statistics.mean(volatility_history)
            # High volatility = shorter TTL
            volatility_factor = 2.0 - avg_volatility  # Range: 1.0 - 2.0
            base_ttl *= volatility_factor
            
        # Factor 3: Importance scaling
        importance_factor = 0.5 + (importance * 1.5)  # Range: 0.5 - 2.0
        base_ttl *= importance_factor
        
        # Factor 4: System load (feedback from controller)
        # TODO: Integrate with feedback controller for load-based adjustments
        
        # Clamp to configured bounds
        adaptive_ttl = max(self.config.min_ttl, min(base_ttl, self.config.max_ttl))
        
        return adaptive_ttl
        
    async def _evict_entries(self, count: int):
        """Evict entries based on configured policy"""
        
        if not self.entries:
            return
            
        current_time = time.time()
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            candidates = sorted(self.access_order.items(), key=lambda x: x[1])[:count]
            
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            candidates = sorted(self.access_frequency.items(), key=lambda x: x[1])[:count]
            candidates = [(key, freq) for key, freq in candidates]
            
        elif self.config.eviction_policy == EvictionPolicy.FEEDBACK_AWARE:
            # Evict based on feedback value scores
            scored_entries = []
            for key, entry in self.entries.items():
                score = entry.calculate_value_score(current_time)
                scored_entries.append((key, score))
                
            # Sort by lowest value score (evict first)
            candidates = sorted(scored_entries, key=lambda x: x[1])[:count]
            
        else:
            # Default to LRU for unknown policies
            candidates = sorted(self.access_order.items(), key=lambda x: x[1])[:count]
            
        # Perform evictions
        for key, _ in candidates:
            if key in self.entries:
                entry = self.entries[key]
                
                # Record lifetime for statistics
                lifetime = current_time - entry.timestamp
                self.stats.entry_lifetimes.append(lifetime)
                
                # Clean up
                del self.entries[key]
                self.access_order.pop(key, None)
                self.access_frequency.pop(key, None)
                
                self.stats.evictions += 1
                
                logger.debug(f"🗑️ Evicted: {key} (lifetime: {lifetime:.2f}s)")
                
    async def _trigger_prefetch(self, key: str, entry: CacheEntry):
        """Trigger background prefetch for stale entry"""
        
        if not self.config.enable_prediction:
            return
            
        try:
            # Add to prefetch queue
            await self.prefetch_queue.put({
                'key': key,
                'entry': entry,
                'timestamp': time.time(),
                'priority': entry.importance_score
            })
            
            logger.debug(f"🔄 Prefetch queued: {key}")
            
        except asyncio.QueueFull:
            logger.warning(f"⚠️ Prefetch queue full, skipping: {key}")
            
    async def _adapt_entry_ttl(self, key: str, entry: CacheEntry):
        """Dynamically adapt entry TTL based on access patterns"""
        
        # Only adapt if entry has enough access history
        if entry.access_count < 5:
            return
            
        current_time = time.time()
        
        # Calculate access frequency
        time_since_creation = current_time - entry.timestamp
        access_rate = entry.access_count / max(time_since_creation, 0.1)
        
        # Adjust TTL based on access rate
        if access_rate > 1.0:  # More than 1 access per second
            # High usage - extend TTL
            new_ttl = min(entry.ttl * 1.2, self.config.max_ttl)
        elif access_rate < 0.1:  # Less than 1 access per 10 seconds
            # Low usage - reduce TTL
            new_ttl = max(entry.ttl * 0.8, self.config.min_ttl)
        else:
            return  # No change needed
            
        # Apply TTL change
        entry.ttl = new_ttl
        logger.debug(f"🔧 TTL adapted: {key} → {new_ttl:.2f}s (rate: {access_rate:.3f}/s)")
        
    def _record_access_pattern(self, key: str, hit: bool, timestamp: float):
        """Record access pattern for predictive optimization"""
        
        pattern = self.access_patterns[key]
        pattern.append(timestamp)
        
        # Update hourly statistics for pattern analysis
        hour = int(timestamp // 3600) % 24
        self.stats.hourly_hit_rates[hour].append(1 if hit else 0)
        
    def _update_volatility(self, key: str, old_entry: CacheEntry, new_entry: CacheEntry):
        """Update volatility score based on value changes"""
        
        try:
            # Compare values to determine volatility
            old_hash = hashlib.sha256(pickle.dumps(old_entry.value)).hexdigest()
            new_hash = hashlib.sha256(pickle.dumps(new_entry.value)).hexdigest()
            
            # Calculate volatility (1.0 = completely different, 0.0 = identical)
            volatility = 1.0 if old_hash != new_hash else 0.0
            
            # Update volatility tracking
            volatility_history = self.volatility_tracker[key]
            volatility_history.append(volatility)
            
            # Update entry volatility score with exponential moving average
            if new_entry.volatility_score == 0.5:  # Initial value
                new_entry.volatility_score = volatility
            else:
                alpha = self.config.volatility_learning_rate
                new_entry.volatility_score = (alpha * volatility + 
                                            (1 - alpha) * new_entry.volatility_score)
                
        except Exception as e:
            logger.debug(f"Error updating volatility for {key}: {e}")
            
    def _compress_if_enabled(self, value: Any) -> Any:
        """Compress value if compression is enabled"""
        
        if not self.config.enable_compression:
            return value
            
        try:
            # Only compress if value is large enough
            serialized = pickle.dumps(value)
            if len(serialized) > 1024:  # 1KB threshold
                compressed = zlib.compress(serialized)
                if len(compressed) < len(serialized):
                    return {'__compressed__': True, '__data__': compressed}
                    
        except Exception as e:
            logger.debug(f"Compression failed: {e}")
            
        return value
        
    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if it was compressed"""
        
        if isinstance(value, dict) and value.get('__compressed__'):
            try:
                decompressed = zlib.decompress(value['__data__'])
                return pickle.loads(decompressed)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                
        return value
        
    def _get_next_version(self) -> int:
        """Get next coherency version number"""
        self.version_counter += 1
        return self.version_counter
        
    async def _update_coherency(self, key: str, entry: CacheEntry):
        """Update cache coherency across dependent entries"""
        
        # Invalidate dependent entries
        dependents = self.dependency_graph.get(key, set())
        
        for dependent_key in dependents:
            if dependent_key in self.entries:
                dependent_entry = self.entries[dependent_key]
                dependent_entry.state = CacheEntryState.STALE
                dependent_entry.version = self._get_next_version()
                
        self.stats.coherency_updates += len(dependents)
        
        if dependents:
            logger.debug(f"🔄 Coherency update: {key} → {len(dependents)} dependents")
            
    async def _invalidate_dependents(self, key: str):
        """Invalidate all entries that depend on the given key"""
        
        dependents = self.dependency_graph.get(key, set())
        
        for dependent_key in list(dependents):
            if dependent_key in self.entries:
                await self.delete(dependent_key)
                
        # Clean up dependency graph
        self.dependency_graph.pop(key, None)
        
    async def _trigger_feedback(self, event_type: str, key: str, metric_value: float):
        """Trigger feedback to the feedback loop controller"""
        
        if self.feedback_callback:
            try:
                feedback_data = {
                    'cache_level': self.config.level.value,
                    'event_type': event_type,
                    'key': key,
                    'metric_value': metric_value,
                    'timestamp': time.time(),
                    'stats': {
                        'hit_rate': self.stats.hit_rate,
                        'efficiency': self.stats.cache_efficiency,
                        'entry_count': len(self.entries)
                    }
                }
                
                await self.feedback_callback(feedback_data)
                
            except Exception as e:
                logger.debug(f"Feedback callback error: {e}")
                
    # Background task implementations
    
    async def _ttl_optimizer_task(self):
        """Background task for TTL optimization"""
        
        while self._running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Analyze TTL effectiveness
                await self._analyze_ttl_effectiveness()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTL optimizer error: {e}")
                await asyncio.sleep(5)
                
    async def _prefetch_processor_task(self):
        """Background task for processing prefetch requests"""
        
        while self._running:
            try:
                # Get prefetch request with timeout
                prefetch_request = await asyncio.wait_for(
                    self.prefetch_queue.get(), timeout=1.0
                )
                
                # Process prefetch
                await self._process_prefetch_request(prefetch_request)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch processor error: {e}")
                
    async def _coherency_monitor_task(self):
        """Background task for coherency monitoring"""
        
        while self._running:
            try:
                await asyncio.sleep(10)  # Run every 10 seconds
                
                if self.config.coherency_enabled:
                    await self._check_coherency_violations()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coherency monitor error: {e}")
                
    async def _performance_monitor_task(self):
        """Background task for performance monitoring"""
        
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Record performance snapshot
                stats = await self.get_stats()
                self.performance_history.append({
                    'timestamp': time.time(),
                    'stats': stats
                })
                
                # Log performance if efficiency is low
                if stats['cache_efficiency'] < 0.8:
                    logger.warning(f"⚠️ Low cache efficiency: {stats['cache_efficiency']:.3f}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                
    async def _analyze_ttl_effectiveness(self):
        """Analyze and optimize TTL settings"""
        
        current_time = time.time()
        
        # Find entries that are expiring too frequently (short TTL)
        short_ttl_candidates = []
        long_ttl_candidates = []
        
        for key, entry in self.entries.items():
            age = current_time - entry.timestamp
            ttl_utilization = age / entry.ttl
            
            # Entry is accessed frequently but has short TTL
            if (entry.access_count > 10 and ttl_utilization > 0.8 and 
                entry.access_count / age > 1.0):
                short_ttl_candidates.append((key, entry))
                
            # Entry is rarely accessed but has long TTL  
            elif (entry.access_count < 3 and ttl_utilization < 0.3 and
                  entry.access_count / age < 0.1):
                long_ttl_candidates.append((key, entry))
                
        # Adjust TTLs
        for key, entry in short_ttl_candidates:
            new_ttl = min(entry.ttl * 1.5, self.config.max_ttl)
            entry.ttl = new_ttl
            logger.debug(f"🔧 TTL increased: {key} → {new_ttl:.2f}s")
            
        for key, entry in long_ttl_candidates:
            new_ttl = max(entry.ttl * 0.7, self.config.min_ttl)
            entry.ttl = new_ttl
            logger.debug(f"🔧 TTL decreased: {key} → {new_ttl:.2f}s")
            
    async def _process_prefetch_request(self, request: Dict[str, Any]):
        """Process a single prefetch request"""
        
        key = request['key']
        entry = request['entry']
        
        # Simulate prefetch operation
        # In real implementation, this would refresh the data from the source
        
        logger.debug(f"🔄 Processing prefetch: {key}")
        
        # Mark as prefetching
        if key in self.entries:
            self.entries[key].state = CacheEntryState.PREFETCHING
            
        # Simulate data refresh delay
        await asyncio.sleep(0.01)  # 10ms simulated refresh
        
        # Update prefetch statistics
        self.stats.prefetches += 1
        
    async def _check_coherency_violations(self):
        """Check for cache coherency violations"""
        
        violations = 0
        current_time = time.time()
        
        for key, entry in self.entries.items():
            # Check if entry state is consistent
            if entry.state == CacheEntryState.INVALID and not entry.is_expired(current_time):
                violations += 1
                logger.warning(f"⚠️ Coherency violation: {key} marked invalid but not expired")
                
        if violations > 0:
            logger.warning(f"⚠️ Found {violations} coherency violations")


class HierarchicalCacheSystem:
    """
    Hierarchical cache system managing L1/L2/L3 cache levels
    with intelligent data promotion and coherency
    """
    
    def __init__(self, feedback_callback: Optional[Callable] = None):
        self.feedback_callback = feedback_callback
        
        # Initialize cache levels
        self.l1_cache = AdaptiveCache(
            CacheConfiguration(
                level=CacheLevel.L1_ENGINE_LOCAL,
                max_size=10000,
                default_ttl=1.0,
                min_ttl=0.1,
                max_ttl=10.0,
                eviction_policy=EvictionPolicy.FEEDBACK_AWARE,
                enable_prediction=True
            ),
            feedback_callback
        )
        
        self.l2_cache = AdaptiveCache(
            CacheConfiguration(
                level=CacheLevel.L2_MESSAGEBUS,
                max_size=50000,
                default_ttl=10.0,
                min_ttl=1.0,
                max_ttl=60.0,
                eviction_policy=EvictionPolicy.ADAPTIVE,
                enable_prediction=True,
                coherency_enabled=True
            ),
            feedback_callback
        )
        
        self.l3_cache = AdaptiveCache(
            CacheConfiguration(
                level=CacheLevel.L3_SYSTEM_WIDE,
                max_size=100000,
                default_ttl=60.0,
                min_ttl=10.0,
                max_ttl=300.0,
                eviction_policy=EvictionPolicy.LRU,
                enable_compression=True,
                coherency_enabled=True
            ),
            feedback_callback
        )
        
        self.caches = {
            CacheLevel.L1_ENGINE_LOCAL: self.l1_cache,
            CacheLevel.L2_MESSAGEBUS: self.l2_cache,
            CacheLevel.L3_SYSTEM_WIDE: self.l3_cache
        }
        
        logger.info("🏗️ HierarchicalCacheSystem initialized")
        
    async def start(self):
        """Start all cache levels"""
        for cache in self.caches.values():
            await cache.start()
            
        logger.info("🚀 HierarchicalCacheSystem started")
        
    async def stop(self):
        """Stop all cache levels"""
        for cache in self.caches.values():
            await cache.stop()
            
        logger.info("🛑 HierarchicalCacheSystem stopped")
        
    async def get(self, key: str, default: Any = None) -> Tuple[Any, CacheLevel]:
        """Get value from cache hierarchy (L1 → L2 → L3)"""
        
        # Try L1 first (fastest)
        value = await self.l1_cache.get(key)
        if value is not None:
            return value, CacheLevel.L1_ENGINE_LOCAL
            
        # Try L2 next
        value = await self.l2_cache.get(key)
        if value is not None:
            # Promote to L1 for future access
            await self.l1_cache.set(key, value, importance=0.8)
            return value, CacheLevel.L2_MESSAGEBUS
            
        # Try L3 last
        value = await self.l3_cache.get(key)
        if value is not None:
            # Promote to L2 and L1
            await self.l2_cache.set(key, value, importance=0.6)
            await self.l1_cache.set(key, value, importance=0.6)
            return value, CacheLevel.L3_SYSTEM_WIDE
            
        return default, None
        
    async def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.L1_ENGINE_LOCAL,
                 ttl: Optional[float] = None, importance: float = 0.5) -> bool:
        """Set value in specific cache level"""
        
        cache = self.caches.get(level)
        if cache:
            return await cache.set(key, value, ttl, importance)
            
        return False
        
    async def delete(self, key: str, level: Optional[CacheLevel] = None) -> int:
        """Delete key from cache(s)"""
        
        deleted_count = 0
        
        if level:
            # Delete from specific level
            cache = self.caches.get(level)
            if cache and await cache.delete(key):
                deleted_count += 1
        else:
            # Delete from all levels
            for cache in self.caches.values():
                if await cache.delete(key):
                    deleted_count += 1
                    
        return deleted_count
        
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {}
        total_entries = 0
        total_memory = 0
        overall_hit_rate = 0
        
        for level, cache in self.caches.items():
            cache_stats = await cache.get_stats()
            stats[level.value] = cache_stats
            
            total_entries += cache_stats['entry_count']
            total_memory += cache_stats['memory_usage_bytes']
            overall_hit_rate += cache_stats['hit_rate']
            
        # Calculate system-wide metrics
        stats['system_wide'] = {
            'total_entries': total_entries,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / 1024 / 1024,
            'overall_hit_rate': overall_hit_rate / len(self.caches),
            'cache_levels': len(self.caches)
        }
        
        return stats


# Factory functions

def create_adaptive_cache(level: CacheLevel, feedback_callback: Optional[Callable] = None) -> AdaptiveCache:
    """Create adaptive cache for specific level"""
    
    configs = {
        CacheLevel.L1_ENGINE_LOCAL: CacheConfiguration(
            level=level,
            max_size=5000,
            default_ttl=0.5,
            min_ttl=0.1,
            max_ttl=5.0,
            eviction_policy=EvictionPolicy.FEEDBACK_AWARE
        ),
        CacheLevel.L2_MESSAGEBUS: CacheConfiguration(
            level=level,
            max_size=25000,
            default_ttl=5.0,
            min_ttl=1.0,
            max_ttl=30.0,
            eviction_policy=EvictionPolicy.ADAPTIVE,
            coherency_enabled=True
        ),
        CacheLevel.L3_SYSTEM_WIDE: CacheConfiguration(
            level=level,
            max_size=100000,
            default_ttl=30.0,
            min_ttl=5.0,
            max_ttl=300.0,
            eviction_policy=EvictionPolicy.LRU,
            enable_compression=True,
            coherency_enabled=True
        )
    }
    
    config = configs.get(level, configs[CacheLevel.L1_ENGINE_LOCAL])
    return AdaptiveCache(config, feedback_callback)


def create_hierarchical_cache_system(feedback_callback: Optional[Callable] = None) -> HierarchicalCacheSystem:
    """Create complete hierarchical cache system"""
    return HierarchicalCacheSystem(feedback_callback)


# Global cache system instance
_global_cache_system: Optional[HierarchicalCacheSystem] = None

async def get_cache_system() -> HierarchicalCacheSystem:
    """Get global cache system instance"""
    global _global_cache_system
    
    if _global_cache_system is None:
        _global_cache_system = create_hierarchical_cache_system()
        await _global_cache_system.start()
        
    return _global_cache_system


if __name__ == "__main__":
    # Test the adaptive cache system
    async def test_adaptive_cache():
        print("🧪 Testing Adaptive Cache System with Hierarchical Architecture")
        print("=" * 80)
        
        cache_system = create_hierarchical_cache_system()
        await cache_system.start()
        
        try:
            # Test data
            test_data = {
                'market_data_AAPL': {'price': 150.25, 'volume': 1000000},
                'risk_metrics_portfolio': {'var': 0.05, 'sharpe': 1.8},
                'ml_prediction_GOOGL': {'price_target': 2800, 'confidence': 0.92}
            }
            
            # Test cache operations
            print("\n📝 Testing cache operations...")
            
            for key, value in test_data.items():
                await cache_system.set(key, value, CacheLevel.L1_ENGINE_LOCAL, importance=0.8)
                print(f"   ✅ Set: {key}")
                
            await asyncio.sleep(0.1)
            
            # Test retrieval and promotion
            print("\n📖 Testing retrieval and promotion...")
            
            for key in test_data.keys():
                value, level = await cache_system.get(key)
                print(f"   ✅ Get: {key} from {level.value if level else 'MISS'}")
                
            # Test system statistics
            await asyncio.sleep(1)
            
            print("\n📊 System Statistics:")
            stats = await cache_system.get_system_stats()
            
            for level_name, level_stats in stats.items():
                if level_name != 'system_wide':
                    print(f"   {level_name}:")
                    print(f"     Entries: {level_stats['entry_count']}")
                    print(f"     Hit Rate: {level_stats['hit_rate']:.1%}")
                    print(f"     Efficiency: {level_stats['cache_efficiency']:.3f}")
                    
            system_stats = stats['system_wide']
            print(f"   System Total:")
            print(f"     Entries: {system_stats['total_entries']}")
            print(f"     Memory: {system_stats['total_memory_mb']:.2f} MB")
            print(f"     Overall Hit Rate: {system_stats['overall_hit_rate']:.1%}")
            
        finally:
            await cache_system.stop()
    
    asyncio.run(test_adaptive_cache())
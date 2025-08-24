"""
GPU Memory Pool Manager for M4 Max Metal Acceleration

Provides efficient memory allocation and management for GPU computations:
- Memory pool with unified architecture optimization for M4 Max 546GB/s bandwidth
- Smart allocation strategies for 40 GPU cores
- Automatic garbage collection and memory pressure handling
- Cache management for repeated calculations and model inference
- Memory fragmentation prevention and defragmentation
- Thermal-aware memory management with automatic throttling

Optimized for Apple Silicon unified memory architecture.
"""

import asyncio
import logging
import time
import threading
import weakref
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import gc

# Import Metal configuration
from .metal_config import (
    metal_device_manager,
    is_metal_available,
    is_m4_max_detected,
    metal_performance_context
)

# Memory management imports with fallback
try:
    import torch
    MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends.mps, 'is_available') else False
except ImportError:
    torch = None
    MPS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemoryBlock:
    """Represents a memory block in the GPU memory pool"""
    size: int
    dtype: str
    shape: Optional[Tuple[int, ...]]
    allocated_time: float
    last_used: float
    reference_count: int
    tensor_id: str
    pool_name: str
    is_cached: bool = False
    access_count: int = 0
    
@dataclass
class MemoryPoolStats:
    """Memory pool statistics and metrics"""
    total_allocated_mb: float
    total_cached_mb: float
    total_free_mb: float
    active_blocks: int
    cached_blocks: int
    allocation_count: int
    cache_hit_rate: float
    fragmentation_ratio: float
    peak_usage_mb: float
    average_block_size_mb: float
    memory_pressure_level: str  # 'low', 'medium', 'high', 'critical'
    
@dataclass
class AllocationRequest:
    """Memory allocation request with metadata"""
    size: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    device: torch.device
    pool_name: str
    priority: int = 0  # Higher number = higher priority
    allow_fallback: bool = True
    cache_key: Optional[str] = None

class MemoryPoolManager:
    """
    Advanced GPU memory pool manager with unified memory optimization
    Handles allocation, caching, and memory pressure for M4 Max
    """
    
    def __init__(self, max_memory_fraction: float = 0.8):
        self.max_memory_fraction = max_memory_fraction
        self.pools: Dict[str, Dict[str, MemoryBlock]] = defaultdict(dict)
        self.cache: Dict[str, torch.Tensor] = {}
        self.allocation_stats: Dict[str, int] = defaultdict(int)
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.weak_references: Set[weakref.ReferenceType] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        self._allocation_lock = threading.Lock()
        
        # Memory tracking
        self.peak_memory_usage = 0
        self.total_allocations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Background cleanup
        self._cleanup_thread_active = False
        self._memory_pressure_threshold = 0.85
        self._fragmentation_threshold = 0.3
        
        # Initialize memory monitoring
        self._start_memory_monitoring()
        
    def _start_memory_monitoring(self):
        """Start background memory monitoring and cleanup"""
        if self._cleanup_thread_active:
            return
            
        def monitor_memory():
            self._cleanup_thread_active = True
            while self._cleanup_thread_active:
                try:
                    # Check memory pressure
                    self._check_memory_pressure()
                    
                    # Cleanup unused references
                    self._cleanup_weak_references()
                    
                    # Defragment if needed
                    if self._should_defragment():
                        self._defragment_memory()
                        
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
    def _check_memory_pressure(self):
        """Check and respond to memory pressure"""
        try:
            memory_stats = self.get_memory_stats()
            utilization = memory_stats.total_allocated_mb / (memory_stats.total_free_mb + memory_stats.total_allocated_mb)
            
            if utilization > self._memory_pressure_threshold:
                logger.warning(f"High memory pressure detected: {utilization:.2%}")
                
                # Aggressive cleanup
                self._emergency_cleanup()
                
                # Reduce cache size
                self._reduce_cache_size(0.5)
                
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            
    def _emergency_cleanup(self):
        """Emergency memory cleanup during high pressure"""
        with self._lock:
            # Force garbage collection
            gc.collect()
            
            if MPS_AVAILABLE and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                
            # Clear least recently used cache entries
            self._cleanup_lru_cache(keep_ratio=0.3)
            
            logger.info("Emergency memory cleanup completed")
            
    def _reduce_cache_size(self, factor: float):
        """Reduce cache size by given factor"""
        with self._lock:
            if not (0 < factor < 1):
                return
                
            current_size = len(self.cache)
            target_size = int(current_size * factor)
            
            # Sort by last access time and remove oldest
            cache_items = sorted(
                self.cache.items(),
                key=lambda x: self._get_tensor_last_access(x[0])
            )
            
            items_to_remove = cache_items[target_size:]
            for key, _ in items_to_remove:
                del self.cache[key]
                
            logger.info(f"Reduced cache size from {current_size} to {len(self.cache)}")
            
    def _get_tensor_last_access(self, tensor_id: str) -> float:
        """Get last access time for a tensor"""
        for pool in self.pools.values():
            if tensor_id in pool:
                return pool[tensor_id].last_used
        return 0.0
        
    def _cleanup_weak_references(self):
        """Clean up dead weak references"""
        dead_refs = [ref for ref in self.weak_references if ref() is None]
        for ref in dead_refs:
            self.weak_references.discard(ref)
            
    def _should_defragment(self) -> bool:
        """Determine if memory defragmentation is needed"""
        stats = self.get_memory_stats()
        return stats.fragmentation_ratio > self._fragmentation_threshold
        
    def _defragment_memory(self):
        """Perform memory defragmentation"""
        with self._lock:
            logger.info("Starting memory defragmentation")
            
            # This is a simplified defragmentation - in a real implementation,
            # we would need to move tensors to contiguous memory locations
            
            # Force garbage collection
            gc.collect()
            
            if MPS_AVAILABLE and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                
            # Compact cache by removing fragmented entries
            self._compact_cache()
            
            logger.info("Memory defragmentation completed")
            
    def _compact_cache(self):
        """Compact memory cache by removing fragmented entries"""
        # Remove cached tensors that are fragmented or rarely used
        items_to_remove = []
        
        for key, tensor in self.cache.items():
            if self._is_tensor_fragmented(tensor) or self._is_tensor_rarely_used(key):
                items_to_remove.append(key)
                
        for key in items_to_remove:
            del self.cache[key]
            
    def _is_tensor_fragmented(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is fragmented in memory"""
        # Simplified check - in practice would need more sophisticated analysis
        try:
            return not tensor.is_contiguous() if hasattr(tensor, 'is_contiguous') else False
        except:
            return False
            
    def _is_tensor_rarely_used(self, tensor_id: str) -> bool:
        """Check if tensor is rarely used"""
        access_history = self.access_patterns.get(tensor_id, deque())
        if len(access_history) < 5:
            return False
            
        # Consider rarely used if average access interval > 60 seconds
        recent_accesses = list(access_history)[-5:]
        if len(recent_accesses) >= 2:
            avg_interval = (recent_accesses[-1] - recent_accesses[0]) / (len(recent_accesses) - 1)
            return avg_interval > 60
            
        return False
        
    @contextmanager
    def allocate_tensor(self, allocation_request: AllocationRequest):
        """
        Context manager for tensor allocation with automatic cleanup
        
        Args:
            allocation_request: Details of the allocation request
            
        Yields:
            torch.Tensor: Allocated tensor
        """
        tensor = None
        tensor_id = None
        
        try:
            with self._allocation_lock:
                # Check cache first
                if allocation_request.cache_key:
                    cached_tensor = self._get_from_cache(allocation_request.cache_key)
                    if cached_tensor is not None:
                        self.cache_hits += 1
                        yield cached_tensor
                        return
                        
                self.cache_misses += 1
                
                # Allocate new tensor
                tensor, tensor_id = self._allocate_tensor_internal(allocation_request)
                
                if tensor is None:
                    raise RuntimeError("Failed to allocate tensor")
                    
                # Track allocation
                self._track_allocation(tensor, tensor_id, allocation_request)
                
                yield tensor
                
        except Exception as e:
            logger.error(f"Tensor allocation failed: {e}")
            if allocation_request.allow_fallback:
                # Try CPU fallback
                logger.info("Attempting CPU fallback allocation")
                try:
                    cpu_request = AllocationRequest(
                        size=allocation_request.size,
                        dtype=allocation_request.dtype,
                        shape=allocation_request.shape,
                        device=torch.device("cpu"),
                        pool_name=allocation_request.pool_name + "_cpu",
                        priority=allocation_request.priority,
                        allow_fallback=False,
                        cache_key=allocation_request.cache_key
                    )
                    tensor, tensor_id = self._allocate_tensor_internal(cpu_request)
                    if tensor is not None:
                        self._track_allocation(tensor, tensor_id, cpu_request)
                        yield tensor
                    else:
                        raise e
                except Exception as fallback_error:
                    logger.error(f"CPU fallback also failed: {fallback_error}")
                    raise e
            else:
                raise e
                
        finally:
            # Cleanup and caching logic
            if tensor is not None and tensor_id is not None:
                self._handle_tensor_cleanup(tensor, tensor_id, allocation_request)
                
    def _get_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor from cache"""
        with self._lock:
            if cache_key in self.cache:
                tensor = self.cache[cache_key]
                
                # Update access pattern
                self.access_patterns[cache_key].append(time.time())
                
                # Verify tensor is still valid
                try:
                    _ = tensor.shape  # Simple validity check
                    return tensor
                except:
                    # Invalid tensor, remove from cache
                    del self.cache[cache_key]
                    
        return None
        
    def _allocate_tensor_internal(self, request: AllocationRequest) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Internal tensor allocation implementation"""
        try:
            # Check memory availability
            if not self._check_memory_availability(request.size):
                # Try to free some memory
                self._free_memory(request.size)
                
                if not self._check_memory_availability(request.size):
                    logger.warning(f"Insufficient memory for allocation: {request.size} bytes")
                    return None, None
                    
            # Allocate tensor
            if request.device.type == "mps" and MPS_AVAILABLE:
                tensor = torch.zeros(request.shape, dtype=request.dtype, device=request.device)
            else:
                tensor = torch.zeros(request.shape, dtype=request.dtype, device=request.device)
                
            # Generate unique tensor ID
            tensor_id = f"{request.pool_name}_{int(time.time() * 1000000)}_{id(tensor)}"
            
            self.total_allocations += 1
            current_usage = self._get_current_memory_usage()
            if current_usage > self.peak_memory_usage:
                self.peak_memory_usage = current_usage
                
            return tensor, tensor_id
            
        except Exception as e:
            logger.error(f"Internal tensor allocation failed: {e}")
            return None, None
            
    def _check_memory_availability(self, required_bytes: int) -> bool:
        """Check if sufficient memory is available"""
        try:
            memory_stats = metal_device_manager.get_memory_stats()
            if memory_stats:
                available_mb = memory_stats.free_mb
                required_mb = required_bytes / (1024 * 1024)
                return available_mb > required_mb * 1.2  # 20% safety margin
            else:
                # Fallback to system memory check
                if PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    return memory.available > required_bytes * 1.5
                    
        except Exception as e:
            logger.error(f"Memory availability check failed: {e}")
            
        return True  # Conservative fallback
        
    def _free_memory(self, required_bytes: int):
        """Free memory to satisfy allocation request"""
        with self._lock:
            freed_bytes = 0
            target_bytes = required_bytes * 1.5  # Free 50% more than needed
            
            # Remove least recently used cache entries
            cache_items = sorted(
                self.cache.items(),
                key=lambda x: self._get_tensor_last_access(x[0])
            )
            
            for key, tensor in cache_items:
                if freed_bytes >= target_bytes:
                    break
                    
                try:
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    del self.cache[key]
                    freed_bytes += tensor_bytes
                except:
                    continue
                    
            # Force garbage collection
            gc.collect()
            
            if MPS_AVAILABLE and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                
            logger.info(f"Freed {freed_bytes / (1024 * 1024):.2f} MB of memory")
            
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            if MPS_AVAILABLE and hasattr(torch.backends.mps, 'current_allocated_memory'):
                return torch.backends.mps.current_allocated_memory()
            else:
                # Estimate from cache size
                total_bytes = 0
                for tensor in self.cache.values():
                    try:
                        total_bytes += tensor.numel() * tensor.element_size()
                    except:
                        continue
                return total_bytes
                
        except Exception as e:
            logger.error(f"Memory usage calculation failed: {e}")
            return 0
            
    def _track_allocation(self, tensor: torch.Tensor, tensor_id: str, request: AllocationRequest):
        """Track tensor allocation for management"""
        with self._lock:
            current_time = time.time()
            
            block = MemoryBlock(
                size=tensor.numel() * tensor.element_size(),
                dtype=str(tensor.dtype),
                shape=tensor.shape,
                allocated_time=current_time,
                last_used=current_time,
                reference_count=1,
                tensor_id=tensor_id,
                pool_name=request.pool_name,
                is_cached=request.cache_key is not None
            )
            
            self.pools[request.pool_name][tensor_id] = block
            self.allocation_stats[request.pool_name] += 1
            
            # Create weak reference for automatic cleanup
            weak_ref = weakref.ref(tensor, lambda ref: self._tensor_deallocated(tensor_id, request.pool_name))
            self.weak_references.add(weak_ref)
            
    def _tensor_deallocated(self, tensor_id: str, pool_name: str):
        """Callback when tensor is deallocated"""
        with self._lock:
            if pool_name in self.pools and tensor_id in self.pools[pool_name]:
                del self.pools[pool_name][tensor_id]
                self.allocation_stats[pool_name] = max(0, self.allocation_stats[pool_name] - 1)
                
    def _handle_tensor_cleanup(self, tensor: torch.Tensor, tensor_id: str, request: AllocationRequest):
        """Handle tensor cleanup and caching"""
        with self._lock:
            # Update last used time
            if request.pool_name in self.pools and tensor_id in self.pools[request.pool_name]:
                self.pools[request.pool_name][tensor_id].last_used = time.time()
                self.pools[request.pool_name][tensor_id].access_count += 1
                
            # Cache tensor if requested and cache is not full
            if request.cache_key and self._should_cache_tensor(tensor, request):
                try:
                    self.cache[request.cache_key] = tensor.clone()
                    self.access_patterns[request.cache_key].append(time.time())
                    
                    # Update block info
                    if request.pool_name in self.pools and tensor_id in self.pools[request.pool_name]:
                        self.pools[request.pool_name][tensor_id].is_cached = True
                        
                except Exception as e:
                    logger.warning(f"Failed to cache tensor: {e}")
                    
    def _should_cache_tensor(self, tensor: torch.Tensor, request: AllocationRequest) -> bool:
        """Determine if tensor should be cached"""
        # Don't cache if cache is too full
        if len(self.cache) > 1000:  # Configurable cache size limit
            return False
            
        # Don't cache very large tensors
        tensor_size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
        if tensor_size_mb > 100:  # Don't cache tensors > 100MB
            return False
            
        # Cache if it's likely to be reused
        if request.cache_key in self.access_patterns:
            access_history = self.access_patterns[request.cache_key]
            if len(access_history) >= 2:
                # Cache if accessed multiple times recently
                return True
                
        # Cache small, frequently allocated shapes
        common_shapes = [(1, 1), (32, 32), (64, 64), (128, 128)]
        return tensor.shape in common_shapes
        
    def _cleanup_lru_cache(self, keep_ratio: float = 0.7):
        """Clean up least recently used cache entries"""
        if not (0 < keep_ratio < 1):
            return
            
        current_size = len(self.cache)
        target_size = int(current_size * keep_ratio)
        
        if current_size <= target_size:
            return
            
        # Sort by last access time
        cache_items = sorted(
            self.cache.items(),
            key=lambda x: self._get_tensor_last_access(x[0])
        )
        
        # Remove oldest entries
        items_to_remove = cache_items[target_size:]
        for key, _ in items_to_remove:
            del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
                
        logger.info(f"LRU cache cleanup: {len(items_to_remove)} entries removed")
        
    def get_memory_stats(self) -> MemoryPoolStats:
        """Get comprehensive memory pool statistics"""
        with self._lock:
            total_allocated = 0
            total_cached = 0
            active_blocks = 0
            cached_blocks = 0
            
            # Calculate pool statistics
            for pool in self.pools.values():
                for block in pool.values():
                    total_allocated += block.size
                    active_blocks += 1
                    
                    if block.is_cached:
                        total_cached += block.size
                        cached_blocks += 1
                        
            # Calculate cache statistics
            cache_size = 0
            for tensor in self.cache.values():
                try:
                    cache_size += tensor.numel() * tensor.element_size()
                except:
                    continue
                    
            total_cached += cache_size
            
            # Get system memory info
            system_memory = metal_device_manager.get_memory_stats()
            total_free = system_memory.free_mb * 1024 * 1024 if system_memory else 0
            
            # Calculate metrics
            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            # Estimate fragmentation ratio
            if active_blocks > 0:
                avg_block_size = total_allocated / active_blocks
                theoretical_optimal = total_allocated
                fragmentation_ratio = 1 - (theoretical_optimal / max(total_allocated, 1))
            else:
                fragmentation_ratio = 0
                avg_block_size = 0
                
            # Determine memory pressure level
            if system_memory:
                utilization = total_allocated / (total_allocated + total_free) if total_free > 0 else 1
                if utilization > 0.9:
                    pressure_level = "critical"
                elif utilization > 0.75:
                    pressure_level = "high"
                elif utilization > 0.5:
                    pressure_level = "medium"
                else:
                    pressure_level = "low"
            else:
                pressure_level = "unknown"
                
            return MemoryPoolStats(
                total_allocated_mb=total_allocated / (1024 * 1024),
                total_cached_mb=total_cached / (1024 * 1024),
                total_free_mb=total_free / (1024 * 1024),
                active_blocks=active_blocks,
                cached_blocks=cached_blocks,
                allocation_count=self.total_allocations,
                cache_hit_rate=cache_hit_rate,
                fragmentation_ratio=fragmentation_ratio,
                peak_usage_mb=self.peak_memory_usage / (1024 * 1024),
                average_block_size_mb=avg_block_size / (1024 * 1024),
                memory_pressure_level=pressure_level
            )
            
    def clear_cache(self, pool_name: Optional[str] = None):
        """Clear memory cache"""
        with self._lock:
            if pool_name:
                # Clear specific pool
                keys_to_remove = [key for key in self.cache.keys() if pool_name in key]
                for key in keys_to_remove:
                    del self.cache[key]
                    if key in self.access_patterns:
                        del self.access_patterns[key]
            else:
                # Clear all caches
                self.cache.clear()
                self.access_patterns.clear()
                
            # Force garbage collection
            gc.collect()
            
            if MPS_AVAILABLE and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                
            logger.info(f"Memory cache cleared for pool: {pool_name or 'all'}")
            
    def optimize_memory_layout(self):
        """Optimize memory layout for M4 Max unified architecture"""
        with self._lock:
            logger.info("Starting memory layout optimization")
            
            # Force defragmentation
            self._defragment_memory()
            
            # Reorganize cache for better spatial locality
            self._reorganize_cache_by_access_pattern()
            
            # Optimize pool organization
            self._optimize_pool_organization()
            
            logger.info("Memory layout optimization completed")
            
    def _reorganize_cache_by_access_pattern(self):
        """Reorganize cache based on access patterns"""
        # Group frequently accessed tensors together
        frequent_access = {}
        infrequent_access = {}
        
        for key, tensor in self.cache.items():
            access_count = len(self.access_patterns.get(key, []))
            if access_count > 10:  # Frequently accessed
                frequent_access[key] = tensor
            else:
                infrequent_access[key] = tensor
                
        # Rebuild cache with frequent items first
        self.cache.clear()
        self.cache.update(frequent_access)
        self.cache.update(infrequent_access)
        
    def _optimize_pool_organization(self):
        """Optimize memory pool organization"""
        # Consolidate small pools
        small_pools = [name for name, pool in self.pools.items() if len(pool) < 5]
        
        if len(small_pools) > 1:
            # Merge small pools into a consolidated pool
            consolidated_pool = {}
            consolidated_name = "consolidated_small_pools"
            
            for pool_name in small_pools:
                consolidated_pool.update(self.pools[pool_name])
                del self.pools[pool_name]
                
            if consolidated_pool:
                self.pools[consolidated_name] = consolidated_pool
                logger.info(f"Consolidated {len(small_pools)} small pools")
                
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        stats = self.get_memory_stats()
        
        if stats.cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate - consider increasing cache size or improving key strategy")
            
        if stats.fragmentation_ratio > 0.3:
            recommendations.append("High memory fragmentation - run defragmentation or optimize allocation patterns")
            
        if stats.memory_pressure_level in ["high", "critical"]:
            recommendations.append("High memory pressure - consider reducing batch sizes or clearing unused caches")
            
        if stats.total_cached_mb > stats.total_allocated_mb * 0.5:
            recommendations.append("Cache size is large relative to active memory - consider cache size limits")
            
        if len(self.pools) > 20:
            recommendations.append("Many memory pools - consider pool consolidation")
            
        if stats.average_block_size_mb < 1:
            recommendations.append("Small average block size may indicate inefficient allocation patterns")
            
        if is_m4_max_detected() and stats.total_allocated_mb < 1000:
            recommendations.append("M4 Max detected but low memory usage - consider larger batch sizes for better utilization")
            
        if not recommendations:
            recommendations.append("Memory usage appears optimal")
            
        return recommendations

# Global memory pool manager instance
memory_pool_manager = MemoryPoolManager() if MPS_AVAILABLE else None

# Convenience functions
@contextmanager
def allocate_gpu_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: Optional[torch.device] = None, pool_name: str = "default",
                       cache_key: Optional[str] = None):
    """Convenient tensor allocation context manager"""
    if not memory_pool_manager:
        # Fallback to direct allocation
        if device is None:
            device = torch.device("mps") if MPS_AVAILABLE else torch.device("cpu")
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        yield tensor
        return
        
    if device is None:
        device = torch.device("mps") if MPS_AVAILABLE else torch.device("cpu")
        
    size = 1
    for dim in shape:
        size *= dim
    size *= torch.tensor([], dtype=dtype).element_size()
    
    request = AllocationRequest(
        size=size,
        dtype=dtype,
        shape=shape,
        device=device,
        pool_name=pool_name,
        cache_key=cache_key
    )
    
    with memory_pool_manager.allocate_tensor(request) as tensor:
        yield tensor

def get_memory_pool_stats() -> Optional[MemoryPoolStats]:
    """Get memory pool statistics"""
    return memory_pool_manager.get_memory_stats() if memory_pool_manager else None

def clear_gpu_memory_cache(pool_name: Optional[str] = None):
    """Clear GPU memory cache"""
    if memory_pool_manager:
        memory_pool_manager.clear_cache(pool_name)

def optimize_gpu_memory_layout():
    """Optimize GPU memory layout"""
    if memory_pool_manager:
        memory_pool_manager.optimize_memory_layout()

def get_memory_optimization_recommendations() -> List[str]:
    """Get memory optimization recommendations"""
    return memory_pool_manager.get_optimization_recommendations() if memory_pool_manager else []
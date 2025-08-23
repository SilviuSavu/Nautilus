"""
Memory Pool Optimization System for Ultra-Performance Trading

Provides:
- Custom memory allocators for high-frequency objects
- Object pooling to minimize garbage collection
- Memory-mapped file optimizations
- Garbage collection optimization strategies
- Memory profiling and leak detection
"""

import asyncio
import logging
import mmap
import os
import sys
import threading
import time
import weakref
import gc
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type, TypeVar
from dataclasses import dataclass
from collections import defaultdict, deque
from contextlib import contextmanager
import struct
from enum import Enum

# Memory profiling imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MemoryPoolType(Enum):
    """Memory pool types for different allocation strategies"""
    STACK = "stack"           # Stack-based allocation (LIFO)
    RING = "ring"             # Ring buffer allocation
    SLAB = "slab"             # Slab allocation for fixed sizes
    BUDDY = "buddy"           # Buddy system for variable sizes
    OBJECT = "object"         # Object-specific pools

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_allocated_bytes: int = 0
    total_freed_bytes: int = 0
    current_usage_bytes: int = 0
    peak_usage_bytes: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0
    fragmentation_ratio: float = 0.0
    gc_collections: int = 0
    gc_time_ms: float = 0.0

@dataclass
class ObjectPoolStats:
    """Object pool usage statistics"""
    pool_name: str
    object_type: str
    pool_size: int
    allocated_objects: int
    free_objects: int
    total_allocations: int
    total_deallocations: int
    avg_allocation_time_us: float = 0.0
    pool_hit_ratio: float = 0.0

class CustomMemoryAllocator:
    """
    Custom memory allocator with different allocation strategies
    """
    
    def __init__(self, pool_type: MemoryPoolType = MemoryPoolType.SLAB, 
                 initial_size: int = 1024 * 1024):  # 1MB default
        self.pool_type = pool_type
        self.initial_size = initial_size
        self.memory_blocks: List[memoryview] = []
        self.free_blocks: Dict[int, List[memoryview]] = defaultdict(list)
        self.allocated_blocks: Dict[int, memoryview] = {}
        self.stats = MemoryStats()
        self._lock = threading.RLock()
        self._next_block_id = 0
        
        # Initialize memory pool
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize the memory pool based on allocation strategy"""
        with self._lock:
            if self.pool_type == MemoryPoolType.SLAB:
                self._initialize_slab_pool()
            elif self.pool_type == MemoryPoolType.BUDDY:
                self._initialize_buddy_pool()
            elif self.pool_type == MemoryPoolType.RING:
                self._initialize_ring_pool()
            else:
                self._initialize_stack_pool()
                
    def _initialize_slab_pool(self):
        """Initialize slab allocator for fixed-size blocks"""
        # Common object sizes in trading systems
        slab_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        for size in slab_sizes:
            num_blocks = self.initial_size // size
            memory_block = bytearray(size * num_blocks)
            
            # Create individual slab blocks
            for i in range(num_blocks):
                start_offset = i * size
                end_offset = (i + 1) * size
                block_view = memoryview(memory_block[start_offset:end_offset])
                self.free_blocks[size].append(block_view)
                
        logger.info(f"Initialized slab allocator with {len(slab_sizes)} slab sizes")
        
    def _initialize_buddy_pool(self):
        """Initialize buddy system allocator"""
        # Buddy system uses power-of-2 sizes
        total_memory = bytearray(self.initial_size)
        self.memory_blocks.append(memoryview(total_memory))
        
        # Start with the largest block
        largest_size = 1
        while largest_size < self.initial_size:
            largest_size *= 2
        largest_size //= 2
        
        self.free_blocks[largest_size].append(memoryview(total_memory))
        logger.info(f"Initialized buddy allocator with {largest_size} byte blocks")
        
    def _initialize_ring_pool(self):
        """Initialize ring buffer allocator"""
        total_memory = bytearray(self.initial_size)
        self.memory_blocks.append(memoryview(total_memory))
        self.ring_head = 0
        self.ring_tail = 0
        logger.info("Initialized ring buffer allocator")
        
    def _initialize_stack_pool(self):
        """Initialize stack-based allocator"""
        total_memory = bytearray(self.initial_size)
        self.memory_blocks.append(memoryview(total_memory))
        self.stack_top = 0
        logger.info("Initialized stack allocator")
        
    def allocate(self, size: int) -> Optional[memoryview]:
        """Allocate memory block of specified size"""
        with self._lock:
            start_time = time.perf_counter()
            
            block = None
            if self.pool_type == MemoryPoolType.SLAB:
                block = self._allocate_slab(size)
            elif self.pool_type == MemoryPoolType.BUDDY:
                block = self._allocate_buddy(size)
            elif self.pool_type == MemoryPoolType.RING:
                block = self._allocate_ring(size)
            else:
                block = self._allocate_stack(size)
                
            if block is not None:
                block_id = self._next_block_id
                self._next_block_id += 1
                self.allocated_blocks[block_id] = block
                
                # Update statistics
                self.stats.allocation_count += 1
                self.stats.total_allocated_bytes += size
                self.stats.current_usage_bytes += size
                self.stats.peak_usage_bytes = max(
                    self.stats.peak_usage_bytes,
                    self.stats.current_usage_bytes
                )
                
                allocation_time = (time.perf_counter() - start_time) * 1_000_000  # microseconds
                
                return block, block_id
                
            return None
            
    def _allocate_slab(self, size: int) -> Optional[memoryview]:
        """Allocate from slab pools"""
        # Find the smallest slab that can fit the size
        for slab_size in sorted(self.free_blocks.keys()):
            if slab_size >= size and self.free_blocks[slab_size]:
                return self.free_blocks[slab_size].pop()
                
        # Need to grow the pool
        return self._grow_slab_pool(size)
        
    def _allocate_buddy(self, size: int) -> Optional[memoryview]:
        """Allocate using buddy system"""
        # Round up to next power of 2
        buddy_size = 1
        while buddy_size < size:
            buddy_size *= 2
            
        # Find available block of required size
        if buddy_size in self.free_blocks and self.free_blocks[buddy_size]:
            return self.free_blocks[buddy_size].pop()
            
        # Try to split larger blocks
        return self._split_buddy_block(buddy_size)
        
    def _allocate_ring(self, size: int) -> Optional[memoryview]:
        """Allocate from ring buffer"""
        if len(self.memory_blocks) == 0:
            return None
            
        memory_block = self.memory_blocks[0]
        total_size = len(memory_block)
        
        # Check if we can fit the allocation
        if self.ring_head + size <= total_size:
            block = memory_block[self.ring_head:self.ring_head + size]
            self.ring_head += size
            return block
        elif size <= total_size:
            # Wrap around
            self.ring_head = 0
            self.ring_tail = 0
            block = memory_block[self.ring_head:self.ring_head + size]
            self.ring_head += size
            return block
            
        return None
        
    def _allocate_stack(self, size: int) -> Optional[memoryview]:
        """Allocate from stack"""
        if len(self.memory_blocks) == 0:
            return None
            
        memory_block = self.memory_blocks[0]
        
        if self.stack_top + size <= len(memory_block):
            block = memory_block[self.stack_top:self.stack_top + size]
            self.stack_top += size
            return block
            
        return None
        
    def _grow_slab_pool(self, size: int) -> Optional[memoryview]:
        """Grow slab pool when needed"""
        # Find appropriate slab size
        slab_size = 32
        while slab_size < size:
            slab_size *= 2
            
        # Allocate new slab
        new_slab_memory = bytearray(slab_size * 100)  # 100 blocks
        for i in range(100):
            start_offset = i * slab_size
            end_offset = (i + 1) * slab_size
            block_view = memoryview(new_slab_memory[start_offset:end_offset])
            self.free_blocks[slab_size].append(block_view)
            
        return self.free_blocks[slab_size].pop()
        
    def _split_buddy_block(self, target_size: int) -> Optional[memoryview]:
        """Split larger buddy blocks to create target size"""
        # Find the smallest available block larger than target
        for block_size in sorted(self.free_blocks.keys(), reverse=True):
            if block_size > target_size and self.free_blocks[block_size]:
                larger_block = self.free_blocks[block_size].pop()
                
                # Split the block
                half_size = block_size // 2
                left_half = larger_block[:half_size]
                right_half = larger_block[half_size:]
                
                if half_size == target_size:
                    # Perfect fit
                    self.free_blocks[half_size].append(right_half)
                    return left_half
                else:
                    # Continue splitting
                    self.free_blocks[half_size].extend([left_half, right_half])
                    return self._split_buddy_block(target_size)
                    
        return None
        
    def deallocate(self, block_id: int) -> bool:
        """Deallocate memory block"""
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False
                
            block = self.allocated_blocks.pop(block_id)
            block_size = len(block)
            
            # Return block to appropriate free list
            if self.pool_type == MemoryPoolType.SLAB:
                # Find matching slab size
                for slab_size in self.free_blocks.keys():
                    if slab_size >= block_size:
                        self.free_blocks[slab_size].append(block)
                        break
            elif self.pool_type == MemoryPoolType.BUDDY:
                self._merge_buddy_blocks(block, block_size)
            else:
                # For stack and ring, we can't easily deallocate individual blocks
                pass
                
            # Update statistics
            self.stats.deallocation_count += 1
            self.stats.total_freed_bytes += block_size
            self.stats.current_usage_bytes -= block_size
            
            return True
            
    def _merge_buddy_blocks(self, block: memoryview, block_size: int):
        """Merge buddy blocks to reduce fragmentation"""
        # Simple implementation - just add to free list
        # Full buddy merging would require more complex tracking
        self.free_blocks[block_size].append(block)
        
    def get_stats(self) -> MemoryStats:
        """Get memory allocator statistics"""
        with self._lock:
            # Calculate fragmentation ratio
            total_free = sum(len(blocks) * size for size, blocks in self.free_blocks.items())
            if self.stats.total_allocated_bytes > 0:
                self.stats.fragmentation_ratio = 1.0 - (total_free / self.initial_size)
                
            return self.stats

class ObjectPoolManager:
    """
    Object pooling manager for high-frequency trading objects
    """
    
    def __init__(self):
        self.pools: Dict[Type, 'ObjectPool'] = {}
        self.pool_stats: Dict[Type, ObjectPoolStats] = {}
        self._lock = threading.RLock()
        
    def create_pool(self, object_type: Type[T], pool_size: int = 1000,
                   factory_function: Optional[Callable[[], T]] = None) -> 'ObjectPool[T]':
        """Create object pool for specific type"""
        with self._lock:
            if object_type not in self.pools:
                pool = ObjectPool(object_type, pool_size, factory_function)
                self.pools[object_type] = pool
                self.pool_stats[object_type] = ObjectPoolStats(
                    pool_name=object_type.__name__,
                    object_type=str(object_type),
                    pool_size=pool_size,
                    allocated_objects=0,
                    free_objects=pool_size,
                    total_allocations=0,
                    total_deallocations=0
                )
                
            return self.pools[object_type]
            
    def get_object(self, object_type: Type[T]) -> Optional[T]:
        """Get object from pool"""
        if object_type in self.pools:
            start_time = time.perf_counter()
            obj = self.pools[object_type].get()
            allocation_time = (time.perf_counter() - start_time) * 1_000_000
            
            # Update statistics
            stats = self.pool_stats[object_type]
            stats.total_allocations += 1
            stats.allocated_objects += 1
            stats.free_objects -= 1
            
            # Update average allocation time (exponential moving average)
            alpha = 0.1
            stats.avg_allocation_time_us = (
                (1 - alpha) * stats.avg_allocation_time_us + 
                alpha * allocation_time
            )
            
            # Update hit ratio
            stats.pool_hit_ratio = (stats.total_allocations - stats.total_deallocations) / stats.total_allocations
            
            return obj
            
        return None
        
    def return_object(self, obj: T) -> bool:
        """Return object to pool"""
        object_type = type(obj)
        if object_type in self.pools:
            success = self.pools[object_type].return_object(obj)
            if success:
                stats = self.pool_stats[object_type]
                stats.total_deallocations += 1
                stats.allocated_objects -= 1
                stats.free_objects += 1
            return success
        return False
        
    def get_all_stats(self) -> Dict[str, ObjectPoolStats]:
        """Get statistics for all pools"""
        return {name: stats for name, stats in self.pool_stats.items()}

class ObjectPool:
    """
    Generic object pool implementation
    """
    
    def __init__(self, object_type: Type[T], pool_size: int = 1000,
                 factory_function: Optional[Callable[[], T]] = None):
        self.object_type = object_type
        self.pool_size = pool_size
        self.factory_function = factory_function or self._default_factory
        self.free_objects: deque = deque()
        self.allocated_objects: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()
        
        # Pre-populate pool
        self._populate_pool()
        
    def _default_factory(self) -> T:
        """Default factory function"""
        try:
            return self.object_type()
        except Exception as e:
            logger.error(f"Failed to create object of type {self.object_type}: {e}")
            return None
            
    def _populate_pool(self):
        """Pre-populate the object pool"""
        for _ in range(self.pool_size):
            obj = self.factory_function()
            if obj is not None:
                self.free_objects.append(obj)
                
    def get(self) -> Optional[T]:
        """Get object from pool"""
        with self._lock:
            if self.free_objects:
                obj = self.free_objects.popleft()
                self.allocated_objects.add(obj)
                
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                    
                return obj
            else:
                # Pool exhausted, create new object
                obj = self.factory_function()
                if obj is not None:
                    self.allocated_objects.add(obj)
                return obj
                
    def return_object(self, obj: T) -> bool:
        """Return object to pool"""
        with self._lock:
            if obj in self.allocated_objects:
                self.allocated_objects.remove(obj)
                
                # Only return to pool if not at capacity
                if len(self.free_objects) < self.pool_size:
                    self.free_objects.append(obj)
                    
                return True
                
        return False
        
    def clear(self):
        """Clear the entire pool"""
        with self._lock:
            self.free_objects.clear()
            self.allocated_objects.clear()

class GCOptimizer:
    """
    Garbage collection optimizer for reduced latency spikes
    """
    
    def __init__(self):
        self.gc_stats = {"collections": 0, "total_time": 0.0}
        self.gc_thresholds = gc.get_threshold()
        self.adaptive_gc = True
        self._lock = threading.RLock()
        
    def optimize_gc_settings(self):
        """Optimize garbage collection settings for trading workloads"""
        # Increase thresholds to reduce GC frequency
        gc.set_threshold(2000, 20, 20)  # More conservative thresholds
        
        # Disable automatic GC for generation 2 (most expensive)
        gc.disable()
        
        logger.info("Optimized GC settings for low-latency trading")
        
    def manual_gc_cycle(self) -> Dict[str, float]:
        """Perform manual GC cycle and return timing information"""
        start_time = time.perf_counter()
        
        # Collect each generation separately for timing
        gen0_collected = gc.collect(0)
        gen0_time = time.perf_counter() - start_time
        
        gen1_start = time.perf_counter()
        gen1_collected = gc.collect(1)
        gen1_time = time.perf_counter() - gen1_start
        
        gen2_start = time.perf_counter()
        gen2_collected = gc.collect(2)
        gen2_time = time.perf_counter() - gen2_start
        
        total_time = time.perf_counter() - start_time
        
        with self._lock:
            self.gc_stats["collections"] += 1
            self.gc_stats["total_time"] += total_time
            
        return {
            "generation_0": {"collected": gen0_collected, "time_ms": gen0_time * 1000},
            "generation_1": {"collected": gen1_collected, "time_ms": gen1_time * 1000},
            "generation_2": {"collected": gen2_collected, "time_ms": gen2_time * 1000},
            "total_time_ms": total_time * 1000
        }
        
    async def adaptive_gc_scheduler(self, target_latency_ms: float = 1.0):
        """Adaptive GC scheduling to meet latency targets"""
        while self.adaptive_gc:
            try:
                # Monitor system conditions
                if self._should_run_gc():
                    gc_results = self.manual_gc_cycle()
                    
                    if gc_results["total_time_ms"] > target_latency_ms:
                        logger.warning(f"GC took {gc_results['total_time_ms']:.2f}ms, "
                                     f"exceeding target of {target_latency_ms}ms")
                        
                # Wait before next check
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Adaptive GC scheduler error: {e}")
                await asyncio.sleep(1.0)
                
    def _should_run_gc(self) -> bool:
        """Determine if GC should run based on memory pressure"""
        if not PSUTIL_AVAILABLE:
            return True  # Conservative approach
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Run GC if memory usage is high
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            return memory_usage_mb > 500  # Run GC if using > 500MB
            
        except Exception:
            return True
            
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get GC statistics"""
        with self._lock:
            return {
                "manual_collections": self.gc_stats["collections"],
                "total_gc_time_ms": self.gc_stats["total_time"] * 1000,
                "avg_gc_time_ms": (self.gc_stats["total_time"] / max(1, self.gc_stats["collections"])) * 1000,
                "current_thresholds": gc.get_threshold(),
                "gc_counts": gc.get_count()
            }

class MemoryMappedFileManager:
    """
    Memory-mapped file manager for efficient large data access
    """
    
    def __init__(self):
        self.mapped_files: Dict[str, mmap.mmap] = {}
        self.file_handles: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    def create_mapped_file(self, file_path: str, size: int, 
                          access_mode: int = mmap.ACCESS_WRITE) -> mmap.mmap:
        """Create memory-mapped file"""
        with self._lock:
            if file_path in self.mapped_files:
                return self.mapped_files[file_path]
                
            # Create or open file
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(b'\x00' * size)  # Pre-allocate space
                    
            file_handle = open(file_path, 'r+b')
            mapped_file = mmap.mmap(file_handle.fileno(), size, access=access_mode)
            
            self.mapped_files[file_path] = mapped_file
            self.file_handles[file_path] = file_handle
            
            logger.info(f"Created memory-mapped file: {file_path} ({size} bytes)")
            return mapped_file
            
    def get_mapped_file(self, file_path: str) -> Optional[mmap.mmap]:
        """Get existing memory-mapped file"""
        return self.mapped_files.get(file_path)
        
    def close_mapped_file(self, file_path: str):
        """Close memory-mapped file"""
        with self._lock:
            if file_path in self.mapped_files:
                self.mapped_files[file_path].close()
                self.file_handles[file_path].close()
                
                del self.mapped_files[file_path]
                del self.file_handles[file_path]
                
                logger.info(f"Closed memory-mapped file: {file_path}")
                
    def close_all_files(self):
        """Close all memory-mapped files"""
        with self._lock:
            for file_path in list(self.mapped_files.keys()):
                self.close_mapped_file(file_path)

class MemoryProfiler:
    """
    Memory profiling and leak detection
    """
    
    def __init__(self):
        self.snapshots: List[Any] = []
        self.tracking_enabled = False
        self.leak_threshold_mb = 100  # MB
        
    def start_tracking(self):
        """Start memory tracking"""
        if TRACEMALLOC_AVAILABLE:
            tracemalloc.start()
            self.tracking_enabled = True
            logger.info("Started memory tracking")
        else:
            logger.warning("tracemalloc not available - memory tracking disabled")
            
    def stop_tracking(self):
        """Stop memory tracking"""
        if TRACEMALLOC_AVAILABLE and self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            logger.info("Stopped memory tracking")
            
    def take_snapshot(self) -> Dict[str, Any]:
        """Take memory snapshot"""
        if not TRACEMALLOC_AVAILABLE or not self.tracking_enabled:
            return {"error": "Memory tracking not available"}
            
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)
        
        # Get top memory consumers
        top_stats = snapshot.statistics('lineno')[:10]
        
        return {
            "timestamp": time.time(),
            "total_traces": len(snapshot.traces),
            "top_consumers": [
                {
                    "file": stat.traceback.format()[-1],
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count
                }
                for stat in top_stats
            ]
        }
        
    def detect_leaks(self) -> Dict[str, Any]:
        """Detect memory leaks by comparing snapshots"""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots for leak detection"}
            
        current = self.snapshots[-1]
        previous = self.snapshots[-2]
        
        # Compare snapshots
        top_stats = current.compare_to(previous, 'lineno')[:10]
        
        leaks_detected = []
        for stat in top_stats:
            size_diff_mb = stat.size_diff / (1024 * 1024)
            if size_diff_mb > self.leak_threshold_mb:
                leaks_detected.append({
                    "file": stat.traceback.format()[-1],
                    "size_diff_mb": size_diff_mb,
                    "count_diff": stat.count_diff
                })
                
        return {
            "leaks_detected": len(leaks_detected) > 0,
            "potential_leaks": leaks_detected,
            "total_size_diff_mb": sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
        }

# Global instances
custom_memory_allocator = CustomMemoryAllocator(MemoryPoolType.SLAB)
object_pool_manager = ObjectPoolManager()
gc_optimizer = GCOptimizer()
memory_mapped_file_manager = MemoryMappedFileManager()
memory_profiler = MemoryProfiler()

# Context managers and decorators
@contextmanager
def pooled_object(object_type: Type[T]):
    """Context manager for automatic object pool management"""
    obj = object_pool_manager.get_object(object_type)
    try:
        yield obj
    finally:
        if obj is not None:
            object_pool_manager.return_object(obj)
            
def memory_optimized(func: Callable) -> Callable:
    """Decorator for memory-optimized function execution"""
    def wrapper(*args, **kwargs):
        # Disable GC during function execution for consistent performance
        gc_was_enabled = gc.isenabled()
        gc.disable()
        
        try:
            return func(*args, **kwargs)
        finally:
            if gc_was_enabled:
                gc.enable()
                
    return wrapper

# Convenience functions
def allocate_aligned_memory(size: int, alignment: int = 64) -> Optional[memoryview]:
    """Allocate cache-aligned memory"""
    aligned_size = ((size + alignment - 1) // alignment) * alignment
    return custom_memory_allocator.allocate(aligned_size)

async def optimize_memory_for_trading():
    """Optimize system memory settings for trading"""
    gc_optimizer.optimize_gc_settings()
    
    # Start adaptive GC scheduler
    asyncio.create_task(gc_optimizer.adaptive_gc_scheduler())
    
    # Start memory profiling if available
    memory_profiler.start_tracking()
    
    logger.info("Memory optimizations applied for trading workloads")
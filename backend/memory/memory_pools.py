"""
Memory Pool System for M4 Max Unified Architecture

Implements specialized memory pools for different workload types with
pre-allocated buffers, smart garbage collection, and trading-aware scheduling.

Key Features:
- Specialized pools for trading data, ML models, analytics
- Pre-allocated buffers for ultra-low latency operations
- Smart garbage collection with trading-aware scheduling
- Memory defragmentation and optimization
- Pool-specific allocation strategies
"""

import asyncio
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
import heapq
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .unified_memory_manager import (
    MemoryWorkloadType, 
    MemoryRegion, 
    MemoryBlock,
    get_unified_memory_manager
)


class PoolStrategy(Enum):
    """Memory pool allocation strategies"""
    FIRST_FIT = "first_fit"           # Fast allocation, may fragment
    BEST_FIT = "best_fit"             # Minimize waste, slower allocation
    BUDDY_SYSTEM = "buddy_system"     # Power-of-2 allocations, good for reuse
    SLAB_ALLOCATOR = "slab_allocator" # Fixed-size objects, very fast
    STACK_ALLOCATOR = "stack_allocator" # LIFO allocation, no fragmentation


class PoolPriority(Enum):
    """Memory pool priority levels"""
    CRITICAL = 1      # Trading data, risk calculations
    HIGH = 2          # Real-time analytics, websockets
    NORMAL = 3        # ML models, general analytics
    LOW = 4           # Historical data, background tasks
    BACKGROUND = 5    # Cleanup, maintenance tasks


@dataclass
class PoolConfig:
    """Configuration for a memory pool"""
    name: str
    workload_type: MemoryWorkloadType
    initial_size: int
    max_size: int
    block_size: int
    strategy: PoolStrategy
    priority: PoolPriority
    prealloc_blocks: int = 0
    auto_grow: bool = True
    auto_shrink: bool = True
    gc_threshold: float = 0.8
    defrag_threshold: float = 0.3
    region_preference: Optional[MemoryRegion] = None
    alignment: int = 64


@dataclass
class PoolBlock:
    """Memory block within a pool"""
    address: int
    size: int
    is_free: bool = True
    allocated_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    next_block: Optional[int] = None
    prev_block: Optional[int] = None


@dataclass
class PoolStatistics:
    """Statistics for a memory pool"""
    total_size: int
    used_size: int
    free_size: int
    block_count: int
    free_blocks: int
    allocation_count: int
    deallocation_count: int
    hit_rate: float
    fragmentation_ratio: float
    avg_allocation_time: float
    peak_usage: int
    gc_runs: int
    defrag_runs: int


class MemoryPool:
    """
    Specialized memory pool for specific workload types
    
    Optimized for M4 Max unified memory architecture with
    workload-specific allocation strategies.
    """
    
    def __init__(self, config: PoolConfig):
        self.config = config
        self.name = config.name
        self.workload_type = config.workload_type
        
        # Memory management
        self.base_address: Optional[int] = None
        self.pool_size = config.initial_size
        self.blocks: Dict[int, PoolBlock] = {}
        self.free_blocks: Set[int] = set()
        self.free_list: List[Tuple[int, int]] = []  # (address, size) sorted by address
        
        # Strategy-specific data structures
        self.buddy_tree: Dict[int, Set[int]] = {}  # size -> set of free blocks
        self.slab_cache: deque = deque()           # Pre-allocated objects
        self.stack_top: int = 0                    # Stack allocator pointer
        
        # Statistics and monitoring
        self.stats = PoolStatistics(
            total_size=0, used_size=0, free_size=0, block_count=0,
            free_blocks=0, allocation_count=0, deallocation_count=0,
            hit_rate=0.0, fragmentation_ratio=0.0, avg_allocation_time=0.0,
            peak_usage=0, gc_runs=0, defrag_runs=0
        )
        
        # Performance tracking
        self.allocation_times: deque = deque(maxlen=1000)
        self.recent_allocations: deque = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background services
        self.gc_thread = threading.Thread(target=self._gc_service, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_service, daemon=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Initialize pool
        self._initialize_pool()
        
        # Start background services
        self.gc_thread.start()
        self.monitor_thread.start()
    
    def allocate(self, size: int, alignment: Optional[int] = None) -> Optional[int]:
        """
        Allocate memory from the pool
        
        Args:
            size: Size in bytes to allocate
            alignment: Memory alignment requirement
        
        Returns:
            Memory address or None if allocation failed
        """
        start_time = time.time()
        
        with self.lock:
            # Use pool-specific alignment or default
            align = alignment or self.config.alignment
            aligned_size = (size + align - 1) & ~(align - 1)
            
            # Try pool-specific allocation strategy
            address = self._allocate_strategy(aligned_size, align)
            
            if address is None and self.config.auto_grow:
                # Try to grow pool
                if self._grow_pool(aligned_size * 2):
                    address = self._allocate_strategy(aligned_size, align)
            
            if address is not None:
                # Update statistics
                self.stats.allocation_count += 1
                self.stats.used_size += aligned_size
                self.stats.free_size -= aligned_size
                
                # Create and register block
                block = PoolBlock(
                    address=address,
                    size=aligned_size,
                    is_free=False,
                    allocated_at=start_time,
                    last_accessed=start_time
                )
                self.blocks[address] = block
                
                # Track allocation time
                allocation_time = time.time() - start_time
                self.allocation_times.append(allocation_time)
                self.recent_allocations.append((address, size, start_time))
                
                # Update peak usage
                if self.stats.used_size > self.stats.peak_usage:
                    self.stats.peak_usage = self.stats.used_size
                
                self.logger.debug(f"Allocated {aligned_size} bytes at {hex(address)} in {allocation_time:.6f}s")
        
        return address
    
    def deallocate(self, address: int) -> bool:
        """
        Deallocate memory back to the pool
        
        Args:
            address: Address to deallocate
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if address not in self.blocks:
                return False
            
            block = self.blocks[address]
            if block.is_free:
                return False
            
            # Mark as free
            block.is_free = True
            self.stats.deallocation_count += 1
            self.stats.used_size -= block.size
            self.stats.free_size += block.size
            
            # Return to strategy-specific free list
            self._deallocate_strategy(address, block.size)
            
            self.logger.debug(f"Deallocated {block.size} bytes at {hex(address)}")
            return True
    
    def get_statistics(self) -> PoolStatistics:
        """Get current pool statistics"""
        with self.lock:
            # Update calculated statistics
            if self.allocation_times:
                self.stats.avg_allocation_time = sum(self.allocation_times) / len(self.allocation_times)
            
            total_allocs = self.stats.allocation_count
            if total_allocs > 0:
                successful_allocs = len([a for a in self.recent_allocations if a[0] is not None])
                self.stats.hit_rate = successful_allocs / min(total_allocs, len(self.recent_allocations))
            
            # Calculate fragmentation
            if self.stats.free_size > 0:
                self.stats.fragmentation_ratio = len(self.free_list) / max(1, self.stats.free_size // self.config.block_size)
            
            # Update counts
            self.stats.block_count = len(self.blocks)
            self.stats.free_blocks = len(self.free_blocks)
            
            return self.stats
    
    def defragment(self) -> int:
        """
        Defragment the pool by coalescing free blocks
        
        Returns:
            Number of blocks coalesced
        """
        with self.lock:
            start_time = time.time()
            coalesced = 0
            
            # Sort free list by address
            self.free_list.sort()
            
            # Coalesce adjacent free blocks
            new_free_list = []
            i = 0
            while i < len(self.free_list):
                addr, size = self.free_list[i]
                
                # Look for adjacent blocks
                while i + 1 < len(self.free_list):
                    next_addr, next_size = self.free_list[i + 1]
                    if addr + size == next_addr:
                        # Adjacent blocks - coalesce
                        size += next_size
                        coalesced += 1
                        i += 1
                    else:
                        break
                
                new_free_list.append((addr, size))
                i += 1
            
            self.free_list = new_free_list
            self.stats.defrag_runs += 1
            
            duration = time.time() - start_time
            self.logger.info(f"Defragmented pool {self.name}: coalesced {coalesced} blocks in {duration:.3f}s")
            
            return coalesced
    
    def force_gc(self) -> int:
        """
        Force garbage collection on the pool
        
        Returns:
            Number of blocks freed
        """
        with self.lock:
            freed = 0
            current_time = time.time()
            
            # Find blocks eligible for cleanup
            cleanup_candidates = []
            for address, block in self.blocks.items():
                if block.is_free:
                    continue
                
                # Age-based cleanup for non-critical workloads
                age = current_time - block.last_accessed
                if age > self._get_gc_threshold():
                    cleanup_candidates.append(address)
            
            # Free old blocks
            for address in cleanup_candidates:
                if self.deallocate(address):
                    freed += 1
            
            self.stats.gc_runs += 1
            self.logger.info(f"GC freed {freed} blocks from pool {self.name}")
            
            return freed
    
    def shrink(self) -> int:
        """
        Shrink the pool by releasing unused memory
        
        Returns:
            Bytes released
        """
        with self.lock:
            if not self.config.auto_shrink:
                return 0
            
            # Calculate shrink amount
            target_size = max(
                self.config.initial_size,
                self.stats.used_size + self.config.block_size * 10
            )
            
            if self.pool_size <= target_size:
                return 0
            
            shrink_amount = self.pool_size - target_size
            
            # Release memory back to unified manager
            if self._shrink_pool(shrink_amount):
                self.logger.info(f"Shrunk pool {self.name} by {shrink_amount} bytes")
                return shrink_amount
            
            return 0
    
    # Private methods
    
    def _initialize_pool(self):
        """Initialize the memory pool"""
        # Allocate initial pool from unified memory manager
        manager = get_unified_memory_manager()
        self.base_address = manager.allocate(
            size=self.config.initial_size,
            workload_type=self.config.workload_type,
            prefer_region=self.config.region_preference,
            alignment=self.config.alignment
        )
        
        if self.base_address is None:
            raise MemoryError(f"Failed to allocate initial pool for {self.name}")
        
        # Initialize strategy-specific structures
        self._initialize_strategy()
        
        # Pre-allocate blocks if configured
        if self.config.prealloc_blocks > 0:
            self._preallocate_blocks()
        
        # Update statistics
        self.stats.total_size = self.config.initial_size
        self.stats.free_size = self.config.initial_size
        
        self.logger.info(f"Initialized pool {self.name} with {self.config.initial_size} bytes")
    
    def _initialize_strategy(self):
        """Initialize strategy-specific data structures"""
        if self.config.strategy == PoolStrategy.BUDDY_SYSTEM:
            # Initialize buddy system with power-of-2 sizes
            size = self.config.block_size
            while size <= self.pool_size:
                self.buddy_tree[size] = set()
                size *= 2
            
            # Add initial free block
            self.buddy_tree[self.pool_size].add(self.base_address)
        
        elif self.config.strategy == PoolStrategy.SLAB_ALLOCATOR:
            # Pre-allocate slab objects
            for _ in range(self.config.prealloc_blocks):
                self.slab_cache.append(self.base_address + len(self.slab_cache) * self.config.block_size)
        
        elif self.config.strategy == PoolStrategy.STACK_ALLOCATOR:
            # Initialize stack pointer
            self.stack_top = self.base_address
        
        else:  # FIRST_FIT or BEST_FIT
            # Add entire pool as single free block
            self.free_list.append((self.base_address, self.pool_size))
    
    def _allocate_strategy(self, size: int, alignment: int) -> Optional[int]:
        """Allocate using pool-specific strategy"""
        if self.config.strategy == PoolStrategy.FIRST_FIT:
            return self._allocate_first_fit(size, alignment)
        
        elif self.config.strategy == PoolStrategy.BEST_FIT:
            return self._allocate_best_fit(size, alignment)
        
        elif self.config.strategy == PoolStrategy.BUDDY_SYSTEM:
            return self._allocate_buddy(size)
        
        elif self.config.strategy == PoolStrategy.SLAB_ALLOCATOR:
            return self._allocate_slab()
        
        elif self.config.strategy == PoolStrategy.STACK_ALLOCATOR:
            return self._allocate_stack(size, alignment)
        
        return None
    
    def _allocate_first_fit(self, size: int, alignment: int) -> Optional[int]:
        """First-fit allocation strategy"""
        for i, (addr, block_size) in enumerate(self.free_list):
            # Check alignment
            aligned_addr = (addr + alignment - 1) & ~(alignment - 1)
            aligned_size = size + (aligned_addr - addr)
            
            if block_size >= aligned_size:
                # Use this block
                self.free_list.pop(i)
                
                # Split block if necessary
                if block_size > aligned_size + alignment:
                    remaining_addr = aligned_addr + size
                    remaining_size = block_size - aligned_size
                    self.free_list.insert(i, (remaining_addr, remaining_size))
                
                return aligned_addr
        
        return None
    
    def _allocate_best_fit(self, size: int, alignment: int) -> Optional[int]:
        """Best-fit allocation strategy"""
        best_index = -1
        best_size = float('inf')
        
        for i, (addr, block_size) in enumerate(self.free_list):
            # Check alignment
            aligned_addr = (addr + alignment - 1) & ~(alignment - 1)
            aligned_size = size + (aligned_addr - addr)
            
            if block_size >= aligned_size and block_size < best_size:
                best_index = i
                best_size = block_size
        
        if best_index >= 0:
            addr, block_size = self.free_list.pop(best_index)
            aligned_addr = (addr + alignment - 1) & ~(alignment - 1)
            aligned_size = size + (aligned_addr - addr)
            
            # Split block if necessary
            if block_size > aligned_size + alignment:
                remaining_addr = aligned_addr + size
                remaining_size = block_size - aligned_size
                self.free_list.append((remaining_addr, remaining_size))
            
            return aligned_addr
        
        return None
    
    def _allocate_buddy(self, size: int) -> Optional[int]:
        """Buddy system allocation"""
        # Find smallest power of 2 that fits the size
        buddy_size = self.config.block_size
        while buddy_size < size:
            buddy_size *= 2
        
        # Look for available block of this size
        while buddy_size <= self.pool_size:
            if buddy_size in self.buddy_tree and self.buddy_tree[buddy_size]:
                addr = self.buddy_tree[buddy_size].pop()
                return addr
            
            buddy_size *= 2
        
        # Try to split a larger block
        return self._split_buddy_block(size)
    
    def _allocate_slab(self) -> Optional[int]:
        """Slab allocator - fixed-size objects"""
        if self.slab_cache:
            return self.slab_cache.popleft()
        
        # Allocate new slab if possible
        if len(self.blocks) * self.config.block_size < self.pool_size:
            addr = self.base_address + len(self.blocks) * self.config.block_size
            return addr
        
        return None
    
    def _allocate_stack(self, size: int, alignment: int) -> Optional[int]:
        """Stack allocator - LIFO allocation"""
        aligned_top = (self.stack_top + alignment - 1) & ~(alignment - 1)
        
        if aligned_top + size <= self.base_address + self.pool_size:
            addr = aligned_top
            self.stack_top = aligned_top + size
            return addr
        
        return None
    
    def _deallocate_strategy(self, address: int, size: int):
        """Deallocate using pool-specific strategy"""
        if self.config.strategy == PoolStrategy.BUDDY_SYSTEM:
            self._deallocate_buddy(address, size)
        elif self.config.strategy == PoolStrategy.SLAB_ALLOCATOR:
            self.slab_cache.append(address)
        elif self.config.strategy == PoolStrategy.STACK_ALLOCATOR:
            # Stack deallocation only works if it's the top allocation
            if address + size == self.stack_top:
                self.stack_top = address
        else:
            # Add to free list for first-fit/best-fit
            self.free_list.append((address, size))
    
    def _split_buddy_block(self, size: int) -> Optional[int]:
        """Split buddy block to create smaller block"""
        # Find a larger block to split
        for buddy_size in sorted(self.buddy_tree.keys(), reverse=True):
            if buddy_size > size and self.buddy_tree[buddy_size]:
                addr = self.buddy_tree[buddy_size].pop()
                
                # Split the block
                half_size = buddy_size // 2
                self.buddy_tree[half_size].add(addr)
                self.buddy_tree[half_size].add(addr + half_size)
                
                # Recursively allocate from the split
                return self._allocate_buddy(size)
        
        return None
    
    def _deallocate_buddy(self, address: int, size: int):
        """Deallocate buddy block and coalesce if possible"""
        buddy_size = self.config.block_size
        while buddy_size < size:
            buddy_size *= 2
        
        # Find buddy address
        buddy_addr = address ^ buddy_size
        
        # Check if buddy is free
        if buddy_addr in self.buddy_tree.get(buddy_size, set()):
            # Coalesce with buddy
            self.buddy_tree[buddy_size].remove(buddy_addr)
            coalesced_addr = min(address, buddy_addr)
            self.buddy_tree[buddy_size * 2].add(coalesced_addr)
            
            # Recursively coalesce larger blocks
            self._deallocate_buddy(coalesced_addr, buddy_size * 2)
        else:
            # Add to free list
            self.buddy_tree[buddy_size].add(address)
    
    def _grow_pool(self, additional_size: int) -> bool:
        """Grow the pool by allocating more memory"""
        if self.pool_size + additional_size > self.config.max_size:
            return False
        
        manager = get_unified_memory_manager()
        new_base = manager.allocate(
            size=additional_size,
            workload_type=self.config.workload_type,
            prefer_region=self.config.region_preference,
            alignment=self.config.alignment
        )
        
        if new_base is None:
            return False
        
        # Add new space to free list
        self.free_list.append((new_base, additional_size))
        self.pool_size += additional_size
        self.stats.total_size += additional_size
        self.stats.free_size += additional_size
        
        return True
    
    def _shrink_pool(self, shrink_size: int) -> bool:
        """Shrink the pool by releasing memory"""
        # Find free blocks at the end of the pool to release
        # This is simplified - real implementation would be more complex
        self.pool_size -= shrink_size
        self.stats.total_size -= shrink_size
        self.stats.free_size -= shrink_size
        return True
    
    def _get_gc_threshold(self) -> float:
        """Get garbage collection threshold based on priority"""
        thresholds = {
            PoolPriority.CRITICAL: 300,      # 5 minutes
            PoolPriority.HIGH: 600,          # 10 minutes
            PoolPriority.NORMAL: 1800,       # 30 minutes
            PoolPriority.LOW: 3600,          # 1 hour
            PoolPriority.BACKGROUND: 7200,   # 2 hours
        }
        return thresholds.get(self.config.priority, 1800)
    
    def _gc_service(self):
        """Background garbage collection service"""
        while True:
            try:
                time.sleep(30)  # Run every 30 seconds
                
                # Check if GC is needed
                if self.stats.free_size / max(1, self.stats.total_size) < 1 - self.config.gc_threshold:
                    freed = self.force_gc()
                    if freed > 0:
                        self.logger.debug(f"GC service freed {freed} blocks")
                
                # Check fragmentation
                if self.get_statistics().fragmentation_ratio > self.config.defrag_threshold:
                    coalesced = self.defragment()
                    if coalesced > 0:
                        self.logger.debug(f"Defrag service coalesced {coalesced} blocks")
                
            except Exception as e:
                self.logger.error(f"GC service error in pool {self.name}: {e}")
    
    def _monitor_service(self):
        """Background monitoring service"""
        while True:
            try:
                time.sleep(60)  # Monitor every minute
                
                stats = self.get_statistics()
                
                # Log performance metrics
                self.logger.info(
                    f"Pool {self.name}: "
                    f"Used={stats.used_size/1024/1024:.1f}MB "
                    f"Free={stats.free_size/1024/1024:.1f}MB "
                    f"Frag={stats.fragmentation_ratio:.2f} "
                    f"Hit={stats.hit_rate:.2f} "
                    f"AvgAlloc={stats.avg_allocation_time*1000:.2f}ms"
                )
                
                # Auto-shrink if configured
                if self.config.auto_shrink and stats.free_size > stats.used_size * 2:
                    self.shrink()
                
            except Exception as e:
                self.logger.error(f"Monitor service error in pool {self.name}: {e}")


class MemoryPoolManager:
    """
    Manages multiple specialized memory pools
    
    Coordinates allocation across pools and handles pool lifecycle.
    """
    
    def __init__(self):
        self.pools: Dict[str, MemoryPool] = {}
        self.workload_pools: Dict[MemoryWorkloadType, str] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default pools
        self._initialize_default_pools()
    
    def create_pool(self, config: PoolConfig) -> MemoryPool:
        """Create a new memory pool"""
        with self.lock:
            if config.name in self.pools:
                raise ValueError(f"Pool {config.name} already exists")
            
            pool = MemoryPool(config)
            self.pools[config.name] = pool
            self.workload_pools[config.workload_type] = config.name
            
            self.logger.info(f"Created pool {config.name} for {config.workload_type}")
            return pool
    
    def get_pool(self, name: str) -> Optional[MemoryPool]:
        """Get pool by name"""
        return self.pools.get(name)
    
    def get_pool_for_workload(self, workload_type: MemoryWorkloadType) -> Optional[MemoryPool]:
        """Get pool for specific workload type"""
        pool_name = self.workload_pools.get(workload_type)
        return self.pools.get(pool_name) if pool_name else None
    
    def allocate_from_workload(
        self,
        workload_type: MemoryWorkloadType,
        size: int,
        alignment: Optional[int] = None
    ) -> Optional[int]:
        """Allocate memory for specific workload type"""
        pool = self.get_pool_for_workload(workload_type)
        return pool.allocate(size, alignment) if pool else None
    
    def get_global_statistics(self) -> Dict[str, PoolStatistics]:
        """Get statistics for all pools"""
        return {name: pool.get_statistics() for name, pool in self.pools.items()}
    
    def defragment_all(self) -> int:
        """Defragment all pools"""
        total_coalesced = 0
        for pool in self.pools.values():
            total_coalesced += pool.defragment()
        return total_coalesced
    
    def force_gc_all(self) -> int:
        """Force garbage collection on all pools"""
        total_freed = 0
        for pool in self.pools.values():
            total_freed += pool.force_gc()
        return total_freed
    
    def _initialize_default_pools(self):
        """Initialize default pools for common workload types"""
        default_configs = [
            PoolConfig(
                name="trading_data_pool",
                workload_type=MemoryWorkloadType.TRADING_DATA,
                initial_size=64 * 1024 * 1024,  # 64MB
                max_size=512 * 1024 * 1024,     # 512MB
                block_size=4096,
                strategy=PoolStrategy.SLAB_ALLOCATOR,
                priority=PoolPriority.CRITICAL,
                prealloc_blocks=1000,
                region_preference=MemoryRegion.CPU_CACHE_FRIENDLY,
                alignment=64
            ),
            PoolConfig(
                name="ml_models_pool",
                workload_type=MemoryWorkloadType.ML_MODELS,
                initial_size=128 * 1024 * 1024,  # 128MB
                max_size=2 * 1024 * 1024 * 1024, # 2GB
                block_size=1024 * 1024,          # 1MB blocks
                strategy=PoolStrategy.BUDDY_SYSTEM,
                priority=PoolPriority.NORMAL,
                region_preference=MemoryRegion.NEURAL_ENGINE,
                alignment=256
            ),
            PoolConfig(
                name="analytics_pool",
                workload_type=MemoryWorkloadType.ANALYTICS,
                initial_size=256 * 1024 * 1024,  # 256MB
                max_size=4 * 1024 * 1024 * 1024, # 4GB
                block_size=64 * 1024,            # 64KB blocks
                strategy=PoolStrategy.FIRST_FIT,
                priority=PoolPriority.HIGH,
                region_preference=MemoryRegion.GPU_OPTIMIZED,
                alignment=128
            ),
            PoolConfig(
                name="websocket_pool",
                workload_type=MemoryWorkloadType.WEBSOCKET_STREAMS,
                initial_size=32 * 1024 * 1024,   # 32MB
                max_size=256 * 1024 * 1024,      # 256MB
                block_size=8192,                 # 8KB blocks
                strategy=PoolStrategy.STACK_ALLOCATOR,
                priority=PoolPriority.HIGH,
                region_preference=MemoryRegion.CPU_CACHE_FRIENDLY,
                alignment=32
            ),
            PoolConfig(
                name="risk_calc_pool",
                workload_type=MemoryWorkloadType.RISK_CALCULATION,
                initial_size=64 * 1024 * 1024,   # 64MB
                max_size=512 * 1024 * 1024,      # 512MB
                block_size=16384,                # 16KB blocks
                strategy=PoolStrategy.BEST_FIT,
                priority=PoolPriority.CRITICAL,
                region_preference=MemoryRegion.CPU_CACHE_FRIENDLY,
                alignment=64
            ),
            PoolConfig(
                name="gpu_acceleration_pool",
                workload_type=MemoryWorkloadType.GPU_ACCELERATION,
                initial_size=512 * 1024 * 1024,  # 512MB
                max_size=8 * 1024 * 1024 * 1024, # 8GB
                block_size=2 * 1024 * 1024,      # 2MB blocks
                strategy=PoolStrategy.BUDDY_SYSTEM,
                priority=PoolPriority.HIGH,
                region_preference=MemoryRegion.GPU_OPTIMIZED,
                alignment=512
            ),
        ]
        
        for config in default_configs:
            try:
                self.create_pool(config)
            except Exception as e:
                self.logger.error(f"Failed to create default pool {config.name}: {e}")


# Global pool manager instance
_pool_manager = None
_pool_manager_lock = threading.Lock()


def get_memory_pool_manager() -> MemoryPoolManager:
    """Get singleton instance of memory pool manager"""
    global _pool_manager
    
    if _pool_manager is None:
        with _pool_manager_lock:
            if _pool_manager is None:
                _pool_manager = MemoryPoolManager()
    
    return _pool_manager


# Convenience functions for pool operations

def allocate_from_pool(
    workload_type: MemoryWorkloadType,
    size: int,
    alignment: Optional[int] = None
) -> Optional[int]:
    """Allocate memory from appropriate pool"""
    manager = get_memory_pool_manager()
    return manager.allocate_from_workload(workload_type, size, alignment)


def get_pool_statistics() -> Dict[str, PoolStatistics]:
    """Get statistics for all pools"""
    manager = get_memory_pool_manager()
    return manager.get_global_statistics()


def defragment_pools() -> int:
    """Defragment all memory pools"""
    manager = get_memory_pool_manager()
    return manager.defragment_all()


def cleanup_pools() -> int:
    """Force garbage collection on all pools"""
    manager = get_memory_pool_manager()
    return manager.force_gc_all()
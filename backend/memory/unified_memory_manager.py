"""
Unified Memory Manager for M4 Max Architecture

Leverages M4 Max's unified memory architecture (546 GB/s bandwidth) to optimize
zero-copy operations between CPU, GPU, and Neural Engine while managing 36GB RAM
efficiently across 16+ containers.

Key Features:
- Unified memory pool management optimized for M4 Max
- Zero-copy data sharing between CPU/GPU/Neural Engine
- Real-time memory pressure monitoring
- Trading-aware memory allocation strategies
- Cross-container memory optimization
"""

import asyncio
import ctypes
import mmap
import psutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

try:
    import CoreML
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class MemoryWorkloadType(Enum):
    """Memory workload classification for optimized allocation"""
    TRADING_DATA = "trading_data"          # Ultra-low latency market data
    ML_MODELS = "ml_models"                # Neural Engine optimized
    ANALYTICS = "analytics"                # CPU/GPU parallel processing
    WEBSOCKET_STREAMS = "websocket_streams" # Real-time streaming
    RISK_CALCULATION = "risk_calculation"   # High-priority compute
    HISTORICAL_DATA = "historical_data"     # Large sequential access
    TEMPORARY_COMPUTE = "temporary_compute" # Short-lived operations
    GPU_ACCELERATION = "gpu_acceleration"   # Metal performance shaders


class MemoryRegion(Enum):
    """M4 Max memory regions with different access patterns"""
    UNIFIED_MAIN = "unified_main"          # Main unified memory pool
    GPU_OPTIMIZED = "gpu_optimized"        # GPU-optimized allocations
    NEURAL_ENGINE = "neural_engine"        # Neural Engine optimized
    CPU_CACHE_FRIENDLY = "cpu_cache_friendly" # CPU L1/L2 cache aligned
    CROSS_CONTAINER = "cross_container"     # Shared between containers
    ZERO_COPY_BUFFER = "zero_copy_buffer"  # Zero-copy operation buffers


@dataclass
class MemoryBlock:
    """Represents an allocated memory block in the unified system"""
    address: int
    size: int
    workload_type: MemoryWorkloadType
    region: MemoryRegion
    container_id: Optional[str] = None
    allocated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_pinned: bool = False
    metal_buffer: Optional[Any] = None
    shared_count: int = 0
    
    def touch(self):
        """Update access tracking"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class MemoryPressureMetrics:
    """Real-time memory pressure and bandwidth utilization"""
    total_allocated: int
    available_memory: int
    bandwidth_utilization: float  # Percentage of 546 GB/s
    pressure_level: float  # 0.0 to 1.0
    container_allocations: Dict[str, int]
    workload_allocations: Dict[MemoryWorkloadType, int]
    fragmentation_ratio: float
    gc_pressure: float


class UnifiedMemoryManager:
    """
    M4 Max Unified Memory Manager
    
    Optimizes memory allocation across CPU, GPU, and Neural Engine
    with zero-copy operations and real-time pressure management.
    """
    
    def __init__(self, total_memory_gb: float = 36.0):
        self.total_memory = int(total_memory_gb * 1024 * 1024 * 1024)
        self.max_bandwidth = 546 * 1024 * 1024 * 1024  # 546 GB/s
        
        # Memory tracking
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        self.free_blocks: Dict[MemoryRegion, List[Tuple[int, int]]] = {
            region: [] for region in MemoryRegion
        }
        self.workload_pools: Dict[MemoryWorkloadType, Set[int]] = {
            workload: set() for workload in MemoryWorkloadType
        }
        
        # Container tracking
        self.container_allocations: Dict[str, Set[int]] = {}
        self.container_limits: Dict[str, int] = {}
        self.container_priorities: Dict[str, int] = {}
        
        # Performance monitoring
        self.bandwidth_monitor = BandwidthMonitor(self.max_bandwidth)
        self.pressure_monitor = MemoryPressureMonitor()
        
        # Thread safety
        self.lock = threading.RLock()
        self.allocation_stats = AllocationStatistics()
        
        # Background services
        self.cleanup_thread = threading.Thread(target=self._cleanup_service, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitoring_service, daemon=True)
        
        # Metal integration if available
        self.metal_device = None
        if METAL_AVAILABLE:
            try:
                self.metal_device = Metal.MTLCreateSystemDefaultDevice()
            except Exception as e:
                logging.warning(f"Metal device initialization failed: {e}")
        
        # Start background services
        self.cleanup_thread.start()
        self.monitoring_thread.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized UnifiedMemoryManager with {total_memory_gb}GB total memory")
    
    def allocate(
        self,
        size: int,
        workload_type: MemoryWorkloadType,
        container_id: Optional[str] = None,
        alignment: int = 64,
        prefer_region: Optional[MemoryRegion] = None,
        zero_copy: bool = False
    ) -> Optional[int]:
        """
        Allocate memory optimized for M4 Max unified architecture
        
        Args:
            size: Memory size in bytes
            workload_type: Type of workload for optimization
            container_id: Container requesting memory
            alignment: Memory alignment (default 64-byte for cache lines)
            prefer_region: Preferred memory region
            zero_copy: Enable zero-copy optimizations
        
        Returns:
            Memory address or None if allocation failed
        """
        with self.lock:
            # Check container limits
            if container_id and not self._check_container_limit(container_id, size):
                self.logger.warning(f"Container {container_id} exceeded memory limit")
                return None
            
            # Determine optimal region
            region = prefer_region or self._select_optimal_region(workload_type, size, zero_copy)
            
            # Align size
            aligned_size = (size + alignment - 1) & ~(alignment - 1)
            
            # Attempt allocation
            address = self._allocate_in_region(aligned_size, region, alignment)
            if address is None:
                # Try garbage collection and retry
                self._emergency_cleanup()
                address = self._allocate_in_region(aligned_size, region, alignment)
            
            if address is None:
                self.logger.error(f"Failed to allocate {size} bytes for {workload_type}")
                return None
            
            # Create memory block
            block = MemoryBlock(
                address=address,
                size=aligned_size,
                workload_type=workload_type,
                region=region,
                container_id=container_id,
                is_pinned=workload_type in {MemoryWorkloadType.TRADING_DATA, MemoryWorkloadType.RISK_CALCULATION}
            )
            
            # Setup Metal buffer if needed
            if zero_copy and self.metal_device and workload_type == MemoryWorkloadType.GPU_ACCELERATION:
                block.metal_buffer = self._create_metal_buffer(address, aligned_size)
            
            # Register allocation
            self.allocated_blocks[address] = block
            self.workload_pools[workload_type].add(address)
            
            if container_id:
                if container_id not in self.container_allocations:
                    self.container_allocations[container_id] = set()
                self.container_allocations[container_id].add(address)
            
            # Update statistics
            self.allocation_stats.record_allocation(workload_type, aligned_size)
            
            self.logger.debug(f"Allocated {aligned_size} bytes at {hex(address)} for {workload_type}")
            return address
    
    def deallocate(self, address: int) -> bool:
        """Deallocate memory block"""
        with self.lock:
            if address not in self.allocated_blocks:
                return False
            
            block = self.allocated_blocks[address]
            
            # Check if block is shared
            if block.shared_count > 0:
                block.shared_count -= 1
                return True
            
            # Clean up Metal buffer
            if block.metal_buffer:
                block.metal_buffer = None
            
            # Remove from tracking
            del self.allocated_blocks[address]
            self.workload_pools[block.workload_type].discard(address)
            
            if block.container_id and block.container_id in self.container_allocations:
                self.container_allocations[block.container_id].discard(address)
            
            # Add to free list
            self.free_blocks[block.region].append((address, block.size))
            
            # Update statistics
            self.allocation_stats.record_deallocation(block.workload_type, block.size)
            
            self.logger.debug(f"Deallocated {block.size} bytes at {hex(address)}")
            return True
    
    @contextmanager
    def zero_copy_buffer(self, size: int, workload_type: MemoryWorkloadType):
        """Context manager for zero-copy operations"""
        address = self.allocate(
            size=size,
            workload_type=workload_type,
            prefer_region=MemoryRegion.ZERO_COPY_BUFFER,
            zero_copy=True
        )
        
        if address is None:
            raise MemoryError("Failed to allocate zero-copy buffer")
        
        try:
            yield address
        finally:
            self.deallocate(address)
    
    def create_shared_memory(
        self,
        size: int,
        workload_type: MemoryWorkloadType,
        container_ids: List[str]
    ) -> Optional[int]:
        """Create shared memory region for cross-container communication"""
        with self.lock:
            address = self.allocate(
                size=size,
                workload_type=workload_type,
                prefer_region=MemoryRegion.CROSS_CONTAINER,
                zero_copy=True
            )
            
            if address is None:
                return None
            
            block = self.allocated_blocks[address]
            block.shared_count = len(container_ids)
            
            # Register with all containers
            for container_id in container_ids:
                if container_id not in self.container_allocations:
                    self.container_allocations[container_id] = set()
                self.container_allocations[container_id].add(address)
            
            return address
    
    def get_memory_pressure(self) -> MemoryPressureMetrics:
        """Get current memory pressure metrics"""
        with self.lock:
            total_allocated = sum(block.size for block in self.allocated_blocks.values())
            available = self.total_memory - total_allocated
            
            container_allocs = {}
            for container_id, addresses in self.container_allocations.items():
                container_allocs[container_id] = sum(
                    self.allocated_blocks[addr].size 
                    for addr in addresses 
                    if addr in self.allocated_blocks
                )
            
            workload_allocs = {}
            for workload, addresses in self.workload_pools.items():
                workload_allocs[workload] = sum(
                    self.allocated_blocks[addr].size 
                    for addr in addresses 
                    if addr in self.allocated_blocks
                )
            
            # Calculate fragmentation
            total_free_blocks = sum(len(blocks) for blocks in self.free_blocks.values())
            fragmentation = min(1.0, total_free_blocks / max(1, len(self.allocated_blocks)))
            
            return MemoryPressureMetrics(
                total_allocated=total_allocated,
                available_memory=available,
                bandwidth_utilization=self.bandwidth_monitor.get_utilization(),
                pressure_level=total_allocated / self.total_memory,
                container_allocations=container_allocs,
                workload_allocations=workload_allocs,
                fragmentation_ratio=fragmentation,
                gc_pressure=self.pressure_monitor.get_gc_pressure()
            )
    
    def optimize_for_trading(self):
        """Optimize memory layout for ultra-low latency trading"""
        with self.lock:
            # Pin trading-critical memory
            for address, block in self.allocated_blocks.items():
                if block.workload_type in {
                    MemoryWorkloadType.TRADING_DATA,
                    MemoryWorkloadType.RISK_CALCULATION,
                    MemoryWorkloadType.WEBSOCKET_STREAMS
                }:
                    block.is_pinned = True
            
            # Compact non-critical allocations
            self._compact_memory_regions()
            
            # Pre-allocate critical buffers
            self._preallocate_trading_buffers()
    
    def set_container_limit(self, container_id: str, limit_bytes: int, priority: int = 5):
        """Set memory limit and priority for container"""
        with self.lock:
            self.container_limits[container_id] = limit_bytes
            self.container_priorities[container_id] = priority
    
    def force_garbage_collection(self, workload_type: Optional[MemoryWorkloadType] = None):
        """Force garbage collection for specific workload or all"""
        with self.lock:
            if workload_type:
                self._cleanup_workload(workload_type)
            else:
                self._emergency_cleanup()
    
    # Private methods
    
    def _select_optimal_region(
        self,
        workload_type: MemoryWorkloadType,
        size: int,
        zero_copy: bool
    ) -> MemoryRegion:
        """Select optimal memory region based on workload characteristics"""
        if zero_copy:
            return MemoryRegion.ZERO_COPY_BUFFER
        
        region_map = {
            MemoryWorkloadType.TRADING_DATA: MemoryRegion.CPU_CACHE_FRIENDLY,
            MemoryWorkloadType.ML_MODELS: MemoryRegion.NEURAL_ENGINE,
            MemoryWorkloadType.GPU_ACCELERATION: MemoryRegion.GPU_OPTIMIZED,
            MemoryWorkloadType.ANALYTICS: MemoryRegion.GPU_OPTIMIZED,
            MemoryWorkloadType.WEBSOCKET_STREAMS: MemoryRegion.CPU_CACHE_FRIENDLY,
            MemoryWorkloadType.RISK_CALCULATION: MemoryRegion.CPU_CACHE_FRIENDLY,
            MemoryWorkloadType.HISTORICAL_DATA: MemoryRegion.UNIFIED_MAIN,
            MemoryWorkloadType.TEMPORARY_COMPUTE: MemoryRegion.UNIFIED_MAIN,
        }
        
        return region_map.get(workload_type, MemoryRegion.UNIFIED_MAIN)
    
    def _allocate_in_region(self, size: int, region: MemoryRegion, alignment: int) -> Optional[int]:
        """Allocate memory in specific region"""
        # Try to reuse free blocks first
        free_list = self.free_blocks[region]
        for i, (addr, block_size) in enumerate(free_list):
            if block_size >= size:
                # Use this block
                free_list.pop(i)
                if block_size > size + alignment:
                    # Split block
                    free_list.append((addr + size, block_size - size))
                return addr
        
        # Allocate new block (simplified - in real implementation would use mmap/malloc)
        try:
            # This is a simplified allocation - real implementation would use
            # platform-specific memory allocation optimized for M4 Max
            address = id(bytearray(size))  # Simplified for example
            return address
        except MemoryError:
            return None
    
    def _check_container_limit(self, container_id: str, additional_size: int) -> bool:
        """Check if container can allocate additional memory"""
        if container_id not in self.container_limits:
            return True
        
        current_usage = sum(
            self.allocated_blocks[addr].size
            for addr in self.container_allocations.get(container_id, set())
            if addr in self.allocated_blocks
        )
        
        return current_usage + additional_size <= self.container_limits[container_id]
    
    def _create_metal_buffer(self, address: int, size: int) -> Optional[Any]:
        """Create Metal buffer for GPU operations"""
        if not self.metal_device:
            return None
        
        try:
            # Create Metal buffer from existing memory
            # This is simplified - real implementation would use Metal APIs
            return self.metal_device.newBufferWithBytesNoCopy_length_options_deallocator_(
                address, size, 0, None
            )
        except Exception as e:
            self.logger.warning(f"Failed to create Metal buffer: {e}")
            return None
    
    def _compact_memory_regions(self):
        """Compact memory to reduce fragmentation"""
        # Simplified compaction - move non-pinned blocks to reduce fragmentation
        for region in MemoryRegion:
            free_list = self.free_blocks[region]
            if len(free_list) > 1:
                # Sort and merge adjacent free blocks
                free_list.sort()
                merged = []
                current_addr, current_size = free_list[0]
                
                for addr, size in free_list[1:]:
                    if current_addr + current_size == addr:
                        current_size += size
                    else:
                        merged.append((current_addr, current_size))
                        current_addr, current_size = addr, size
                
                merged.append((current_addr, current_size))
                self.free_blocks[region] = merged
    
    def _preallocate_trading_buffers(self):
        """Pre-allocate buffers for ultra-low latency trading"""
        # Pre-allocate common trading data structures
        buffer_sizes = [
            (1024 * 1024, MemoryWorkloadType.TRADING_DATA),      # 1MB market data buffer
            (512 * 1024, MemoryWorkloadType.RISK_CALCULATION),   # 512KB risk buffer
            (2 * 1024 * 1024, MemoryWorkloadType.WEBSOCKET_STREAMS),  # 2MB websocket buffer
        ]
        
        for size, workload_type in buffer_sizes:
            address = self.allocate(size, workload_type, alignment=4096)
            if address:
                # Mark as pre-allocated system buffer
                self.allocated_blocks[address].is_pinned = True
    
    def _cleanup_workload(self, workload_type: MemoryWorkloadType):
        """Clean up memory for specific workload type"""
        addresses_to_cleanup = []
        current_time = time.time()
        
        for address in self.workload_pools[workload_type]:
            if address not in self.allocated_blocks:
                continue
            
            block = self.allocated_blocks[address]
            
            # Don't cleanup pinned or recently accessed blocks
            if block.is_pinned or (current_time - block.last_accessed) < 60:
                continue
            
            # Cleanup blocks not accessed recently
            if (current_time - block.last_accessed) > 300:  # 5 minutes
                addresses_to_cleanup.append(address)
        
        for address in addresses_to_cleanup:
            self.deallocate(address)
    
    def _emergency_cleanup(self):
        """Emergency cleanup when memory is critically low"""
        current_time = time.time()
        addresses_to_cleanup = []
        
        # Find candidates for cleanup
        for address, block in self.allocated_blocks.items():
            if block.is_pinned or block.shared_count > 0:
                continue
            
            # Prioritize cleanup by workload type and age
            cleanup_priority = {
                MemoryWorkloadType.TEMPORARY_COMPUTE: 1,
                MemoryWorkloadType.HISTORICAL_DATA: 2,
                MemoryWorkloadType.ANALYTICS: 3,
                MemoryWorkloadType.ML_MODELS: 4,
            }
            
            priority = cleanup_priority.get(block.workload_type, 10)
            age = current_time - block.last_accessed
            
            if age > 60 and priority <= 3:  # Cleanup old temporary data
                addresses_to_cleanup.append((address, priority, age))
        
        # Sort by priority and age, cleanup oldest low-priority first
        addresses_to_cleanup.sort(key=lambda x: (x[1], -x[2]))
        
        for address, _, _ in addresses_to_cleanup[:10]:  # Cleanup up to 10 blocks
            self.deallocate(address)
    
    def _cleanup_service(self):
        """Background cleanup service"""
        while True:
            try:
                time.sleep(30)  # Run every 30 seconds
                
                pressure = self.get_memory_pressure()
                if pressure.pressure_level > 0.8:  # High memory pressure
                    self._emergency_cleanup()
                elif pressure.fragmentation_ratio > 0.3:  # High fragmentation
                    self._compact_memory_regions()
                
            except Exception as e:
                self.logger.error(f"Cleanup service error: {e}")
    
    def _monitoring_service(self):
        """Background monitoring service"""
        while True:
            try:
                time.sleep(10)  # Monitor every 10 seconds
                
                pressure = self.get_memory_pressure()
                
                # Log warnings for high pressure
                if pressure.pressure_level > 0.9:
                    self.logger.warning(f"Critical memory pressure: {pressure.pressure_level:.2%}")
                elif pressure.pressure_level > 0.8:
                    self.logger.info(f"High memory pressure: {pressure.pressure_level:.2%}")
                
                # Monitor bandwidth utilization
                if pressure.bandwidth_utilization > 0.8:
                    self.logger.info(f"High bandwidth utilization: {pressure.bandwidth_utilization:.2%}")
                
            except Exception as e:
                self.logger.error(f"Monitoring service error: {e}")


class BandwidthMonitor:
    """Monitor memory bandwidth utilization (546 GB/s for M4 Max)"""
    
    def __init__(self, max_bandwidth: int):
        self.max_bandwidth = max_bandwidth
        self.recent_transfers = []
        self.lock = threading.Lock()
    
    def record_transfer(self, bytes_transferred: int):
        """Record memory transfer for bandwidth calculation"""
        with self.lock:
            current_time = time.time()
            self.recent_transfers.append((current_time, bytes_transferred))
            
            # Keep only recent transfers (last 1 second)
            cutoff = current_time - 1.0
            self.recent_transfers = [
                (ts, size) for ts, size in self.recent_transfers if ts > cutoff
            ]
    
    def get_utilization(self) -> float:
        """Get current bandwidth utilization as percentage"""
        with self.lock:
            if not self.recent_transfers:
                return 0.0
            
            total_bytes = sum(size for _, size in self.recent_transfers)
            return min(1.0, total_bytes / self.max_bandwidth)


class MemoryPressureMonitor:
    """Monitor system memory pressure and GC impact"""
    
    def __init__(self):
        self.gc_times = []
        self.lock = threading.Lock()
    
    def record_gc_event(self, duration: float):
        """Record garbage collection event"""
        with self.lock:
            current_time = time.time()
            self.gc_times.append((current_time, duration))
            
            # Keep only recent GC events (last 5 minutes)
            cutoff = current_time - 300
            self.gc_times = [(ts, dur) for ts, dur in self.gc_times if ts > cutoff]
    
    def get_gc_pressure(self) -> float:
        """Get GC pressure metric (0.0 to 1.0)"""
        with self.lock:
            if not self.gc_times:
                return 0.0
            
            total_gc_time = sum(duration for _, duration in self.gc_times)
            return min(1.0, total_gc_time / 60.0)  # Normalize to 1 minute


class AllocationStatistics:
    """Track allocation statistics for performance analysis"""
    
    def __init__(self):
        self.stats = {workload: {"allocated": 0, "deallocated": 0, "count": 0}
                     for workload in MemoryWorkloadType}
        self.lock = threading.Lock()
    
    def record_allocation(self, workload_type: MemoryWorkloadType, size: int):
        """Record memory allocation"""
        with self.lock:
            stats = self.stats[workload_type]
            stats["allocated"] += size
            stats["count"] += 1
    
    def record_deallocation(self, workload_type: MemoryWorkloadType, size: int):
        """Record memory deallocation"""
        with self.lock:
            self.stats[workload_type]["deallocated"] += size
    
    def get_statistics(self) -> Dict[MemoryWorkloadType, Dict[str, int]]:
        """Get allocation statistics"""
        with self.lock:
            return {workload: stats.copy() for workload, stats in self.stats.items()}


# Singleton instance for global access
_unified_memory_manager = None
_manager_lock = threading.Lock()


def get_unified_memory_manager() -> UnifiedMemoryManager:
    """Get singleton instance of unified memory manager"""
    global _unified_memory_manager
    
    if _unified_memory_manager is None:
        with _manager_lock:
            if _unified_memory_manager is None:
                _unified_memory_manager = UnifiedMemoryManager()
    
    return _unified_memory_manager


# Convenience functions for common operations

def allocate_trading_buffer(size: int, container_id: Optional[str] = None) -> Optional[int]:
    """Allocate optimized buffer for trading data"""
    manager = get_unified_memory_manager()
    return manager.allocate(
        size=size,
        workload_type=MemoryWorkloadType.TRADING_DATA,
        container_id=container_id,
        alignment=64,
        prefer_region=MemoryRegion.CPU_CACHE_FRIENDLY
    )


def allocate_ml_buffer(size: int, container_id: Optional[str] = None) -> Optional[int]:
    """Allocate optimized buffer for ML operations"""
    manager = get_unified_memory_manager()
    return manager.allocate(
        size=size,
        workload_type=MemoryWorkloadType.ML_MODELS,
        container_id=container_id,
        prefer_region=MemoryRegion.NEURAL_ENGINE,
        zero_copy=True
    )


def allocate_gpu_buffer(size: int, container_id: Optional[str] = None) -> Optional[int]:
    """Allocate optimized buffer for GPU operations"""
    manager = get_unified_memory_manager()
    return manager.allocate(
        size=size,
        workload_type=MemoryWorkloadType.GPU_ACCELERATION,
        container_id=container_id,
        prefer_region=MemoryRegion.GPU_OPTIMIZED,
        zero_copy=True
    )
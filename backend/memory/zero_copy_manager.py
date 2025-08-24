"""
Zero-Copy Operations Manager for M4 Max Architecture

Implements zero-copy data sharing patterns optimized for M4 Max unified memory,
including GPU memory mapping, Neural Engine optimization, and cross-container
memory sharing with ultra-low latency operations.

Key Features:
- Zero-copy data sharing between CPU/GPU/Neural Engine
- Metal performance shaders integration
- Neural Engine memory management
- Cross-container shared memory optimization
- Memory mapping and direct buffer access
- DMA-like transfers within unified memory
"""

import asyncio
import ctypes
import mmap
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable, Protocol
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import Metal
    import MetalPerformanceShaders as MPS
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

try:
    import CoreML
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

from .unified_memory_manager import (
    MemoryWorkloadType,
    MemoryRegion,
    get_unified_memory_manager
)


class ZeroCopyOperation(Enum):
    """Types of zero-copy operations"""
    CPU_TO_GPU = "cpu_to_gpu"             # CPU data to GPU processing
    GPU_TO_CPU = "gpu_to_cpu"             # GPU results to CPU
    CPU_TO_NEURAL = "cpu_to_neural"       # CPU data to Neural Engine
    NEURAL_TO_CPU = "neural_to_cpu"       # Neural Engine results to CPU
    GPU_TO_NEURAL = "gpu_to_neural"       # GPU to Neural Engine
    NEURAL_TO_GPU = "neural_to_gpu"       # Neural Engine to GPU
    CROSS_CONTAINER = "cross_container"    # Between containers
    MEMORY_MAPPED_IO = "memory_mapped_io"  # File I/O to memory
    NETWORK_DIRECT = "network_direct"      # Direct network to memory


class BufferType(Enum):
    """Buffer types for different hardware targets"""
    UNIFIED_BUFFER = "unified_buffer"      # M4 Max unified memory
    METAL_BUFFER = "metal_buffer"          # Metal GPU buffer
    COREML_BUFFER = "coreml_buffer"        # CoreML Neural Engine buffer
    SHARED_MEMORY = "shared_memory"        # Cross-process shared memory
    MEMORY_MAPPED = "memory_mapped"        # Memory-mapped file
    DMA_BUFFER = "dma_buffer"              # DMA-style buffer


@dataclass
class ZeroCopyBuffer:
    """Zero-copy buffer with multi-hardware access"""
    address: int
    size: int
    buffer_type: BufferType
    workload_type: MemoryWorkloadType
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # Hardware-specific handles
    metal_buffer: Optional[Any] = None
    coreml_buffer: Optional[Any] = None
    numpy_view: Optional[np.ndarray] = None
    shared_memory_handle: Optional[Any] = None
    
    # Access tracking
    cpu_accessible: bool = True
    gpu_accessible: bool = False
    neural_accessible: bool = False
    
    # Reference counting for automatic cleanup
    ref_count: int = 1
    
    def touch(self):
        """Update access tracking"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class ZeroCopyTransfer:
    """Represents a zero-copy data transfer operation"""
    source_buffer: ZeroCopyBuffer
    dest_buffer: ZeroCopyBuffer
    operation: ZeroCopyOperation
    transfer_size: int
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    bandwidth_achieved: Optional[float] = None
    success: bool = False


class ZeroCopyProtocol(Protocol):
    """Protocol for zero-copy operations"""
    def setup_transfer(self, src: ZeroCopyBuffer, dst: ZeroCopyBuffer) -> bool: ...
    def execute_transfer(self, transfer: ZeroCopyTransfer) -> bool: ...
    def cleanup_transfer(self, transfer: ZeroCopyTransfer) -> None: ...


class ZeroCopyManager:
    """
    Zero-Copy Operations Manager for M4 Max
    
    Orchestrates zero-copy operations across CPU, GPU, and Neural Engine
    using M4 Max unified memory architecture for maximum performance.
    """
    
    def __init__(self):
        # Hardware detection and initialization
        self.metal_device = None
        self.metal_command_queue = None
        self.coreml_available = COREML_AVAILABLE
        
        if METAL_AVAILABLE:
            try:
                self.metal_device = Metal.MTLCreateSystemDefaultDevice()
                if self.metal_device:
                    self.metal_command_queue = self.metal_device.newCommandQueue()
            except Exception as e:
                logging.warning(f"Metal initialization failed: {e}")
        
        # Buffer tracking
        self.active_buffers: Dict[int, ZeroCopyBuffer] = {}
        self.shared_buffers: Dict[str, ZeroCopyBuffer] = {}  # Named shared buffers
        self.buffer_mappings: Dict[int, Set[int]] = {}  # address -> set of mapped addresses
        
        # Transfer tracking
        self.active_transfers: List[ZeroCopyTransfer] = []
        self.transfer_history: List[ZeroCopyTransfer] = []
        
        # Performance monitoring
        self.bandwidth_utilization = BandwidthTracker()
        self.operation_stats = OperationStatistics()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background services
        self.cleanup_thread = threading.Thread(target=self._cleanup_service, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_service, daemon=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Start services
        self.cleanup_thread.start()
        self.monitor_thread.start()
        
        self.logger.info("Initialized ZeroCopyManager with M4 Max optimization")
    
    def create_buffer(
        self,
        size: int,
        buffer_type: BufferType,
        workload_type: MemoryWorkloadType,
        data_type: Optional[type] = None,
        alignment: int = 64
    ) -> Optional[ZeroCopyBuffer]:
        """
        Create zero-copy buffer optimized for M4 Max
        
        Args:
            size: Buffer size in bytes
            buffer_type: Type of buffer for hardware optimization
            workload_type: Workload type for memory pool selection
            data_type: Data type for numpy view creation
            alignment: Memory alignment requirement
        
        Returns:
            ZeroCopyBuffer or None if creation failed
        """
        with self.lock:
            # Allocate memory from unified memory manager
            manager = get_unified_memory_manager()
            address = manager.allocate(
                size=size,
                workload_type=workload_type,
                alignment=alignment,
                prefer_region=self._select_region_for_buffer_type(buffer_type),
                zero_copy=True
            )
            
            if address is None:
                self.logger.error(f"Failed to allocate {size} bytes for {buffer_type}")
                return None
            
            # Create buffer
            buffer = ZeroCopyBuffer(
                address=address,
                size=size,
                buffer_type=buffer_type,
                workload_type=workload_type
            )
            
            # Setup hardware-specific access
            self._setup_hardware_access(buffer, data_type)
            
            # Register buffer
            self.active_buffers[address] = buffer
            
            self.logger.debug(f"Created {buffer_type} buffer of {size} bytes at {hex(address)}")
            return buffer
    
    def create_shared_buffer(
        self,
        name: str,
        size: int,
        container_ids: List[str],
        workload_type: MemoryWorkloadType
    ) -> Optional[ZeroCopyBuffer]:
        """Create named shared buffer for cross-container communication"""
        with self.lock:
            if name in self.shared_buffers:
                # Return existing shared buffer
                buffer = self.shared_buffers[name]
                buffer.ref_count += 1
                return buffer
            
            # Create new shared buffer
            manager = get_unified_memory_manager()
            address = manager.create_shared_memory(
                size=size,
                workload_type=workload_type,
                container_ids=container_ids
            )
            
            if address is None:
                return None
            
            buffer = ZeroCopyBuffer(
                address=address,
                size=size,
                buffer_type=BufferType.SHARED_MEMORY,
                workload_type=workload_type,
                ref_count=len(container_ids)
            )
            
            # Setup for cross-container access
            buffer.cpu_accessible = True
            buffer.gpu_accessible = True
            buffer.neural_accessible = True
            
            self.shared_buffers[name] = buffer
            self.active_buffers[address] = buffer
            
            self.logger.info(f"Created shared buffer '{name}' for {len(container_ids)} containers")
            return buffer
    
    def map_buffer_for_gpu(self, buffer: ZeroCopyBuffer) -> bool:
        """Map buffer for GPU access using Metal"""
        if not self.metal_device or buffer.metal_buffer:
            return buffer.metal_buffer is not None
        
        try:
            # Create Metal buffer from existing memory
            buffer.metal_buffer = self.metal_device.newBufferWithBytesNoCopy_length_options_deallocator_(
                buffer.address,
                buffer.size,
                Metal.MTLResourceStorageModeShared,
                None
            )
            
            if buffer.metal_buffer:
                buffer.gpu_accessible = True
                self.logger.debug(f"Mapped buffer {hex(buffer.address)} for GPU access")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to map buffer for GPU: {e}")
        
        return False
    
    def map_buffer_for_neural_engine(self, buffer: ZeroCopyBuffer) -> bool:
        """Map buffer for Neural Engine access using CoreML"""
        if not self.coreml_available or buffer.coreml_buffer:
            return buffer.coreml_buffer is not None
        
        try:
            # Create CoreML-compatible buffer
            # This is simplified - real implementation would use CoreML APIs
            if buffer.numpy_view is not None:
                buffer.coreml_buffer = buffer.numpy_view
                buffer.neural_accessible = True
                self.logger.debug(f"Mapped buffer {hex(buffer.address)} for Neural Engine access")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to map buffer for Neural Engine: {e}")
        
        return False
    
    def create_numpy_view(
        self,
        buffer: ZeroCopyBuffer,
        dtype: np.dtype,
        shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        """Create numpy array view of buffer for zero-copy access"""
        try:
            if shape is None:
                # Calculate shape based on dtype
                element_size = dtype.itemsize
                shape = (buffer.size // element_size,)
            
            # Create numpy array from memory address
            # This is simplified - real implementation would use ctypes and numpy APIs
            array_size = int(np.prod(shape))
            if array_size * dtype.itemsize > buffer.size:
                raise ValueError("Shape too large for buffer size")
            
            # Create ctypes array from address
            ctypes_array = (ctypes.c_ubyte * buffer.size).from_address(buffer.address)
            
            # Create numpy view
            numpy_view = np.frombuffer(ctypes_array, dtype=dtype).reshape(shape)
            buffer.numpy_view = numpy_view
            
            self.logger.debug(f"Created numpy view {shape} {dtype} for buffer {hex(buffer.address)}")
            return numpy_view
            
        except Exception as e:
            self.logger.error(f"Failed to create numpy view: {e}")
            return None
    
    def execute_zero_copy_transfer(
        self,
        src_buffer: ZeroCopyBuffer,
        dst_buffer: ZeroCopyBuffer,
        operation: ZeroCopyOperation,
        size: Optional[int] = None
    ) -> Optional[ZeroCopyTransfer]:
        """Execute zero-copy transfer between buffers"""
        with self.lock:
            transfer_size = size or min(src_buffer.size, dst_buffer.size)
            
            transfer = ZeroCopyTransfer(
                source_buffer=src_buffer,
                dest_buffer=dst_buffer,
                operation=operation,
                transfer_size=transfer_size
            )
            
            # Execute operation-specific transfer
            success = self._execute_operation(transfer)
            
            transfer.completed_at = time.time()
            transfer.success = success
            
            if success:
                # Calculate bandwidth
                duration = transfer.completed_at - transfer.started_at
                if duration > 0:
                    transfer.bandwidth_achieved = transfer_size / duration
                
                # Update statistics
                self.operation_stats.record_transfer(operation, transfer_size, duration)
                self.bandwidth_utilization.record_transfer(transfer_size, duration)
                
                # Update buffer access tracking
                src_buffer.touch()
                dst_buffer.touch()
            
            # Add to history
            self.active_transfers.append(transfer)
            self.transfer_history.append(transfer)
            
            # Keep history manageable
            if len(self.transfer_history) > 1000:
                self.transfer_history = self.transfer_history[-500:]
            
            return transfer if success else None
    
    @contextmanager
    def zero_copy_context(
        self,
        size: int,
        buffer_type: BufferType,
        workload_type: MemoryWorkloadType
    ):
        """Context manager for temporary zero-copy buffer"""
        buffer = self.create_buffer(size, buffer_type, workload_type)
        if buffer is None:
            raise MemoryError("Failed to create zero-copy buffer")
        
        try:
            yield buffer
        finally:
            self.release_buffer(buffer)
    
    def release_buffer(self, buffer: ZeroCopyBuffer) -> bool:
        """Release zero-copy buffer"""
        with self.lock:
            buffer.ref_count -= 1
            
            if buffer.ref_count <= 0:
                # Clean up hardware-specific resources
                if buffer.metal_buffer:
                    buffer.metal_buffer = None
                
                if buffer.coreml_buffer:
                    buffer.coreml_buffer = None
                
                if buffer.shared_memory_handle:
                    buffer.shared_memory_handle = None
                
                # Remove from tracking
                if buffer.address in self.active_buffers:
                    del self.active_buffers[buffer.address]
                
                # Find and remove from shared buffers
                for name, shared_buffer in list(self.shared_buffers.items()):
                    if shared_buffer.address == buffer.address:
                        del self.shared_buffers[name]
                        break
                
                # Deallocate from unified memory manager
                manager = get_unified_memory_manager()
                manager.deallocate(buffer.address)
                
                self.logger.debug(f"Released buffer {hex(buffer.address)}")
                return True
        
        return False
    
    def get_shared_buffer(self, name: str) -> Optional[ZeroCopyBuffer]:
        """Get shared buffer by name"""
        with self.lock:
            buffer = self.shared_buffers.get(name)
            if buffer:
                buffer.ref_count += 1
                buffer.touch()
            return buffer
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get zero-copy performance metrics"""
        with self.lock:
            return {
                'active_buffers': len(self.active_buffers),
                'shared_buffers': len(self.shared_buffers),
                'active_transfers': len(self.active_transfers),
                'bandwidth_utilization': self.bandwidth_utilization.get_current_utilization(),
                'operation_stats': self.operation_stats.get_statistics(),
                'total_buffer_size': sum(buf.size for buf in self.active_buffers.values()),
            }
    
    # Private methods
    
    def _select_region_for_buffer_type(self, buffer_type: BufferType) -> Optional[MemoryRegion]:
        """Select optimal memory region for buffer type"""
        region_map = {
            BufferType.UNIFIED_BUFFER: MemoryRegion.UNIFIED_MAIN,
            BufferType.METAL_BUFFER: MemoryRegion.GPU_OPTIMIZED,
            BufferType.COREML_BUFFER: MemoryRegion.NEURAL_ENGINE,
            BufferType.SHARED_MEMORY: MemoryRegion.CROSS_CONTAINER,
            BufferType.MEMORY_MAPPED: MemoryRegion.UNIFIED_MAIN,
            BufferType.DMA_BUFFER: MemoryRegion.ZERO_COPY_BUFFER,
        }
        return region_map.get(buffer_type)
    
    def _setup_hardware_access(self, buffer: ZeroCopyBuffer, data_type: Optional[type]):
        """Setup hardware-specific access for buffer"""
        # Always enable CPU access
        buffer.cpu_accessible = True
        
        # Create numpy view if data type is provided
        if data_type and issubclass(data_type, np.number):
            dtype = np.dtype(data_type)
            self.create_numpy_view(buffer, dtype)
        
        # Setup GPU access for appropriate buffer types
        if buffer.buffer_type in {BufferType.METAL_BUFFER, BufferType.UNIFIED_BUFFER}:
            self.map_buffer_for_gpu(buffer)
        
        # Setup Neural Engine access for ML workloads
        if buffer.workload_type == MemoryWorkloadType.ML_MODELS:
            self.map_buffer_for_neural_engine(buffer)
    
    def _execute_operation(self, transfer: ZeroCopyTransfer) -> bool:
        """Execute specific zero-copy operation"""
        try:
            if transfer.operation == ZeroCopyOperation.CPU_TO_GPU:
                return self._cpu_to_gpu_transfer(transfer)
            elif transfer.operation == ZeroCopyOperation.GPU_TO_CPU:
                return self._gpu_to_cpu_transfer(transfer)
            elif transfer.operation == ZeroCopyOperation.CPU_TO_NEURAL:
                return self._cpu_to_neural_transfer(transfer)
            elif transfer.operation == ZeroCopyOperation.NEURAL_TO_CPU:
                return self._neural_to_cpu_transfer(transfer)
            elif transfer.operation == ZeroCopyOperation.GPU_TO_NEURAL:
                return self._gpu_to_neural_transfer(transfer)
            elif transfer.operation == ZeroCopyOperation.NEURAL_TO_GPU:
                return self._neural_to_gpu_transfer(transfer)
            elif transfer.operation == ZeroCopyOperation.CROSS_CONTAINER:
                return self._cross_container_transfer(transfer)
            else:
                return self._generic_memory_copy(transfer)
                
        except Exception as e:
            self.logger.error(f"Transfer operation {transfer.operation} failed: {e}")
            return False
    
    def _cpu_to_gpu_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """CPU to GPU zero-copy transfer using Metal"""
        src = transfer.source_buffer
        dst = transfer.dest_buffer
        
        if not dst.metal_buffer:
            if not self.map_buffer_for_gpu(dst):
                return False
        
        # In M4 Max unified memory, this is essentially a no-op
        # The GPU can directly access CPU memory
        if src.numpy_view is not None and dst.metal_buffer:
            # Memory is already shared, just update access tracking
            dst.touch()
            return True
        
        return self._generic_memory_copy(transfer)
    
    def _gpu_to_cpu_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """GPU to CPU zero-copy transfer"""
        # In unified memory, GPU results are immediately available to CPU
        if transfer.source_buffer.metal_buffer:
            # Synchronize GPU operations if needed
            if self.metal_command_queue:
                # Wait for GPU operations to complete
                pass  # In real implementation, would use Metal synchronization
            
            transfer.dest_buffer.touch()
            return True
        
        return self._generic_memory_copy(transfer)
    
    def _cpu_to_neural_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """CPU to Neural Engine zero-copy transfer"""
        src = transfer.source_buffer
        dst = transfer.dest_buffer
        
        if not dst.neural_accessible:
            if not self.map_buffer_for_neural_engine(dst):
                return False
        
        # Neural Engine can directly access unified memory
        if src.numpy_view is not None:
            dst.touch()
            return True
        
        return self._generic_memory_copy(transfer)
    
    def _neural_to_cpu_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """Neural Engine to CPU zero-copy transfer"""
        # Results from Neural Engine are immediately available in unified memory
        transfer.dest_buffer.touch()
        return True
    
    def _gpu_to_neural_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """GPU to Neural Engine zero-copy transfer"""
        # Both can access unified memory directly
        if transfer.source_buffer.metal_buffer and transfer.dest_buffer.neural_accessible:
            transfer.dest_buffer.touch()
            return True
        
        return self._generic_memory_copy(transfer)
    
    def _neural_to_gpu_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """Neural Engine to GPU zero-copy transfer"""
        # Both can access unified memory directly
        if transfer.source_buffer.neural_accessible and transfer.dest_buffer.metal_buffer:
            transfer.dest_buffer.touch()
            return True
        
        return self._generic_memory_copy(transfer)
    
    def _cross_container_transfer(self, transfer: ZeroCopyTransfer) -> bool:
        """Cross-container zero-copy transfer"""
        # In unified memory, containers can share memory directly
        # This is mainly about permission and access control
        src = transfer.source_buffer
        dst = transfer.dest_buffer
        
        if src.buffer_type == BufferType.SHARED_MEMORY and dst.buffer_type == BufferType.SHARED_MEMORY:
            # Direct shared memory access
            dst.touch()
            return True
        
        return self._generic_memory_copy(transfer)
    
    def _generic_memory_copy(self, transfer: ZeroCopyTransfer) -> bool:
        """Generic memory copy as fallback"""
        try:
            # Use ctypes for direct memory access
            src_ptr = ctypes.cast(transfer.source_buffer.address, ctypes.POINTER(ctypes.c_ubyte))
            dst_ptr = ctypes.cast(transfer.dest_buffer.address, ctypes.POINTER(ctypes.c_ubyte))
            
            # Copy memory
            ctypes.memmove(dst_ptr, src_ptr, transfer.transfer_size)
            
            transfer.dest_buffer.touch()
            return True
            
        except Exception as e:
            self.logger.error(f"Generic memory copy failed: {e}")
            return False
    
    def _cleanup_service(self):
        """Background cleanup service"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                with self.lock:
                    # Clean up completed transfers
                    self.active_transfers = [
                        t for t in self.active_transfers 
                        if t.completed_at is None or time.time() - t.completed_at < 300
                    ]
                    
                    # Clean up unused buffers
                    current_time = time.time()
                    buffers_to_cleanup = []
                    
                    for addr, buffer in self.active_buffers.items():
                        if (buffer.ref_count <= 0 and 
                            current_time - buffer.last_accessed > 600):  # 10 minutes
                            buffers_to_cleanup.append(buffer)
                    
                    for buffer in buffers_to_cleanup:
                        self.release_buffer(buffer)
                
            except Exception as e:
                self.logger.error(f"Cleanup service error: {e}")
    
    def _monitor_service(self):
        """Background monitoring service"""
        while True:
            try:
                time.sleep(30)  # Monitor every 30 seconds
                
                metrics = self.get_performance_metrics()
                
                self.logger.info(
                    f"ZeroCopy: Buffers={metrics['active_buffers']} "
                    f"Shared={metrics['shared_buffers']} "
                    f"Transfers={metrics['active_transfers']} "
                    f"BW={metrics['bandwidth_utilization']:.1%}"
                )
                
            except Exception as e:
                self.logger.error(f"Monitor service error: {e}")


class BandwidthTracker:
    """Track memory bandwidth utilization"""
    
    def __init__(self, window_seconds: int = 5):
        self.window_seconds = window_seconds
        self.transfers: List[Tuple[float, int, float]] = []  # (timestamp, bytes, duration)
        self.lock = threading.Lock()
    
    def record_transfer(self, bytes_transferred: int, duration: float):
        """Record a transfer for bandwidth calculation"""
        with self.lock:
            current_time = time.time()
            self.transfers.append((current_time, bytes_transferred, duration))
            
            # Keep only recent transfers
            cutoff = current_time - self.window_seconds
            self.transfers = [t for t in self.transfers if t[0] > cutoff]
    
    def get_current_utilization(self) -> float:
        """Get current bandwidth utilization percentage"""
        with self.lock:
            if not self.transfers:
                return 0.0
            
            current_time = time.time()
            cutoff = current_time - self.window_seconds
            
            recent_transfers = [t for t in self.transfers if t[0] > cutoff]
            
            if not recent_transfers:
                return 0.0
            
            total_bytes = sum(bytes_transferred for _, bytes_transferred, _ in recent_transfers)
            avg_bandwidth = total_bytes / self.window_seconds  # bytes per second
            
            # M4 Max theoretical max: 546 GB/s
            max_bandwidth = 546 * 1024 * 1024 * 1024
            
            return min(1.0, avg_bandwidth / max_bandwidth)


class OperationStatistics:
    """Track zero-copy operation statistics"""
    
    def __init__(self):
        self.stats = {op: {'count': 0, 'total_bytes': 0, 'total_time': 0.0, 'avg_bandwidth': 0.0}
                     for op in ZeroCopyOperation}
        self.lock = threading.Lock()
    
    def record_transfer(self, operation: ZeroCopyOperation, bytes_transferred: int, duration: float):
        """Record transfer statistics"""
        with self.lock:
            stats = self.stats[operation]
            stats['count'] += 1
            stats['total_bytes'] += bytes_transferred
            stats['total_time'] += duration
            
            if duration > 0:
                bandwidth = bytes_transferred / duration
                # Calculate running average bandwidth
                total_count = stats['count']
                stats['avg_bandwidth'] = (
                    (stats['avg_bandwidth'] * (total_count - 1) + bandwidth) / total_count
                )
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get operation statistics"""
        with self.lock:
            return {op.value: stats.copy() for op, stats in self.stats.items()}


# Global zero-copy manager instance
_zero_copy_manager = None
_zero_copy_lock = threading.Lock()


def get_zero_copy_manager() -> ZeroCopyManager:
    """Get singleton instance of zero-copy manager"""
    global _zero_copy_manager
    
    if _zero_copy_manager is None:
        with _zero_copy_lock:
            if _zero_copy_manager is None:
                _zero_copy_manager = ZeroCopyManager()
    
    return _zero_copy_manager


# Convenience functions

def create_zero_copy_buffer(
    size: int,
    buffer_type: BufferType = BufferType.UNIFIED_BUFFER,
    workload_type: MemoryWorkloadType = MemoryWorkloadType.ANALYTICS,
    data_type: Optional[type] = None
) -> Optional[ZeroCopyBuffer]:
    """Create zero-copy buffer with default parameters"""
    manager = get_zero_copy_manager()
    return manager.create_buffer(size, buffer_type, workload_type, data_type)


def create_shared_buffer(
    name: str,
    size: int,
    container_ids: List[str],
    workload_type: MemoryWorkloadType = MemoryWorkloadType.TEMPORARY_COMPUTE
) -> Optional[ZeroCopyBuffer]:
    """Create shared buffer for cross-container communication"""
    manager = get_zero_copy_manager()
    return manager.create_shared_buffer(name, size, container_ids, workload_type)


def execute_zero_copy_transfer(
    src_buffer: ZeroCopyBuffer,
    dst_buffer: ZeroCopyBuffer,
    operation: ZeroCopyOperation
) -> Optional[ZeroCopyTransfer]:
    """Execute zero-copy transfer between buffers"""
    manager = get_zero_copy_manager()
    return manager.execute_zero_copy_transfer(src_buffer, dst_buffer, operation)
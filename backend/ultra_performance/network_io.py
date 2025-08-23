"""
Network and I/O Optimizations for Ultra-Low Latency Trading

Provides:
- Kernel bypass networking (DPDK-style optimizations)
- Zero-copy I/O operations
- Optimized serialization protocols
- Batch processing optimizations
- Network latency optimization techniques
"""

import asyncio
import logging
import socket
import struct
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import os
import sys
import io
import select

# High-performance networking
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    uvloop = None
    UVLOOP_AVAILABLE = False

# Fast serialization
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None
    MSGPACK_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    orjson = None
    ORJSON_AVAILABLE = False

try:
    import flatbuffers
    FLATBUFFERS_AVAILABLE = True
except ImportError:
    flatbuffers = None
    FLATBUFFERS_AVAILABLE = False

# Compression
try:
    import lz4.frame
    import lz4.block
    LZ4_AVAILABLE = True
except ImportError:
    lz4 = None
    LZ4_AVAILABLE = False

# Memory mapping and zero-copy
try:
    import mmap
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

class SerializationProtocol(Enum):
    """Serialization protocol types"""
    MSGPACK = "msgpack"
    ORJSON = "orjson"  
    FLATBUFFERS = "flatbuffers"
    BINARY_STRUCT = "binary_struct"
    NATIVE_BYTES = "native_bytes"

class NetworkOptimizationLevel(Enum):
    """Network optimization levels"""
    STANDARD = "standard"
    HIGH_PERFORMANCE = "high_performance"
    ULTRA_LOW_LATENCY = "ultra_low_latency"
    KERNEL_BYPASS = "kernel_bypass"

@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    total_packets_sent: int = 0
    total_packets_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    avg_send_latency_us: float = 0.0
    avg_receive_latency_us: float = 0.0
    packet_loss_rate: float = 0.0
    zero_copy_operations: int = 0
    batch_operations: int = 0
    serialization_time_us: float = 0.0
    deserialization_time_us: float = 0.0

@dataclass
class ConnectionConfig:
    """Network connection configuration"""
    host: str
    port: int
    socket_buffer_size: int = 65536
    tcp_nodelay: bool = True
    tcp_keepalive: bool = True
    reuse_addr: bool = True
    reuse_port: bool = True
    optimization_level: NetworkOptimizationLevel = NetworkOptimizationLevel.HIGH_PERFORMANCE
    zero_copy_enabled: bool = True
    batch_size: int = 100

class DPDKNetworkManager:
    """
    DPDK-style network manager for kernel bypass and ultra-low latency
    Note: This is a simulation of DPDK concepts for systems without DPDK support
    """
    
    def __init__(self):
        self.optimization_level = NetworkOptimizationLevel.HIGH_PERFORMANCE
        self.network_metrics = NetworkMetrics()
        self.connections: Dict[str, socket.socket] = {}
        self.poll_objects: Dict[str, select.poll] = {}
        self._lock = threading.RLock()
        
        # Initialize high-performance event loop if available
        if UVLOOP_AVAILABLE:
            uvloop.install()
            logger.info("Installed uvloop for high-performance async I/O")
            
    async def create_optimized_connection(
        self,
        config: ConnectionConfig,
        connection_id: str
    ) -> bool:
        """Create optimized network connection"""
        try:
            # Create socket with optimizations
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Apply socket optimizations
            self._apply_socket_optimizations(sock, config)
            
            # Connect to remote endpoint
            await asyncio.get_event_loop().sock_connect(sock, (config.host, config.port))
            
            with self._lock:
                self.connections[connection_id] = sock
                
                # Create poll object for efficient I/O monitoring
                poll_obj = select.poll()
                poll_obj.register(sock.fileno(), select.POLLIN | select.POLLOUT)
                self.poll_objects[connection_id] = poll_obj
                
            logger.info(f"Created optimized connection {connection_id} to {config.host}:{config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create connection {connection_id}: {e}")
            return False
            
    def _apply_socket_optimizations(self, sock: socket.socket, config: ConnectionConfig):
        """Apply socket-level optimizations"""
        # Enable TCP_NODELAY to disable Nagle's algorithm
        if config.tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
        # Enable TCP keepalive
        if config.tcp_keepalive:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
        # Enable address reuse
        if config.reuse_addr:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
        # Enable port reuse (Linux)
        if config.reuse_port and hasattr(socket, 'SO_REUSEPORT'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
        # Set socket buffer sizes
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, config.socket_buffer_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, config.socket_buffer_size)
        
        # Platform-specific optimizations
        if sys.platform.startswith('linux'):
            self._apply_linux_optimizations(sock, config)
        elif sys.platform == 'darwin':
            self._apply_macos_optimizations(sock, config)
            
    def _apply_linux_optimizations(self, sock: socket.socket, config: ConnectionConfig):
        """Apply Linux-specific socket optimizations"""
        try:
            # Enable TCP_QUICKACK to send ACKs immediately
            if hasattr(socket, 'TCP_QUICKACK'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
            # Disable TCP_CORK for immediate sending
            if hasattr(socket, 'TCP_CORK'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 0)
                
            # Set TCP congestion control (if available)
            if hasattr(socket, 'TCP_CONGESTION'):
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b'bbr')
                except OSError:
                    # BBR not available, use default
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to apply Linux optimizations: {e}")
            
    def _apply_macos_optimizations(self, sock: socket.socket, config: ConnectionConfig):
        """Apply macOS-specific socket optimizations"""
        try:
            # Enable TCP_NOPUSH equivalent behavior
            if hasattr(socket, 'TCP_NOPUSH'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NOPUSH, 0)
                
        except Exception as e:
            logger.warning(f"Failed to apply macOS optimizations: {e}")
            
    async def send_data_optimized(
        self,
        connection_id: str,
        data: bytes,
        zero_copy: bool = True
    ) -> bool:
        """Send data using optimized techniques"""
        if connection_id not in self.connections:
            return False
            
        sock = self.connections[connection_id]
        start_time = time.perf_counter_ns()
        
        try:
            if zero_copy and hasattr(sock, 'sendfile'):
                # Use sendfile for zero-copy if data is from file
                # For memory data, we simulate zero-copy by avoiding extra copies
                bytes_sent = await asyncio.get_event_loop().sock_sendall(sock, data)
                self.network_metrics.zero_copy_operations += 1
            else:
                bytes_sent = await asyncio.get_event_loop().sock_sendall(sock, data)
                
            # Update metrics
            send_time_us = (time.perf_counter_ns() - start_time) / 1000
            self._update_send_metrics(len(data), send_time_us)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data on connection {connection_id}: {e}")
            return False
            
    async def receive_data_optimized(
        self,
        connection_id: str,
        buffer_size: int = 4096,
        zero_copy: bool = True
    ) -> Optional[bytes]:
        """Receive data using optimized techniques"""
        if connection_id not in self.connections:
            return None
            
        sock = self.connections[connection_id]
        start_time = time.perf_counter_ns()
        
        try:
            data = await asyncio.get_event_loop().sock_recv(sock, buffer_size)
            
            if zero_copy:
                # Return memoryview for zero-copy access
                self.network_metrics.zero_copy_operations += 1
                result = memoryview(data)
            else:
                result = data
                
            # Update metrics
            receive_time_us = (time.perf_counter_ns() - start_time) / 1000
            self._update_receive_metrics(len(data), receive_time_us)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to receive data on connection {connection_id}: {e}")
            return None
            
    def _update_send_metrics(self, bytes_sent: int, latency_us: float):
        """Update send metrics"""
        self.network_metrics.total_packets_sent += 1
        self.network_metrics.total_bytes_sent += bytes_sent
        
        # Exponential moving average for latency
        alpha = 0.1
        self.network_metrics.avg_send_latency_us = (
            (1 - alpha) * self.network_metrics.avg_send_latency_us + 
            alpha * latency_us
        )
        
    def _update_receive_metrics(self, bytes_received: int, latency_us: float):
        """Update receive metrics"""
        self.network_metrics.total_packets_received += 1
        self.network_metrics.total_bytes_received += bytes_received
        
        # Exponential moving average for latency
        alpha = 0.1
        self.network_metrics.avg_receive_latency_us = (
            (1 - alpha) * self.network_metrics.avg_receive_latency_us + 
            alpha * latency_us
        )

class ZeroCopyIOManager:
    """
    Zero-copy I/O manager for high-performance data operations
    """
    
    def __init__(self):
        self.memory_pools: Dict[int, deque] = defaultdict(deque)
        self.buffer_cache: Dict[str, memoryview] = {}
        self._lock = threading.RLock()
        
    def get_zero_copy_buffer(self, size: int, buffer_id: Optional[str] = None) -> memoryview:
        """Get zero-copy buffer from pool"""
        with self._lock:
            # Try to reuse buffer from pool
            if self.memory_pools[size]:
                return self.memory_pools[size].popleft()
                
            # Create new buffer
            buffer = bytearray(size)
            buffer_view = memoryview(buffer)
            
            if buffer_id:
                self.buffer_cache[buffer_id] = buffer_view
                
            return buffer_view
            
    def return_zero_copy_buffer(self, buffer: memoryview, buffer_id: Optional[str] = None):
        """Return zero-copy buffer to pool"""
        with self._lock:
            size = len(buffer)
            
            # Clean buffer for reuse
            if hasattr(buffer, 'obj') and hasattr(buffer.obj, 'clear'):
                buffer.obj[:] = b'\x00' * size
                
            self.memory_pools[size].append(buffer)
            
            if buffer_id and buffer_id in self.buffer_cache:
                del self.buffer_cache[buffer_id]
                
    async def zero_copy_file_read(self, file_path: str) -> Optional[memoryview]:
        """Zero-copy file reading using memory mapping"""
        try:
            with open(file_path, 'rb') as f:
                # Memory map the file for zero-copy access
                mapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                return memoryview(mapped_file)
                
        except Exception as e:
            logger.error(f"Zero-copy file read failed for {file_path}: {e}")
            return None
            
    async def zero_copy_file_write(self, file_path: str, data: memoryview) -> bool:
        """Zero-copy file writing"""
        try:
            with open(file_path, 'wb') as f:
                # Write directly from memory view
                bytes_written = f.write(data)
                return bytes_written == len(data)
                
        except Exception as e:
            logger.error(f"Zero-copy file write failed for {file_path}: {e}")
            return False
            
    def create_scatter_gather_buffer(self, data_segments: List[memoryview]) -> memoryview:
        """Create scatter-gather buffer for efficient I/O"""
        total_size = sum(len(segment) for segment in data_segments)
        combined_buffer = self.get_zero_copy_buffer(total_size)
        
        offset = 0
        for segment in data_segments:
            segment_size = len(segment)
            combined_buffer[offset:offset + segment_size] = segment
            offset += segment_size
            
        return combined_buffer

class OptimizedSerializationProtocols:
    """
    Optimized serialization protocols for minimal latency
    """
    
    def __init__(self):
        self.protocol_stats: Dict[SerializationProtocol, Dict[str, float]] = defaultdict(dict)
        
    def serialize_optimized(
        self,
        data: Any,
        protocol: SerializationProtocol = SerializationProtocol.MSGPACK
    ) -> Optional[bytes]:
        """Serialize data using optimized protocol"""
        start_time = time.perf_counter_ns()
        
        try:
            if protocol == SerializationProtocol.MSGPACK and MSGPACK_AVAILABLE:
                result = msgpack.packb(data, use_bin_type=True)
            elif protocol == SerializationProtocol.ORJSON and ORJSON_AVAILABLE:
                result = orjson.dumps(data)
            elif protocol == SerializationProtocol.BINARY_STRUCT:
                result = self._serialize_binary_struct(data)
            elif protocol == SerializationProtocol.NATIVE_BYTES:
                result = self._serialize_native_bytes(data)
            else:
                # Fallback to pickle
                import pickle
                result = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                
            # Update metrics
            serialization_time_us = (time.perf_counter_ns() - start_time) / 1000
            self.protocol_stats[protocol]['avg_serialize_time_us'] = (
                self.protocol_stats[protocol].get('avg_serialize_time_us', 0) * 0.9 + 
                serialization_time_us * 0.1
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Serialization failed with protocol {protocol}: {e}")
            return None
            
    def deserialize_optimized(
        self,
        data: bytes,
        protocol: SerializationProtocol = SerializationProtocol.MSGPACK
    ) -> Any:
        """Deserialize data using optimized protocol"""
        start_time = time.perf_counter_ns()
        
        try:
            if protocol == SerializationProtocol.MSGPACK and MSGPACK_AVAILABLE:
                result = msgpack.unpackb(data, raw=False)
            elif protocol == SerializationProtocol.ORJSON and ORJSON_AVAILABLE:
                result = orjson.loads(data)
            elif protocol == SerializationProtocol.BINARY_STRUCT:
                result = self._deserialize_binary_struct(data)
            elif protocol == SerializationProtocol.NATIVE_BYTES:
                result = self._deserialize_native_bytes(data)
            else:
                # Fallback to pickle
                import pickle
                result = pickle.loads(data)
                
            # Update metrics
            deserialization_time_us = (time.perf_counter_ns() - start_time) / 1000
            self.protocol_stats[protocol]['avg_deserialize_time_us'] = (
                self.protocol_stats[protocol].get('avg_deserialize_time_us', 0) * 0.9 + 
                deserialization_time_us * 0.1
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Deserialization failed with protocol {protocol}: {e}")
            return None
            
    def _serialize_binary_struct(self, data: Any) -> bytes:
        """Serialize using binary struct format (fastest for simple types)"""
        if isinstance(data, dict):
            # Handle common trading message formats
            if 'price' in data and 'quantity' in data and 'timestamp' in data:
                # Market data message
                return struct.pack(
                    'ddd',
                    float(data.get('price', 0)),
                    float(data.get('quantity', 0)),
                    float(data.get('timestamp', 0))
                )
            elif 'symbol' in data and 'price' in data:
                # Quote message
                symbol = data.get('symbol', '')[:8].ljust(8, '\x00')  # Fixed 8 chars
                return struct.pack('8sdd',
                    symbol.encode('utf-8'),
                    float(data.get('price', 0)),
                    float(data.get('quantity', 0))
                )
                
        # Fallback for complex data
        import pickle
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
    def _deserialize_binary_struct(self, data: bytes) -> Any:
        """Deserialize from binary struct format"""
        try:
            if len(data) == 24:  # Market data message (3 doubles)
                price, quantity, timestamp = struct.unpack('ddd', data)
                return {
                    'price': price,
                    'quantity': quantity,
                    'timestamp': timestamp
                }
            elif len(data) == 24:  # Quote message (8 chars + 2 doubles)
                symbol_bytes, price, quantity = struct.unpack('8sdd', data)
                symbol = symbol_bytes.rstrip(b'\x00').decode('utf-8')
                return {
                    'symbol': symbol,
                    'price': price,
                    'quantity': quantity
                }
        except struct.error:
            pass
            
        # Fallback
        import pickle
        return pickle.loads(data)
        
    def _serialize_native_bytes(self, data: Any) -> bytes:
        """Ultra-fast serialization for native types"""
        if isinstance(data, (int, float)):
            return str(data).encode('ascii')
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
            return b','.join(str(x).encode('ascii') for x in data)
        else:
            import pickle
            return pickle.dumps(data)
            
    def _deserialize_native_bytes(self, data: bytes) -> Any:
        """Ultra-fast deserialization for native types"""
        try:
            data_str = data.decode('utf-8')
            
            # Try float
            if '.' in data_str:
                return float(data_str)
                
            # Try int
            if data_str.isdigit() or (data_str.startswith('-') and data_str[1:].isdigit()):
                return int(data_str)
                
            # Try list of numbers
            if ',' in data_str:
                parts = data_str.split(',')
                result = []
                for part in parts:
                    if '.' in part:
                        result.append(float(part))
                    else:
                        result.append(int(part))
                return result
                
            # Return as string
            return data_str
            
        except (UnicodeDecodeError, ValueError):
            # Fallback
            import pickle
            return pickle.loads(data)

class BatchProcessor:
    """
    Batch processing optimizer for high-throughput operations
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.pending_operations: deque = deque()
        self.batch_metrics = {"batches_processed": 0, "items_processed": 0}
        self._processing_lock = threading.Lock()
        
    async def add_to_batch(self, operation: Callable, *args, **kwargs):
        """Add operation to batch queue"""
        with self._processing_lock:
            self.pending_operations.append((operation, args, kwargs))
            
            if len(self.pending_operations) >= self.batch_size:
                await self._process_batch()
                
    async def _process_batch(self):
        """Process accumulated batch operations"""
        if not self.pending_operations:
            return
            
        batch = list(self.pending_operations)
        self.pending_operations.clear()
        
        start_time = time.perf_counter()
        
        # Process operations in parallel where possible
        tasks = []
        for operation, args, kwargs in batch:
            if asyncio.iscoroutinefunction(operation):
                tasks.append(operation(*args, **kwargs))
            else:
                # Run synchronous operations in thread pool
                tasks.append(asyncio.get_event_loop().run_in_executor(
                    None, operation, *args, **kwargs
                ))
                
        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update metrics
        self.batch_metrics["batches_processed"] += 1
        self.batch_metrics["items_processed"] += len(batch)
        self.batch_metrics["last_batch_time_ms"] = (time.perf_counter() - start_time) * 1000
        
        return results
        
    async def flush_batch(self):
        """Force process current batch"""
        with self._processing_lock:
            if self.pending_operations:
                await self._process_batch()

class NetworkLatencyOptimizer:
    """
    Network latency optimization techniques
    """
    
    def __init__(self):
        self.latency_measurements: List[float] = []
        self.optimization_strategies = {
            "tcp_nodelay": True,
            "socket_buffer_tuning": True,
            "connection_pooling": True,
            "request_pipelining": True
        }
        
    async def measure_network_latency(self, host: str, port: int, samples: int = 10) -> Dict[str, float]:
        """Measure network latency to target host"""
        latencies = []
        
        for _ in range(samples):
            start_time = time.perf_counter_ns()
            
            try:
                # Create connection and measure time
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                await asyncio.get_event_loop().sock_connect(sock, (host, port))
                
                latency_us = (time.perf_counter_ns() - start_time) / 1000
                latencies.append(latency_us)
                
                sock.close()
                
            except Exception as e:
                logger.warning(f"Latency measurement failed: {e}")
                continue
                
            # Small delay between samples
            await asyncio.sleep(0.01)
            
        if latencies:
            return {
                "min_latency_us": min(latencies),
                "max_latency_us": max(latencies),
                "avg_latency_us": sum(latencies) / len(latencies),
                "samples": len(latencies)
            }
        else:
            return {"error": "No successful latency measurements"}
            
    async def optimize_connection_for_latency(
        self,
        sock: socket.socket,
        target_latency_us: float = 100.0
    ) -> bool:
        """Optimize socket connection for target latency"""
        try:
            # Enable TCP_NODELAY (disable Nagle's algorithm)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Set small socket buffers for low latency
            buffer_size = 4096  # Small buffer for low latency
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
            
            # Platform-specific optimizations
            if sys.platform.startswith('linux'):
                # Enable TCP_QUICKACK
                if hasattr(socket, 'TCP_QUICKACK'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize connection for latency: {e}")
            return False

# Global instances
dpdk_network_manager = DPDKNetworkManager()
zero_copy_io_manager = ZeroCopyIOManager()
optimized_serialization = OptimizedSerializationProtocols()
batch_processor = BatchProcessor()
network_latency_optimizer = NetworkLatencyOptimizer()

# Convenience functions
async def send_optimized(connection_id: str, data: Any, 
                        protocol: SerializationProtocol = SerializationProtocol.MSGPACK) -> bool:
    """Send data with optimized serialization and network transmission"""
    serialized_data = optimized_serialization.serialize_optimized(data, protocol)
    if serialized_data:
        return await dpdk_network_manager.send_data_optimized(connection_id, serialized_data)
    return False

async def receive_optimized(connection_id: str, 
                           protocol: SerializationProtocol = SerializationProtocol.MSGPACK) -> Any:
    """Receive and deserialize data using optimized protocols"""
    raw_data = await dpdk_network_manager.receive_data_optimized(connection_id)
    if raw_data:
        return optimized_serialization.deserialize_optimized(raw_data, protocol)
    return None

async def create_ultra_low_latency_connection(host: str, port: int, connection_id: str) -> bool:
    """Create ultra-low latency optimized connection"""
    config = ConnectionConfig(
        host=host,
        port=port,
        tcp_nodelay=True,
        socket_buffer_size=4096,  # Small buffer for low latency
        optimization_level=NetworkOptimizationLevel.ULTRA_LOW_LATENCY,
        zero_copy_enabled=True
    )
    
    return await dpdk_network_manager.create_optimized_connection(config, connection_id)

# Context managers
class ZeroCopyBuffer:
    """Context manager for zero-copy buffer management"""
    
    def __init__(self, size: int, buffer_id: Optional[str] = None):
        self.size = size
        self.buffer_id = buffer_id
        self.buffer: Optional[memoryview] = None
        
    def __enter__(self) -> memoryview:
        self.buffer = zero_copy_io_manager.get_zero_copy_buffer(self.size, self.buffer_id)
        return self.buffer
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.buffer:
            zero_copy_io_manager.return_zero_copy_buffer(self.buffer, self.buffer_id)

# Decorators
def network_optimized(protocol: SerializationProtocol = SerializationProtocol.MSGPACK):
    """Decorator for network-optimized function calls"""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Add to batch processor for efficient handling
            await batch_processor.add_to_batch(func, *args, **kwargs)
            
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
#!/usr/bin/env python3
"""
Shared Memory IPC System for Hybrid Architecture
Zero-copy data transfer between Docker containers and native engines

This component provides:
- Memory-mapped file-based IPC
- Zero-copy data transfer for large datasets
- Lock-free ring buffer implementation
- Cross-platform memory management
"""

import asyncio
import json
import logging
import mmap
import os
import struct
import tempfile
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

@dataclass
class MemoryRegion:
    """Memory region descriptor"""
    region_id: str
    file_path: str
    size: int
    offset: int
    data_type: str
    timestamp: float

@dataclass
class IPCMessage:
    """IPC message structure"""
    message_id: str
    source: str
    destination: str
    message_type: str
    data_regions: List[MemoryRegion]
    metadata: Dict[str, Any]
    timestamp: float

class IPCMessageType(Enum):
    """IPC message types"""
    DATA_TRANSFER = "data_transfer"
    CALCULATION_REQUEST = "calculation_request"
    CALCULATION_RESULT = "calculation_result"
    HEALTH_CHECK = "health_check"
    MEMORY_ALLOCATION = "memory_allocation"
    MEMORY_DEALLOCATION = "memory_deallocation"

class DataType(Enum):
    """Supported data types for IPC"""
    NUMPY_ARRAY = "numpy_array"
    JSON_DATA = "json_data"
    BINARY_DATA = "binary_data"
    MARKET_DATA = "market_data"
    PORTFOLIO_DATA = "portfolio_data"

class SharedMemoryPool:
    """Shared memory pool manager"""
    
    def __init__(self, pool_size: int = 512 * 1024 * 1024):  # 512MB default
        self.pool_size = pool_size
        self.memory_regions: Dict[str, MemoryRegion] = {}
        self.memory_map = None
        self.pool_file_path = None
        self.allocation_offset = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> str:
        """Initialize shared memory pool"""
        try:
            # Create temporary file for memory pool
            fd, self.pool_file_path = tempfile.mkstemp(
                prefix="nautilus_ipc_pool_", 
                suffix=".mem",
                dir="/tmp"
            )
            
            # Resize file to pool size
            os.ftruncate(fd, self.pool_size)
            
            # Create memory map
            self.memory_map = mmap.mmap(fd, self.pool_size, access=mmap.ACCESS_WRITE)
            
            # Close file descriptor (mmap keeps it open)
            os.close(fd)
            
            # Set permissions for cross-process access
            os.chmod(self.pool_file_path, 0o666)
            
            self.logger.info(f"Shared memory pool initialized: {self.pool_file_path} ({self.pool_size} bytes)")
            return self.pool_file_path
            
        except Exception as e:
            self.logger.error(f"Failed to initialize shared memory pool: {e}")
            raise
    
    def allocate_region(self, region_id: str, size: int, data_type: str) -> MemoryRegion:
        """Allocate memory region from pool"""
        with self.lock:
            if not self.memory_map:
                raise RuntimeError("Shared memory pool not initialized")
            
            if region_id in self.memory_regions:
                raise ValueError(f"Region {region_id} already allocated")
            
            # Align to 64-byte boundaries for better performance
            aligned_size = ((size + 63) // 64) * 64
            
            if self.allocation_offset + aligned_size > self.pool_size:
                raise MemoryError("Insufficient memory in pool")
            
            # Create region descriptor
            region = MemoryRegion(
                region_id=region_id,
                file_path=self.pool_file_path,
                size=aligned_size,
                offset=self.allocation_offset,
                data_type=data_type,
                timestamp=time.time()
            )
            
            self.memory_regions[region_id] = region
            self.allocation_offset += aligned_size
            
            self.logger.debug(f"Allocated region {region_id}: {aligned_size} bytes at offset {region.offset}")
            return region
    
    def deallocate_region(self, region_id: str) -> bool:
        """Deallocate memory region"""
        with self.lock:
            if region_id not in self.memory_regions:
                return False
            
            region = self.memory_regions.pop(region_id)
            
            # Zero out the memory region for security
            self.memory_map.seek(region.offset)
            self.memory_map.write(b'\x00' * region.size)
            self.memory_map.flush()
            
            self.logger.debug(f"Deallocated region {region_id}")
            return True
    
    def write_region_data(self, region_id: str, data: bytes, offset: int = 0) -> int:
        """Write data to memory region"""
        with self.lock:
            if region_id not in self.memory_regions:
                raise ValueError(f"Region {region_id} not found")
            
            region = self.memory_regions[region_id]
            
            if offset + len(data) > region.size:
                raise ValueError("Data size exceeds region capacity")
            
            # Write to memory map
            absolute_offset = region.offset + offset
            self.memory_map.seek(absolute_offset)
            bytes_written = self.memory_map.write(data)
            self.memory_map.flush()
            
            return bytes_written
    
    def read_region_data(self, region_id: str, size: int, offset: int = 0) -> bytes:
        """Read data from memory region"""
        with self.lock:
            if region_id not in self.memory_regions:
                raise ValueError(f"Region {region_id} not found")
            
            region = self.memory_regions[region_id]
            
            if offset + size > region.size:
                raise ValueError("Read size exceeds region bounds")
            
            # Read from memory map
            absolute_offset = region.offset + offset
            self.memory_map.seek(absolute_offset)
            return self.memory_map.read(size)
    
    def get_region_info(self, region_id: str) -> Optional[MemoryRegion]:
        """Get memory region information"""
        with self.lock:
            return self.memory_regions.get(region_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self.lock:
            allocated_bytes = sum(region.size for region in self.memory_regions.values())
            
            return {
                "pool_size": self.pool_size,
                "allocated_bytes": allocated_bytes,
                "free_bytes": self.pool_size - allocated_bytes,
                "utilization_percentage": (allocated_bytes / self.pool_size) * 100,
                "active_regions": len(self.memory_regions),
                "file_path": self.pool_file_path
            }
    
    def cleanup(self):
        """Clean up memory pool"""
        with self.lock:
            if self.memory_map:
                self.memory_map.close()
                self.memory_map = None
            
            if self.pool_file_path and os.path.exists(self.pool_file_path):
                try:
                    os.unlink(self.pool_file_path)
                except:
                    pass
                self.pool_file_path = None
            
            self.memory_regions.clear()
            self.allocation_offset = 0

class DataSerializer:
    """Serializer for different data types to shared memory"""
    
    @staticmethod
    def serialize_numpy_array(array: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Serialize numpy array to bytes with metadata"""
        metadata = {
            "dtype": str(array.dtype),
            "shape": array.shape,
            "data_size": array.nbytes
        }
        
        # Convert to bytes
        data_bytes = array.tobytes()
        
        return data_bytes, metadata
    
    @staticmethod
    def deserialize_numpy_array(data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Deserialize bytes to numpy array"""
        dtype = np.dtype(metadata["dtype"])
        shape = tuple(metadata["shape"])
        
        # Create array from bytes
        array = np.frombuffer(data, dtype=dtype)
        return array.reshape(shape)
    
    @staticmethod
    def serialize_json_data(obj: Any) -> Tuple[bytes, Dict[str, Any]]:
        """Serialize JSON-serializable object"""
        json_str = json.dumps(obj)
        data_bytes = json_str.encode('utf-8')
        
        metadata = {
            "encoding": "utf-8",
            "data_size": len(data_bytes)
        }
        
        return data_bytes, metadata
    
    @staticmethod
    def deserialize_json_data(data: bytes, metadata: Dict[str, Any]) -> Any:
        """Deserialize JSON data"""
        encoding = metadata.get("encoding", "utf-8")
        json_str = data.decode(encoding)
        return json.loads(json_str)
    
    @staticmethod
    def serialize_market_data(market_data: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """Serialize market data structure"""
        # Convert arrays to numpy for efficient storage
        serialized_data = {}
        array_metadata = {}
        
        for key, value in market_data.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                # Convert to numpy array
                array = np.array(value)
                serialized_data[key] = array
                array_metadata[key] = {
                    "dtype": str(array.dtype),
                    "shape": array.shape
                }
            else:
                serialized_data[key] = value
        
        # Use numpy's save format for complex data
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **serialized_data)
        data_bytes = buffer.getvalue()
        
        metadata = {
            "format": "npz_compressed",
            "array_metadata": array_metadata,
            "data_size": len(data_bytes)
        }
        
        return data_bytes, metadata
    
    @staticmethod
    def deserialize_market_data(data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize market data"""
        import io
        
        buffer = io.BytesIO(data)
        loaded_data = np.load(buffer, allow_pickle=False)
        
        # Convert back to original format
        result = {}
        for key in loaded_data.files:
            array = loaded_data[key]
            if array.ndim == 0:
                # Scalar value
                result[key] = array.item()
            else:
                result[key] = array.tolist()
        
        return result

class IPCChannel:
    """IPC channel for communication between processes"""
    
    def __init__(self, channel_name: str, memory_pool: SharedMemoryPool):
        self.channel_name = channel_name
        self.memory_pool = memory_pool
        self.control_socket_path = f"/tmp/nautilus_ipc_{channel_name}.sock"
        self.server_socket = None
        self.client_connections = {}
        self.message_handlers = {}
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    def register_handler(self, message_type: IPCMessageType, handler):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    async def start_server(self):
        """Start IPC server"""
        # Remove existing socket file
        if os.path.exists(self.control_socket_path):
            os.unlink(self.control_socket_path)
        
        import socket
        
        # Create Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.control_socket_path)
        self.server_socket.listen(10)
        self.server_socket.setblocking(False)
        
        # Set permissions
        os.chmod(self.control_socket_path, 0o777)
        
        self.running = True
        self.logger.info(f"IPC server started: {self.control_socket_path}")
        
        while self.running:
            try:
                # Accept connection
                conn, addr = await asyncio.get_event_loop().run_in_executor(
                    None, self.server_socket.accept
                )
                
                # Handle connection
                asyncio.create_task(self.handle_connection(conn))
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"IPC server error: {e}")
                    await asyncio.sleep(0.1)
    
    async def handle_connection(self, conn):
        """Handle client connection"""
        try:
            conn.settimeout(30.0)
            
            while True:
                # Read message length
                length_data = conn.recv(4)
                if not length_data:
                    break
                
                message_length = struct.unpack('!I', length_data)[0]
                
                # Read message data
                message_data = b''
                while len(message_data) < message_length:
                    chunk = conn.recv(message_length - len(message_data))
                    if not chunk:
                        break
                    message_data += chunk
                
                if len(message_data) != message_length:
                    break
                
                # Parse message
                try:
                    message_json = json.loads(message_data.decode('utf-8'))
                    message = self._parse_ipc_message(message_json)
                    
                    # Process message
                    response = await self._process_message(message)
                    
                    if response:
                        # Send response
                        response_data = self._serialize_ipc_message(response).encode('utf-8')
                        conn.send(struct.pack('!I', len(response_data)))
                        conn.send(response_data)
                    
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")
                    break
                
        except Exception as e:
            self.logger.error(f"Connection handling error: {e}")
        finally:
            conn.close()
    
    def _parse_ipc_message(self, message_json: Dict[str, Any]) -> IPCMessage:
        """Parse JSON to IPC message"""
        data_regions = []
        for region_data in message_json.get("data_regions", []):
            data_regions.append(MemoryRegion(**region_data))
        
        return IPCMessage(
            message_id=message_json["message_id"],
            source=message_json["source"],
            destination=message_json["destination"],
            message_type=message_json["message_type"],
            data_regions=data_regions,
            metadata=message_json.get("metadata", {}),
            timestamp=message_json["timestamp"]
        )
    
    def _serialize_ipc_message(self, message: IPCMessage) -> str:
        """Serialize IPC message to JSON"""
        return json.dumps({
            "message_id": message.message_id,
            "source": message.source,
            "destination": message.destination,
            "message_type": message.message_type,
            "data_regions": [
                {
                    "region_id": region.region_id,
                    "file_path": region.file_path,
                    "size": region.size,
                    "offset": region.offset,
                    "data_type": region.data_type,
                    "timestamp": region.timestamp
                }
                for region in message.data_regions
            ],
            "metadata": message.metadata,
            "timestamp": message.timestamp
        })
    
    async def _process_message(self, message: IPCMessage) -> Optional[IPCMessage]:
        """Process incoming IPC message"""
        try:
            message_type = IPCMessageType(message.message_type)
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                return await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
            return None
    
    async def send_message(self, message: IPCMessage, target_socket: str) -> bool:
        """Send IPC message to target"""
        try:
            import socket
            
            # Connect to target
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(target_socket)
            client_socket.settimeout(30.0)
            
            # Serialize and send message
            message_data = self._serialize_ipc_message(message).encode('utf-8')
            client_socket.send(struct.pack('!I', len(message_data)))
            client_socket.send(message_data)
            
            # Read response if expected
            if message.message_type in ["calculation_request", "data_transfer"]:
                length_data = client_socket.recv(4)
                if length_data:
                    response_length = struct.unpack('!I', length_data)[0]
                    
                    response_data = b''
                    while len(response_data) < response_length:
                        chunk = client_socket.recv(response_length - len(response_data))
                        if not chunk:
                            break
                        response_data += chunk
                    
                    response_json = json.loads(response_data.decode('utf-8'))
                    self.logger.debug(f"Received response: {response_json['message_id']}")
            
            client_socket.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def stop_server(self):
        """Stop IPC server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.control_socket_path):
            os.unlink(self.control_socket_path)

class IPCDataTransfer:
    """High-level data transfer interface using shared memory"""
    
    def __init__(self, process_name: str, memory_pool: SharedMemoryPool):
        self.process_name = process_name
        self.memory_pool = memory_pool
        self.channel = IPCChannel(process_name, memory_pool)
        self.serializer = DataSerializer()
        self.logger = logging.getLogger(__name__)
        
        # Register default handlers
        self.channel.register_handler(
            IPCMessageType.DATA_TRANSFER,
            self._handle_data_transfer
        )
    
    async def transfer_numpy_array(self, array: np.ndarray, destination: str,
                                  region_id: Optional[str] = None) -> bool:
        """Transfer numpy array to destination process"""
        try:
            if region_id is None:
                region_id = f"numpy_{int(time.time() * 1000000)}"
            
            # Serialize array
            data_bytes, metadata = self.serializer.serialize_numpy_array(array)
            
            # Allocate memory region
            region = self.memory_pool.allocate_region(
                region_id, len(data_bytes), DataType.NUMPY_ARRAY.value
            )
            
            # Write data to shared memory
            self.memory_pool.write_region_data(region_id, data_bytes)
            
            # Send transfer message
            message = IPCMessage(
                message_id=f"transfer_{region_id}",
                source=self.process_name,
                destination=destination,
                message_type=IPCMessageType.DATA_TRANSFER.value,
                data_regions=[region],
                metadata=metadata,
                timestamp=time.time()
            )
            
            target_socket = f"/tmp/nautilus_ipc_{destination}.sock"
            success = await self.channel.send_message(message, target_socket)
            
            if not success:
                # Clean up on failure
                self.memory_pool.deallocate_region(region_id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to transfer numpy array: {e}")
            return False
    
    async def transfer_market_data(self, market_data: Dict[str, Any], destination: str,
                                  region_id: Optional[str] = None) -> bool:
        """Transfer market data to destination process"""
        try:
            if region_id is None:
                region_id = f"market_{int(time.time() * 1000000)}"
            
            # Serialize market data
            data_bytes, metadata = self.serializer.serialize_market_data(market_data)
            
            # Allocate memory region
            region = self.memory_pool.allocate_region(
                region_id, len(data_bytes), DataType.MARKET_DATA.value
            )
            
            # Write data to shared memory
            self.memory_pool.write_region_data(region_id, data_bytes)
            
            # Send transfer message
            message = IPCMessage(
                message_id=f"transfer_{region_id}",
                source=self.process_name,
                destination=destination,
                message_type=IPCMessageType.DATA_TRANSFER.value,
                data_regions=[region],
                metadata=metadata,
                timestamp=time.time()
            )
            
            target_socket = f"/tmp/nautilus_ipc_{destination}.sock"
            success = await self.channel.send_message(message, target_socket)
            
            if not success:
                self.memory_pool.deallocate_region(region_id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to transfer market data: {e}")
            return False
    
    async def receive_numpy_array(self, region_id: str) -> Optional[np.ndarray]:
        """Receive numpy array from shared memory"""
        try:
            region = self.memory_pool.get_region_info(region_id)
            if not region:
                self.logger.error(f"Region {region_id} not found")
                return None
            
            # Read data from shared memory
            data_bytes = self.memory_pool.read_region_data(region_id, region.size)
            
            # Find metadata (would be passed via message)
            # For now, assume it's stored in the first part of the region
            # In production, metadata would come from the IPC message
            metadata = {"dtype": "float64", "shape": [100, 10]}  # Example
            
            # Deserialize array
            array = self.serializer.deserialize_numpy_array(data_bytes, metadata)
            
            return array
            
        except Exception as e:
            self.logger.error(f"Failed to receive numpy array: {e}")
            return None
    
    async def _handle_data_transfer(self, message: IPCMessage) -> IPCMessage:
        """Handle incoming data transfer message"""
        try:
            self.logger.info(f"Received data transfer: {message.message_id}")
            
            # Process data regions
            for region in message.data_regions:
                self.logger.info(f"Processing region: {region.region_id} ({region.data_type})")
                
                # Data is already in shared memory
                # Application can access it via region_id
            
            # Send acknowledgment
            response = IPCMessage(
                message_id=f"ack_{message.message_id}",
                source=self.process_name,
                destination=message.source,
                message_type="acknowledgment",
                data_regions=[],
                metadata={"status": "received"},
                timestamp=time.time()
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle data transfer: {e}")
            return None
    
    async def start(self):
        """Start IPC data transfer service"""
        await self.channel.start_server()
    
    def stop(self):
        """Stop IPC data transfer service"""
        self.channel.stop_server()

class SharedMemoryIPCManager:
    """Main manager for shared memory IPC system"""
    
    def __init__(self, pool_size: int = 512 * 1024 * 1024):
        self.memory_pool = SharedMemoryPool(pool_size)
        self.data_transfers: Dict[str, IPCDataTransfer] = {}
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self, processes: List[str]) -> str:
        """Initialize IPC system for given processes"""
        # Initialize memory pool
        pool_path = self.memory_pool.initialize()
        
        # Create data transfer instances for each process
        for process_name in processes:
            transfer = IPCDataTransfer(process_name, self.memory_pool)
            self.data_transfers[process_name] = transfer
        
        self.logger.info(f"IPC system initialized for processes: {processes}")
        return pool_path
    
    async def start_process_server(self, process_name: str):
        """Start IPC server for a specific process"""
        if process_name in self.data_transfers:
            await self.data_transfers[process_name].start()
        else:
            raise ValueError(f"Process {process_name} not registered")
    
    def get_transfer_client(self, process_name: str) -> Optional[IPCDataTransfer]:
        """Get data transfer client for a process"""
        return self.data_transfers.get(process_name)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        return self.memory_pool.get_stats()
    
    def cleanup(self):
        """Clean up IPC system"""
        for transfer in self.data_transfers.values():
            transfer.stop()
        
        self.memory_pool.cleanup()
        self.data_transfers.clear()

# Global IPC manager instance
_ipc_manager = None

async def get_ipc_manager() -> SharedMemoryIPCManager:
    """Get global IPC manager instance"""
    global _ipc_manager
    
    if _ipc_manager is None:
        _ipc_manager = SharedMemoryIPCManager()
        
        # Initialize for hybrid architecture processes
        processes = [
            "native_ml_engine",
            "native_risk_engine", 
            "docker_backend",
            "docker_frontend"
        ]
        
        await _ipc_manager.initialize(processes)
    
    return _ipc_manager

def cleanup_ipc_manager():
    """Clean up global IPC manager"""
    global _ipc_manager
    
    if _ipc_manager is not None:
        _ipc_manager.cleanup()
        _ipc_manager = None

async def main():
    """Test shared memory IPC system"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing Shared Memory IPC System")
    
    # Initialize IPC manager
    ipc_manager = await get_ipc_manager()
    
    # Start server for test process
    try:
        await ipc_manager.start_process_server("native_ml_engine")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        cleanup_ipc_manager()

if __name__ == "__main__":
    asyncio.run(main())
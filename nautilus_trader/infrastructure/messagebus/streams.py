#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Redis Stream Manager for Enhanced MessageBus

Provides Redis Streams functionality with consumer groups, automatic
scaling, and monitoring capabilities for the enhanced MessageBus system.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
from redis.asyncio.client import ResponseT

from nautilus_trader.infrastructure.messagebus.config import EnhancedMessageBusConfig, StreamConfig


class StreamState(Enum):
    """Stream processing states"""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class StreamMetrics:
    """Redis Stream metrics"""
    stream_name: str
    length: int
    consumer_groups: int
    pending_messages: int
    last_id: str
    created_at: float
    messages_per_second: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class ConsumerGroupInfo:
    """Consumer group information"""
    name: str
    stream_name: str
    consumers: int
    pending: int
    last_delivered_id: str
    lag: int


@dataclass
class StreamMessage:
    """Redis Stream message"""
    stream_name: str
    message_id: str
    data: Dict[str, Any]
    timestamp: float
    consumer_group: Optional[str] = None
    consumer_name: Optional[str] = None


class RedisStreamManager:
    """
    Redis Stream Manager for Enhanced MessageBus
    
    Provides enterprise-grade Redis Streams functionality with:
    - Consumer groups for distributed processing
    - Automatic stream lifecycle management
    - Performance monitoring and health checks
    - Error handling and recovery
    - Horizontal scaling capabilities
    """
    
    def __init__(self, config: EnhancedMessageBusConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RedisStreamManager")
        
        # Redis connection
        self._redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        
        # Stream management
        self.active_streams: Dict[str, StreamConfig] = {}
        self.consumer_groups: Dict[str, List[str]] = {}  # stream_name -> [group_names]
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # State
        self.state = StreamState.STOPPED
        self.connected = False
        self.start_time = time.time()
        
        # Performance tracking
        self._message_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._last_metric_time = time.time()
        
    async def connect(self) -> None:
        """Connect to Redis with connection pooling"""
        try:
            if self._redis and self.connected:
                return
                
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                max_connections=self.config.connection_pool_size,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Create Redis client
            self._redis = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self._redis.ping()
            self.connected = True
            self.state = StreamState.ACTIVE
            
            # Start background tasks
            if self.config.enable_metrics:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            if self.config.enable_health_checks:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            if self.config.auto_cleanup_enabled:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.state = StreamState.ERROR
            raise
            
    async def close(self) -> None:
        """Close Redis connection and cleanup"""
        try:
            self.state = StreamState.STOPPED
            
            # Cancel background tasks
            for task in [self._monitoring_task, self._health_check_task, self._cleanup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close Redis connection
            if self._redis:
                await self._redis.aclose()
                self._redis = None
                
            if self._connection_pool:
                await self._connection_pool.aclose()
                self._connection_pool = None
                
            self.connected = False
            self.logger.info("Redis connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing Redis connection: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Redis"""
        return self.connected and self._redis is not None
    
    async def create_stream(self, stream_name: str, stream_config: Optional[StreamConfig] = None) -> bool:
        """
        Create a Redis Stream with configuration
        
        Args:
            stream_name: Name of the stream
            stream_config: Optional stream configuration
            
        Returns:
            True if stream created successfully
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Use provided config or create default
            config = stream_config or StreamConfig(
                name=stream_name,
                max_length=self.config.stream_max_length,
                retention_policy="maxlen"
            )
            
            # Create stream by adding a placeholder message (will be trimmed)
            placeholder_data = {
                "type": "stream_init",
                "stream_name": stream_name,
                "created_at": str(time.time()),
                "config": json.dumps(asdict(config))
            }
            
            await self._redis.xadd(stream_name, placeholder_data, maxlen=config.max_length)
            
            # Store stream configuration
            self.active_streams[stream_name] = config
            self._message_counts[stream_name] = 0
            self._error_counts[stream_name] = 0
            
            # Initialize metrics
            self.stream_metrics[stream_name] = StreamMetrics(
                stream_name=stream_name,
                length=1,
                consumer_groups=0,
                pending_messages=0,
                last_id="0-0",
                created_at=time.time()
            )
            
            self.logger.info(f"Created stream: {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create stream {stream_name}: {e}")
            self._error_counts[stream_name] = self._error_counts.get(stream_name, 0) + 1
            return False
    
    async def delete_stream(self, stream_name: str) -> bool:
        """
        Delete a Redis Stream and cleanup
        
        Args:
            stream_name: Name of the stream to delete
            
        Returns:
            True if stream deleted successfully
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Delete consumer groups first
            if stream_name in self.consumer_groups:
                for group_name in self.consumer_groups[stream_name]:
                    try:
                        await self._redis.xgroup_destroy(stream_name, group_name)
                    except Exception as e:
                        self.logger.warning(f"Error deleting consumer group {group_name}: {e}")
                
                del self.consumer_groups[stream_name]
            
            # Delete the stream
            deleted = await self._redis.delete(stream_name)
            
            # Cleanup local state
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]
            if stream_name in self.stream_metrics:
                del self.stream_metrics[stream_name]
            if stream_name in self._message_counts:
                del self._message_counts[stream_name]
            if stream_name in self._error_counts:
                del self._error_counts[stream_name]
            
            self.logger.info(f"Deleted stream: {stream_name}")
            return bool(deleted)
            
        except Exception as e:
            self.logger.error(f"Failed to delete stream {stream_name}: {e}")
            return False
    
    async def add_to_stream(self, stream_name: str, data: Dict[str, Any], 
                          message_id: str = "*") -> Optional[str]:
        """
        Add message to Redis Stream
        
        Args:
            stream_name: Name of the stream
            data: Message data
            message_id: Message ID (default: auto-generate)
            
        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = str(time.time())
            
            # Add message ID if not present
            if "message_id" not in data:
                data["message_id"] = str(uuid.uuid4())
            
            # Get stream configuration
            config = self.active_streams.get(stream_name, StreamConfig(name=stream_name))
            
            # Add to stream with length limit
            result = await self._redis.xadd(
                stream_name, 
                data, 
                id=message_id,
                maxlen=config.max_length if config.max_length > 0 else None
            )
            
            # Update metrics
            self._message_counts[stream_name] = self._message_counts.get(stream_name, 0) + 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to add message to stream {stream_name}: {e}")
            self._error_counts[stream_name] = self._error_counts.get(stream_name, 0) + 1
            return None
    
    async def read_stream(self, stream_name: str, count: int = 10, 
                         start_id: str = "0") -> List[StreamMessage]:
        """
        Read messages from Redis Stream
        
        Args:
            stream_name: Name of the stream
            count: Maximum number of messages to read
            start_id: Starting message ID
            
        Returns:
            List of stream messages
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Read from stream
            messages = await self._redis.xread({stream_name: start_id}, count=count, block=0)
            
            result = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    result.append(StreamMessage(
                        stream_name=stream,
                        message_id=msg_id,
                        data=fields,
                        timestamp=float(fields.get("timestamp", time.time()))
                    ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to read from stream {stream_name}: {e}")
            self._error_counts[stream_name] = self._error_counts.get(stream_name, 0) + 1
            return []
    
    async def create_consumer_group(self, stream_name: str, group_name: str, 
                                  start_id: str = "$") -> bool:
        """
        Create consumer group for Redis Stream
        
        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            start_id: Starting position for the group
            
        Returns:
            True if group created successfully
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Create consumer group
            await self._redis.xgroup_create(stream_name, group_name, start_id, mkstream=True)
            
            # Update local tracking
            if stream_name not in self.consumer_groups:
                self.consumer_groups[stream_name] = []
            if group_name not in self.consumer_groups[stream_name]:
                self.consumer_groups[stream_name].append(group_name)
            
            self.logger.info(f"Created consumer group {group_name} for stream {stream_name}")
            return True
            
        except Exception as e:
            # Ignore if group already exists
            if "BUSYGROUP" in str(e):
                self.logger.debug(f"Consumer group {group_name} already exists for {stream_name}")
                return True
            else:
                self.logger.error(f"Failed to create consumer group {group_name}: {e}")
                return False
    
    async def read_consumer_group(self, stream_name: str, group_name: str, 
                                consumer_name: str, count: int = 10) -> List[StreamMessage]:
        """
        Read messages from consumer group
        
        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            consumer_name: Name of the consumer
            count: Maximum number of messages to read
            
        Returns:
            List of stream messages
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Read from consumer group
            messages = await self._redis.xreadgroup(
                group_name, 
                consumer_name,
                {stream_name: ">"},
                count=count,
                block=0
            )
            
            result = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    result.append(StreamMessage(
                        stream_name=stream,
                        message_id=msg_id,
                        data=fields,
                        timestamp=float(fields.get("timestamp", time.time())),
                        consumer_group=group_name,
                        consumer_name=consumer_name
                    ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to read from consumer group {group_name}: {e}")
            return []
    
    async def ack_message(self, stream_name: str, group_name: str, *message_ids: str) -> int:
        """
        Acknowledge processed messages
        
        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            *message_ids: Message IDs to acknowledge
            
        Returns:
            Number of messages acknowledged
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            return await self._redis.xack(stream_name, group_name, *message_ids)
        except Exception as e:
            self.logger.error(f"Failed to acknowledge messages: {e}")
            return 0
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """
        Get Redis Stream information
        
        Args:
            stream_name: Name of the stream
            
        Returns:
            Stream information dictionary
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            info = await self._redis.xinfo_stream(stream_name)
            
            return {
                "name": stream_name,
                "length": info["length"],
                "radix_tree_keys": info["radix-tree-keys"],
                "radix_tree_nodes": info["radix-tree-nodes"],
                "last_generated_id": info["last-generated-id"],
                "groups": info["groups"],
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stream info for {stream_name}: {e}")
            return {}
    
    async def get_consumer_group_info(self, stream_name: str) -> List[ConsumerGroupInfo]:
        """
        Get consumer group information for stream
        
        Args:
            stream_name: Name of the stream
            
        Returns:
            List of consumer group information
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            groups = await self._redis.xinfo_groups(stream_name)
            
            result = []
            for group_info in groups:
                result.append(ConsumerGroupInfo(
                    name=group_info["name"],
                    stream_name=stream_name,
                    consumers=group_info["consumers"],
                    pending=group_info["pending"],
                    last_delivered_id=group_info["last-delivered-id"],
                    lag=group_info.get("lag", 0)
                ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get consumer group info for {stream_name}: {e}")
            return []
    
    async def trim_stream(self, stream_name: str, max_length: Optional[int] = None) -> int:
        """
        Trim stream to maximum length
        
        Args:
            stream_name: Name of the stream
            max_length: Maximum length (uses config default if None)
            
        Returns:
            Number of messages trimmed
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Redis")
            
        try:
            config = self.active_streams.get(stream_name, StreamConfig(name=stream_name))
            trim_length = max_length or config.max_length
            
            if trim_length <= 0:
                return 0
                
            return await self._redis.xtrim(stream_name, maxlen=trim_length, approximate=True)
            
        except Exception as e:
            self.logger.error(f"Failed to trim stream {stream_name}: {e}")
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Redis Stream Manager metrics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate messages per second
        time_diff = current_time - self._last_metric_time
        if time_diff > 0:
            for stream_name in self.stream_metrics:
                count = self._message_counts.get(stream_name, 0)
                self.stream_metrics[stream_name].messages_per_second = count / time_diff
                
                error_count = self._error_counts.get(stream_name, 0)
                total_ops = count + error_count
                self.stream_metrics[stream_name].error_rate = (error_count / total_ops * 100) if total_ops > 0 else 0.0
        
        self._last_metric_time = current_time
        
        return {
            "manager_metrics": {
                "state": self.state.value,
                "connected": self.connected,
                "uptime_seconds": uptime,
                "active_streams": len(self.active_streams),
                "total_consumer_groups": sum(len(groups) for groups in self.consumer_groups.values()),
                "total_messages_processed": sum(self._message_counts.values()),
                "total_errors": sum(self._error_counts.values())
            },
            "stream_metrics": {name: metrics.to_dict() for name, metrics in self.stream_metrics.items()},
            "redis_config": {
                "host": self.config.redis_host,
                "port": self.config.redis_port,
                "db": self.config.redis_db,
                "pool_size": self.config.connection_pool_size
            }
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Redis Stream Manager"""
        try:
            # Test Redis connection
            if not self.is_connected():
                return {
                    "healthy": False,
                    "status": "disconnected",
                    "message": "Not connected to Redis"
                }
            
            # Ping Redis
            await self._redis.ping()
            
            # Check stream health
            unhealthy_streams = []
            for stream_name in self.active_streams:
                try:
                    await self._redis.xinfo_stream(stream_name)
                except Exception as e:
                    unhealthy_streams.append(f"{stream_name}: {str(e)}")
            
            healthy = len(unhealthy_streams) == 0
            
            return {
                "healthy": healthy,
                "status": self.state.value,
                "active_streams": len(self.active_streams),
                "unhealthy_streams": unhealthy_streams,
                "redis_connected": self.connected,
                "uptime_seconds": time.time() - self.start_time
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.state == StreamState.ACTIVE:
            try:
                # Update stream metrics
                for stream_name in self.active_streams:
                    try:
                        info = await self.get_stream_info(stream_name)
                        if info:
                            if stream_name in self.stream_metrics:
                                self.stream_metrics[stream_name].length = info["length"]
                                self.stream_metrics[stream_name].last_id = info["last_generated_id"]
                                self.stream_metrics[stream_name].consumer_groups = info["groups"]
                    except Exception as e:
                        self.logger.debug(f"Error updating metrics for {stream_name}: {e}")
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while self.state == StreamState.ACTIVE:
            try:
                # Check Redis connection
                if self._redis:
                    await self._redis.ping()
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Health check failed: {e}")
                # Try to reconnect if connection lost
                if "Connection closed" in str(e) or "Connection refused" in str(e):
                    try:
                        await self.connect()
                    except Exception:
                        pass
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.state == StreamState.ACTIVE:
            try:
                # Trim streams if needed
                for stream_name, config in self.active_streams.items():
                    if config.max_length > 0:
                        await self.trim_stream(stream_name)
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
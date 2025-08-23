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

import asyncio
import json
import logging
import re
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Callable, Pattern
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis.asyncio as redis

from nautilus_trader.infrastructure.messagebus.config import (
    EnhancedMessageBusConfig,
    MessagePriority,
    BufferConfig
)

logger = logging.getLogger(__name__)

@dataclass
class MessageMetrics:
    """Metrics for MessageBus performance monitoring"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_per_second: float = 0.0
    buffer_flushes: int = 0
    pattern_matches: int = 0
    avg_latency_ms: float = 0.0
    buffer_utilization: float = 0.0
    error_count: int = 0
    connected: bool = False
    last_activity: datetime = None

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()

class PriorityBuffer:
    """Priority-based message buffer with configurable flushing"""
    
    def __init__(self, config: BufferConfig, priority: MessagePriority):
        self.config = config
        self.priority = priority
        self.buffer: deque = deque(maxlen=config.max_size)
        self.last_flush = time.time()
        self.total_messages = 0
        self.flush_count = 0
        
    def add_message(self, topic: str, message: bytes) -> bool:
        """Add message to buffer, returns True if buffer should be flushed"""
        self.buffer.append((topic, message, time.time()))
        self.total_messages += 1
        
        # Check flush conditions
        current_time = time.time()
        time_since_flush = (current_time - self.last_flush) * 1000  # ms
        buffer_size = len(self.buffer)
        
        should_flush = (
            time_since_flush >= self.config.flush_interval_ms or
            buffer_size >= self.config.high_water_mark or
            buffer_size >= self.config.max_size * 0.9  # 90% full
        )
        
        return should_flush
    
    def get_messages(self) -> List[tuple]:
        """Get all messages and clear buffer"""
        messages = list(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()
        self.flush_count += 1
        return messages
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def utilization(self) -> float:
        """Get buffer utilization percentage"""
        return len(self.buffer) / self.config.max_size

class PatternMatcher:
    """High-performance pattern matching for topic routing"""
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
    def add_pattern(self, pattern: str, handler: Callable):
        """Add a pattern and its handler"""
        if pattern not in self.patterns:
            # Compile glob pattern to regex
            regex_pattern = pattern.replace('.', r'\.')
            regex_pattern = regex_pattern.replace('*', '.*')
            regex_pattern = regex_pattern.replace('?', '.')
            self.patterns[pattern] = re.compile(f"^{regex_pattern}$")
        
        self.pattern_handlers[pattern].append(handler)
    
    def match_topic(self, topic: str) -> List[Callable]:
        """Find all handlers that match the given topic"""
        matching_handlers = []
        
        for pattern, compiled_pattern in self.patterns.items():
            if compiled_pattern.match(topic):
                matching_handlers.extend(self.pattern_handlers[pattern])
        
        return matching_handlers
    
    def remove_pattern(self, pattern: str, handler: Callable):
        """Remove a pattern handler"""
        if pattern in self.pattern_handlers:
            handlers = self.pattern_handlers[pattern]
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                del self.pattern_handlers[pattern]
                if pattern in self.patterns:
                    del self.patterns[pattern]

class BufferedMessageBusClient:
    """Enhanced MessageBus client with priority queues, buffering, and pattern matching"""
    
    def __init__(self, config: EnhancedMessageBusConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        
        # Priority buffers
        self.buffers: Dict[MessagePriority, PriorityBuffer] = {}
        self._initialize_buffers()
        
        # Pattern matching
        self.pattern_matcher = PatternMatcher()
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        
        # Worker management
        self.worker_count = self.config.min_workers
        self.worker_tasks: List[asyncio.Task] = []
        
        # Metrics and monitoring
        self.metrics = MessageMetrics()
        self.last_metrics_update = time.time()
        
        # Message queue for receiving
        self.receive_queue: asyncio.Queue = asyncio.Queue()
        
        # Health monitoring
        self.last_heartbeat = time.time()
        self.connection_failures = 0
        
        # Buffer flushing
        self.buffer_lock = asyncio.Lock()
        
    def _initialize_buffers(self):
        """Initialize priority buffers"""
        for priority in MessagePriority:
            buffer_config = self.config.get_buffer_config(priority)
            self.buffers[priority] = PriorityBuffer(buffer_config, priority)
    
    async def connect(self):
        """Connect to Redis and start background tasks"""
        try:
            # Create Redis connection
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.command_timeout,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.running = True
            self.metrics.connected = True
            self.connection_failures = 0
            
            # Start background tasks
            asyncio.create_task(self._buffer_flush_loop())
            asyncio.create_task(self._metrics_update_loop())
            asyncio.create_task(self._health_monitoring_loop())
            
            # Start initial workers
            await self._scale_workers()
            
            logger.info(f"Enhanced MessageBus connected to Redis {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.metrics.connected = False
            self.connection_failures += 1
            raise
    
    async def disconnect(self):
        """Disconnect and cleanup resources"""
        self.running = False
        
        # Stop workers
        for task in self.worker_tasks:
            task.cancel()
        
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Flush all buffers
        await self._flush_all_buffers()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        self.metrics.connected = False
        logger.info("Enhanced MessageBus disconnected")
    
    async def close(self):
        """Alias for disconnect"""
        await self.disconnect()
    
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.running and self.metrics.connected
    
    async def publish(self, topic: str, message: bytes, priority: MessagePriority = MessagePriority.NORMAL):
        """Publish message with priority-based buffering"""
        if not self.running:
            raise RuntimeError("MessageBus client not connected")
        
        try:
            # Add to appropriate priority buffer
            buffer = self.buffers[priority]
            should_flush = buffer.add_message(topic, message)
            
            # Update metrics
            self.metrics.messages_sent += 1
            self.metrics.last_activity = datetime.utcnow()
            
            # Trigger flush if needed
            if should_flush:
                asyncio.create_task(self._flush_buffer(priority))
            
            # Auto-scale workers if needed
            if self.config.auto_scale_enabled:
                await self._check_auto_scaling()
                
        except Exception as e:
            logger.error(f"Error publishing message to {topic}: {e}")
            self.metrics.error_count += 1
            raise
    
    async def subscribe(self, pattern: str, handler: Optional[Callable] = None):
        """Subscribe to topic pattern"""
        if not self.running:
            raise RuntimeError("MessageBus client not connected")
        
        try:
            # Add pattern to matcher
            if handler:
                self.pattern_matcher.add_pattern(pattern, handler)
            else:
                # Default handler queues to receive_queue
                async def queue_handler(topic: str, message: bytes):
                    await self.receive_queue.put((topic, message))
                
                self.pattern_matcher.add_pattern(pattern, queue_handler)
            
            # Subscribe to Redis pattern
            pubsub = self.redis_client.pubsub()
            await pubsub.psubscribe(pattern)
            
            # Start pattern listener if not already running
            asyncio.create_task(self._pattern_listener(pubsub, pattern))
            
            logger.debug(f"Subscribed to pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Error subscribing to pattern {pattern}: {e}")
            self.metrics.error_count += 1
            raise
    
    async def unsubscribe(self, pattern: str, handler: Optional[Callable] = None):
        """Unsubscribe from topic pattern"""
        if handler:
            self.pattern_matcher.remove_pattern(pattern, handler)
        else:
            # Remove all handlers for pattern
            if pattern in self.pattern_matcher.pattern_handlers:
                del self.pattern_matcher.pattern_handlers[pattern]
                if pattern in self.pattern_matcher.patterns:
                    del self.pattern_matcher.patterns[pattern]
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Receive message from subscription queue"""
        try:
            if timeout:
                topic, message = await asyncio.wait_for(self.receive_queue.get(), timeout=timeout)
            else:
                topic, message = await self.receive_queue.get()
            
            self.metrics.messages_received += 1
            self.metrics.last_activity = datetime.utcnow()
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            self.metrics.error_count += 1
            return None
    
    async def _flush_buffer(self, priority: MessagePriority):
        """Flush specific priority buffer"""
        async with self.buffer_lock:
            try:
                buffer = self.buffers[priority]
                messages = buffer.get_messages()
                
                if not messages:
                    return
                
                # Batch publish to Redis
                pipe = self.redis_client.pipeline()
                for topic, message, timestamp in messages:
                    pipe.publish(topic, message)
                
                await pipe.execute()
                
                # Update metrics
                self.metrics.buffer_flushes += 1
                
                logger.debug(f"Flushed {len(messages)} messages from {priority.name} buffer")
                
            except Exception as e:
                logger.error(f"Error flushing {priority.name} buffer: {e}")
                self.metrics.error_count += 1
    
    async def _flush_all_buffers(self):
        """Flush all priority buffers"""
        for priority in MessagePriority:
            await self._flush_buffer(priority)
    
    async def _buffer_flush_loop(self):
        """Background loop for periodic buffer flushing"""
        while self.running:
            try:
                # Check each buffer for flush conditions
                for priority, buffer in self.buffers.items():
                    current_time = time.time()
                    time_since_flush = (current_time - buffer.last_flush) * 1000
                    
                    if (buffer.size() > 0 and 
                        time_since_flush >= buffer.config.flush_interval_ms):
                        await self._flush_buffer(priority)
                
                # Sleep for minimum flush interval
                min_interval = min(
                    buffer.config.flush_interval_ms 
                    for buffer in self.buffers.values()
                ) / 1000  # Convert to seconds
                
                await asyncio.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Error in buffer flush loop: {e}")
                await asyncio.sleep(1)
    
    async def _pattern_listener(self, pubsub, pattern: str):
        """Listen for messages on subscribed patterns"""
        try:
            async for message in pubsub.listen():
                if message['type'] == 'pmessage':
                    topic = message['channel'].decode('utf-8')
                    data = message['data']
                    
                    # Find matching handlers
                    handlers = self.pattern_matcher.match_topic(topic)
                    
                    # Execute handlers
                    for handler in handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(topic, data)
                            else:
                                handler(topic, data)
                            
                            self.metrics.pattern_matches += 1
                            
                        except Exception as e:
                            logger.error(f"Error in pattern handler for {topic}: {e}")
                            self.metrics.error_count += 1
                            
        except Exception as e:
            logger.error(f"Error in pattern listener for {pattern}: {e}")
        finally:
            await pubsub.close()
    
    async def _check_auto_scaling(self):
        """Check if worker scaling is needed"""
        if not self.config.auto_scale_enabled:
            return
        
        # Calculate total buffer utilization
        total_utilization = sum(
            buffer.utilization() for buffer in self.buffers.values()
        ) / len(self.buffers)
        
        # Scale up if utilization is high
        if (total_utilization > self.config.scale_up_threshold and 
            self.worker_count < self.config.max_workers):
            await self._add_worker()
        
        # Scale down if utilization is low
        elif (total_utilization < self.config.scale_down_threshold and 
              self.worker_count > self.config.min_workers):
            await self._remove_worker()
    
    async def _add_worker(self):
        """Add a new worker task"""
        worker_id = len(self.worker_tasks)
        task = asyncio.create_task(self._worker_loop(worker_id))
        self.worker_tasks.append(task)
        self.worker_count += 1
        logger.debug(f"Added worker {worker_id}, total workers: {self.worker_count}")
    
    async def _remove_worker(self):
        """Remove a worker task"""
        if self.worker_tasks:
            task = self.worker_tasks.pop()
            task.cancel()
            self.worker_count -= 1
            logger.debug(f"Removed worker, total workers: {self.worker_count}")
    
    async def _scale_workers(self):
        """Initial worker scaling"""
        for i in range(self.config.min_workers):
            await self._add_worker()
    
    async def _worker_loop(self, worker_id: int):
        """Background worker for message processing"""
        logger.debug(f"Started worker {worker_id}")
        
        try:
            while self.running:
                # Worker processing logic would go here
                # For now, just sleep to maintain worker presence
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
        except Exception as e:
            logger.error(f"Error in worker {worker_id}: {e}")
    
    async def _metrics_update_loop(self):
        """Background loop for metrics updates"""
        while self.running:
            try:
                current_time = time.time()
                time_delta = current_time - self.last_metrics_update
                
                if time_delta > 0:
                    # Calculate messages per second
                    self.metrics.messages_per_second = (
                        self.metrics.messages_sent / time_delta
                    ) if time_delta > 0 else 0
                
                # Calculate buffer utilization
                self.metrics.buffer_utilization = sum(
                    buffer.utilization() for buffer in self.buffers.values()
                ) / len(self.buffers)
                
                self.last_metrics_update = current_time
                
                # Sleep for metrics interval
                await asyncio.sleep(self.config.metrics_interval_ms / 1000)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitoring_loop(self):
        """Background loop for health monitoring"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check heartbeat
                if current_time - self.last_heartbeat > self.config.heartbeat_timeout:
                    logger.warning("MessageBus heartbeat timeout, checking connection")
                    
                    try:
                        await self.redis_client.ping()
                        self.last_heartbeat = current_time
                        self.connection_failures = 0
                    except Exception as e:
                        logger.error(f"Connection check failed: {e}")
                        self.connection_failures += 1
                        
                        if self.connection_failures >= self.config.max_consecutive_failures:
                            logger.error("Max connection failures reached, attempting reconnect")
                            await self._attempt_reconnect()
                
                # Sleep for health check interval
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _attempt_reconnect(self):
        """Attempt to reconnect to Redis"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            await asyncio.sleep(1)  # Brief pause before reconnect
            await self.connect()
            
            logger.info("Successfully reconnected to Redis")
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get comprehensive health check information"""
        try:
            # Test Redis connection
            redis_healthy = False
            try:
                await self.redis_client.ping()
                redis_healthy = True
            except Exception:
                pass
            
            return {
                "status": "healthy" if redis_healthy and self.running else "unhealthy",
                "connected": redis_healthy,
                "running": self.running,
                "worker_count": self.worker_count,
                "buffer_utilization": self.metrics.buffer_utilization,
                "messages_per_second": self.metrics.messages_per_second,
                "error_rate": self.metrics.error_count / max(1, self.metrics.messages_sent + self.metrics.messages_received),
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None,
                "connection_failures": self.connection_failures,
                "uptime": time.time() - self.last_metrics_update
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "messages_per_second": self.metrics.messages_per_second,
            "buffer_flushes": self.metrics.buffer_flushes,
            "pattern_matches": self.metrics.pattern_matches,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "buffer_utilization": self.metrics.buffer_utilization,
            "error_count": self.metrics.error_count,
            "connected": self.metrics.connected,
            "worker_count": self.worker_count,
            "buffer_sizes": {
                priority.name: buffer.size() 
                for priority, buffer in self.buffers.items()
            }
        }
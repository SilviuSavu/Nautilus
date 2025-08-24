# Optimized MessageBus for Real Data Scenarios
# High-performance message bus implementation optimized for real market data processing
# Part of the 3-4x performance improvement initiative

import asyncio
import logging
import time
import json
import msgpack
import lz4.frame
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as redis
from collections import defaultdict, deque

# Import optimization components
try:
    from ..cache.redis_optimization_layer import get_redis_cache, CacheStrategy
    REDIS_CACHE_AVAILABLE = True
except ImportError:
    REDIS_CACHE_AVAILABLE = False

try:
    from ..serialization.optimized_serializers import get_default_serializer
    OPTIMIZED_SERIALIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZED_SERIALIZATION_AVAILABLE = False

try:
    from ..routing.enhanced_hardware_router import get_enhanced_router, OptimizationTarget
    ENHANCED_ROUTING_AVAILABLE = True
except ImportError:
    ENHANCED_ROUTING_AVAILABLE = False

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels for processing order"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class MessageType(Enum):
    """Message type classification for routing optimization"""
    MARKET_DATA = "market_data"
    TRADE_EXECUTION = "trade_execution"
    RISK_CALCULATION = "risk_calculation"
    ANALYTICS = "analytics"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"

class ProcessingMode(Enum):
    """Message processing modes for optimization"""
    IMMEDIATE = "immediate"        # Process immediately, highest latency sensitivity
    BATCHED = "batched"           # Batch processing for throughput optimization
    SCHEDULED = "scheduled"       # Scheduled processing for efficiency
    BACKGROUND = "background"     # Background processing, lowest priority

@dataclass
class OptimizedMessage:
    """Optimized message structure with performance enhancements"""
    id: str
    type: MessageType
    priority: MessagePriority
    payload: Any
    timestamp_ns: int
    sender: str
    routing_key: str
    processing_mode: ProcessingMode = ProcessingMode.IMMEDIATE
    compression_ratio: Optional[float] = None
    serialization_format: str = "msgpack"
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    processing_time_ms: Optional[float] = None
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageMetrics:
    """Message processing performance metrics"""
    message_id: str
    type: MessageType
    priority: MessagePriority
    processing_time_ms: float
    serialization_time_ms: float
    routing_time_ms: float
    queue_time_ms: float
    hardware_used: str
    compression_achieved: Optional[float]
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MessageBusConfig:
    """Optimized MessageBus configuration"""
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 1  # Use separate DB for messagebus
    max_connections: int = 50
    
    # Performance optimization settings
    enable_compression: bool = True
    enable_hardware_routing: bool = True
    enable_batch_processing: bool = True
    enable_priority_queues: bool = True
    
    # Batch processing configuration
    batch_size: int = 100
    batch_timeout_ms: int = 50
    max_batch_wait_ms: int = 100
    
    # Queue configuration
    max_queue_size: int = 10000
    queue_overflow_strategy: str = "drop_oldest"  # drop_oldest, drop_newest, reject
    
    # Performance targets
    target_latency_ms: float = 10.0
    target_throughput_msgs_per_sec: int = 10000
    
    # Retry configuration
    default_retry_delay_ms: int = 100
    max_retry_delay_ms: int = 5000
    retry_backoff_factor: float = 1.5

class OptimizedMessageBus:
    """
    High-performance MessageBus optimized for real market data processing
    
    Features:
    - Binary serialization with compression for large payloads
    - Hardware-aware message routing for optimal processing
    - Priority queues with intelligent batching
    - Adaptive compression based on message characteristics
    - Real-time performance monitoring and optimization
    - Circuit breaker pattern for reliability
    - Zero-copy operations where possible
    """
    
    def __init__(self, config: MessageBusConfig = None):
        self.config = config or MessageBusConfig()
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        
        # Optimization components
        self.redis_cache = None
        self.serializer = None
        self.router = None
        
        # Message processing
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.priority_queues: Dict[MessagePriority, deque] = {
            priority: deque() for priority in MessagePriority
        }
        
        # Batching system
        self.batch_buffers: Dict[str, List[OptimizedMessage]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
        # Performance monitoring
        self.message_metrics: List[MessageMetrics] = []
        self.performance_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "average_latency_ms": 0.0,
            "throughput_msgs_per_sec": 0.0,
            "compression_ratio": 0.0,
            "hardware_utilization": {},
            "queue_sizes": {}
        }
        
        # Circuit breaker
        self.circuit_breaker_enabled = True
        self.failure_threshold = 10
        self.failure_count = 0
        self.circuit_open_until: Optional[datetime] = None
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="msgbus-worker")
        
        logger.info(f"Optimized MessageBus initialized - Target: {self.config.target_latency_ms}ms latency, {self.config.target_throughput_msgs_per_sec} msg/sec throughput")
    
    async def initialize(self) -> None:
        """Initialize MessageBus with all optimization components"""
        try:
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize optimization components
            if REDIS_CACHE_AVAILABLE and self.config.enable_hardware_routing:
                self.redis_cache = await get_redis_cache()
                logger.info("✅ Redis cache layer integrated")
            
            if OPTIMIZED_SERIALIZATION_AVAILABLE:
                self.serializer = get_default_serializer()
                logger.info("✅ Optimized serialization integrated")
            
            if ENHANCED_ROUTING_AVAILABLE and self.config.enable_hardware_routing:
                self.router = await get_enhanced_router()
                logger.info("✅ Enhanced hardware routing integrated")
            
            # Start background processing tasks
            await self._start_background_tasks()
            
            logger.info("✅ Optimized MessageBus initialization complete")
            
        except Exception as e:
            logger.error(f"❌ MessageBus initialization failed: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection pool"""
        self.redis_pool = redis.ConnectionPool(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            max_connections=self.config.max_connections,
            decode_responses=False  # Handle binary data
        )
        
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        # Test connection
        await self.redis_client.ping()
        logger.info("✅ Redis MessageBus connection established")
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        # Message processing task
        task = asyncio.create_task(self._message_processor())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Performance monitoring task
        task = asyncio.create_task(self._performance_monitor())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Batch processing task
        if self.config.enable_batch_processing:
            task = asyncio.create_task(self._batch_processor())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        # Queue maintenance task
        task = asyncio.create_task(self._queue_maintainer())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        logger.info("✅ Background processing tasks started")
    
    async def publish(
        self,
        message_type: MessageType,
        payload: Any,
        routing_key: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        processing_mode: ProcessingMode = ProcessingMode.IMMEDIATE,
        hardware_requirements: Dict[str, Any] = None
    ) -> str:
        """
        Publish optimized message with intelligent routing
        
        Args:
            message_type: Type of message for routing optimization
            payload: Message payload (will be optimally serialized)
            routing_key: Message routing identifier
            priority: Message priority level
            processing_mode: Processing mode for optimization
            hardware_requirements: Specific hardware requirements
        
        Returns:
            Message ID for tracking
        """
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if self._is_circuit_open():
                raise Exception("MessageBus circuit breaker is open")
            
            # Generate message ID
            message_id = self._generate_message_id(routing_key, payload)
            
            # Create optimized message
            message = OptimizedMessage(
                id=message_id,
                type=message_type,
                priority=priority,
                payload=payload,
                timestamp_ns=time.time_ns(),
                sender="messagebus",
                routing_key=routing_key,
                processing_mode=processing_mode,
                hardware_requirements=hardware_requirements or {}
            )
            
            # Optimize message serialization
            await self._optimize_message_serialization(message)
            
            # Route message based on processing mode
            if processing_mode == ProcessingMode.IMMEDIATE:
                await self._process_immediate_message(message)
            elif processing_mode == ProcessingMode.BATCHED:
                await self._add_to_batch(message)
            else:
                await self._queue_message(message)
            
            # Record metrics
            processing_time = (time.time() - start_time) * 1000
            await self._record_message_metrics(message, processing_time, True)
            
            self.performance_stats["messages_processed"] += 1
            
            return message_id
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            await self._record_message_metrics(None, processing_time, False, str(e))
            
            self.performance_stats["messages_failed"] += 1
            self.failure_count += 1
            
            # Check circuit breaker threshold
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
            
            logger.error(f"Message publishing failed: {e}")
            raise
    
    async def _optimize_message_serialization(self, message: OptimizedMessage) -> None:
        """Optimize message serialization based on payload characteristics"""
        try:
            if not self.serializer:
                # Fallback to msgpack
                serialized_data = msgpack.packb(message.payload)
                message.serialization_format = "msgpack"
            else:
                # Use optimized serializer
                serialized_data, metrics = self.serializer.serialize(message.payload)
                message.compression_ratio = metrics.compression_ratio
                message.serialization_format = metrics.format_used
            
            # Apply additional compression for large messages
            if self.config.enable_compression and len(serialized_data) > 1024:
                compressed_data = lz4.frame.compress(serialized_data)
                if len(compressed_data) < len(serialized_data):
                    message.payload = compressed_data
                    message.compression_ratio = len(serialized_data) / len(compressed_data)
                    message.serialization_format += "_lz4"
                else:
                    message.payload = serialized_data
            else:
                message.payload = serialized_data
        
        except Exception as e:
            logger.warning(f"Message serialization optimization failed: {e}")
            # Fallback to basic JSON
            message.payload = json.dumps(message.payload).encode()
            message.serialization_format = "json"
    
    async def _process_immediate_message(self, message: OptimizedMessage) -> None:
        """Process message immediately with hardware optimization"""
        try:
            # Apply hardware routing if available
            if self.router and self.config.enable_hardware_routing:
                routing_strategy = await self.router.route_workload(
                    workload_type=message.type.value,
                    context=self._create_optimization_context(message),
                    optimization_target=OptimizationTarget.LATENCY
                )
                message.hardware_requirements.update(routing_strategy.optimization_flags)
            
            # Process message with handlers
            handlers = self.message_handlers.get(message.type, [])
            
            if handlers:
                # Execute handlers in parallel for better throughput
                tasks = [self._execute_handler(handler, message) for handler in handlers]
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Store in Redis for consumer retrieval
                await self._store_message_in_redis(message)
        
        except Exception as e:
            logger.error(f"Immediate message processing failed: {e}")
            raise
    
    async def _add_to_batch(self, message: OptimizedMessage) -> None:
        """Add message to batch buffer for efficient processing"""
        batch_key = f"{message.type.value}_{message.priority.value}"
        self.batch_buffers[batch_key].append(message)
        
        # Check if batch is ready for processing
        if len(self.batch_buffers[batch_key]) >= self.config.batch_size:
            await self._process_batch(batch_key)
        elif batch_key not in self.batch_timers:
            # Start batch timer
            timer = asyncio.create_task(self._batch_timer(batch_key))
            self.batch_timers[batch_key] = timer
    
    async def _batch_timer(self, batch_key: str) -> None:
        """Timer for batch processing timeout"""
        try:
            await asyncio.sleep(self.config.batch_timeout_ms / 1000.0)
            if batch_key in self.batch_buffers and self.batch_buffers[batch_key]:
                await self._process_batch(batch_key)
        except asyncio.CancelledError:
            pass
        finally:
            self.batch_timers.pop(batch_key, None)
    
    async def _process_batch(self, batch_key: str) -> None:
        """Process a batch of messages efficiently"""
        if batch_key not in self.batch_buffers or not self.batch_buffers[batch_key]:
            return
        
        batch = self.batch_buffers[batch_key]
        self.batch_buffers[batch_key] = []
        
        # Cancel batch timer if exists
        if batch_key in self.batch_timers:
            self.batch_timers[batch_key].cancel()
            del self.batch_timers[batch_key]
        
        try:
            # Optimize batch for hardware processing
            if self.router and self.config.enable_hardware_routing:
                # Batch processing often benefits from GPU acceleration
                routing_strategy = await self.router.route_workload(
                    workload_type="batch_processing",
                    context=self._create_batch_optimization_context(batch),
                    optimization_target=OptimizationTarget.THROUGHPUT
                )
                
                # Apply optimization flags to all messages in batch
                for message in batch:
                    message.hardware_requirements.update(routing_strategy.optimization_flags)
            
            # Process batch efficiently
            await self._execute_batch_processing(batch)
            
            logger.debug(f"Processed batch of {len(batch)} messages for {batch_key}")
        
        except Exception as e:
            logger.error(f"Batch processing failed for {batch_key}: {e}")
            # Requeue messages individually for retry
            for message in batch:
                await self._queue_message(message)
    
    async def _queue_message(self, message: OptimizedMessage) -> None:
        """Queue message in priority queue"""
        if self.config.enable_priority_queues:
            queue = self.priority_queues[message.priority]
            
            # Check queue size limits
            if len(queue) >= self.config.max_queue_size:
                await self._handle_queue_overflow(message.priority)
            
            queue.append(message)
        else:
            # Store directly in Redis
            await self._store_message_in_redis(message)
    
    async def _handle_queue_overflow(self, priority: MessagePriority) -> None:
        """Handle queue overflow according to configured strategy"""
        queue = self.priority_queues[priority]
        
        if self.config.queue_overflow_strategy == "drop_oldest":
            dropped = queue.popleft()
            logger.warning(f"Dropped oldest message {dropped.id} due to queue overflow")
        elif self.config.queue_overflow_strategy == "drop_newest":
            # Message will be dropped by not being added
            logger.warning(f"Dropping newest message due to queue overflow")
        else:  # reject
            raise Exception(f"Queue {priority.name} is full (max size: {self.config.max_queue_size})")
    
    async def _message_processor(self) -> None:
        """Background message processor for queued messages"""
        while True:
            try:
                # Process messages by priority (highest first)
                for priority in reversed(list(MessagePriority)):
                    queue = self.priority_queues[priority]
                    
                    while queue:
                        message = queue.popleft()
                        
                        try:
                            await self._process_queued_message(message)
                        except Exception as e:
                            logger.error(f"Queued message processing failed: {e}")
                            
                            # Retry logic
                            message.retry_count += 1
                            if message.retry_count < message.max_retries:
                                # Exponential backoff
                                delay = min(
                                    self.config.default_retry_delay_ms * (self.config.retry_backoff_factor ** message.retry_count),
                                    self.config.max_retry_delay_ms
                                ) / 1000.0
                                
                                await asyncio.sleep(delay)
                                queue.append(message)
                            else:
                                logger.error(f"Message {message.id} exceeded max retries, dropping")
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
            
            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1.0)  # Longer delay on error
    
    async def _performance_monitor(self) -> None:
        """Monitor and update performance statistics"""
        last_message_count = 0
        last_update_time = time.time()
        
        while True:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
                current_time = time.time()
                current_message_count = self.performance_stats["messages_processed"]
                
                # Calculate throughput
                time_delta = current_time - last_update_time
                message_delta = current_message_count - last_message_count
                
                if time_delta > 0:
                    self.performance_stats["throughput_msgs_per_sec"] = message_delta / time_delta
                
                # Update average latency from recent metrics
                recent_metrics = self.message_metrics[-100:] if len(self.message_metrics) > 100 else self.message_metrics
                if recent_metrics:
                    total_latency = sum(m.processing_time_ms for m in recent_metrics)
                    self.performance_stats["average_latency_ms"] = total_latency / len(recent_metrics)
                    
                    # Update compression ratio
                    compressed_metrics = [m for m in recent_metrics if m.compression_achieved]
                    if compressed_metrics:
                        total_compression = sum(m.compression_achieved for m in compressed_metrics)
                        self.performance_stats["compression_ratio"] = total_compression / len(compressed_metrics)
                
                # Update queue sizes
                self.performance_stats["queue_sizes"] = {
                    priority.name: len(queue) for priority, queue in self.priority_queues.items()
                }
                
                last_message_count = current_message_count
                last_update_time = current_time
                
                logger.debug(f"MessageBus Performance - Throughput: {self.performance_stats['throughput_msgs_per_sec']:.1f} msg/sec, Latency: {self.performance_stats['average_latency_ms']:.2f}ms")
            
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register message handler for specific message type"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for {message_type.value}")
    
    def unregister_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Unregister message handler"""
        if handler in self.message_handlers[message_type]:
            self.message_handlers[message_type].remove(handler)
            logger.info(f"Unregistered handler for {message_type.value}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive MessageBus performance statistics"""
        return {
            "message_processing": dict(self.performance_stats),
            "configuration": {
                "batch_processing_enabled": self.config.enable_batch_processing,
                "hardware_routing_enabled": self.config.enable_hardware_routing,
                "compression_enabled": self.config.enable_compression,
                "priority_queues_enabled": self.config.enable_priority_queues,
                "target_latency_ms": self.config.target_latency_ms,
                "target_throughput": self.config.target_throughput_msgs_per_sec
            },
            "circuit_breaker": {
                "enabled": self.circuit_breaker_enabled,
                "failure_count": self.failure_count,
                "circuit_open": self._is_circuit_open(),
                "circuit_open_until": self.circuit_open_until.isoformat() if self.circuit_open_until else None
            },
            "optimization_components": {
                "redis_cache_available": self.redis_cache is not None,
                "optimized_serializer_available": self.serializer is not None,
                "enhanced_router_available": self.router is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MessageBus health and all integrated components"""
        try:
            health = {
                "messagebus": "healthy",
                "circuit_breaker": "closed" if not self._is_circuit_open() else "open",
                "components": {}
            }
            
            # Test Redis connection
            await self.redis_client.ping()
            health["components"]["redis"] = "healthy"
            
            # Check optimization components
            if self.redis_cache:
                cache_health = await self.redis_cache.health_check()
                health["components"]["redis_cache"] = cache_health["status"]
            
            if self.router:
                router_health = await self.router.health_check()
                health["components"]["enhanced_router"] = router_health.get("enhanced_router", "unknown")
            
            # Check queue health
            total_queued = sum(len(queue) for queue in self.priority_queues.values())
            health["queues"] = {
                "total_messages_queued": total_queued,
                "queue_health": "healthy" if total_queued < self.config.max_queue_size * 0.8 else "near_capacity"
            }
            
            return health
        
        except Exception as e:
            return {
                "messagebus": "unhealthy",
                "error": str(e)
            }
    
    # Helper methods
    
    def _generate_message_id(self, routing_key: str, payload: Any) -> str:
        """Generate unique message ID"""
        content = f"{routing_key}_{time.time_ns()}_{hash(str(payload))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_breaker_enabled:
            return False
        
        if self.circuit_open_until and datetime.now() < self.circuit_open_until:
            return True
        elif self.circuit_open_until and datetime.now() >= self.circuit_open_until:
            # Circuit breaker timeout expired, reset
            self.circuit_open_until = None
            self.failure_count = 0
        
        return False
    
    def _open_circuit(self) -> None:
        """Open circuit breaker"""
        self.circuit_open_until = datetime.now() + timedelta(seconds=30)
        logger.warning("MessageBus circuit breaker opened due to excessive failures")
    
    async def close(self) -> None:
        """Clean up MessageBus resources"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            # Close thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("Optimized MessageBus closed")
        
        except Exception as e:
            logger.error(f"Error closing MessageBus: {e}")
    
    # Additional helper methods (stubs for complex operations)
    
    def _create_optimization_context(self, message: OptimizedMessage):
        """Create optimization context for hardware routing"""
        # Would create detailed context based on message characteristics
        pass
    
    def _create_batch_optimization_context(self, batch: List[OptimizedMessage]):
        """Create optimization context for batch processing"""
        # Would analyze batch characteristics for optimal processing
        pass
    
    async def _execute_handler(self, handler: Callable, message: OptimizedMessage):
        """Execute message handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                await asyncio.get_event_loop().run_in_executor(self.executor, handler, message)
        except Exception as e:
            logger.error(f"Handler execution failed: {e}")
    
    async def _execute_batch_processing(self, batch: List[OptimizedMessage]):
        """Execute optimized batch processing"""
        # Would implement efficient batch processing logic
        for message in batch:
            await self._process_queued_message(message)
    
    async def _process_queued_message(self, message: OptimizedMessage):
        """Process a queued message"""
        # Would implement queued message processing
        await self._process_immediate_message(message)
    
    async def _store_message_in_redis(self, message: OptimizedMessage):
        """Store message in Redis for consumer retrieval"""
        try:
            key = f"msgbus:{message.routing_key}:{message.id}"
            value = msgpack.packb({
                "id": message.id,
                "type": message.type.value,
                "priority": message.priority.value,
                "payload": message.payload,
                "timestamp_ns": message.timestamp_ns,
                "sender": message.sender
            })
            
            # Set with expiration
            ttl = 3600  # 1 hour
            await self.redis_client.setex(key, ttl, value)
        
        except Exception as e:
            logger.error(f"Failed to store message in Redis: {e}")
    
    async def _record_message_metrics(self, message: Optional[OptimizedMessage], processing_time_ms: float, success: bool, error_message: str = None):
        """Record message processing metrics"""
        try:
            if message:
                metrics = MessageMetrics(
                    message_id=message.id,
                    type=message.type,
                    priority=message.priority,
                    processing_time_ms=processing_time_ms,
                    serialization_time_ms=0.0,  # Would be measured
                    routing_time_ms=0.0,        # Would be measured
                    queue_time_ms=0.0,          # Would be measured
                    hardware_used="unknown",    # Would be determined
                    compression_achieved=message.compression_ratio,
                    success=success,
                    error_message=error_message
                )
                
                self.message_metrics.append(metrics)
                
                # Keep only recent metrics to prevent memory growth
                if len(self.message_metrics) > 1000:
                    self.message_metrics = self.message_metrics[-500:]
        
        except Exception as e:
            logger.warning(f"Failed to record message metrics: {e}")

# Global optimized messagebus instance
_optimized_messagebus: Optional[OptimizedMessageBus] = None

async def get_optimized_messagebus(config: MessageBusConfig = None) -> OptimizedMessageBus:
    """Get or create optimized MessageBus instance"""
    global _optimized_messagebus
    
    if _optimized_messagebus is None:
        _optimized_messagebus = OptimizedMessageBus(config)
        await _optimized_messagebus.initialize()
    
    return _optimized_messagebus
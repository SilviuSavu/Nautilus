"""
SME MessageBus Integration

Enhanced MessageBus routing for SME-accelerated operations with intelligent
workload distribution and performance optimization.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import redis.asyncio as redis

from ..acceleration.sme.sme_hardware_router import SMEHardwareRouter, SMEWorkloadCharacteristics, SMEWorkloadType

logger = logging.getLogger(__name__)

class SMEMessageType(Enum):
    MATRIX_OPERATION = "sme_matrix_operation"
    COVARIANCE_REQUEST = "sme_covariance_request"
    CORRELATION_REQUEST = "sme_correlation_request"
    PERFORMANCE_METRIC = "sme_performance_metric"
    HARDWARE_STATUS = "sme_hardware_status"
    ROUTING_DECISION = "sme_routing_decision"

@dataclass
class SMEMessage:
    """SME-enhanced Message"""
    id: str
    message_type: SMEMessageType
    source_engine: str
    target_engine: Optional[str]
    payload: Dict[str, Any]
    priority: int  # 1=low, 2=medium, 3=high
    sme_workload: Optional[SMEWorkloadCharacteristics] = None
    timestamp: float = 0.0
    processing_time_ms: float = 0.0

@dataclass
class SMEBusMetrics:
    """SME MessageBus Performance Metrics"""
    total_messages: int = 0
    sme_accelerated_messages: int = 0
    average_routing_time_ms: float = 0.0
    peak_throughput_mps: float = 0.0  # messages per second
    hardware_utilization: Dict[str, float] = None

class SMEEnhancedMessageBus:
    """SME-Enhanced Universal MessageBus Client"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
        # SME Integration
        self.sme_hardware_router = SMEHardwareRouter()
        self.sme_enabled = False
        
        # Message tracking
        self.message_queue = asyncio.Queue()
        self.pending_messages = {}
        self.message_metrics = SMEBusMetrics(hardware_utilization={})
        
        # SME-specific queues
        self.sme_priority_queue = asyncio.PriorityQueue()
        self.sme_processing_queue = asyncio.Queue()
        
        # Performance tracking
        self.throughput_counter = 0
        self.throughput_window_start = time.time()
        self.routing_times = []
        
    async def initialize(self) -> bool:
        """Initialize SME-enhanced MessageBus"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            # Initialize SME hardware routing
            self.sme_enabled = await self.sme_hardware_router.initialize_sme_routing()
            
            if self.sme_enabled:
                logger.info("✅ SME-enhanced MessageBus initialized")
            else:
                logger.info("✅ MessageBus initialized (SME disabled)")
                
            # Start background tasks
            asyncio.create_task(self._sme_message_processor())
            asyncio.create_task(self._metrics_updater())
            
            return True
            
        except Exception as e:
            logger.error(f"SME MessageBus initialization failed: {e}")
            return False
    
    async def send_sme_message(self, message: SMEMessage) -> bool:
        """Send SME-enhanced message with intelligent routing"""
        try:
            start_time = time.perf_counter()
            
            # Set timestamp
            message.timestamp = time.time()
            
            # Route SME workload if present
            if message.sme_workload and self.sme_enabled:
                routing_decision = await self.sme_hardware_router.route_matrix_workload(
                    message.sme_workload
                )
                
                # Add routing information to payload
                message.payload['sme_routing'] = {
                    'primary_resource': routing_decision.primary_resource.value,
                    'use_jit_kernels': routing_decision.use_jit_kernels,
                    'estimated_speedup': routing_decision.estimated_speedup,
                    'estimated_time_ms': routing_decision.estimated_execution_time_ms
                }
            
            # Determine routing strategy
            if message.priority >= 3:  # High priority
                await self._send_priority_message(message)
            elif message.sme_workload and self.sme_enabled:
                await self._send_sme_message(message)
            else:
                await self._send_standard_message(message)
            
            # Record routing time
            routing_time = (time.perf_counter() - start_time) * 1000
            self.routing_times.append(routing_time)
            
            # Update metrics
            self.message_metrics.total_messages += 1
            if message.sme_workload:
                self.message_metrics.sme_accelerated_messages += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SME message: {e}")
            return False
    
    async def _send_priority_message(self, message: SMEMessage) -> None:
        """Send high-priority message via priority queue"""
        priority_score = (3 - message.priority) * 1000 + int(time.time() * 1000) % 1000
        await self.sme_priority_queue.put((priority_score, message))
        
        # Also publish to Redis for immediate delivery
        channel = f"priority:{message.target_engine}" if message.target_engine else "priority:broadcast"
        await self.redis_client.publish(channel, json.dumps(asdict(message), default=str))
    
    async def _send_sme_message(self, message: SMEMessage) -> None:
        """Send SME-accelerated message via SME processing queue"""
        await self.sme_processing_queue.put(message)
        
        # Publish to Redis with SME routing info
        channel = f"sme:{message.target_engine}" if message.target_engine else "sme:broadcast"
        await self.redis_client.publish(channel, json.dumps(asdict(message), default=str))
    
    async def _send_standard_message(self, message: SMEMessage) -> None:
        """Send standard message via regular MessageBus"""
        channel = f"engine:{message.target_engine}" if message.target_engine else "engine:broadcast"
        await self.redis_client.publish(channel, json.dumps(asdict(message), default=str))
    
    async def _sme_message_processor(self) -> None:
        """Background processor for SME messages"""
        try:
            while True:
                # Process priority messages first
                if not self.sme_priority_queue.empty():
                    _, priority_message = await self.sme_priority_queue.get()
                    await self._process_sme_message(priority_message)
                
                # Process SME messages
                elif not self.sme_processing_queue.empty():
                    sme_message = await self.sme_processing_queue.get()
                    await self._process_sme_message(sme_message)
                
                else:
                    await asyncio.sleep(0.001)  # 1ms sleep to prevent busy waiting
                    
        except Exception as e:
            logger.error(f"SME message processor error: {e}")
    
    async def _process_sme_message(self, message: SMEMessage) -> None:
        """Process individual SME message"""
        try:
            processing_start = time.perf_counter()
            
            # Update throughput counter
            self.throughput_counter += 1
            
            # Log SME message processing (debug level)
            logger.debug(f"Processing SME message: {message.message_type.value} "
                        f"from {message.source_engine}")
            
            # Track processing time
            processing_time = (time.perf_counter() - processing_start) * 1000
            message.processing_time_ms = processing_time
            
            # Store processed message for potential retrieval
            self.pending_messages[message.id] = message
            
            # Clean old messages (keep last 1000)
            if len(self.pending_messages) > 1000:
                oldest_ids = sorted(self.pending_messages.keys())[:100]
                for old_id in oldest_ids:
                    del self.pending_messages[old_id]
                    
        except Exception as e:
            logger.error(f"SME message processing failed: {e}")
    
    async def subscribe_to_sme_messages(self, 
                                      engine_name: str, 
                                      message_types: List[SMEMessageType],
                                      callback) -> None:
        """Subscribe to SME messages for specific engine"""
        try:
            pubsub = self.redis_client.pubsub()
            
            # Subscribe to relevant channels
            channels = []
            for msg_type in message_types:
                channels.extend([
                    f"sme:{engine_name}",
                    f"priority:{engine_name}",
                    f"engine:{engine_name}",
                    "sme:broadcast",
                    "priority:broadcast",
                    "engine:broadcast"
                ])
            
            for channel in set(channels):  # Remove duplicates
                await pubsub.subscribe(channel)
            
            # Message processing loop
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        msg_data = json.loads(message['data'])
                        sme_msg = SMEMessage(**msg_data)
                        
                        # Filter by message types
                        if sme_msg.message_type in message_types:
                            await callback(sme_msg)
                            
                    except Exception as e:
                        logger.error(f"Error processing subscribed message: {e}")
                        
        except Exception as e:
            logger.error(f"SME subscription failed: {e}")
    
    async def send_matrix_operation_request(self,
                                          source_engine: str,
                                          target_engine: str,
                                          operation_type: str,
                                          matrix_data: Dict,
                                          priority: int = 2) -> str:
        """Send matrix operation request with SME routing"""
        
        # Create workload characteristics
        sme_workload = SMEWorkloadCharacteristics(
            operation_type=operation_type,
            matrix_dimensions=tuple(matrix_data.get('dimensions', (1, 1))),
            precision="fp32",
            workload_type=SMEWorkloadType.MEDIUM_MATRIX,
            priority=priority
        )
        
        # Create SME message
        message = SMEMessage(
            id=f"matrix_{int(time.time() * 1000000)}",
            message_type=SMEMessageType.MATRIX_OPERATION,
            source_engine=source_engine,
            target_engine=target_engine,
            payload={
                'operation_type': operation_type,
                'matrix_data': matrix_data,
                'request_timestamp': time.time()
            },
            priority=priority,
            sme_workload=sme_workload
        )
        
        # Send message
        success = await self.send_sme_message(message)
        
        return message.id if success else None
    
    async def send_performance_metric(self,
                                    engine_name: str,
                                    metric_name: str,
                                    metric_value: float,
                                    metadata: Optional[Dict] = None) -> bool:
        """Send performance metric via SME MessageBus"""
        
        message = SMEMessage(
            id=f"metric_{engine_name}_{int(time.time() * 1000000)}",
            message_type=SMEMessageType.PERFORMANCE_METRIC,
            source_engine=engine_name,
            target_engine=None,  # Broadcast
            payload={
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metadata': metadata or {},
                'timestamp': time.time()
            },
            priority=1  # Low priority for metrics
        )
        
        return await self.send_sme_message(message)
    
    async def _metrics_updater(self) -> None:
        """Background task to update SME MessageBus metrics"""
        try:
            while True:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Calculate throughput
                current_time = time.time()
                time_window = current_time - self.throughput_window_start
                
                if time_window > 0:
                    throughput = self.throughput_counter / time_window
                    if throughput > self.message_metrics.peak_throughput_mps:
                        self.message_metrics.peak_throughput_mps = throughput
                
                # Calculate average routing time
                if self.routing_times:
                    avg_routing_time = sum(self.routing_times) / len(self.routing_times)
                    self.message_metrics.average_routing_time_ms = avg_routing_time
                    
                    # Keep only recent routing times
                    self.routing_times = self.routing_times[-1000:]
                
                # Get SME hardware utilization
                if self.sme_enabled:
                    utilization = await self.sme_hardware_router.get_sme_utilization()
                    self.message_metrics.hardware_utilization = {
                        resource.value: util for resource, util in utilization.items()
                    }
                
                # Reset counters periodically
                if time_window > 60:  # Reset every minute
                    self.throughput_counter = 0
                    self.throughput_window_start = current_time
                    
        except Exception as e:
            logger.error(f"Metrics updater error: {e}")
    
    async def get_sme_bus_metrics(self) -> SMEBusMetrics:
        """Get SME MessageBus performance metrics"""
        return self.message_metrics
    
    async def get_message_queue_status(self) -> Dict:
        """Get SME message queue status"""
        return {
            "priority_queue_size": self.sme_priority_queue.qsize(),
            "sme_processing_queue_size": self.sme_processing_queue.qsize(),
            "pending_messages": len(self.pending_messages),
            "sme_enabled": self.sme_enabled,
            "redis_connected": self.redis_client is not None
        }
    
    async def send_sme_hardware_status(self, engine_name: str, hardware_stats: Dict) -> bool:
        """Send SME hardware status update"""
        
        message = SMEMessage(
            id=f"hw_status_{engine_name}_{int(time.time() * 1000000)}",
            message_type=SMEMessageType.HARDWARE_STATUS,
            source_engine=engine_name,
            target_engine=None,  # Broadcast to monitoring
            payload={
                'hardware_stats': hardware_stats,
                'timestamp': time.time(),
                'engine_name': engine_name
            },
            priority=2  # Medium priority
        )
        
        return await self.send_sme_message(message)
    
    async def broadcast_sme_routing_decision(self, 
                                           routing_info: Dict, 
                                           source_engine: str) -> bool:
        """Broadcast SME routing decision to interested engines"""
        
        message = SMEMessage(
            id=f"routing_{source_engine}_{int(time.time() * 1000000)}",
            message_type=SMEMessageType.ROUTING_DECISION,
            source_engine=source_engine,
            target_engine=None,  # Broadcast
            payload={
                'routing_decision': routing_info,
                'timestamp': time.time()
            },
            priority=1  # Low priority for routing info
        )
        
        return await self.send_sme_message(message)
    
    async def optimize_message_routing(self) -> Dict:
        """Optimize SME message routing based on performance data"""
        try:
            if not self.sme_enabled:
                return {"status": "sme_disabled"}
            
            # Get SME routing statistics
            routing_stats = await self.sme_hardware_router.get_routing_statistics()
            
            # Optimize routing strategy
            await self.sme_hardware_router.optimize_routing_strategy()
            
            # Update MessageBus routing parameters based on performance
            if self.message_metrics.average_routing_time_ms > 10.0:
                # If routing is slow, increase batch sizes
                logger.info("Optimizing MessageBus routing for high latency")
            
            if self.message_metrics.peak_throughput_mps > 1000:
                # If throughput is high, optimize for concurrent processing
                logger.info("Optimizing MessageBus for high throughput")
            
            return {
                "status": "optimized",
                "routing_stats": routing_stats,
                "current_metrics": asdict(self.message_metrics)
            }
            
        except Exception as e:
            logger.error(f"MessageBus routing optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close(self) -> None:
        """Close SME MessageBus connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            logger.info("✅ SME MessageBus closed")
            
        except Exception as e:
            logger.error(f"SME MessageBus close error: {e}")

# Utility functions for SME MessageBus integration

async def create_sme_messagebus(redis_host: str = "redis", redis_port: int = 6379) -> SMEEnhancedMessageBus:
    """Create and initialize SME-enhanced MessageBus"""
    messagebus = SMEEnhancedMessageBus(redis_host, redis_port)
    
    if await messagebus.initialize():
        return messagebus
    else:
        raise RuntimeError("Failed to initialize SME MessageBus")

async def send_matrix_operation(messagebus: SMEEnhancedMessageBus,
                              source: str,
                              target: str,
                              operation: str,
                              matrix_data: Dict,
                              priority: int = 2) -> Optional[str]:
    """Convenience function to send matrix operation request"""
    return await messagebus.send_matrix_operation_request(
        source, target, operation, matrix_data, priority
    )

async def broadcast_performance_metric(messagebus: SMEEnhancedMessageBus,
                                     engine_name: str,
                                     metric_name: str,
                                     metric_value: float,
                                     metadata: Optional[Dict] = None) -> bool:
    """Convenience function to broadcast performance metric"""
    return await messagebus.send_performance_metric(
        engine_name, metric_name, metric_value, metadata
    )
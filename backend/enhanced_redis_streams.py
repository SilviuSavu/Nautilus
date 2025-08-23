"""
Enhanced Redis Streams Management
Advanced stream operations inspired by NautilusTrader's Redis implementation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
import json

import redis.asyncio as redis
from enhanced_messagebus_client import MessageBusMessage, MessageBusConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamInfo:
    """Information about a Redis stream"""
    name: str
    length: int
    last_generated_id: str
    first_entry_id: str
    last_entry_id: str
    max_deleted_entry_id: str
    entries_added: int
    radix_tree_keys: int
    radix_tree_nodes: int
    groups: int
    last_trimmed: Optional[datetime] = None
    consumer_groups: List[str] = field(default_factory=list)


@dataclass  
class ConsumerGroupInfo:
    """Information about a consumer group"""
    name: str
    consumers: int
    pending: int
    last_delivered_id: str
    entries_read: int
    lag: int


@dataclass
class StreamMetrics:
    """Stream performance metrics"""
    stream_name: str
    messages_per_second: float
    bytes_per_second: float
    average_message_size: float
    peak_throughput: float
    consumer_lag: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class RedisStreamManager:
    """
    Advanced Redis stream management inspired by NautilusTrader
    Handles multi-stream operations, automatic management, and performance monitoring
    """
    
    def __init__(self, config: MessageBusConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        
        # Stream tracking
        self.managed_streams: Set[str] = set()
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        self.last_metrics_update = time.time()
        
        # Consumer group management
        self.consumer_groups: Dict[str, ConsumerGroupInfo] = {}
        
        # Auto-management settings
        self.auto_trim_enabled = config.autotrim_mins > 0
        self.metrics_interval = 60  # Update metrics every minute
        
    async def initialize(self, redis_client: redis.Redis) -> None:
        """Initialize stream manager with Redis client"""
        self.redis_client = redis_client
        
        # Discover existing streams
        await self._discover_existing_streams()
        
        # Setup consumer groups for managed streams
        for stream_name in self.managed_streams:
            await self._ensure_consumer_group(stream_name)
        
        logger.info(f"Redis stream manager initialized with {len(self.managed_streams)} streams")
    
    async def _discover_existing_streams(self) -> None:
        """Discover existing streams matching our patterns"""
        try:
            # Find streams matching our prefix
            pattern = f"{self.config.stream_key}*"
            stream_keys = await self.redis_client.keys(pattern)
            
            for key in stream_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                # Verify it's actually a stream
                try:
                    await self.redis_client.xlen(key)
                    self.managed_streams.add(key)
                    logger.debug(f"Discovered existing stream: {key}")
                except redis.RedisError:
                    continue  # Not a stream
                    
        except Exception as e:
            logger.error(f"Error discovering streams: {e}")
    
    async def _ensure_consumer_group(self, stream_name: str) -> None:
        """Ensure consumer group exists for stream"""
        try:
            await self.redis_client.xgroup_create(
                stream_name,
                self.config.consumer_group,
                id='0',
                mkstream=True
            )
            logger.debug(f"Ensured consumer group {self.config.consumer_group} for stream {stream_name}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {self.config.consumer_group} already exists for {stream_name}")
            else:
                logger.error(f"Error creating consumer group: {e}")
    
    async def create_stream(self, stream_name: str) -> bool:
        """Create a new stream with initial setup"""
        try:
            # Add dummy entry to create stream, then remove it
            entry_id = await self.redis_client.xadd(
                stream_name, 
                {"_init": "true"},
                id="*"
            )
            
            # Remove the dummy entry
            await self.redis_client.xdel(stream_name, entry_id)
            
            # Setup consumer group
            await self._ensure_consumer_group(stream_name)
            
            # Track the stream
            self.managed_streams.add(stream_name)
            
            logger.info(f"Created and configured stream: {stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating stream {stream_name}: {e}")
            return False
    
    async def get_stream_info(self, stream_name: str) -> Optional[StreamInfo]:
        """Get detailed information about a stream"""
        try:
            info = await self.redis_client.xinfo_stream(stream_name)
            
            # Extract consumer groups
            groups_info = await self.redis_client.xinfo_groups(stream_name)
            consumer_groups = [group['name'].decode('utf-8') if isinstance(group['name'], bytes) else group['name'] 
                             for group in groups_info]
            
            return StreamInfo(
                name=stream_name,
                length=info.get('length', 0),
                last_generated_id=info.get('last-generated-id', '0-0'),
                first_entry_id=info.get('first-entry', {}).get('id', '0-0') if info.get('first-entry') else '0-0',
                last_entry_id=info.get('last-entry', {}).get('id', '0-0') if info.get('last-entry') else '0-0',
                max_deleted_entry_id=info.get('max-deleted-entry-id', '0-0'),
                entries_added=info.get('entries-added', 0),
                radix_tree_keys=info.get('radix-tree-keys', 0),
                radix_tree_nodes=info.get('radix-tree-nodes', 0),
                groups=len(consumer_groups),
                consumer_groups=consumer_groups
            )
            
        except Exception as e:
            logger.error(f"Error getting stream info for {stream_name}: {e}")
            return None
    
    async def get_consumer_group_info(self, stream_name: str, group_name: str) -> Optional[ConsumerGroupInfo]:
        """Get consumer group information"""
        try:
            groups_info = await self.redis_client.xinfo_groups(stream_name)
            
            for group in groups_info:
                if group['name'].decode('utf-8') == group_name:
                    return ConsumerGroupInfo(
                        name=group_name,
                        consumers=group.get('consumers', 0),
                        pending=group.get('pending', 0),
                        last_delivered_id=group.get('last-delivered-id', '0-0'),
                        entries_read=group.get('entries-read', 0),
                        lag=group.get('lag', 0)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting consumer group info: {e}")
            return None
    
    async def trim_stream(self, stream_name: str, max_len: Optional[int] = None, 
                         min_id: Optional[str] = None, approximate: bool = True) -> int:
        """Trim stream using XTRIM command"""
        try:
            if max_len is not None:
                # Trim by maximum length
                result = await self.redis_client.xtrim(
                    stream_name, 
                    maxlen=max_len, 
                    approximate=approximate
                )
            elif min_id is not None:
                # Trim by minimum ID (time-based)
                result = await self.redis_client.xtrim(
                    stream_name,
                    minid=min_id,
                    approximate=approximate
                )
            else:
                return 0
            
            logger.debug(f"Trimmed {result} entries from stream {stream_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error trimming stream {stream_name}: {e}")
            return 0
    
    async def auto_trim_streams(self) -> Dict[str, int]:
        """Auto-trim all managed streams based on retention policy"""
        if not self.auto_trim_enabled:
            return {}
        
        results = {}
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.config.autotrim_mins)
        cutoff_timestamp_ms = int(cutoff_time.timestamp() * 1000)
        min_id = f"{cutoff_timestamp_ms}-0"
        
        for stream_name in self.managed_streams:
            try:
                trimmed = await self.trim_stream(stream_name, min_id=min_id, approximate=True)
                results[stream_name] = trimmed
                
                # Update stream info
                if stream_name in self.stream_metrics:
                    # Mark as trimmed for metrics
                    pass
                    
            except Exception as e:
                logger.error(f"Error auto-trimming stream {stream_name}: {e}")
                results[stream_name] = -1
        
        logger.info(f"Auto-trimmed {len(results)} streams, removed {sum(v for v in results.values() if v > 0)} total entries")
        return results
    
    async def update_stream_metrics(self) -> None:
        """Update performance metrics for all managed streams"""
        current_time = time.time()
        
        for stream_name in self.managed_streams:
            try:
                # Get stream info
                info = await self.get_stream_info(stream_name)
                if not info:
                    continue
                
                # Calculate metrics
                if stream_name in self.stream_metrics:
                    old_metrics = self.stream_metrics[stream_name]
                    time_delta = current_time - self.last_metrics_update
                    
                    if time_delta > 0:
                        # Calculate messages per second
                        entries_delta = info.entries_added - (old_metrics.messages_per_second * time_delta)
                        messages_per_second = entries_delta / time_delta
                        
                        # Estimate bytes per second (rough approximation)
                        avg_message_size = old_metrics.average_message_size or 1000  # Default 1KB
                        bytes_per_second = messages_per_second * avg_message_size
                        
                        # Update metrics
                        self.stream_metrics[stream_name] = StreamMetrics(
                            stream_name=stream_name,
                            messages_per_second=messages_per_second,
                            bytes_per_second=bytes_per_second,
                            average_message_size=avg_message_size,
                            peak_throughput=max(old_metrics.peak_throughput, messages_per_second),
                            last_updated=datetime.utcnow()
                        )
                else:
                    # Initialize metrics
                    self.stream_metrics[stream_name] = StreamMetrics(
                        stream_name=stream_name,
                        messages_per_second=0.0,
                        bytes_per_second=0.0,
                        average_message_size=1000.0,  # Default estimate
                        peak_throughput=0.0
                    )
                
            except Exception as e:
                logger.error(f"Error updating metrics for stream {stream_name}: {e}")
        
        self.last_metrics_update = current_time
    
    async def get_stream_metrics(self, stream_name: str) -> Optional[StreamMetrics]:
        """Get performance metrics for a specific stream"""
        return self.stream_metrics.get(stream_name)
    
    async def get_all_metrics(self) -> Dict[str, StreamMetrics]:
        """Get performance metrics for all managed streams"""
        await self.update_stream_metrics()
        return self.stream_metrics.copy()
    
    async def consume_from_multiple_streams(self, 
                                          stream_keys: List[str], 
                                          consumer_name: str,
                                          block_ms: int = 1000,
                                          count: int = 10) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Consume from multiple streams simultaneously"""
        try:
            # Prepare last IDs for each stream (start from end)
            last_ids = ['$' for _ in stream_keys]
            
            # Read from multiple streams
            result = await self.redis_client.xreadgroup(
                self.config.consumer_group,
                consumer_name,
                {stream: last_id for stream, last_id in zip(stream_keys, last_ids)},
                count=count,
                block=block_ms
            )
            
            # Process results
            processed_results = []
            for stream_name, messages in result:
                if isinstance(stream_name, bytes):
                    stream_name = stream_name.decode('utf-8')
                
                processed_messages = []
                for message_id, fields in messages:
                    processed_message = {
                        'id': message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id,
                        'fields': {
                            k.decode('utf-8') if isinstance(k, bytes) else k: 
                            v.decode('utf-8') if isinstance(v, bytes) else v 
                            for k, v in fields.items()
                        }
                    }
                    processed_messages.append(processed_message)
                
                processed_results.append((stream_name, processed_messages))
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error consuming from multiple streams: {e}")
            return []
    
    async def acknowledge_messages(self, stream_name: str, message_ids: List[str]) -> int:
        """Acknowledge processed messages"""
        try:
            result = await self.redis_client.xack(
                stream_name,
                self.config.consumer_group,
                *message_ids
            )
            logger.debug(f"Acknowledged {result} messages from {stream_name}")
            return result
        except Exception as e:
            logger.error(f"Error acknowledging messages: {e}")
            return 0
    
    async def get_pending_messages(self, stream_name: str, 
                                  consumer_name: Optional[str] = None,
                                  min_idle_time: int = 0) -> List[Dict[str, Any]]:
        """Get pending (unacknowledged) messages"""
        try:
            if consumer_name:
                # Get pending for specific consumer
                result = await self.redis_client.xpending_range(
                    stream_name,
                    self.config.consumer_group,
                    consumer=consumer_name,
                    min='-',
                    max='+',
                    count=100
                )
            else:
                # Get overall pending info
                result = await self.redis_client.xpending(
                    stream_name,
                    self.config.consumer_group
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting pending messages: {e}")
            return []
    
    async def claim_stale_messages(self, stream_name: str, 
                                  consumer_name: str,
                                  min_idle_time_ms: int = 60000) -> List[Dict[str, Any]]:
        """Claim messages that have been pending too long"""
        try:
            # Get pending messages older than min_idle_time
            pending = await self.redis_client.xpending_range(
                stream_name,
                self.config.consumer_group,
                min='-',
                max='+',
                count=100,
                min_idle_time=min_idle_time_ms
            )
            
            if not pending:
                return []
            
            # Extract message IDs
            message_ids = [msg['message_id'] for msg in pending]
            
            # Claim the messages
            claimed = await self.redis_client.xclaim(
                stream_name,
                self.config.consumer_group,
                consumer_name,
                min_idle_time_ms,
                message_ids
            )
            
            logger.info(f"Claimed {len(claimed)} stale messages from {stream_name}")
            return claimed
            
        except Exception as e:
            logger.error(f"Error claiming stale messages: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all managed streams"""
        health_info = {
            'status': 'healthy',
            'managed_streams': len(self.managed_streams),
            'streams': {},
            'consumer_groups': {},
            'issues': []
        }
        
        for stream_name in self.managed_streams:
            try:
                # Check stream exists and get info
                info = await self.get_stream_info(stream_name)
                if info:
                    health_info['streams'][stream_name] = {
                        'length': info.length,
                        'groups': info.groups,
                        'status': 'healthy'
                    }
                    
                    # Check consumer group health
                    group_info = await self.get_consumer_group_info(stream_name, self.config.consumer_group)
                    if group_info:
                        health_info['consumer_groups'][f"{stream_name}:{self.config.consumer_group}"] = {
                            'consumers': group_info.consumers,
                            'pending': group_info.pending,
                            'lag': group_info.lag,
                            'status': 'healthy' if group_info.lag < 1000 else 'degraded'
                        }
                        
                        # Check for high lag
                        if group_info.lag > 1000:
                            health_info['issues'].append(f"High lag ({group_info.lag}) in {stream_name}")
                else:
                    health_info['streams'][stream_name] = {'status': 'error'}
                    health_info['issues'].append(f"Cannot access stream {stream_name}")
                    
            except Exception as e:
                health_info['streams'][stream_name] = {'status': 'error', 'error': str(e)}
                health_info['issues'].append(f"Error checking {stream_name}: {e}")
        
        # Overall health status
        if health_info['issues']:
            health_info['status'] = 'degraded' if len(health_info['issues']) < len(self.managed_streams) else 'error'
        
        return health_info


class StreamConsumer:
    """Advanced stream consumer with automatic message handling"""
    
    def __init__(self, stream_manager: RedisStreamManager, consumer_name: str):
        self.stream_manager = stream_manager
        self.consumer_name = consumer_name
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.running = False
        
    def add_handler(self, topic_pattern: str, handler: Callable):
        """Add message handler for specific topic pattern"""
        if topic_pattern not in self.message_handlers:
            self.message_handlers[topic_pattern] = []
        self.message_handlers[topic_pattern].append(handler)
    
    async def start_consuming(self, stream_names: List[str]):
        """Start consuming from specified streams"""
        self.running = True
        logger.info(f"Starting consumer {self.consumer_name} for streams: {stream_names}")
        
        while self.running:
            try:
                # Consume messages from multiple streams
                results = await self.stream_manager.consume_from_multiple_streams(
                    stream_names, 
                    self.consumer_name,
                    block_ms=1000,
                    count=10
                )
                
                # Process messages
                for stream_name, messages in results:
                    await self._process_messages(stream_name, messages)
                    
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_messages(self, stream_name: str, messages: List[Dict[str, Any]]):
        """Process messages from a stream"""
        message_ids = []
        
        for message in messages:
            try:
                # Extract message data
                fields = message['fields']
                topic = fields.get('topic', '')
                payload = fields.get('payload', '')
                
                # Create MessageBusMessage
                if payload:
                    try:
                        payload_data = json.loads(payload)
                    except json.JSONDecodeError:
                        payload_data = payload
                else:
                    payload_data = {}
                
                msg = MessageBusMessage(
                    topic=topic,
                    payload=payload_data,
                    timestamp=int(fields.get('timestamp', 0)),
                    message_type=fields.get('message_type', 'data'),
                    message_id=fields.get('message_id', message['id'])
                )
                
                # Find and call handlers
                handled = False
                for pattern, handlers in self.message_handlers.items():
                    if self._matches_pattern(pattern, topic):
                        for handler in handlers:
                            try:
                                await handler(msg)
                                handled = True
                            except Exception as e:
                                logger.error(f"Error in message handler: {e}")
                
                if handled:
                    message_ids.append(message['id'])
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        # Acknowledge processed messages
        if message_ids:
            await self.stream_manager.acknowledge_messages(stream_name, message_ids)
    
    def _matches_pattern(self, pattern: str, topic: str) -> bool:
        """Simple pattern matching for topics"""
        if pattern == '*':
            return True
        if '*' not in pattern:
            return pattern == topic
        
        # Simple wildcard matching
        if pattern.endswith('*'):
            return topic.startswith(pattern[:-1])
        if pattern.startswith('*'):
            return topic.endswith(pattern[1:])
        
        return False
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        logger.info(f"Consumer {self.consumer_name} stopped")
"""
MessageBus Client for NautilusTrader Integration
Provides WebSocket connection to NautilusTrader's MessageBus via Redis streams.
"""

import asyncio
import json
import logging
import time
from asyncio import Queue
from enum import Enum
from typing import Any, Callable, List
from datetime import datetime

import redis.asyncio as redis
from pydantic import BaseModel


class ConnectionState(Enum):
    """Connection states for MessageBus client"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MessageBusMessage(BaseModel):
    """Standard message format for MessageBus"""
    topic: str
    payload: dict[str, Any]
    timestamp: int
    message_type: str = "data"


class ConnectionStatus(BaseModel):
    """Connection status information"""
    state: ConnectionState
    connected_at: datetime | None = None
    last_message_at: datetime | None = None
    reconnect_attempts: int = 0
    error_message: str | None = None
    messages_received: int = 0


class MessageBusClient:
    """
    MessageBus client that connects to NautilusTrader via Redis streams
    with automatic reconnection and health monitoring.
    """
    
    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0, stream_key: str = "nautilus-streams", consumer_group: str = "dashboard-group", consumer_name: str = "dashboard-consumer", max_reconnect_attempts: int = 10, reconnect_base_delay: float = 1.0, reconnect_max_delay: float = 60.0, connection_timeout: float = 5.0, health_check_interval: float = 30.0, ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        
        # Connection management
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        
        # State
        self._redis_client: redis.Redis | None = None
        self._connection_status = ConnectionStatus(state=ConnectionState.DISCONNECTED)
        self._message_handlers: list[Callable[[MessageBusMessage], None]] = []
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._message_queue: Queue = Queue()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    @property
    def connection_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self._connection_status
        
    @property
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._connection_status.state == ConnectionState.CONNECTED
        
    def add_message_handler(self, handler: Callable[[MessageBusMessage], None]) -> None:
        """Add a message handler callback"""
        self._message_handlers.append(handler)
        
    def remove_message_handler(self, handler: Callable[[MessageBusMessage], None]) -> None:
        """Remove a message handler callback"""
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
            
    async def start(self) -> None:
        """Start the MessageBus client"""
        if self._running:
            return
            
        self.logger.info("Starting MessageBus client...")
        self._running = True
        
        # Start connection and monitoring tasks
        connection_task = asyncio.create_task(self._connection_manager())
        message_task = asyncio.create_task(self._message_processor()) 
        consume_task = asyncio.create_task(self._consume_messages())
        health_task = asyncio.create_task(self._health_monitor())
        
        self._tasks.extend([connection_task, message_task, consume_task, health_task])
        
    async def stop(self) -> None:
        """Stop the MessageBus client"""
        if not self._running:
            return
            
        self.logger.info("Stopping MessageBus client...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        # Close Redis connection
        if self._redis_client:
            await self._redis_client.aclose()
            
        self._connection_status.state = ConnectionState.DISCONNECTED
        self.logger.info("MessageBus client stopped")
        
    async def _connection_manager(self) -> None:
        """Manage Redis connection with automatic reconnection"""
        reconnect_attempts = 0
        
        while self._running:
            try:
                if not self._redis_client or not await self._is_redis_connected():
                    self._connection_status.state = ConnectionState.CONNECTING
                    self.logger.info(f"Connecting to Redis at {self.redis_host}:{self.redis_port}")
                    
                    # Create Redis client
                    self._redis_client = redis.Redis(
                        host=self.redis_host, port=self.redis_port, db=self.redis_db, socket_connect_timeout=self.connection_timeout, socket_keepalive=True, socket_keepalive_options={}, retry_on_timeout=True, decode_responses=True
                    )
                    
                    # Test connection
                    await asyncio.wait_for(
                        self._redis_client.ping(), timeout=self.connection_timeout
                    )
                    
                    # Setup consumer group
                    await self._setup_consumer_group()
                    
                    # Connection successful
                    self._connection_status.state = ConnectionState.CONNECTED
                    self._connection_status.connected_at = datetime.now()
                    self._connection_status.error_message = None
                    reconnect_attempts = 0
                    
                    self.logger.info("Successfully connected to MessageBus")
                    
                    # Wait while connected - check periodically for disconnection
                    while self._running and await self._is_redis_connected():
                        await asyncio.sleep(5.0)  # Check every 5 seconds
                    
            except Exception as e:
                reconnect_attempts += 1
                self._connection_status.state = ConnectionState.RECONNECTING
                self._connection_status.reconnect_attempts = reconnect_attempts
                self._connection_status.error_message = str(e)
                
                self.logger.error(f"Connection failed: {e}")
                
                if reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error("Max reconnection attempts reached")
                    self._connection_status.state = ConnectionState.ERROR
                    break
                    
                # Exponential backoff
                delay = min(
                    self.reconnect_base_delay * (2 ** (reconnect_attempts - 1)), self.reconnect_max_delay
                )
                self.logger.info(f"Reconnecting in {delay:.1f} seconds (attempt {reconnect_attempts})")
                await asyncio.sleep(delay)
                
    async def _is_redis_connected(self) -> bool:
        """Check if Redis connection is active"""
        if not self._redis_client:
            return False
            
        try:
            await self._redis_client.ping()
            return True
        except Exception:
            return False
            
    async def _setup_consumer_group(self) -> None:
        """Setup Redis consumer group for reading streams"""
        try:
            # Create consumer group (ignore if already exists)
            await self._redis_client.xgroup_create(
                self.stream_key, self.consumer_group, id="0", mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
                
    async def _consume_messages(self) -> None:
        """Consume messages from Redis streams"""
        while self._running:
            # Wait for connection to be established
            if not self.is_connected or not self._redis_client:
                await asyncio.sleep(0.1)
                continue
                
            try:
                # Read from streams with timeout
                streams = await self._redis_client.xreadgroup(
                    self.consumer_group, self.consumer_name, {self.stream_key: ">"}, count=10, block=1000  # 1 second timeout
                )
                
                for stream_name, messages in streams:
                    for message_id, fields in messages:
                        await self._process_stream_message(message_id, fields)
                        
                        # Acknowledge message
                        await self._redis_client.xack(
                            self.stream_key, self.consumer_group, message_id
                        )
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error consuming messages: {e}")
                # On error, wait a bit before retrying
                await asyncio.sleep(1.0)
                
    async def _process_stream_message(self, message_id: str, fields: dict[str, str]) -> None:
        """Process a message from Redis stream"""
        try:
            # Parse message fields with better error handling
            topic = fields.get("topic", "unknown")
            payload_str = fields.get("payload", "{}")
            
            # Handle timestamp parsing more robustly
            try:
                timestamp = int(fields.get("timestamp", time.time_ns()))
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid timestamp in message {message_id}, using current time")
                timestamp = time.time_ns()
                
            message_type = fields.get("type", "data")
            
            # Parse payload JSON with better error handling
            try:
                payload = json.loads(payload_str) if payload_str else {}
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON payload in message {message_id}: {e}")
                payload = {"raw_payload": payload_str, "parse_error": str(e)}
            
            # Create message object
            message = MessageBusMessage(
                topic=topic, payload=payload, timestamp=timestamp, message_type=message_type
            )
            
            # Update stats atomically
            self._connection_status.messages_received += 1
            self._connection_status.last_message_at = datetime.now()
            
            # Queue message for processing
            await self._message_queue.put(message)
            
        except Exception as e:
            self.logger.error(f"Error processing stream message {message_id}: {e}")
            # Don't re-raise to avoid breaking the message consumption loop
            
    async def _message_processor(self) -> None:
        """Process queued messages and call handlers"""
        while self._running:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(), timeout=1.0
                )
                
                # Call all message handlers
                for handler in self._message_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
                
    async def _health_monitor(self) -> None:
        """Monitor connection health"""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.is_connected and not await self._is_redis_connected():
                    self.logger.warning("Health check failed - connection lost")
                    self._connection_status.state = ConnectionState.DISCONNECTED
                    
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    async def broadcast_event(self, event_type: str, data: dict[str, Any]) -> bool:
        """
        Broadcast an event to all connected WebSocket clients via Redis
        
        Args:
            event_type: Type of event (e.g., 'nautilus_engine_status')
            data: Event data to broadcast
            
        Returns:
            bool: True if broadcast succeeded, False otherwise
        """
        try:
            if not self.is_connected:
                self.logger.warning("Cannot broadcast event - not connected to Redis")
                return False
            
            # Create message for broadcasting
            message = {
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to Redis stream for WebSocket clients to consume
            stream_name = "websocket:broadcasts"
            await self.redis_client.xadd(stream_name, message)
            
            self.logger.debug(f"Broadcasted event '{event_type}' to WebSocket clients")
            return True
            
        except Exception as e:
            self.logger.error(f"Error broadcasting event: {e}")
            return False
    
    async def send_message(self, topic: str, payload: dict[str, Any], message_type: str = "data") -> bool:
        """
        Send a message to a specific topic via Redis
        
        Args:
            topic: Message topic
            payload: Message payload
            message_type: Type of message
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            if not self.is_connected:
                self.logger.warning("Cannot send message - not connected to Redis")
                return False
            
            message = {
                "topic": topic,
                "payload": json.dumps(payload),
                "timestamp": int(time.time()),
                "message_type": message_type
            }
            
            # Send to Redis stream
            stream_name = f"messages:{topic}"
            await self.redis_client.xadd(stream_name, message)
            
            self.logger.debug(f"Sent message to topic '{topic}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False


# Global client instance
messagebus_client = MessageBusClient()


def get_messagebus_client() -> MessageBusClient:
    """Get the global MessageBus client instance"""
    return messagebus_client
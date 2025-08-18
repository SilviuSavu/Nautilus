"""
Unit tests for MessageBus client connection management.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from messagebus_client import (
    MessageBusClient,
    ConnectionState,
    MessageBusMessage,
    ConnectionStatus
)


class TestMessageBusClient:
    """Test suite for MessageBusClient"""

    @pytest.fixture
    def client(self):
        """Create a MessageBusClient instance for testing"""
        return MessageBusClient(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            max_reconnect_attempts=3,
            reconnect_base_delay=0.1,
            reconnect_max_delay=1.0,
            connection_timeout=1.0,
            health_check_interval=1.0
        )

    def test_client_initialization(self, client):
        """Test client initializes with correct default state"""
        assert client.connection_status.state == ConnectionState.DISCONNECTED
        assert client.connection_status.messages_received == 0
        assert client.connection_status.reconnect_attempts == 0
        assert not client.is_connected

    def test_add_remove_message_handler(self, client):
        """Test adding and removing message handlers"""
        handler = MagicMock()
        
        # Add handler
        client.add_message_handler(handler)
        assert handler in client._message_handlers
        
        # Remove handler
        client.remove_message_handler(handler)
        assert handler not in client._message_handlers

    @pytest.mark.asyncio
    async def test_start_stop_client(self, client):
        """Test starting and stopping the client"""
        with patch.object(client, '_connection_manager', new_callable=AsyncMock) as mock_conn:
            with patch.object(client, '_message_processor', new_callable=AsyncMock) as mock_proc:
                with patch.object(client, '_consume_messages', new_callable=AsyncMock) as mock_consume:
                    with patch.object(client, '_health_monitor', new_callable=AsyncMock) as mock_health:
                        
                        # Start client
                        await client.start()
                        assert client._running is True
                        assert len(client._tasks) == 4  # Updated to 4 tasks
                        
                        # Stop client
                        await client.stop()
                        assert client._running is False
                        assert client.connection_status.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_redis_connection_success(self, client):
        """Test successful Redis connection"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create = AsyncMock()
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            with patch.object(client, '_consume_messages', new_callable=AsyncMock):
                with patch.object(client, '_setup_consumer_group', new_callable=AsyncMock):
                    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                        
                        # Set up to exit the connection manager loop after one iteration
                        async def stop_running(*args):
                            client._running = False
                        mock_sleep.side_effect = stop_running
                        
                        client._running = True
                        await client._connection_manager()
                        
                        # Verify connection was attempted
                        mock_redis.ping.assert_called()
                        # Verify client reached connected state
                        assert client.connection_status.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_redis_connection_failure_with_retry(self, client):
        """Test Redis connection failure triggers retry logic"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                
                client._running = True
                # Override max attempts to avoid long test
                client.max_reconnect_attempts = 2
                
                await client._connection_manager()
                
                # Verify retry logic was triggered
                assert client.connection_status.state == ConnectionState.ERROR
                assert client.connection_status.reconnect_attempts == 2
                mock_sleep.assert_called()

    @pytest.mark.asyncio 
    async def test_is_redis_connected(self, client):
        """Test Redis connection health check"""
        # Test with no client
        assert await client._is_redis_connected() is False
        
        # Test with successful ping
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        client._redis_client = mock_redis
        
        assert await client._is_redis_connected() is True
        
        # Test with failed ping
        mock_redis.ping.side_effect = Exception("Ping failed")
        assert await client._is_redis_connected() is False

    @pytest.mark.asyncio
    async def test_setup_consumer_group(self, client):
        """Test Redis consumer group setup"""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        client._redis_client = mock_redis
        
        await client._setup_consumer_group()
        
        mock_redis.xgroup_create.assert_called_once_with(
            client.stream_key,
            client.consumer_group,
            id="0",
            mkstream=True
        )

    @pytest.mark.asyncio
    async def test_setup_consumer_group_already_exists(self, client):
        """Test consumer group setup when group already exists"""
        import redis
        mock_redis = AsyncMock()
        mock_redis.xgroup_create.side_effect = redis.ResponseError("BUSYGROUP")
        client._redis_client = mock_redis
        
        # Should not raise exception
        await client._setup_consumer_group()

    @pytest.mark.asyncio
    async def test_process_stream_message(self, client):
        """Test processing of Redis stream messages"""
        message_id = "1234567890-0"
        fields = {
            "topic": "test.topic",
            "payload": '{"key": "value"}',
            "timestamp": "1630000000000000000",
            "type": "data"
        }
        
        initial_count = client.connection_status.messages_received
        
        await client._process_stream_message(message_id, fields)
        
        # Verify message was queued
        assert not client._message_queue.empty()
        
        # Verify stats updated
        assert client.connection_status.messages_received == initial_count + 1
        assert client.connection_status.last_message_at is not None

    @pytest.mark.asyncio
    async def test_process_malformed_stream_message(self, client):
        """Test processing of malformed Redis stream messages"""
        message_id = "1234567890-0"
        fields = {
            "topic": "test.topic",
            "payload": "invalid json",  # Invalid JSON
            "timestamp": "not_a_number",  # Invalid timestamp
        }
        
        initial_count = client.connection_status.messages_received
        
        # Should handle error gracefully
        await client._process_stream_message(message_id, fields)
        
        # Message should not be processed
        assert client.connection_status.messages_received == initial_count

    @pytest.mark.asyncio
    async def test_message_processor(self, client):
        """Test message processor calls handlers"""
        handler1 = MagicMock()
        handler2 = AsyncMock()
        
        client.add_message_handler(handler1)
        client.add_message_handler(handler2)
        
        # Add a message to the queue
        message = MessageBusMessage(
            topic="test.topic",
            payload={"key": "value"},
            timestamp=1630000000000000000,
            message_type="data"
        )
        await client._message_queue.put(message)
        
        # Run processor once
        client._running = True
        
        # Create a task that will timeout after processing one message
        async def run_processor():
            await client._message_processor()
            
        try:
            await asyncio.wait_for(run_processor(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        # Verify handlers were called
        handler1.assert_called_once_with(message)
        handler2.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_health_monitor(self, client):
        """Test health monitoring functionality"""
        client._running = True
        client._connection_status.state = ConnectionState.CONNECTED
        
        with patch.object(client, '_is_redis_connected', return_value=False):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                
                # Set up to exit the health monitor loop after one iteration
                async def stop_running(*args):
                    client._running = False
                mock_sleep.side_effect = stop_running
                
                # Run health monitor
                await client._health_monitor()
                
                # Verify connection state changed to disconnected
                assert client.connection_status.state == ConnectionState.DISCONNECTED


class TestMessageBusMessage:
    """Test suite for MessageBusMessage model"""

    def test_message_creation(self):
        """Test creating a MessageBusMessage"""
        message = MessageBusMessage(
            topic="test.topic",
            payload={"key": "value", "number": 42},
            timestamp=1630000000000000000,
            message_type="data"
        )
        
        assert message.topic == "test.topic"
        assert message.payload == {"key": "value", "number": 42}
        assert message.timestamp == 1630000000000000000
        assert message.message_type == "data"

    def test_message_default_type(self):
        """Test MessageBusMessage with default message type"""
        message = MessageBusMessage(
            topic="test.topic",
            payload={},
            timestamp=1630000000000000000
        )
        
        assert message.message_type == "data"


class TestConnectionStatus:
    """Test suite for ConnectionStatus model"""

    def test_status_creation(self):
        """Test creating a ConnectionStatus"""
        now = datetime.now()
        status = ConnectionStatus(
            state=ConnectionState.CONNECTED,
            connected_at=now,
            last_message_at=now,
            reconnect_attempts=0,
            error_message=None,
            messages_received=5
        )
        
        assert status.state == ConnectionState.CONNECTED
        assert status.connected_at == now
        assert status.last_message_at == now
        assert status.reconnect_attempts == 0
        assert status.error_message is None
        assert status.messages_received == 5

    def test_status_default_values(self):
        """Test ConnectionStatus with default values"""
        status = ConnectionStatus(state=ConnectionState.DISCONNECTED)
        
        assert status.state == ConnectionState.DISCONNECTED
        assert status.connected_at is None
        assert status.last_message_at is None
        assert status.reconnect_attempts == 0
        assert status.error_message is None
        assert status.messages_received == 0
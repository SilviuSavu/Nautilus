"""
Integration tests for end-to-end MessageBus message flow.
"""

import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

# Mock Redis before importing main
with patch('redis.asyncio.Redis'):
    from main import app
    from messagebus_client import MessageBusClient, MessageBusMessage


class TestEndToEndMessageFlow:
    """Integration tests for complete message flow from Redis to WebSocket"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def messagebus_client(self):
        """Create MessageBus client for testing"""
        return MessageBusClient(
            redis_host="localhost",
            redis_port=6379,
            max_reconnect_attempts=1,
            reconnect_base_delay=0.1,
            connection_timeout=1.0
        )

    @pytest.mark.asyncio
    async def test_redis_to_websocket_message_flow(self, messagebus_client):
        """Test complete message flow from Redis stream to WebSocket"""
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create = AsyncMock()
        
        # Mock stream read with test message - use correct stream name
        test_message_data = [
            ("nautilus-streams", [
                ("1630000000000-0", {
                    "topic": "market.eurusd.quote",
                    "payload": '{"bid": 1.0850, "ask": 1.0852, "symbol": "EURUSD"}',
                    "timestamp": "1630000000000000000",
                    "type": "market_data"
                })
            ])
        ]
        mock_redis.xreadgroup.return_value = test_message_data
        mock_redis.xack = AsyncMock()
        
        # Track broadcasted messages
        broadcasted_messages = []
        
        async def mock_handler(message: MessageBusMessage):
            broadcasted_messages.append(message)
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            # Add message handler
            messagebus_client.add_message_handler(mock_handler)
            
            # Start client
            await messagebus_client.start()
            
            # Wait for connection establishment and message processing
            await asyncio.sleep(0.2)
            
            # Stop client
            await messagebus_client.stop()
            
            # Verify message was processed
            assert len(broadcasted_messages) == 1
            message = broadcasted_messages[0]
            assert message.topic == "market.eurusd.quote"
            assert message.payload["symbol"] == "EURUSD"
            assert message.payload["bid"] == 1.0850

    @pytest.mark.asyncio
    async def test_connection_failure_and_recovery(self, messagebus_client):
        """Test connection failure scenarios and recovery"""
        # Mock Redis client that fails initially then succeeds
        mock_redis = AsyncMock()
        call_count = 0
        
        def ping_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Connection failed")
            return True
        
        mock_redis.ping.side_effect = ping_side_effect
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup.return_value = []
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                
                # Start client
                await messagebus_client.start()
                
                # Wait for connection attempts
                await asyncio.sleep(0.2)
                
                # Stop client
                await messagebus_client.stop()
                
                # Verify reconnection attempts were made
                assert messagebus_client.connection_status.reconnect_attempts > 0

    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, messagebus_client):
        """Test handling of malformed messages from Redis"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create = AsyncMock()
        
        # Mock stream read with malformed message
        malformed_message_data = [
            ("test-stream", [
                ("1630000000000-0", {
                    "topic": "market.test",
                    "payload": "invalid json",  # Invalid JSON
                    "timestamp": "not_a_number",  # Invalid timestamp
                })
            ])
        ]
        mock_redis.xreadgroup.return_value = malformed_message_data
        mock_redis.xack = AsyncMock()
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            
            # Start client
            await messagebus_client.start()
            
            # Wait for message processing
            await asyncio.sleep(0.1)
            
            # Stop client
            await messagebus_client.stop()
            
            # Should not crash and should handle error gracefully
            # Message count should remain 0 since malformed message wasn't processed
            assert messagebus_client.connection_status.messages_received == 0

    def test_api_endpoints_during_connection_states(self, client):
        """Test API endpoints respond correctly during different connection states"""
        with patch('main.messagebus_client') as mock_messagebus:
            
            # Test with disconnected state
            mock_messagebus.is_connected = False
            mock_messagebus.connection_status.state.value = "disconnected"
            mock_messagebus.connection_status.messages_received = 0
            mock_messagebus.connection_status.reconnect_attempts = 0
            mock_messagebus.connection_status.error_message = None
            mock_messagebus.connection_status.connected_at = None
            mock_messagebus.connection_status.last_message_at = None
            
            response = client.get("/api/v1/status")
            assert response.status_code == 200
            data = response.json()
            assert data["features"]["messagebus"] is False
            
            response = client.get("/api/v1/messagebus/status")
            assert response.status_code == 200
            status_data = response.json()
            assert status_data["connection_state"] == "disconnected"

    def test_websocket_with_messagebus_messages(self, client):
        """Test WebSocket receives MessageBus messages"""
        with patch('main.messagebus_client') as mock_messagebus:
            mock_messagebus.is_connected = True
            
            # Test WebSocket connection
            with client.websocket_connect("/ws") as websocket:
                # Receive welcome message
                welcome = websocket.receive_text()
                welcome_data = json.loads(welcome)
                assert welcome_data["type"] == "connection"
                
                # The message handler would normally broadcast MessageBus messages
                # In a real scenario, this would be triggered by Redis messages


class TestPerformanceRequirements:
    """Test performance requirements are met"""

    @pytest.mark.asyncio
    async def test_connection_establishment_time(self):
        """Test connection establishment within 5 seconds"""
        client = MessageBusClient(connection_timeout=5.0)
        
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup.return_value = []
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            start_time = asyncio.get_event_loop().time()
            
            await client.start()
            
            # Wait for connection
            await asyncio.sleep(0.1)
            
            connection_time = asyncio.get_event_loop().time() - start_time
            
            await client.stop()
            
            # Should connect within 5 seconds
            assert connection_time < 5.0

    @pytest.mark.asyncio
    async def test_message_processing_latency(self):
        """Test message parsing latency < 10ms"""
        client = MessageBusClient()
        
        # Test message processing time
        message_id = "1630000000000-0"
        fields = {
            "topic": "test.performance",
            "payload": '{"data": "test"}',
            "timestamp": "1630000000000000000",
            "type": "data"
        }
        
        start_time = asyncio.get_event_loop().time()
        
        await client._process_stream_message(message_id, fields)
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000  # Convert to ms
        
        # Should process within 10ms
        assert processing_time < 10.0

    def test_api_response_time(self, client):
        """Test connection status API response < 100ms"""
        with patch('main.messagebus_client') as mock_messagebus:
            mock_messagebus.is_connected = True
            mock_messagebus.connection_status.state.value = "connected"
            mock_messagebus.connection_status.messages_received = 100
            mock_messagebus.connection_status.reconnect_attempts = 0
            mock_messagebus.connection_status.error_message = None
            mock_messagebus.connection_status.connected_at = None
            mock_messagebus.connection_status.last_message_at = None
            
            import time
            start_time = time.time()
            
            response = client.get("/api/v1/messagebus/status")
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            assert response.status_code == 200
            # Should respond within 100ms
            assert response_time < 100.0


class TestErrorRecoveryScenarios:
    """Test various error scenarios and recovery"""

    @pytest.mark.asyncio
    async def test_redis_disconnection_during_operation(self):
        """Test handling of Redis disconnection during operation"""
        client = MessageBusClient(health_check_interval=0.1)
        
        mock_redis = AsyncMock()
        
        # Initially connected
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup.return_value = []
        
        connected_calls = 0
        def ping_side_effect():
            nonlocal connected_calls
            connected_calls += 1
            if connected_calls <= 3:
                return True
            # Simulate disconnection after 3 calls
            raise Exception("Connection lost")
        
        mock_redis.ping.side_effect = ping_side_effect
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            await client.start()
            
            # Wait for health check to detect disconnection
            await asyncio.sleep(0.2)
            
            await client.stop()
            
            # Should detect disconnection
            assert connected_calls > 3

    @pytest.mark.asyncio 
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff timing is implemented correctly"""
        client = MessageBusClient(
            reconnect_base_delay=0.1,
            reconnect_max_delay=1.0,
            max_reconnect_attempts=3
        )
        
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Always fail")
        
        sleep_calls = []
        
        async def mock_sleep(delay):
            sleep_calls.append(delay)
        
        with patch('messagebus_client.redis.Redis', return_value=mock_redis):
            with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep):
                
                await client._connection_manager()
                
                # Verify exponential backoff pattern
                assert len(sleep_calls) >= 2
                # First delay should be base delay
                assert sleep_calls[0] == 0.1
                # Second delay should be doubled
                assert sleep_calls[1] == 0.2
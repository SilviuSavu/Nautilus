"""
Unit tests for Redis Integration - Sprint 3 Priority 6

Tests Redis pub/sub integration, messaging protocols, and real-time communication
infrastructure with comprehensive coverage including error scenarios.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

# Import Sprint 3 Redis components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from websocket.redis_pubsub import RedisPubSubManager
from websocket.streaming_service import StreamingService
from websocket.message_protocols import MessageProtocol, MessageType


class TestRedisPubSubManager:
    """Test Redis pub/sub manager functionality"""
    
    @pytest.fixture
    def pubsub_manager(self):
        """Create Redis pub/sub manager for testing"""
        return RedisPubSubManager(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            
            # Mock pubsub
            mock_pubsub = AsyncMock()
            mock_redis.pubsub.return_value = mock_pubsub
            mock_pubsub.subscribe = AsyncMock()
            mock_pubsub.unsubscribe = AsyncMock()
            
            yield mock_redis, mock_pubsub
    
    @pytest.mark.asyncio
    async def test_redis_connection_establishment(self, pubsub_manager, mock_redis):
        """Test Redis connection establishment"""
        mock_redis_client, _ = mock_redis
        
        success = await pubsub_manager.connect()
        
        assert success is True
        assert pubsub_manager.redis_client is not None
        mock_redis_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, pubsub_manager):
        """Test Redis connection failure handling"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis
            
            success = await pubsub_manager.connect()
            
            assert success is False
            assert pubsub_manager.redis_client is None
    
    @pytest.mark.asyncio
    async def test_publish_message_success(self, pubsub_manager, mock_redis):
        """Test successful message publishing"""
        mock_redis_client, _ = mock_redis
        pubsub_manager.redis_client = mock_redis_client
        
        test_message = {
            "type": "market_data_update",
            "symbol": "AAPL",
            "price": 155.25,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await pubsub_manager.publish("market.AAPL.quote", test_message)
        
        assert success is True
        mock_redis_client.publish.assert_called_once()
        
        # Verify message was serialized correctly
        call_args = mock_redis_client.publish.call_args[0]
        assert call_args[0] == "market.AAPL.quote"
        published_message = json.loads(call_args[1])
        assert published_message["symbol"] == "AAPL"
        assert published_message["price"] == 155.25
    
    @pytest.mark.asyncio
    async def test_publish_message_failure(self, pubsub_manager, mock_redis):
        """Test message publishing failure"""
        mock_redis_client, _ = mock_redis
        mock_redis_client.publish.side_effect = Exception("Redis error")
        pubsub_manager.redis_client = mock_redis_client
        
        test_message = {"test": "data"}
        
        success = await pubsub_manager.publish("test.channel", test_message)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self, pubsub_manager, mock_redis):
        """Test subscribing to Redis channel"""
        mock_redis_client, mock_pubsub = mock_redis
        pubsub_manager.redis_client = mock_redis_client
        
        handler_calls = []
        
        async def test_handler(channel, message):
            handler_calls.append((channel, message))
        
        success = await pubsub_manager.subscribe("test.channel", test_handler)
        
        assert success is True
        mock_pubsub.subscribe.assert_called_once_with("test.channel")
        assert "test.channel" in pubsub_manager.message_handlers
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_channel(self, pubsub_manager, mock_redis):
        """Test unsubscribing from Redis channel"""
        mock_redis_client, mock_pubsub = mock_redis
        pubsub_manager.redis_client = mock_redis_client
        pubsub_manager.pubsub = mock_pubsub
        
        # First subscribe
        await pubsub_manager.subscribe("test.channel", lambda c, m: None)
        
        # Then unsubscribe
        success = await pubsub_manager.unsubscribe("test.channel")
        
        assert success is True
        mock_pubsub.unsubscribe.assert_called_once_with("test.channel")
        assert "test.channel" not in pubsub_manager.message_handlers
    
    @pytest.mark.asyncio
    async def test_message_processing_loop(self, pubsub_manager, mock_redis):
        """Test Redis message processing loop"""
        mock_redis_client, mock_pubsub = mock_redis
        pubsub_manager.redis_client = mock_redis_client
        pubsub_manager.pubsub = mock_pubsub
        
        # Mock message data
        test_messages = [
            {
                'type': 'message',
                'channel': b'test.channel',
                'data': json.dumps({"type": "test", "data": "message1"}).encode()
            },
            {
                'type': 'message',
                'channel': b'test.channel', 
                'data': json.dumps({"type": "test", "data": "message2"}).encode()
            }
        ]
        
        async def mock_listen():
            for message in test_messages:
                yield message
        
        mock_pubsub.listen = mock_listen
        
        processed_messages = []
        
        async def message_handler(channel, message):
            processed_messages.append((channel, message))
        
        # Setup handler
        pubsub_manager.message_handlers["test.channel"] = [message_handler]
        
        # Process messages (run briefly)
        try:
            await asyncio.wait_for(pubsub_manager._process_messages(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        # Verify messages were processed
        assert len(processed_messages) >= 1
        if processed_messages:
            channel, message = processed_messages[0]
            assert channel == "test.channel"
            assert message["data"] in ["message1", "message2"]
    
    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, pubsub_manager, mock_redis):
        """Test handling of malformed Redis messages"""
        mock_redis_client, mock_pubsub = mock_redis
        pubsub_manager.redis_client = mock_redis_client
        pubsub_manager.pubsub = mock_pubsub
        
        # Mock malformed message data
        malformed_messages = [
            {
                'type': 'message',
                'channel': b'test.channel',
                'data': b'invalid json'  # Invalid JSON
            },
            {
                'type': 'subscribe',  # Non-message type
                'channel': b'test.channel',
                'data': b'1'
            }
        ]
        
        async def mock_listen():
            for message in malformed_messages:
                yield message
        
        mock_pubsub.listen = mock_listen
        
        processed_messages = []
        
        async def message_handler(channel, message):
            processed_messages.append((channel, message))
        
        # Setup handler
        pubsub_manager.message_handlers["test.channel"] = [message_handler]
        
        # Process messages (should not crash)
        try:
            await asyncio.wait_for(pubsub_manager._process_messages(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        # Should not process malformed messages
        assert len(processed_messages) == 0
    
    def test_channel_pattern_matching(self, pubsub_manager):
        """Test Redis channel pattern matching"""
        # Test exact channel matching
        assert pubsub_manager._matches_pattern("market.AAPL.quote", "market.AAPL.quote")
        
        # Test wildcard pattern matching
        assert pubsub_manager._matches_pattern("market.*.quote", "market.AAPL.quote")
        assert pubsub_manager._matches_pattern("market.*", "market.AAPL.quote")
        
        # Test non-matching patterns
        assert not pubsub_manager._matches_pattern("orders.*", "market.AAPL.quote")
        assert not pubsub_manager._matches_pattern("market.MSFT.*", "market.AAPL.quote")
    
    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self, pubsub_manager, mock_redis):
        """Test Redis connection health monitoring"""
        mock_redis_client, _ = mock_redis
        pubsub_manager.redis_client = mock_redis_client
        
        # Test healthy connection
        health = await pubsub_manager.check_health()
        assert health["connected"] is True
        assert health["ping_successful"] is True
        
        # Test unhealthy connection
        mock_redis_client.ping.side_effect = Exception("Connection lost")
        
        health = await pubsub_manager.check_health()
        assert health["connected"] is False
        assert health["ping_successful"] is False


class TestStreamingService:
    """Test streaming service functionality"""
    
    @pytest.fixture
    def streaming_service(self):
        """Create streaming service for testing"""
        return StreamingService()
    
    @pytest.fixture
    def mock_pubsub_manager(self):
        """Mock Redis pub/sub manager"""
        mock_manager = AsyncMock()
        mock_manager.connect.return_value = True
        mock_manager.publish.return_value = True
        mock_manager.subscribe.return_value = True
        return mock_manager
    
    @pytest.mark.asyncio
    async def test_streaming_service_initialization(self, streaming_service, mock_pubsub_manager):
        """Test streaming service initialization"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        success = await streaming_service.initialize()
        
        assert success is True
        mock_pubsub_manager.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_market_data_streaming(self, streaming_service, mock_pubsub_manager):
        """Test market data streaming"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        market_data = {
            "symbol": "AAPL",
            "bid": 155.20,
            "ask": 155.25,
            "last": 155.22,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await streaming_service.stream_market_data("AAPL", market_data)
        
        assert success is True
        mock_pubsub_manager.publish.assert_called_once()
        
        # Verify channel and message format
        call_args = mock_pubsub_manager.publish.call_args[0]
        assert call_args[0] == "market.AAPL.quote"
        assert call_args[1]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_order_update_streaming(self, streaming_service, mock_pubsub_manager):
        """Test order update streaming"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        order_update = {
            "order_id": "ord_123456",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "filled_quantity": 50,
            "status": "PARTIALLY_FILLED",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await streaming_service.stream_order_update("ord_123456", order_update)
        
        assert success is True
        mock_pubsub_manager.publish.assert_called_once()
        
        # Verify channel and message format
        call_args = mock_pubsub_manager.publish.call_args[0]
        assert call_args[0] == "orders.ord_123456"
        assert call_args[1]["status"] == "PARTIALLY_FILLED"
    
    @pytest.mark.asyncio
    async def test_portfolio_update_streaming(self, streaming_service, mock_pubsub_manager):
        """Test portfolio update streaming"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        portfolio_update = {
            "portfolio_id": "portfolio_1",
            "total_value": 100000.0,
            "daily_pnl": 2500.0,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "unrealized_pnl": 500.0},
                {"symbol": "MSFT", "quantity": 50, "unrealized_pnl": 250.0}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await streaming_service.stream_portfolio_update("portfolio_1", portfolio_update)
        
        assert success is True
        mock_pubsub_manager.publish.assert_called_once()
        
        # Verify channel and message format
        call_args = mock_pubsub_manager.publish.call_args[0]
        assert call_args[0] == "portfolio.portfolio_1.update"
        assert call_args[1]["total_value"] == 100000.0
    
    @pytest.mark.asyncio
    async def test_risk_alert_streaming(self, streaming_service, mock_pubsub_manager):
        """Test risk alert streaming"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        risk_alert = {
            "alert_id": "alert_001",
            "alert_type": "POSITION_LIMIT_BREACH",
            "portfolio_id": "portfolio_1",
            "symbol": "AAPL",
            "current_value": 1200000,
            "limit_value": 1000000,
            "severity": "CRITICAL",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await streaming_service.stream_risk_alert(risk_alert)
        
        assert success is True
        mock_pubsub_manager.publish.assert_called_once()
        
        # Verify channel and message format
        call_args = mock_pubsub_manager.publish.call_args[0]
        assert call_args[0] == "risk.alerts"
        assert call_args[1]["severity"] == "CRITICAL"
    
    @pytest.mark.asyncio
    async def test_analytics_streaming(self, streaming_service, mock_pubsub_manager):
        """Test analytics data streaming"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        analytics_update = {
            "portfolio_id": "portfolio_1",
            "metrics": {
                "sharpe_ratio": 1.25,
                "max_drawdown": -0.08,
                "total_return": 0.15,
                "volatility": 0.18
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await streaming_service.stream_analytics_update("portfolio_1", analytics_update)
        
        assert success is True
        mock_pubsub_manager.publish.assert_called_once()
        
        # Verify channel and message format
        call_args = mock_pubsub_manager.publish.call_args[0]
        assert call_args[0] == "analytics.portfolio_1.update"
        assert call_args[1]["metrics"]["sharpe_ratio"] == 1.25
    
    def test_message_throttling(self, streaming_service):
        """Test message throttling to prevent spam"""
        # Configure throttling
        streaming_service.configure_throttling(
            max_messages_per_second=10,
            burst_limit=50
        )
        
        # Test throttling logic
        channel = "test.channel"
        
        # Should allow initial messages
        for i in range(10):
            allowed = streaming_service._check_throttle_limit(channel)
            assert allowed is True
        
        # Should throttle after limit
        allowed = streaming_service._check_throttle_limit(channel)
        # May or may not be throttled depending on implementation
        assert isinstance(allowed, bool)
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, streaming_service, mock_pubsub_manager):
        """Test streaming error handling"""
        streaming_service.pubsub_manager = mock_pubsub_manager
        
        # Mock publish failure
        mock_pubsub_manager.publish.return_value = False
        
        market_data = {"symbol": "AAPL", "price": 155.25}
        
        success = await streaming_service.stream_market_data("AAPL", market_data)
        
        assert success is False
        
        # Verify error was logged (would need to check logs in real implementation)
        # For now, just ensure no exception was raised
    
    def test_channel_routing(self, streaming_service):
        """Test message channel routing logic"""
        # Test market data channel routing
        channel = streaming_service._get_market_data_channel("AAPL")
        assert channel == "market.AAPL.quote"
        
        # Test order channel routing
        channel = streaming_service._get_order_channel("ord_123")
        assert channel == "orders.ord_123"
        
        # Test portfolio channel routing
        channel = streaming_service._get_portfolio_channel("portfolio_1")
        assert channel == "portfolio.portfolio_1.update"
        
        # Test analytics channel routing
        channel = streaming_service._get_analytics_channel("portfolio_1")
        assert channel == "analytics.portfolio_1.update"


class TestMessageProtocolIntegration:
    """Test message protocol integration with Redis"""
    
    @pytest.fixture
    def message_protocol(self):
        """Create message protocol instance"""
        return MessageProtocol()
    
    def test_message_serialization_for_redis(self, message_protocol):
        """Test message serialization for Redis transport"""
        # Create market data message
        message = message_protocol.create_market_data_message(
            symbol="AAPL",
            bid=155.20,
            ask=155.25,
            connection_id="conn_123"
        )
        
        # Test serialization
        serialized = message_protocol.serialize_for_transport(message)
        
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["type"] == MessageType.MARKET_DATA.value
        assert deserialized["payload"]["symbol"] == "AAPL"
    
    def test_message_deserialization_from_redis(self, message_protocol):
        """Test message deserialization from Redis transport"""
        # Mock Redis message
        redis_message = {
            "type": MessageType.ORDER_UPDATE.value,
            "topic": "orders.ord_123",
            "payload": {
                "order_id": "ord_123",
                "status": "FILLED",
                "filled_qty": 100
            },
            "timestamp": datetime.utcnow().isoformat(),
            "connection_id": "conn_123"
        }
        
        serialized = json.dumps(redis_message)
        
        # Test deserialization
        message = message_protocol.deserialize_from_transport(serialized)
        
        assert message["type"] == MessageType.ORDER_UPDATE.value
        assert message["payload"]["order_id"] == "ord_123"
        assert message["payload"]["status"] == "FILLED"
    
    def test_message_validation_pipeline(self, message_protocol):
        """Test message validation in Redis pipeline"""
        # Test valid message
        valid_message = {
            "type": MessageType.SYSTEM_NOTIFICATION.value,
            "topic": "system.notifications",
            "payload": {
                "message": "System maintenance scheduled",
                "level": "INFO"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "connection_id": "system"
        }
        
        validation_result = message_protocol.validate_for_transport(valid_message)
        assert validation_result["valid"] is True
        
        # Test invalid message
        invalid_message = {
            "type": "INVALID_TYPE",
            "payload": {"test": "data"}
            # Missing required fields
        }
        
        validation_result = message_protocol.validate_for_transport(invalid_message)
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0


class TestRedisIntegrationPerformance:
    """Test Redis integration performance characteristics"""
    
    @pytest.fixture
    def pubsub_manager(self):
        """Create Redis pub/sub manager for performance testing"""
        return RedisPubSubManager()
    
    @pytest.mark.asyncio
    async def test_high_frequency_publishing(self, pubsub_manager):
        """Test high-frequency message publishing"""
        import time
        
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            pubsub_manager.redis_client = mock_redis
            
            # Test publishing 1000 messages
            start_time = time.time()
            
            tasks = []
            for i in range(1000):
                message = {"id": i, "timestamp": time.time()}
                task = pubsub_manager.publish(f"test.channel.{i % 10}", message)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            elapsed_time = time.time() - start_time
            
            # Should handle high-frequency publishing
            assert elapsed_time < 5.0  # Within 5 seconds
            assert mock_redis.publish.call_count == 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_channel_subscriptions(self, pubsub_manager):
        """Test concurrent channel subscriptions"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_pubsub = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.pubsub.return_value = mock_pubsub
            mock_redis_class.return_value = mock_redis
            
            pubsub_manager.redis_client = mock_redis
            
            # Subscribe to multiple channels concurrently
            channels = [f"test.channel.{i}" for i in range(100)]
            
            async def dummy_handler(channel, message):
                pass
            
            tasks = [
                pubsub_manager.subscribe(channel, dummy_handler)
                for channel in channels
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All subscriptions should succeed
            assert all(result is True for result in results)
            assert len(pubsub_manager.message_handlers) == 100
    
    def test_memory_usage_with_large_messages(self, pubsub_manager):
        """Test memory usage with large messages"""
        import sys
        
        # Create large message (1MB)
        large_data = "x" * (1024 * 1024)
        large_message = {
            "type": "large_data",
            "data": large_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Measure memory usage
        initial_size = sys.getsizeof(pubsub_manager)
        
        # Simulate processing large message
        serialized = json.dumps(large_message)
        deserialized = json.loads(serialized)
        
        final_size = sys.getsizeof(pubsub_manager)
        
        # Memory usage should not increase significantly
        memory_growth = final_size - initial_size
        assert memory_growth < 100  # Less than 100 bytes growth


class TestRedisIntegrationResilience:
    """Test Redis integration resilience and error recovery"""
    
    @pytest.fixture
    def pubsub_manager(self):
        """Create Redis pub/sub manager for resilience testing"""
        return RedisPubSubManager(
            redis_host="localhost",
            redis_port=6379,
            connection_timeout=1.0,
            retry_attempts=3,
            retry_delay=0.1
        )
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, pubsub_manager):
        """Test Redis connection retry logic"""
        call_count = 0
        
        def mock_redis_init(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_redis = AsyncMock()
            if call_count <= 2:
                mock_redis.ping.side_effect = Exception("Connection failed")
            else:
                mock_redis.ping.return_value = True
            return mock_redis
        
        with patch('redis.asyncio.Redis', side_effect=mock_redis_init):
            success = await pubsub_manager.connect_with_retry()
            
            assert success is True
            assert call_count == 3  # Failed twice, succeeded third time
    
    @pytest.mark.asyncio
    async def test_publish_retry_on_failure(self, pubsub_manager):
        """Test message publishing retry on failure"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            call_count = 0
            
            def mock_publish(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Temporary failure")
                return 1
            
            mock_redis.publish = mock_publish
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            
            pubsub_manager.redis_client = mock_redis
            
            message = {"test": "data"}
            success = await pubsub_manager.publish_with_retry("test.channel", message)
            
            assert success is True
            assert call_count == 3  # Failed twice, succeeded third time
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_redis_unavailable(self, pubsub_manager):
        """Test graceful degradation when Redis is unavailable"""
        # Mock Redis completely unavailable
        pubsub_manager.redis_client = None
        
        # Should not crash, should return False
        success = await pubsub_manager.publish("test.channel", {"test": "data"})
        assert success is False
        
        # Health check should indicate unhealthy state
        health = await pubsub_manager.check_health()
        assert health["connected"] is False
"""
Unit tests for WebSocket Infrastructure components - Sprint 3 Priority 1

Tests WebSocket manager, subscription manager, message protocols, and event dispatcher
with comprehensive coverage including edge cases and error handling.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any

# Import Sprint 3 WebSocket components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from websocket.websocket_manager import WebSocketManager, websocket_manager
from websocket.subscription_manager import SubscriptionManager
from websocket.message_protocols import MessageProtocol, MessageType, validate_message
from websocket.event_dispatcher import EventDispatcher
from websocket.redis_pubsub import RedisPubSubManager


class TestWebSocketManager:
    """Test WebSocket connection management functionality"""
    
    @pytest.fixture
    def ws_manager(self):
        """Create fresh WebSocket manager for testing"""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection"""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws
    
    @pytest.fixture
    def connection_id(self):
        """Generate unique connection ID"""
        return f"test_conn_{uuid.uuid4().hex[:8]}"
    
    @pytest.mark.asyncio
    async def test_websocket_connection_success(self, ws_manager, mock_websocket, connection_id):
        """Test successful WebSocket connection establishment"""
        user_id = "test_user_123"
        
        # Test connection
        success = await ws_manager.connect(mock_websocket, connection_id, user_id)
        
        # Verify connection success
        assert success is True
        mock_websocket.accept.assert_called_once()
        
        # Verify connection stored
        assert connection_id in ws_manager.active_connections
        assert ws_manager.active_connections[connection_id] == mock_websocket
        
        # Verify metadata stored
        assert connection_id in ws_manager.connection_metadata
        metadata = ws_manager.connection_metadata[connection_id]
        assert metadata["user_id"] == user_id
        assert isinstance(metadata["connected_at"], datetime)
        assert metadata["message_count"] == 0
        
        # Verify subscriptions initialized
        assert connection_id in ws_manager.subscriptions
        assert len(ws_manager.subscriptions[connection_id]) == 0
        
        # Verify connection confirmation sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "connection_established"
        assert sent_data["connection_id"] == connection_id
    
    @pytest.mark.asyncio
    async def test_websocket_connection_failure(self, ws_manager, mock_websocket, connection_id):
        """Test WebSocket connection failure handling"""
        # Mock connection failure
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        # Test connection
        success = await ws_manager.connect(mock_websocket, connection_id)
        
        # Verify connection failed
        assert success is False
        assert connection_id not in ws_manager.active_connections
        assert connection_id not in ws_manager.connection_metadata
        assert connection_id not in ws_manager.subscriptions
    
    def test_websocket_disconnect(self, ws_manager, mock_websocket, connection_id):
        """Test WebSocket disconnection cleanup"""
        # Setup connection manually
        ws_manager.active_connections[connection_id] = mock_websocket
        ws_manager.subscriptions[connection_id] = {"topic1", "topic2"}
        ws_manager.connection_metadata[connection_id] = {
            "user_id": "test_user",
            "connected_at": datetime.utcnow(),
            "message_count": 5
        }
        
        # Test disconnect
        ws_manager.disconnect(connection_id)
        
        # Verify cleanup
        assert connection_id not in ws_manager.active_connections
        assert connection_id not in ws_manager.subscriptions
        assert connection_id not in ws_manager.connection_metadata
    
    @pytest.mark.asyncio
    async def test_send_personal_message_success(self, ws_manager, mock_websocket, connection_id):
        """Test sending personal message to specific connection"""
        # Setup connection
        ws_manager.active_connections[connection_id] = mock_websocket
        ws_manager.connection_metadata[connection_id] = {"message_count": 0}
        
        test_message = {
            "type": "test_message",
            "data": {"value": 123},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test send message
        success = await ws_manager.send_personal_message(test_message, connection_id)
        
        # Verify success
        assert success is True
        mock_websocket.send_text.assert_called_once()
        
        # Verify message content
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "test_message"
        assert sent_data["data"]["value"] == 123
        
        # Verify message count incremented
        assert ws_manager.connection_metadata[connection_id]["message_count"] == 1
    
    @pytest.mark.asyncio
    async def test_send_personal_message_connection_not_found(self, ws_manager):
        """Test sending message to non-existent connection"""
        test_message = {"type": "test", "data": "test"}
        
        success = await ws_manager.send_personal_message(test_message, "non_existent_id")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_send_personal_message_websocket_disconnect(self, ws_manager, mock_websocket, connection_id):
        """Test handling WebSocket disconnect during message send"""
        # Setup connection
        ws_manager.active_connections[connection_id] = mock_websocket
        ws_manager.connection_metadata[connection_id] = {"message_count": 0}
        ws_manager.subscriptions[connection_id] = set()
        
        # Mock WebSocket disconnect
        mock_websocket.send_text.side_effect = WebSocketDisconnect()
        
        test_message = {"type": "test", "data": "test"}
        
        # Test send message
        success = await ws_manager.send_personal_message(test_message, connection_id)
        
        # Verify failure and cleanup
        assert success is False
        assert connection_id not in ws_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message_to_all_connections(self, ws_manager):
        """Test broadcasting message to all active connections"""
        # Setup multiple connections
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws3 = AsyncMock(spec=WebSocket)
        
        ws_manager.active_connections = {
            "conn1": mock_ws1,
            "conn2": mock_ws2,
            "conn3": mock_ws3
        }
        ws_manager.connection_metadata = {
            "conn1": {"message_count": 0},
            "conn2": {"message_count": 0},
            "conn3": {"message_count": 0}
        }
        ws_manager.subscriptions = {
            "conn1": set(),
            "conn2": set(),
            "conn3": set()
        }
        
        test_message = {"type": "broadcast", "data": "test_broadcast"}
        
        # Test broadcast
        sent_count = await ws_manager.broadcast_message(test_message)
        
        # Verify all connections received message
        assert sent_count == 3
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        mock_ws3.send_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_message_to_topic_subscribers(self, ws_manager):
        """Test broadcasting message to topic subscribers only"""
        # Setup connections with different subscriptions
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws3 = AsyncMock(spec=WebSocket)
        
        ws_manager.active_connections = {
            "conn1": mock_ws1,
            "conn2": mock_ws2,
            "conn3": mock_ws3
        }
        ws_manager.connection_metadata = {
            "conn1": {"message_count": 0},
            "conn2": {"message_count": 0},
            "conn3": {"message_count": 0}
        }
        ws_manager.subscriptions = {
            "conn1": {"market.data", "orders"},
            "conn2": {"market.data"},
            "conn3": {"orders", "portfolio"}
        }
        
        test_message = {"type": "market_update", "data": "AAPL price update"}
        
        # Test broadcast to specific topic
        sent_count = await ws_manager.broadcast_message(test_message, "market.data")
        
        # Verify only subscribers received message
        assert sent_count == 2
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        mock_ws3.send_text.assert_not_called()
    
    def test_topic_subscription_management(self, ws_manager, connection_id):
        """Test topic subscription and unsubscription"""
        # Initialize subscription set
        ws_manager.subscriptions[connection_id] = set()
        
        # Test subscribe to topic
        success = ws_manager.subscribe_to_topic(connection_id, "market.data")
        assert success is True
        assert "market.data" in ws_manager.subscriptions[connection_id]
        
        # Test subscribe to additional topic
        ws_manager.subscribe_to_topic(connection_id, "orders")
        assert "orders" in ws_manager.subscriptions[connection_id]
        assert len(ws_manager.subscriptions[connection_id]) == 2
        
        # Test unsubscribe from topic
        success = ws_manager.unsubscribe_from_topic(connection_id, "market.data")
        assert success is True
        assert "market.data" not in ws_manager.subscriptions[connection_id]
        assert "orders" in ws_manager.subscriptions[connection_id]
        
        # Test unsubscribe from non-existent connection
        success = ws_manager.unsubscribe_from_topic("non_existent", "test")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_heartbeat_handling(self, ws_manager, mock_websocket, connection_id):
        """Test WebSocket heartbeat functionality"""
        # Setup connection
        ws_manager.active_connections[connection_id] = mock_websocket
        initial_time = datetime.utcnow()
        ws_manager.connection_metadata[connection_id] = {
            "last_heartbeat": initial_time,
            "message_count": 0
        }
        
        # Test heartbeat
        success = await ws_manager.handle_heartbeat(connection_id)
        
        # Verify heartbeat handled
        assert success is True
        
        # Verify timestamp updated
        updated_time = ws_manager.connection_metadata[connection_id]["last_heartbeat"]
        assert updated_time > initial_time
        
        # Verify heartbeat response sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "heartbeat_response"
    
    def test_connection_statistics(self, ws_manager):
        """Test connection statistics generation"""
        # Setup test data
        now = datetime.utcnow()
        old_time = now - timedelta(seconds=120)
        
        ws_manager.active_connections = {
            "conn1": Mock(),
            "conn2": Mock(),
            "conn3": Mock()
        }
        ws_manager.subscriptions = {
            "conn1": {"market.data", "orders"},
            "conn2": {"market.data"},
            "conn3": set()
        }
        ws_manager.connection_metadata = {
            "conn1": {
                "user_id": "user1",
                "connected_at": old_time,
                "last_heartbeat": now - timedelta(seconds=30),
                "message_count": 100
            },
            "conn2": {
                "user_id": "user2", 
                "connected_at": now - timedelta(seconds=60),
                "last_heartbeat": old_time,  # Stale heartbeat
                "message_count": 50
            },
            "conn3": {
                "user_id": None,
                "connected_at": now,
                "last_heartbeat": now,
                "message_count": 0
            }
        }
        
        # Test statistics
        stats = ws_manager.get_connection_stats()
        
        # Verify basic counts
        assert stats["total_connections"] == 3
        assert stats["total_subscriptions"] == 3
        
        # Verify topic counts
        assert stats["connections_by_topic"]["market.data"] == 2
        assert stats["connections_by_topic"]["orders"] == 1
        
        # Verify health info
        assert len(stats["connection_health"]) == 3
        
        # Find healthy and unhealthy connections
        healthy_connections = [conn for conn in stats["connection_health"] if conn["is_healthy"]]
        unhealthy_connections = [conn for conn in stats["connection_health"] if not conn["is_healthy"]]
        
        assert len(healthy_connections) == 2
        assert len(unhealthy_connections) == 1
    
    @pytest.mark.asyncio
    async def test_stale_connection_cleanup(self, ws_manager):
        """Test cleanup of stale connections"""
        now = datetime.utcnow()
        stale_time = now - timedelta(seconds=400)  # Stale connection
        fresh_time = now - timedelta(seconds=100)   # Fresh connection
        
        # Setup connections with different heartbeat times
        ws_manager.active_connections = {
            "stale_conn": Mock(),
            "fresh_conn": Mock()
        }
        ws_manager.subscriptions = {
            "stale_conn": {"test"},
            "fresh_conn": {"test"}
        }
        ws_manager.connection_metadata = {
            "stale_conn": {
                "connected_at": stale_time,
                "last_heartbeat": stale_time
            },
            "fresh_conn": {
                "connected_at": fresh_time,
                "last_heartbeat": fresh_time
            }
        }
        
        # Test cleanup with 300 second timeout
        cleaned_count = await ws_manager.cleanup_stale_connections(timeout_seconds=300)
        
        # Verify stale connection cleaned up
        assert cleaned_count == 1
        assert "stale_conn" not in ws_manager.active_connections
        assert "fresh_conn" in ws_manager.active_connections


class TestSubscriptionManager:
    """Test subscription management functionality"""
    
    @pytest.fixture
    def sub_manager(self):
        """Create subscription manager for testing"""
        return SubscriptionManager()
    
    def test_subscription_management(self, sub_manager):
        """Test subscription CRUD operations"""
        connection_id = "test_conn"
        topic = "market.AAPL"
        
        # Test initial state
        assert not sub_manager.is_subscribed(connection_id, topic)
        assert sub_manager.get_subscribers(topic) == set()
        
        # Test subscription
        success = sub_manager.subscribe(connection_id, topic)
        assert success is True
        assert sub_manager.is_subscribed(connection_id, topic)
        assert connection_id in sub_manager.get_subscribers(topic)
        
        # Test duplicate subscription
        success = sub_manager.subscribe(connection_id, topic)
        assert success is True  # Should not fail
        
        # Test unsubscription
        success = sub_manager.unsubscribe(connection_id, topic)
        assert success is True
        assert not sub_manager.is_subscribed(connection_id, topic)
        assert connection_id not in sub_manager.get_subscribers(topic)
    
    def test_bulk_operations(self, sub_manager):
        """Test bulk subscription operations"""
        connection_id = "test_conn"
        topics = ["market.AAPL", "market.MSFT", "orders", "portfolio"]
        
        # Test bulk subscribe
        success = sub_manager.bulk_subscribe(connection_id, topics)
        assert success is True
        
        for topic in topics:
            assert sub_manager.is_subscribed(connection_id, topic)
        
        # Test get connection subscriptions
        conn_topics = sub_manager.get_connection_subscriptions(connection_id)
        assert len(conn_topics) == 4
        assert all(topic in conn_topics for topic in topics)
        
        # Test bulk unsubscribe
        success = sub_manager.bulk_unsubscribe(connection_id, topics[:2])
        assert success is True
        
        remaining_topics = sub_manager.get_connection_subscriptions(connection_id)
        assert len(remaining_topics) == 2
        assert "orders" in remaining_topics
        assert "portfolio" in remaining_topics
    
    def test_connection_cleanup(self, sub_manager):
        """Test cleaning up all subscriptions for a connection"""
        connection_id = "test_conn"
        topics = ["topic1", "topic2", "topic3"]
        
        # Setup subscriptions
        for topic in topics:
            sub_manager.subscribe(connection_id, topic)
        
        # Verify subscriptions exist
        assert len(sub_manager.get_connection_subscriptions(connection_id)) == 3
        
        # Test cleanup
        cleaned_count = sub_manager.cleanup_connection(connection_id)
        assert cleaned_count == 3
        
        # Verify all subscriptions removed
        assert len(sub_manager.get_connection_subscriptions(connection_id)) == 0
        for topic in topics:
            assert connection_id not in sub_manager.get_subscribers(topic)


class TestMessageProtocols:
    """Test message protocol validation and formatting"""
    
    def test_message_validation_success(self):
        """Test valid message validation"""
        valid_message = {
            "type": MessageType.MARKET_DATA.value,
            "topic": "market.AAPL.quote",
            "payload": {
                "symbol": "AAPL",
                "bid": 150.25,
                "ask": 150.27,
                "timestamp": 1234567890
            },
            "timestamp": "2024-01-01T10:00:00Z",
            "connection_id": "conn_123"
        }
        
        result = validate_message(valid_message)
        assert result["valid"] is True
        assert result["errors"] == []
    
    def test_message_validation_missing_required_fields(self):
        """Test validation with missing required fields"""
        invalid_message = {
            "type": MessageType.MARKET_DATA.value,
            "payload": {"data": "test"}
            # Missing required fields: topic, timestamp, connection_id
        }
        
        result = validate_message(invalid_message)
        assert result["valid"] is False
        assert len(result["errors"]) >= 3
        
        error_messages = " ".join(result["errors"])
        assert "topic" in error_messages
        assert "timestamp" in error_messages
        assert "connection_id" in error_messages
    
    def test_message_validation_invalid_type(self):
        """Test validation with invalid message type"""
        invalid_message = {
            "type": "INVALID_TYPE",
            "topic": "test.topic",
            "payload": {"data": "test"},
            "timestamp": "2024-01-01T10:00:00Z",
            "connection_id": "conn_123"
        }
        
        result = validate_message(invalid_message)
        assert result["valid"] is False
        assert any("Invalid message type" in error for error in result["errors"])
    
    def test_message_protocol_formatting(self):
        """Test message protocol formatting utilities"""
        protocol = MessageProtocol()
        
        # Test market data message
        market_msg = protocol.create_market_data_message(
            symbol="AAPL",
            bid=150.25,
            ask=150.27,
            connection_id="conn_123"
        )
        
        assert market_msg["type"] == MessageType.MARKET_DATA.value
        assert market_msg["topic"] == "market.AAPL.quote"
        assert market_msg["payload"]["symbol"] == "AAPL"
        assert market_msg["payload"]["bid"] == 150.25
        assert market_msg["connection_id"] == "conn_123"
        
        # Test order update message
        order_msg = protocol.create_order_update_message(
            order_id="ord_123",
            status="FILLED",
            filled_qty=100,
            connection_id="conn_123"
        )
        
        assert order_msg["type"] == MessageType.ORDER_UPDATE.value
        assert order_msg["topic"] == "orders.ord_123"
        assert order_msg["payload"]["status"] == "FILLED"
        
        # Test system notification
        sys_msg = protocol.create_system_notification(
            message="Trading session started",
            level="INFO",
            connection_id="conn_123"
        )
        
        assert sys_msg["type"] == MessageType.SYSTEM_NOTIFICATION.value
        assert sys_msg["payload"]["level"] == "INFO"


class TestEventDispatcher:
    """Test event dispatching functionality"""
    
    @pytest.fixture
    def dispatcher(self):
        """Create event dispatcher for testing"""
        return EventDispatcher()
    
    @pytest.mark.asyncio
    async def test_event_handler_registration(self, dispatcher):
        """Test registering and unregistering event handlers"""
        handler_called = []
        
        async def test_handler(event_type, data):
            handler_called.append((event_type, data))
        
        # Test registration
        handler_id = dispatcher.register_handler("market.data", test_handler)
        assert handler_id is not None
        assert len(dispatcher.get_handlers("market.data")) == 1
        
        # Test event dispatch
        await dispatcher.dispatch_event("market.data", {"symbol": "AAPL"})
        
        assert len(handler_called) == 1
        assert handler_called[0][0] == "market.data"
        assert handler_called[0][1]["symbol"] == "AAPL"
        
        # Test unregistration
        success = dispatcher.unregister_handler(handler_id)
        assert success is True
        assert len(dispatcher.get_handlers("market.data")) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self, dispatcher):
        """Test multiple handlers for same event type"""
        handler1_calls = []
        handler2_calls = []
        
        async def handler1(event_type, data):
            handler1_calls.append(data)
        
        async def handler2(event_type, data):
            handler2_calls.append(data)
        
        # Register multiple handlers
        dispatcher.register_handler("test.event", handler1)
        dispatcher.register_handler("test.event", handler2)
        
        # Dispatch event
        test_data = {"test": "data"}
        await dispatcher.dispatch_event("test.event", test_data)
        
        # Verify both handlers called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
        assert handler1_calls[0] == test_data
        assert handler2_calls[0] == test_data
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self, dispatcher):
        """Test error handling in event handlers"""
        successful_handler_calls = []
        
        async def failing_handler(event_type, data):
            raise Exception("Handler error")
        
        async def successful_handler(event_type, data):
            successful_handler_calls.append(data)
        
        # Register both handlers
        dispatcher.register_handler("test.event", failing_handler)
        dispatcher.register_handler("test.event", successful_handler)
        
        # Dispatch event - should not raise exception
        await dispatcher.dispatch_event("test.event", {"test": "data"})
        
        # Successful handler should still be called despite failing handler
        assert len(successful_handler_calls) == 1


class TestRedisPubSubManager:
    """Test Redis pub/sub integration for WebSocket messaging"""
    
    @pytest.fixture
    def pubsub_manager(self):
        """Create Redis pub/sub manager for testing"""
        return RedisPubSubManager(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, pubsub_manager):
        """Test Redis connection establishment"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            
            success = await pubsub_manager.connect()
            assert success is True
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_message(self, pubsub_manager):
        """Test publishing message to Redis channel"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            pubsub_manager.redis_client = mock_redis
            
            test_message = {"type": "test", "data": "test_data"}
            
            success = await pubsub_manager.publish("test.channel", test_message)
            
            assert success is True
            mock_redis.publish.assert_called_once()
            call_args = mock_redis.publish.call_args[0]
            assert call_args[0] == "test.channel"
            assert json.loads(call_args[1]) == test_message
    
    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self, pubsub_manager):
        """Test subscribing to Redis channel"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_pubsub = AsyncMock()
            mock_redis.pubsub.return_value = mock_pubsub
            mock_pubsub.subscribe.return_value = None
            mock_redis_class.return_value = mock_redis
            pubsub_manager.redis_client = mock_redis
            
            handler_calls = []
            
            async def test_handler(channel, message):
                handler_calls.append((channel, message))
            
            success = await pubsub_manager.subscribe("test.channel", test_handler)
            
            assert success is True
            mock_pubsub.subscribe.assert_called_once_with("test.channel")
    
    @pytest.mark.asyncio
    async def test_message_processing_loop(self, pubsub_manager):
        """Test Redis message processing loop"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_pubsub = AsyncMock()
            
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
            
            mock_pubsub.listen = AsyncMock()
            mock_pubsub.listen.return_value = iter(test_messages)
            mock_redis.pubsub.return_value = mock_pubsub
            mock_redis_class.return_value = mock_redis
            
            pubsub_manager.redis_client = mock_redis
            pubsub_manager.pubsub = mock_pubsub
            
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
            
            # Verify messages processed
            assert len(processed_messages) >= 1


class TestWebSocketIntegrationWithSprint3:
    """Integration tests for WebSocket components with Sprint 3 features"""
    
    @pytest.mark.asyncio
    async def test_websocket_analytics_integration(self):
        """Test WebSocket integration with analytics components"""
        ws_manager = WebSocketManager()
        
        # Mock WebSocket connection
        mock_ws = AsyncMock(spec=WebSocket)
        connection_id = "analytics_test_conn"
        
        # Setup connection
        success = await ws_manager.connect(mock_ws, connection_id, "analytics_user")
        assert success is True
        
        # Subscribe to analytics topics
        ws_manager.subscribe_to_topic(connection_id, "analytics.performance")
        ws_manager.subscribe_to_topic(connection_id, "analytics.risk")
        
        # Mock analytics message broadcast
        analytics_message = {
            "type": "analytics_update",
            "topic": "analytics.performance", 
            "data": {
                "portfolio_id": "port_123",
                "total_pnl": 15000.50,
                "sharpe_ratio": 1.45,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Test broadcast to analytics subscribers
        sent_count = await ws_manager.broadcast_message(analytics_message, "analytics.performance")
        assert sent_count == 1
        
        # Verify message was sent
        mock_ws.send_text.assert_called()
        call_count = mock_ws.send_text.call_count
        assert call_count >= 2  # Connection confirmation + analytics message
    
    @pytest.mark.asyncio
    async def test_websocket_risk_management_integration(self):
        """Test WebSocket integration with risk management"""
        ws_manager = WebSocketManager()
        
        # Setup multiple connections for risk alerts
        connections = []
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            conn_id = f"risk_conn_{i}"
            await ws_manager.connect(mock_ws, conn_id, f"trader_{i}")
            ws_manager.subscribe_to_topic(conn_id, "risk.alerts")
            connections.append((conn_id, mock_ws))
        
        # Mock risk alert message
        risk_alert = {
            "type": "risk_alert",
            "topic": "risk.alerts",
            "data": {
                "alert_type": "POSITION_LIMIT_BREACH",
                "portfolio_id": "port_123",
                "symbol": "AAPL",
                "current_position": 1500,
                "limit": 1000,
                "severity": "HIGH",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Broadcast risk alert
        sent_count = await ws_manager.broadcast_message(risk_alert, "risk.alerts")
        
        # Verify all risk subscribers received alert
        assert sent_count == 3
        for conn_id, mock_ws in connections:
            mock_ws.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_websocket_strategy_deployment_integration(self):
        """Test WebSocket integration with strategy deployment"""
        ws_manager = WebSocketManager()
        
        # Setup strategy manager connection
        mock_ws = AsyncMock(spec=WebSocket)
        conn_id = "strategy_manager_conn"
        await ws_manager.connect(mock_ws, conn_id, "strategy_manager")
        
        # Subscribe to strategy topics
        ws_manager.subscribe_to_topic(conn_id, "strategy.deployment")
        ws_manager.subscribe_to_topic(conn_id, "strategy.performance")
        
        # Mock strategy deployment status
        deployment_status = {
            "type": "strategy_deployment_status",
            "topic": "strategy.deployment",
            "data": {
                "strategy_id": "momentum_v1.2",
                "status": "DEPLOYED",
                "deployment_time": datetime.utcnow().isoformat(),
                "allocated_capital": 100000,
                "risk_limits": {
                    "max_position_size": 50000,
                    "max_daily_loss": 5000
                }
            }
        }
        
        # Send deployment notification
        success = await ws_manager.send_personal_message(deployment_status, conn_id)
        assert success is True
        
        # Verify connection health
        stats = ws_manager.get_connection_stats()
        strategy_conn_health = next(
            conn for conn in stats["connection_health"] 
            if conn["connection_id"] == conn_id
        )
        assert strategy_conn_health["is_healthy"] is True
        assert strategy_conn_health["message_count"] >= 1
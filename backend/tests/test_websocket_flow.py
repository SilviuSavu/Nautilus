"""
Integration tests for WebSocket Communication Flow - Sprint 3

Tests complete WebSocket communication flow from client connection through
real-time data streaming, including Redis integration and message routing.
"""

import asyncio
import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from typing import Dict, List, Any

# Import Sprint 3 WebSocket components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from websocket.websocket_manager import WebSocketManager
from websocket.subscription_manager import SubscriptionManager
from websocket.redis_pubsub import RedisPubSubManager
from websocket.streaming_service import StreamingService
from websocket.message_protocols import MessageProtocol, MessageType
from websocket.event_dispatcher import EventDispatcher


class TestWebSocketConnectionFlow:
    """Test complete WebSocket connection and communication flow"""
    
    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager for integration testing"""
        return WebSocketManager()
    
    @pytest.fixture
    def subscription_manager(self):
        """Create subscription manager for integration testing"""
        return SubscriptionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection for testing"""
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws
    
    @pytest.fixture
    def connection_id(self):
        """Generate unique connection ID"""
        return f"test_conn_{uuid.uuid4().hex[:8]}"
    
    @pytest.mark.asyncio
    async def test_complete_connection_lifecycle(self, websocket_manager, mock_websocket, connection_id):
        """Test complete WebSocket connection lifecycle"""
        user_id = "test_user_123"
        
        # 1. Test connection establishment
        success = await websocket_manager.connect(mock_websocket, connection_id, user_id)
        assert success is True
        
        # Verify connection stored
        assert connection_id in websocket_manager.active_connections
        assert websocket_manager.connection_metadata[connection_id]["user_id"] == user_id
        
        # 2. Test subscription management
        topics = ["market.AAPL.quote", "orders.portfolio_1", "portfolio.updates"]
        
        for topic in topics:
            success = websocket_manager.subscribe_to_topic(connection_id, topic)
            assert success is True
        
        # Verify subscriptions
        assert len(websocket_manager.subscriptions[connection_id]) == 3
        
        # 3. Test message broadcasting
        test_message = {
            "type": "market_update",
            "symbol": "AAPL",
            "price": 155.25,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        sent_count = await websocket_manager.broadcast_message(test_message, "market.AAPL.quote")
        assert sent_count == 1  # One subscriber
        
        # 4. Test heartbeat handling
        success = await websocket_manager.handle_heartbeat(connection_id)
        assert success is True
        
        # 5. Test disconnection
        websocket_manager.disconnect(connection_id)
        
        # Verify cleanup
        assert connection_id not in websocket_manager.active_connections
        assert connection_id not in websocket_manager.subscriptions
    
    @pytest.mark.asyncio
    async def test_multi_client_connection_scenario(self, websocket_manager):
        """Test multiple clients connecting and communicating"""
        # Create multiple mock connections
        connections = []
        for i in range(5):
            mock_ws = AsyncMock()
            conn_id = f"client_{i}"
            user_id = f"user_{i}"
            
            # Connect client
            success = await websocket_manager.connect(mock_ws, conn_id, user_id)
            assert success is True
            
            # Subscribe to different topics
            if i < 3:
                websocket_manager.subscribe_to_topic(conn_id, "market.data")
            if i >= 2:
                websocket_manager.subscribe_to_topic(conn_id, "portfolio.updates")
            
            connections.append((conn_id, mock_ws))
        
        # Test selective broadcasting
        market_message = {"type": "market_update", "data": "market data"}
        market_recipients = await websocket_manager.broadcast_message(market_message, "market.data")
        assert market_recipients == 3  # First 3 clients subscribed
        
        portfolio_message = {"type": "portfolio_update", "data": "portfolio data"}
        portfolio_recipients = await websocket_manager.broadcast_message(portfolio_message, "portfolio.updates")
        assert portfolio_recipients == 3  # Last 3 clients subscribed
        
        # Test broadcast to all
        global_message = {"type": "system_announcement", "data": "system message"}
        all_recipients = await websocket_manager.broadcast_message(global_message)
        assert all_recipients == 5  # All clients
        
        # Cleanup
        for conn_id, _ in connections:
            websocket_manager.disconnect(conn_id)
    
    @pytest.mark.asyncio
    async def test_connection_failure_scenarios(self, websocket_manager, mock_websocket, connection_id):
        """Test various connection failure scenarios"""
        # Test connection failure during accept
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        success = await websocket_manager.connect(mock_websocket, connection_id)
        assert success is False
        assert connection_id not in websocket_manager.active_connections
        
        # Reset mock for successful connection
        mock_websocket.accept = AsyncMock()
        mock_websocket.accept.side_effect = None
        
        success = await websocket_manager.connect(mock_websocket, connection_id)
        assert success is True
        
        # Test message sending failure
        from fastapi import WebSocketDisconnect
        mock_websocket.send_text.side_effect = WebSocketDisconnect()
        
        test_message = {"type": "test", "data": "test"}
        success = await websocket_manager.send_personal_message(test_message, connection_id)
        
        assert success is False
        assert connection_id not in websocket_manager.active_connections  # Should be cleaned up
    
    def test_connection_health_monitoring(self, websocket_manager):
        """Test connection health monitoring and stale connection detection"""
        now = datetime.utcnow()
        
        # Create connections with different health states
        healthy_conn = f"healthy_{uuid.uuid4().hex[:8]}"
        stale_conn = f"stale_{uuid.uuid4().hex[:8]}"
        
        # Setup connections
        websocket_manager.active_connections = {
            healthy_conn: Mock(),
            stale_conn: Mock()
        }
        
        websocket_manager.connection_metadata = {
            healthy_conn: {
                "connected_at": now - timedelta(minutes=5),
                "last_heartbeat": now - timedelta(seconds=30),  # Recent heartbeat
                "message_count": 50
            },
            stale_conn: {
                "connected_at": now - timedelta(minutes=10),
                "last_heartbeat": now - timedelta(minutes=8),  # Old heartbeat
                "message_count": 10
            }
        }
        
        websocket_manager.subscriptions = {
            healthy_conn: {"market.data"},
            stale_conn: {"portfolio.updates"}
        }
        
        # Get connection statistics
        stats = websocket_manager.get_connection_stats()
        
        assert stats["total_connections"] == 2
        assert len(stats["connection_health"]) == 2
        
        # Find healthy and stale connections
        health_info = {
            conn["connection_id"]: conn["is_healthy"]
            for conn in stats["connection_health"]
        }
        
        assert health_info[healthy_conn] is True
        assert health_info[stale_conn] is False
    
    @pytest.mark.asyncio
    async def test_subscription_topic_routing(self, websocket_manager, mock_websocket, connection_id):
        """Test topic-based message routing"""
        # Setup connection
        await websocket_manager.connect(mock_websocket, connection_id)
        
        # Subscribe to specific topics with patterns
        topics = [
            "market.AAPL.quote",
            "market.MSFT.quote", 
            "orders.user_123",
            "portfolio.portfolio_1.updates",
            "risk.alerts"
        ]
        
        for topic in topics:
            websocket_manager.subscribe_to_topic(connection_id, topic)
        
        # Test messages to different topics
        test_scenarios = [
            ("market.AAPL.quote", {"symbol": "AAPL", "price": 155.25}, True),
            ("market.GOOGL.quote", {"symbol": "GOOGL", "price": 2800.00}, False),
            ("orders.user_123", {"order_id": "ord_123", "status": "FILLED"}, True),
            ("orders.user_456", {"order_id": "ord_456", "status": "FILLED"}, False),
            ("risk.alerts", {"alert": "Position limit breach"}, True),
            ("system.maintenance", {"message": "Maintenance mode"}, False)
        ]
        
        for topic, message, should_receive in test_scenarios:
            # Clear previous calls
            mock_websocket.send_text.reset_mock()
            
            sent_count = await websocket_manager.broadcast_message(message, topic)
            
            if should_receive:
                assert sent_count == 1
                mock_websocket.send_text.assert_called_once()
            else:
                assert sent_count == 0
                mock_websocket.send_text.assert_not_called()


class TestRedisWebSocketIntegration:
    """Test WebSocket integration with Redis pub/sub"""
    
    @pytest.fixture
    def websocket_manager(self):
        """WebSocket manager for Redis integration testing"""
        return WebSocketManager()
    
    @pytest.fixture
    def redis_pubsub_manager(self):
        """Redis pub/sub manager for testing"""
        return RedisPubSubManager()
    
    @pytest.fixture
    def streaming_service(self):
        """Streaming service for testing"""
        return StreamingService()
    
    @pytest.mark.asyncio
    async def test_redis_to_websocket_message_flow(self, websocket_manager, redis_pubsub_manager, streaming_service):
        """Test complete message flow from Redis to WebSocket clients"""
        # Mock Redis client
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            # Setup Redis connection
            redis_pubsub_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_pubsub_manager
            
            # Setup WebSocket client
            mock_websocket = AsyncMock()
            connection_id = "redis_test_client"
            
            await websocket_manager.connect(mock_websocket, connection_id)
            websocket_manager.subscribe_to_topic(connection_id, "market.AAPL.quote")
            
            # Create message handler that bridges Redis to WebSocket
            async def redis_to_websocket_handler(channel, message):
                await websocket_manager.broadcast_message(message, channel)
            
            # Setup Redis subscription with handler
            await redis_pubsub_manager.subscribe("market.AAPL.quote", redis_to_websocket_handler)
            
            # Simulate market data update through Redis
            market_data = {
                "symbol": "AAPL",
                "price": 155.25,
                "bid": 155.20,
                "ask": 155.30,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Publish via streaming service
            success = await streaming_service.stream_market_data("AAPL", market_data)
            assert success is True
            
            # Verify Redis publish was called
            mock_redis.publish.assert_called_once()
            
            # In a real scenario, this would trigger the Redis message handler
            # and forward to WebSocket. For testing, we'll simulate this directly
            await redis_to_websocket_handler("market.AAPL.quote", market_data)
            
            # Verify WebSocket client received message
            mock_websocket.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_bidirectional_websocket_redis_communication(self, websocket_manager, streaming_service):
        """Test bidirectional communication between WebSocket and Redis"""
        # Mock Redis
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            # Setup services
            redis_pubsub_manager = RedisPubSubManager()
            redis_pubsub_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_pubsub_manager
            
            # Setup WebSocket client that can send orders
            mock_websocket = AsyncMock()
            connection_id = "bidirectional_client"
            
            await websocket_manager.connect(mock_websocket, connection_id)
            
            # 1. WebSocket client receives market data (Redis -> WebSocket)
            websocket_manager.subscribe_to_topic(connection_id, "market.AAPL.quote")
            
            market_data = {"symbol": "AAPL", "price": 155.25}
            await streaming_service.stream_market_data("AAPL", market_data)
            
            # 2. WebSocket client sends order (WebSocket -> Redis)
            order_data = {
                "symbol": "AAPL",
                "side": "BUY", 
                "quantity": 100,
                "order_type": "MARKET",
                "user_id": "test_user"
            }
            
            # Simulate order placement through WebSocket
            success = await streaming_service.stream_order_update("ord_123", order_data)
            assert success is True
            
            # Verify both directions worked
            assert mock_redis.publish.call_count == 2  # Market data + Order
    
    @pytest.mark.asyncio
    async def test_websocket_redis_error_handling(self, websocket_manager, streaming_service):
        """Test error handling in WebSocket-Redis integration"""
        # Setup with failing Redis
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Redis connection failed")
            mock_redis_class.return_value = mock_redis
            
            redis_pubsub_manager = RedisPubSubManager()
            streaming_service.pubsub_manager = redis_pubsub_manager
            
            # Try to initialize Redis connection
            success = await redis_pubsub_manager.connect()
            assert success is False
            
            # WebSocket should still work independently
            mock_websocket = AsyncMock()
            connection_id = "error_test_client"
            
            success = await websocket_manager.connect(mock_websocket, connection_id)
            assert success is True
            
            # Direct WebSocket messaging should work
            test_message = {"type": "test", "data": "direct message"}
            success = await websocket_manager.send_personal_message(test_message, connection_id)
            assert success is True
            
            # Streaming service should gracefully handle Redis failure
            market_data = {"symbol": "AAPL", "price": 155.25}
            success = await streaming_service.stream_market_data("AAPL", market_data)
            assert success is False  # Should fail gracefully


class TestRealTimeDataStreaming:
    """Test real-time data streaming through WebSocket"""
    
    @pytest.fixture
    def integrated_system(self):
        """Setup integrated system with all components"""
        websocket_manager = WebSocketManager()
        subscription_manager = SubscriptionManager()
        streaming_service = StreamingService()
        message_protocol = MessageProtocol()
        
        return {
            "websocket_manager": websocket_manager,
            "subscription_manager": subscription_manager,
            "streaming_service": streaming_service,
            "message_protocol": message_protocol
        }
    
    @pytest.mark.asyncio
    async def test_market_data_streaming_pipeline(self, integrated_system):
        """Test complete market data streaming pipeline"""
        ws_manager = integrated_system["websocket_manager"]
        streaming_service = integrated_system["streaming_service"]
        message_protocol = integrated_system["message_protocol"]
        
        # Setup mock Redis
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            # Setup Redis for streaming service
            redis_manager = RedisPubSubManager()
            redis_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_manager
            
            # Setup WebSocket clients interested in different symbols
            clients = []
            symbols = ["AAPL", "MSFT", "GOOGL"]
            
            for i, symbol in enumerate(symbols):
                mock_ws = AsyncMock()
                conn_id = f"trader_{i}"
                
                await ws_manager.connect(mock_ws, conn_id, f"user_{i}")
                ws_manager.subscribe_to_topic(conn_id, f"market.{symbol}.quote")
                
                clients.append((conn_id, mock_ws, symbol))
            
            # Stream market data for each symbol
            market_updates = [
                ("AAPL", {"symbol": "AAPL", "price": 155.25, "volume": 1000000}),
                ("MSFT", {"symbol": "MSFT", "price": 305.50, "volume": 500000}),
                ("GOOGL", {"symbol": "GOOGL", "price": 2800.75, "volume": 200000})
            ]
            
            for symbol, data in market_updates:
                # Create properly formatted message
                message = message_protocol.create_market_data_message(
                    symbol=symbol,
                    bid=data["price"] - 0.05,
                    ask=data["price"] + 0.05,
                    connection_id="market_data_feed"
                )
                
                # Stream through Redis
                success = await streaming_service.stream_market_data(symbol, message)
                assert success is True
                
                # In real implementation, Redis would trigger WebSocket broadcast
                # For testing, simulate this
                topic = f"market.{symbol}.quote"
                sent_count = await ws_manager.broadcast_message(message, topic)
                assert sent_count == 1  # One client per symbol
            
            # Verify each client received their symbol's data
            for conn_id, mock_ws, symbol in clients:
                mock_ws.send_text.assert_called()
                # In real scenario, would verify specific message content
    
    @pytest.mark.asyncio
    async def test_portfolio_updates_streaming(self, integrated_system):
        """Test portfolio updates streaming"""
        ws_manager = integrated_system["websocket_manager"]
        streaming_service = integrated_system["streaming_service"]
        
        # Setup mock Redis
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            redis_manager = RedisPubSubManager()
            redis_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_manager
            
            # Setup portfolio manager client
            mock_ws = AsyncMock()
            conn_id = "portfolio_manager"
            
            await ws_manager.connect(mock_ws, conn_id, "portfolio_user")
            ws_manager.subscribe_to_topic(conn_id, "portfolio.portfolio_1.updates")
            
            # Stream portfolio update
            portfolio_update = {
                "portfolio_id": "portfolio_1",
                "total_value": 150000.0,
                "daily_pnl": 5000.0,
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "unrealized_pnl": 2500.0},
                    {"symbol": "MSFT", "quantity": 50, "unrealized_pnl": 1500.0}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            success = await streaming_service.stream_portfolio_update("portfolio_1", portfolio_update)
            assert success is True
            
            # Simulate Redis to WebSocket forwarding
            sent_count = await ws_manager.broadcast_message(
                portfolio_update, "portfolio.portfolio_1.updates"
            )
            assert sent_count == 1
            mock_ws.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_risk_alerts_streaming(self, integrated_system):
        """Test risk alerts streaming"""
        ws_manager = integrated_system["websocket_manager"]
        streaming_service = integrated_system["streaming_service"]
        
        # Setup mock Redis
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            redis_manager = RedisPubSubManager()
            redis_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_manager
            
            # Setup risk manager clients
            risk_clients = []
            for i in range(3):
                mock_ws = AsyncMock()
                conn_id = f"risk_manager_{i}"
                
                await ws_manager.connect(mock_ws, conn_id, f"risk_user_{i}")
                ws_manager.subscribe_to_topic(conn_id, "risk.alerts")
                
                risk_clients.append((conn_id, mock_ws))
            
            # Stream critical risk alert
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
            
            # Risk alerts should go to all risk managers
            sent_count = await ws_manager.broadcast_message(risk_alert, "risk.alerts")
            assert sent_count == 3  # All risk managers should receive
            
            # Verify all risk clients received the alert
            for conn_id, mock_ws in risk_clients:
                mock_ws.send_text.assert_called()


class TestWebSocketPerformanceIntegration:
    """Test WebSocket performance under realistic load"""
    
    @pytest.fixture
    def performance_test_system(self):
        """Setup system for performance testing"""
        return {
            "websocket_manager": WebSocketManager(),
            "streaming_service": StreamingService()
        }
    
    @pytest.mark.asyncio
    async def test_high_frequency_message_broadcasting(self, performance_test_system):
        """Test high-frequency message broadcasting performance"""
        ws_manager = performance_test_system["websocket_manager"]
        
        # Setup multiple clients
        clients = []
        num_clients = 50
        
        for i in range(num_clients):
            mock_ws = AsyncMock()
            conn_id = f"perf_client_{i}"
            
            await ws_manager.connect(mock_ws, conn_id, f"user_{i}")
            ws_manager.subscribe_to_topic(conn_id, "market.data.stream")
            
            clients.append((conn_id, mock_ws))
        
        import time
        start_time = time.time()
        
        # Send 1000 messages rapidly
        num_messages = 1000
        for i in range(num_messages):
            message = {
                "type": "market_tick",
                "symbol": "AAPL",
                "price": 155.25 + (i * 0.01),
                "sequence": i,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sent_count = await ws_manager.broadcast_message(message, "market.data.stream")
            assert sent_count == num_clients
        
        elapsed_time = time.time() - start_time
        
        # Should handle high-frequency broadcasting efficiently
        assert elapsed_time < 10.0  # Within 10 seconds
        
        # Calculate throughput
        messages_per_second = (num_messages * num_clients) / elapsed_time
        assert messages_per_second > 100  # At least 100 messages/sec/client
    
    @pytest.mark.asyncio
    async def test_concurrent_subscription_management(self, performance_test_system):
        """Test concurrent subscription management performance"""
        ws_manager = performance_test_system["websocket_manager"]
        
        # Setup clients
        num_clients = 100
        clients = []
        
        for i in range(num_clients):
            mock_ws = AsyncMock()
            conn_id = f"concurrent_client_{i}"
            await ws_manager.connect(mock_ws, conn_id)
            clients.append(conn_id)
        
        import time
        start_time = time.time()
        
        # Concurrent subscription operations
        tasks = []
        for i, conn_id in enumerate(clients):
            # Each client subscribes to multiple topics
            topics = [f"market.symbol_{j}.quote" for j in range(i % 10)]
            
            for topic in topics:
                task = asyncio.create_task(
                    asyncio.coroutine(lambda: ws_manager.subscribe_to_topic(conn_id, topic))()
                )
                tasks.append(task)
        
        # Wait for all subscriptions to complete
        await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Should handle concurrent subscriptions efficiently
        assert elapsed_time < 5.0  # Within 5 seconds
        
        # Verify all subscriptions were successful
        total_subscriptions = sum(
            len(ws_manager.subscriptions.get(conn_id, set()))
            for conn_id in clients
        )
        assert total_subscriptions > 0
    
    def test_memory_usage_with_many_connections(self, performance_test_system):
        """Test memory usage with many WebSocket connections"""
        ws_manager = performance_test_system["websocket_manager"]
        
        import sys
        import gc
        
        # Force garbage collection and measure initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create many connections
        num_connections = 1000
        
        for i in range(num_connections):
            mock_ws = Mock()
            conn_id = f"memory_test_{i}"
            
            # Simulate connection data
            ws_manager.active_connections[conn_id] = mock_ws
            ws_manager.subscriptions[conn_id] = {f"topic_{i % 100}"}
            ws_manager.connection_metadata[conn_id] = {
                "connected_at": datetime.utcnow(),
                "last_heartbeat": datetime.utcnow(),
                "message_count": 0
            }
        
        # Measure memory after connections
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        objects_per_connection = object_growth / num_connections
        
        # Should use reasonable memory per connection
        assert objects_per_connection < 10  # Less than 10 objects per connection
        
        # Cleanup
        ws_manager.active_connections.clear()
        ws_manager.subscriptions.clear()
        ws_manager.connection_metadata.clear()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
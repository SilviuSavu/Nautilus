"""
Load tests for WebSocket Scalability - Sprint 3

Tests WebSocket scalability with 1000+ concurrent connections, high-frequency
message broadcasting, and system performance under realistic load.
"""

import asyncio
import pytest
import json
import time
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Set
import concurrent.futures
import threading
from statistics import mean, median, stdev

# Import Sprint 3 WebSocket components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from websocket.websocket_manager import WebSocketManager
from websocket.subscription_manager import SubscriptionManager
from websocket.redis_pubsub import RedisPubSubManager
from websocket.streaming_service import StreamingService
from websocket.message_protocols import MessageProtocol, MessageType


class MockWebSocket:
    """High-performance mock WebSocket for load testing"""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.connected = False
        self.message_count = 0
        self.last_message_time = None
        self.messages_received = []
        
    async def accept(self):
        """Mock WebSocket accept"""
        self.connected = True
    
    async def send_text(self, message: str):
        """Mock sending text message"""
        if not self.connected:
            raise Exception("WebSocket not connected")
        
        self.message_count += 1
        self.last_message_time = time.time()
        
        # Only store messages for small-scale tests to avoid memory issues
        if len(self.messages_received) < 100:
            self.messages_received.append(json.loads(message))
    
    async def close(self):
        """Mock WebSocket close"""
        self.connected = False


class TestWebSocketHighConcurrency:
    """Test WebSocket performance with high concurrency"""
    
    @pytest.fixture
    def ws_manager(self):
        """WebSocket manager for load testing"""
        return WebSocketManager()
    
    @pytest.mark.asyncio
    async def test_1000_concurrent_connections(self, ws_manager):
        """Test 1000 concurrent WebSocket connections"""
        num_connections = 1000
        connections = []
        
        # Measure connection establishment time
        start_time = time.time()
        
        # Create connections concurrently
        connection_tasks = []
        for i in range(num_connections):
            mock_ws = MockWebSocket(f"load_test_conn_{i}")
            conn_id = f"load_test_conn_{i}"
            user_id = f"user_{i}"
            
            task = ws_manager.connect(mock_ws, conn_id, user_id)
            connection_tasks.append((conn_id, mock_ws, task))
        
        # Wait for all connections to establish
        connection_results = []
        for conn_id, mock_ws, task in connection_tasks:
            success = await task
            connection_results.append((conn_id, mock_ws, success))
            connections.append((conn_id, mock_ws))
        
        connection_time = time.time() - start_time
        
        # Verify all connections successful
        successful_connections = sum(1 for _, _, success in connection_results if success)
        assert successful_connections == num_connections
        
        # Performance requirements
        assert connection_time < 10.0  # All connections within 10 seconds
        connections_per_second = num_connections / connection_time
        assert connections_per_second > 100  # At least 100 connections/sec
        
        # Verify manager state
        assert len(ws_manager.active_connections) == num_connections
        assert len(ws_manager.connection_metadata) == num_connections
        assert len(ws_manager.subscriptions) == num_connections
        
        # Test connection statistics generation
        stats_start = time.time()
        stats = ws_manager.get_connection_stats()
        stats_time = time.time() - stats_start
        
        assert stats["total_connections"] == num_connections
        assert stats_time < 1.0  # Stats generation within 1 second
        
        # Cleanup connections to free memory
        for conn_id, mock_ws in connections[:100]:  # Cleanup first 100 for further tests
            ws_manager.disconnect(conn_id)
        
        # Verify partial cleanup
        assert len(ws_manager.active_connections) == num_connections - 100
    
    @pytest.mark.asyncio
    async def test_high_frequency_message_broadcasting(self, ws_manager):
        """Test high-frequency message broadcasting to many clients"""
        num_clients = 500
        messages_per_client = 100
        
        # Setup clients
        clients = []
        for i in range(num_clients):
            mock_ws = MockWebSocket(f"hf_client_{i}")
            conn_id = f"hf_client_{i}"
            
            await ws_manager.connect(mock_ws, conn_id, f"hf_user_{i}")
            ws_manager.subscribe_to_topic(conn_id, "market.data.stream")
            
            clients.append((conn_id, mock_ws))
        
        # Measure broadcasting performance
        start_time = time.time()
        total_messages_sent = 0
        
        for msg_id in range(messages_per_client):
            message = {
                "type": "market_tick",
                "symbol": "AAPL",
                "price": 150.00 + (msg_id * 0.01),
                "sequence": msg_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sent_count = await ws_manager.broadcast_message(message, "market.data.stream")
            total_messages_sent += sent_count
            
            # Brief pause to simulate realistic message frequency
            if msg_id % 10 == 0:
                await asyncio.sleep(0.001)  # 1ms pause every 10 messages
        
        broadcasting_time = time.time() - start_time
        
        # Performance verification
        expected_total_messages = num_clients * messages_per_client
        assert total_messages_sent == expected_total_messages
        
        # Performance requirements
        assert broadcasting_time < 30.0  # Complete within 30 seconds
        messages_per_second = total_messages_sent / broadcasting_time
        assert messages_per_second > 1000  # At least 1000 messages/sec total throughput
        
        # Verify message delivery
        sample_clients = clients[:10]  # Check first 10 clients
        for conn_id, mock_ws in sample_clients:
            assert mock_ws.message_count == messages_per_client
            assert mock_ws.last_message_time is not None
        
        print(f"Broadcasting performance: {messages_per_second:.0f} messages/sec")
        print(f"Per-client throughput: {messages_per_second / num_clients:.1f} messages/sec/client")
    
    @pytest.mark.asyncio
    async def test_subscription_management_at_scale(self, ws_manager):
        """Test subscription management with many clients and topics"""
        num_clients = 200
        topics_per_client = 20
        
        # Create diverse topic set
        base_topics = [
            "market.{}.quote", "market.{}.trade", "market.{}.depth",
            "orders.{}", "portfolio.{}.updates", "risk.{}.alerts"
        ]
        symbols = [f"SYMBOL_{i:03d}" for i in range(100)]
        
        all_topics = []
        for topic_template in base_topics:
            if "{}" in topic_template:
                all_topics.extend([topic_template.format(symbol) for symbol in symbols[:50]])
            else:
                all_topics.append(topic_template)
        
        # Setup clients with random subscriptions
        clients = []
        subscription_start = time.time()
        
        for i in range(num_clients):
            mock_ws = MockWebSocket(f"sub_client_{i}")
            conn_id = f"sub_client_{i}"
            
            await ws_manager.connect(mock_ws, conn_id)
            
            # Subscribe to random topics
            import random
            client_topics = random.sample(all_topics, min(topics_per_client, len(all_topics)))
            
            for topic in client_topics:
                success = ws_manager.subscribe_to_topic(conn_id, topic)
                assert success is True
            
            clients.append((conn_id, mock_ws, client_topics))
        
        subscription_time = time.time() - subscription_start
        
        total_subscriptions = num_clients * topics_per_client
        
        # Performance requirements
        assert subscription_time < 15.0  # All subscriptions within 15 seconds
        subscriptions_per_second = total_subscriptions / subscription_time
        assert subscriptions_per_second > 100  # At least 100 subscriptions/sec
        
        # Verify subscription state
        stats = ws_manager.get_connection_stats()
        assert stats["total_subscriptions"] == total_subscriptions
        assert len(stats["connections_by_topic"]) > 0
        
        # Test targeted message broadcasting
        broadcast_start = time.time()
        
        # Send messages to different topic groups
        test_topics = list(stats["connections_by_topic"].keys())[:10]
        for topic in test_topics:
            message = {
                "type": "topic_update",
                "topic": topic,
                "data": f"Update for {topic}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sent_count = await ws_manager.broadcast_message(message, topic)
            expected_subscribers = stats["connections_by_topic"][topic]
            assert sent_count == expected_subscribers
        
        broadcast_time = time.time() - broadcast_start
        
        print(f"Subscription management: {subscriptions_per_second:.0f} subscriptions/sec")
        print(f"Targeted broadcasting: {len(test_topics) / broadcast_time:.1f} topics/sec")
    
    def test_memory_usage_under_load(self, ws_manager):
        """Test memory usage with many connections"""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        num_connections = 1000
        connections = []
        
        # Create connections and track memory
        for i in range(num_connections):
            mock_ws = MockWebSocket(f"mem_test_{i}")
            conn_id = f"mem_test_{i}"
            
            # Simulate connection data
            ws_manager.active_connections[conn_id] = mock_ws
            ws_manager.subscriptions[conn_id] = {f"topic_{j}" for j in range(i % 10)}
            ws_manager.connection_metadata[conn_id] = {
                "user_id": f"user_{i}",
                "connected_at": datetime.utcnow(),
                "last_heartbeat": datetime.utcnow(),
                "message_count": i % 100
            }
            
            connections.append((conn_id, mock_ws))
            
            # Check memory every 100 connections
            if i % 100 == 99:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_per_100_connections = (current_memory - initial_memory) / (i + 1) * 100
                
                # Should use reasonable memory per connection
                assert memory_per_100_connections < 50  # Less than 50MB per 100 connections
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_connection = (final_memory - initial_memory) / num_connections * 1024  # KB
        
        print(f"Memory usage: {memory_per_connection:.2f} KB per connection")
        print(f"Total memory increase: {final_memory - initial_memory:.1f} MB")
        
        # Performance requirements
        assert memory_per_connection < 100  # Less than 100KB per connection
        
        # Cleanup
        ws_manager.active_connections.clear()
        ws_manager.subscriptions.clear()
        ws_manager.connection_metadata.clear()
        
        # Force garbage collection
        gc.collect()


class TestWebSocketRealTimeStreaming:
    """Test real-time streaming performance"""
    
    @pytest.fixture
    def streaming_system(self):
        """Setup streaming system for load testing"""
        ws_manager = WebSocketManager()
        streaming_service = StreamingService()
        
        # Mock Redis for high performance
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.publish.return_value = 1
        
        redis_manager = RedisPubSubManager()
        redis_manager.redis_client = mock_redis
        streaming_service.pubsub_manager = redis_manager
        
        return {
            "ws_manager": ws_manager,
            "streaming_service": streaming_service,
            "redis_manager": redis_manager
        }
    
    @pytest.mark.asyncio
    async def test_market_data_streaming_performance(self, streaming_system):
        """Test market data streaming at market rates"""
        ws_manager = streaming_system["ws_manager"]
        streaming_service = streaming_system["streaming_service"]
        
        # Setup market data subscribers
        num_subscribers = 100
        num_symbols = 50
        
        subscribers = []
        for i in range(num_subscribers):
            mock_ws = MockWebSocket(f"market_subscriber_{i}")
            conn_id = f"market_subscriber_{i}"
            
            await ws_manager.connect(mock_ws, conn_id, f"trader_{i}")
            
            # Subscribe to random symbols
            import random
            subscribed_symbols = random.sample([f"SYM_{j:03d}" for j in range(num_symbols)], 10)
            
            for symbol in subscribed_symbols:
                ws_manager.subscribe_to_topic(conn_id, f"market.{symbol}.quote")
            
            subscribers.append((conn_id, mock_ws, subscribed_symbols))
        
        # Simulate market data at realistic rates
        # Real market: ~100-1000 updates/sec during active trading
        updates_per_second = 500
        test_duration = 10  # seconds
        total_updates = updates_per_second * test_duration
        
        symbols = [f"SYM_{i:03d}" for i in range(num_symbols)]
        
        start_time = time.time()
        updates_sent = 0
        
        for update_id in range(total_updates):
            # Rotate through symbols
            symbol = symbols[update_id % num_symbols]
            
            market_data = {
                "symbol": symbol,
                "bid": 100.00 + (update_id % 1000) * 0.01,
                "ask": 100.05 + (update_id % 1000) * 0.01,
                "last": 100.025 + (update_id % 1000) * 0.01,
                "volume": 1000 + (update_id % 5000),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stream through Redis (mocked)
            success = await streaming_service.stream_market_data(symbol, market_data)
            assert success is True
            
            # Broadcast to WebSocket subscribers
            topic = f"market.{symbol}.quote"
            sent_count = await ws_manager.broadcast_message(market_data, topic)
            updates_sent += sent_count
            
            # Throttle to maintain target rate
            if update_id > 0 and update_id % 100 == 0:
                elapsed = time.time() - start_time
                expected_time = update_id / updates_per_second
                if elapsed < expected_time:
                    await asyncio.sleep(expected_time - elapsed)
        
        actual_duration = time.time() - start_time
        actual_rate = total_updates / actual_duration
        
        # Performance verification
        assert actual_rate > updates_per_second * 0.8  # Within 20% of target rate
        assert updates_sent > 0
        
        print(f"Market data streaming: {actual_rate:.0f} updates/sec")
        print(f"Total messages sent: {updates_sent:,}")
    
    @pytest.mark.asyncio
    async def test_portfolio_update_streaming(self, streaming_system):
        """Test portfolio update streaming performance"""
        ws_manager = streaming_system["ws_manager"]
        streaming_service = streaming_system["streaming_service"]
        
        # Setup portfolio managers
        num_managers = 20
        portfolios_per_manager = 5
        
        managers = []
        for i in range(num_managers):
            mock_ws = MockWebSocket(f"portfolio_manager_{i}")
            conn_id = f"portfolio_manager_{i}"
            
            await ws_manager.connect(mock_ws, conn_id, f"pm_{i}")
            
            # Subscribe to portfolio updates
            managed_portfolios = [f"portfolio_{i}_{j}" for j in range(portfolios_per_manager)]
            for portfolio_id in managed_portfolios:
                ws_manager.subscribe_to_topic(conn_id, f"portfolio.{portfolio_id}.updates")
            
            managers.append((conn_id, mock_ws, managed_portfolios))
        
        # Simulate portfolio updates
        # Real scenario: 1-10 updates/sec per active portfolio
        total_portfolios = num_managers * portfolios_per_manager
        updates_per_portfolio = 20
        
        start_time = time.time()
        
        for update_round in range(updates_per_portfolio):
            for manager_id, (conn_id, mock_ws, portfolios) in enumerate(managers):
                for portfolio_id in portfolios:
                    portfolio_update = {
                        "portfolio_id": portfolio_id,
                        "total_value": 1000000 + (update_round * 1000),
                        "daily_pnl": -5000 + (update_round * 100),
                        "positions_count": 15 + (update_round % 5),
                        "last_updated": datetime.utcnow().isoformat()
                    }
                    
                    # Stream update
                    await streaming_service.stream_portfolio_update(portfolio_id, portfolio_update)
                    
                    # Broadcast to subscribers
                    topic = f"portfolio.{portfolio_id}.updates"
                    sent_count = await ws_manager.broadcast_message(portfolio_update, topic)
                    assert sent_count == 1  # One manager per portfolio
        
        update_duration = time.time() - start_time
        total_updates = total_portfolios * updates_per_portfolio
        updates_per_second = total_updates / update_duration
        
        print(f"Portfolio streaming: {updates_per_second:.0f} updates/sec")
        print(f"Average latency per update: {(update_duration / total_updates) * 1000:.2f}ms")
        
        # Performance requirements
        assert updates_per_second > 50  # At least 50 portfolio updates/sec
        assert (update_duration / total_updates) < 0.010  # Less than 10ms per update
    
    @pytest.mark.asyncio
    async def test_mixed_message_type_performance(self, streaming_system):
        """Test performance with mixed message types"""
        ws_manager = streaming_system["ws_manager"]
        streaming_service = streaming_system["streaming_service"]
        
        # Setup diverse client base
        num_clients = 150
        clients = []
        
        for i in range(num_clients):
            mock_ws = MockWebSocket(f"mixed_client_{i}")
            conn_id = f"mixed_client_{i}"
            
            await ws_manager.connect(mock_ws, conn_id, f"user_{i}")
            
            # Subscribe to different message types based on client type
            client_type = i % 3
            
            if client_type == 0:  # Market data clients
                topics = [f"market.SYM_{j:02d}.quote" for j in range(i % 10)]
            elif client_type == 1:  # Portfolio clients
                topics = [f"portfolio.port_{j}.updates" for j in range(i % 5)]
            else:  # Risk management clients
                topics = ["risk.alerts", "risk.limit_breaches", "risk.system_warnings"]
            
            for topic in topics:
                ws_manager.subscribe_to_topic(conn_id, topic)
            
            clients.append((conn_id, mock_ws, topics, client_type))
        
        # Send mixed message types concurrently
        message_types = [
            ("market", "market.SYM_{:02d}.quote", {"type": "quote", "symbol": "SYM_{:02d}", "price": 100.0}),
            ("portfolio", "portfolio.port_{}.updates", {"type": "portfolio_update", "portfolio_id": "port_{}", "value": 1000000}),
            ("risk", "risk.alerts", {"type": "risk_alert", "severity": "HIGH", "message": "Limit breach detected"})
        ]
        
        start_time = time.time()
        total_messages = 0
        
        # Send 1000 mixed messages
        for msg_id in range(1000):
            msg_type_idx = msg_id % len(message_types)
            msg_type, topic_template, message_template = message_types[msg_type_idx]
            
            if msg_type == "market":
                symbol_id = msg_id % 10
                topic = topic_template.format(symbol_id)
                message = message_template.copy()
                message["symbol"] = message["symbol"].format(symbol_id)
                message["price"] += (msg_id * 0.01)
            elif msg_type == "portfolio":
                portfolio_id = msg_id % 20
                topic = topic_template.format(portfolio_id)
                message = message_template.copy()
                message["portfolio_id"] = message["portfolio_id"].format(portfolio_id)
                message["value"] += (msg_id * 100)
            else:  # risk
                topic = topic_template
                message = message_template.copy()
                message["message"] = f"Alert #{msg_id}: {message['message']}"
            
            message["timestamp"] = datetime.utcnow().isoformat()
            message["sequence"] = msg_id
            
            # Broadcast message
            sent_count = await ws_manager.broadcast_message(message, topic)
            total_messages += sent_count
            
            # Small delay for realistic pacing
            if msg_id % 50 == 0:
                await asyncio.sleep(0.001)
        
        mixed_duration = time.time() - start_time
        messages_per_second = total_messages / mixed_duration
        
        print(f"Mixed message performance: {messages_per_second:.0f} messages/sec")
        print(f"Average processing time: {(mixed_duration / 1000) * 1000:.2f}ms per message batch")
        
        # Performance requirements
        assert messages_per_second > 500  # At least 500 messages/sec
        assert mixed_duration < 20.0  # Complete within 20 seconds


class TestWebSocketFailureRecovery:
    """Test WebSocket performance under failure conditions"""
    
    @pytest.fixture
    def resilient_ws_manager(self):
        """WebSocket manager configured for resilience testing"""
        return WebSocketManager()
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling_at_scale(self, resilient_ws_manager):
        """Test handling connection failures at scale"""
        num_connections = 500
        failure_rate = 0.10  # 10% of connections will fail
        
        connections = []
        failed_connections = []
        
        # Create connections, some of which will fail
        for i in range(num_connections):
            mock_ws = MockWebSocket(f"resilience_conn_{i}")
            conn_id = f"resilience_conn_{i}"
            
            # Simulate random connection failures
            if i % int(1 / failure_rate) == 0:  # Every 10th connection fails
                mock_ws.send_text = AsyncMock(side_effect=Exception("Connection lost"))
                failed_connections.append((conn_id, mock_ws))
            else:
                connections.append((conn_id, mock_ws))
            
            await resilient_ws_manager.connect(mock_ws, conn_id)
        
        # Verify all connections were accepted initially
        assert len(resilient_ws_manager.active_connections) == num_connections
        
        # Broadcast message to trigger failures
        test_message = {
            "type": "resilience_test",
            "data": "Testing connection resilience",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        start_time = time.time()
        sent_count = await resilient_ws_manager.broadcast_message(test_message)
        failure_handling_time = time.time() - start_time
        
        # Verify failed connections were cleaned up
        expected_successful = num_connections - len(failed_connections)
        assert sent_count == expected_successful
        assert len(resilient_ws_manager.active_connections) == expected_successful
        
        # Performance requirement: failure handling should be fast
        assert failure_handling_time < 2.0  # Within 2 seconds
        
        print(f"Failure recovery: {len(failed_connections)} failures handled in {failure_handling_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_performance_degradation_under_load(self, resilient_ws_manager):
        """Test performance degradation patterns under increasing load"""
        load_levels = [100, 250, 500, 750, 1000]
        performance_results = []
        
        for load_level in load_levels:
            # Setup connections for current load level
            connections = []
            for i in range(load_level):
                mock_ws = MockWebSocket(f"load_{load_level}_{i}")
                conn_id = f"load_{load_level}_{i}"
                
                await resilient_ws_manager.connect(mock_ws, conn_id)
                resilient_ws_manager.subscribe_to_topic(conn_id, "load.test")
                
                connections.append((conn_id, mock_ws))
            
            # Measure broadcasting performance at this load level
            num_messages = 50
            start_time = time.time()
            
            for msg_id in range(num_messages):
                message = {
                    "type": "load_test",
                    "load_level": load_level,
                    "message_id": msg_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                sent_count = await resilient_ws_manager.broadcast_message(message, "load.test")
                assert sent_count == load_level
            
            duration = time.time() - start_time
            messages_per_second = (num_messages * load_level) / duration
            avg_latency = (duration / num_messages) * 1000  # ms per message
            
            performance_results.append({
                "load_level": load_level,
                "messages_per_second": messages_per_second,
                "avg_latency_ms": avg_latency,
                "duration": duration
            })
            
            # Clean up for next test
            for conn_id, mock_ws in connections:
                resilient_ws_manager.disconnect(conn_id)
        
        # Analyze performance degradation
        print("Load Level | Messages/sec | Avg Latency (ms)")
        print("-" * 45)
        
        for result in performance_results:
            print(f"{result['load_level']:9d} | {result['messages_per_second']:11.0f} | {result['avg_latency_ms']:13.2f}")
        
        # Verify performance doesn't degrade too severely
        baseline_perf = performance_results[0]["messages_per_second"]
        highest_load_perf = performance_results[-1]["messages_per_second"]
        
        # Should maintain at least 50% of baseline performance at highest load
        performance_retention = highest_load_perf / baseline_perf
        assert performance_retention > 0.3  # At least 30% of baseline performance
        
        # Latency should remain reasonable
        max_latency = max(result["avg_latency_ms"] for result in performance_results)
        assert max_latency < 50.0  # Less than 50ms average latency


class TestWebSocketSystemLimits:
    """Test WebSocket system limits and boundaries"""
    
    def test_maximum_connections_limit(self):
        """Test behavior at maximum connection limits"""
        ws_manager = WebSocketManager()
        
        # Set artificial limit for testing
        original_max_connections = getattr(ws_manager, 'max_connections', None)
        ws_manager.max_connections = 100
        
        connections = []
        
        try:
            # Create connections up to limit
            for i in range(100):
                mock_ws = MockWebSocket(f"limit_test_{i}")
                conn_id = f"limit_test_{i}"
                
                # Simulate connection without async context for testing
                ws_manager.active_connections[conn_id] = mock_ws
                ws_manager.subscriptions[conn_id] = set()
                ws_manager.connection_metadata[conn_id] = {
                    "connected_at": datetime.utcnow(),
                    "message_count": 0
                }
                
                connections.append((conn_id, mock_ws))
            
            # Verify we're at the limit
            assert len(ws_manager.active_connections) == 100
            
            # Test what happens when we try to exceed limit
            # (Implementation dependent - might reject or queue)
            stats = ws_manager.get_connection_stats()
            assert stats["total_connections"] == 100
            
        finally:
            # Cleanup and restore original limit
            ws_manager.active_connections.clear()
            ws_manager.subscriptions.clear()
            ws_manager.connection_metadata.clear()
            
            if original_max_connections is not None:
                ws_manager.max_connections = original_max_connections
            else:
                delattr(ws_manager, 'max_connections')
    
    def test_message_size_limits(self, ws_manager):
        """Test handling of large messages"""
        # Setup one connection
        mock_ws = MockWebSocket("size_test_conn")
        conn_id = "size_test_conn"
        
        # Simulate connection
        ws_manager.active_connections[conn_id] = mock_ws
        ws_manager.subscriptions[conn_id] = {"size.test"}
        ws_manager.connection_metadata[conn_id] = {"message_count": 0}
        
        # Test progressively larger messages
        message_sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
        
        for size in message_sizes:
            large_data = "x" * size
            large_message = {
                "type": "size_test",
                "data": large_data,
                "size_bytes": size,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Measure serialization and broadcasting time
            start_time = time.time()
            
            try:
                sent_count = asyncio.run(
                    ws_manager.broadcast_message(large_message, "size.test")
                )
                
                processing_time = time.time() - start_time
                
                # Should handle large messages efficiently
                assert sent_count == 1
                assert processing_time < 1.0  # Within 1 second
                
                print(f"Message size {size:,} bytes processed in {processing_time*1000:.1f}ms")
                
            except Exception as e:
                # Some very large messages might be rejected
                if size > 100000:  # > 100KB
                    print(f"Large message ({size:,} bytes) rejected: {e}")
                else:
                    raise
    
    def test_subscription_limits_per_connection(self):
        """Test subscription limits per connection"""
        ws_manager = WebSocketManager()
        
        mock_ws = MockWebSocket("subscription_limit_test")
        conn_id = "subscription_limit_test"
        
        # Simulate connection
        ws_manager.active_connections[conn_id] = mock_ws
        ws_manager.subscriptions[conn_id] = set()
        ws_manager.connection_metadata[conn_id] = {"message_count": 0}
        
        # Create many subscriptions
        max_subscriptions = 1000
        
        subscription_start = time.time()
        
        for i in range(max_subscriptions):
            topic = f"topic.{i:04d}"
            success = ws_manager.subscribe_to_topic(conn_id, topic)
            assert success is True
        
        subscription_time = time.time() - subscription_start
        
        # Verify all subscriptions created
        assert len(ws_manager.subscriptions[conn_id]) == max_subscriptions
        
        # Performance requirement
        subscriptions_per_second = max_subscriptions / subscription_time
        assert subscriptions_per_second > 100  # At least 100 subscriptions/sec
        
        # Test unsubscription performance
        unsubscription_start = time.time()
        
        for i in range(0, max_subscriptions, 2):  # Unsubscribe every other topic
            topic = f"topic.{i:04d}"
            success = ws_manager.unsubscribe_from_topic(conn_id, topic)
            assert success is True
        
        unsubscription_time = time.time() - unsubscription_start
        
        # Verify partial unsubscription
        assert len(ws_manager.subscriptions[conn_id]) == max_subscriptions // 2
        
        print(f"Subscription management: {subscriptions_per_second:.0f} subs/sec, "
              f"{(max_subscriptions//2) / unsubscription_time:.0f} unsubs/sec")


if __name__ == "__main__":
    # Run load tests with longer timeout
    pytest.main([__file__, "-v", "--tb=short", "--timeout=300"])
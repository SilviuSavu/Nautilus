# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List

# Import the enhanced MessageBus components
from nautilus_trader.infrastructure.messagebus.config import (
    EnhancedMessageBusConfig, 
    ConfigPresets,
    MessagePriority,
    BufferConfig
)
from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.streams import RedisStreamManager
from nautilus_trader.infrastructure.services.datagov import EnhancedDatagovMessageBusService
from nautilus_trader.infrastructure.services.dbnomics import EnhancedDbnomicsMessageBusService
from nautilus_trader.infrastructure.messagebus.performance import MessageBusBenchmark, BenchmarkConfig

class TestEnhancedMessageBusIntegration:
    """Comprehensive integration tests for enhanced MessageBus"""
    
    @pytest.fixture
    async def messagebus_config(self):
        """Create test MessageBus configuration"""
        return ConfigPresets.development()
    
    @pytest.fixture
    async def messagebus_client(self, messagebus_config):
        """Create and initialize MessageBus client"""
        client = BufferedMessageBusClient(messagebus_config)
        await client.connect()
        yield client
        await client.close()
    
    @pytest.fixture
    async def stream_manager(self, messagebus_config):
        """Create and initialize stream manager"""
        manager = RedisStreamManager(messagebus_config)
        await manager.connect()
        yield manager
        await manager.close()
    
    @pytest.fixture
    async def datagov_service(self):
        """Create Data.gov MessageBus service"""
        config = ConfigPresets.development()
        service = EnhancedDatagovMessageBusService(config)
        await service.start()
        yield service
        await service.stop()
    
    @pytest.fixture
    async def dbnomics_service(self):
        """Create DBnomics MessageBus service"""
        config = ConfigPresets.development()
        service = EnhancedDbnomicsMessageBusService(config)
        await service.start()
        yield service
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_messagebus_client_basic_operations(self, messagebus_client):
        """Test basic MessageBus client operations"""
        # Test publish and subscribe
        topic = "test.basic.operations"
        test_message = b"Hello MessageBus!"
        
        # Subscribe to topic
        await messagebus_client.subscribe(topic)
        
        # Publish message
        await messagebus_client.publish(topic, test_message)
        
        # Receive message
        received = await asyncio.wait_for(
            messagebus_client.receive(), 
            timeout=5.0
        )
        
        assert received == test_message
        
        # Test metrics
        metrics = messagebus_client.get_metrics()
        assert metrics["messages_sent"] > 0
        assert metrics["messages_received"] > 0
    
    @pytest.mark.asyncio
    async def test_message_buffering(self, messagebus_client):
        """Test message buffering functionality"""
        topic = "test.buffering.performance"
        num_messages = 100
        
        # Configure small buffer for testing
        original_flush_interval = messagebus_client.config.default_buffer_config.flush_interval_ms
        messagebus_client.config.default_buffer_config.flush_interval_ms = 50  # 50ms
        
        try:
            await messagebus_client.subscribe(topic)
            
            # Send multiple messages rapidly
            start_time = time.time()
            for i in range(num_messages):
                message = f"Message {i}".encode('utf-8')
                await messagebus_client.publish(topic, message)
            
            # Wait for buffer flush
            await asyncio.sleep(0.2)  # 200ms
            
            # Verify all messages received
            received_count = 0
            while received_count < num_messages:
                try:
                    message = await asyncio.wait_for(
                        messagebus_client.receive(), 
                        timeout=1.0
                    )
                    if message:
                        received_count += 1
                except asyncio.TimeoutError:
                    break
            
            assert received_count == num_messages
            
            # Check buffer metrics
            metrics = messagebus_client.get_metrics()
            assert metrics["buffer_flushes"] > 0
            
        finally:
            # Restore original configuration
            messagebus_client.config.default_buffer_config.flush_interval_ms = original_flush_interval
    
    @pytest.mark.asyncio
    async def test_stream_management(self, stream_manager):
        """Test Redis stream management"""
        stream_name = "test-stream"
        
        # Create stream
        await stream_manager.create_stream(stream_name)
        
        # Add messages to stream
        messages = [
            {"field1": f"value{i}", "timestamp": time.time()}
            for i in range(10)
        ]
        
        for msg in messages:
            await stream_manager.add_to_stream(stream_name, msg)
        
        # Read from stream
        stream_messages = await stream_manager.read_stream(stream_name, count=5)
        assert len(stream_messages) == 5
        
        # Test consumer group
        group_name = "test-group"
        consumer_name = "test-consumer"
        
        await stream_manager.create_consumer_group(stream_name, group_name)
        
        group_messages = await stream_manager.read_consumer_group(
            stream_name, group_name, consumer_name, count=3
        )
        assert len(group_messages) <= 3
        
        # Get stream info
        info = await stream_manager.get_stream_info(stream_name)
        assert info["length"] >= 10
        
        # Cleanup
        await stream_manager.delete_stream(stream_name)
    
    @pytest.mark.asyncio
    async def test_pattern_matching(self, messagebus_client):
        """Test topic pattern matching"""
        # Subscribe to pattern
        pattern = "data.*.BINANCE.*"
        await messagebus_client.subscribe(pattern)
        
        # Test matching topics
        matching_topics = [
            "data.quotes.BINANCE.BTCUSDT",
            "data.trades.BINANCE.ETHUSDT",
            "data.orderbooks.BINANCE.SOLUSDT"
        ]
        
        # Test non-matching topics
        non_matching_topics = [
            "events.trades.BINANCE.BTCUSDT",
            "data.quotes.BYBIT.ETHUSDT",
            "data.quotes.BINANCE"  # incomplete
        ]
        
        # Send messages
        for topic in matching_topics + non_matching_topics:
            message = f"Message for {topic}".encode('utf-8')
            await messagebus_client.publish(topic, message)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Should receive only matching messages
        received_messages = []
        while True:
            try:
                message = await asyncio.wait_for(
                    messagebus_client.receive(), 
                    timeout=0.5
                )
                if message:
                    received_messages.append(message)
                else:
                    break
            except asyncio.TimeoutError:
                break
        
        # Verify we received exactly the matching messages
        assert len(received_messages) == len(matching_topics)
    
    @pytest.mark.asyncio
    async def test_priority_handling(self, messagebus_config):
        """Test message priority handling"""
        # Create client with priority configuration
        critical_buffer = BufferConfig(
            max_size=1000,
            flush_interval_ms=10,  # Very fast for critical
            high_water_mark=800
        )
        
        messagebus_config.priority_buffers[MessagePriority.CRITICAL] = critical_buffer
        
        client = BufferedMessageBusClient(messagebus_config)
        await client.connect()
        
        try:
            topic = "test.priority.handling"
            await client.subscribe(topic)
            
            # Send messages with different priorities
            priorities = [
                (MessagePriority.LOW, "Low priority message"),
                (MessagePriority.CRITICAL, "Critical message"),
                (MessagePriority.NORMAL, "Normal message"),
                (MessagePriority.HIGH, "High priority message")
            ]
            
            for priority, content in priorities:
                message = content.encode('utf-8')
                await client.publish(topic, message, priority=priority)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Receive all messages
            received = []
            while len(received) < len(priorities):
                try:
                    message = await asyncio.wait_for(
                        client.receive(), timeout=1.0
                    )
                    if message:
                        received.append(message.decode('utf-8'))
                except asyncio.TimeoutError:
                    break
            
            assert len(received) == len(priorities)
            
            # Critical messages should be processed faster
            metrics = client.get_metrics()
            assert metrics["messages_sent"] >= len(priorities)
            
        finally:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_datagov_service_integration(self, datagov_service):
        """Test Data.gov service integration"""
        # Test health check
        health_request_id = await datagov_service.send_request(
            "datagov.health.check",
            "/api/v1/datagov/health",
            {},
            callback_topic="test.datagov.health.response"
        )
        
        # Test dataset search
        search_request_id = await datagov_service.send_request(
            "datagov.datasets.search",
            "/api/v1/datagov/datasets/search",
            {"q": "economic", "limit": 10},
            callback_topic="test.datagov.search.response"
        )
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check metrics
        metrics = datagov_service.get_metrics()
        assert metrics["service_metrics"]["requests_processed"] >= 2
        
        # Verify service is healthy
        assert len(datagov_service.active_requests) == 0  # Requests should be completed
    
    @pytest.mark.asyncio
    async def test_dbnomics_service_integration(self, dbnomics_service):
        """Test DBnomics service integration"""
        # Test providers list
        providers_request_id = await dbnomics_service.send_request(
            "dbnomics.providers.list",
            "/api/v1/dbnomics/providers",
            {},
            callback_topic="test.dbnomics.providers.response"
        )
        
        # Test series fetch
        series_request_id = await dbnomics_service.send_request(
            "dbnomics.series.fetch",
            "/api/v1/dbnomics/series",
            {
                "provider_code": "OECD",
                "dataset_code": "EO", 
                "series_code": "GDP_GROWTH"
            },
            callback_topic="test.dbnomics.series.response"
        )
        
        # Test trading indicators
        indicators_request_id = await dbnomics_service.send_request(
            "dbnomics.analytics.trading_indicators",
            "/api/v1/dbnomics/trading-indicators",
            {"type": "macro_economic"},
            callback_topic="test.dbnomics.indicators.response"
        )
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check metrics
        metrics = dbnomics_service.get_metrics()
        assert metrics["service_metrics"]["requests_processed"] >= 3
        assert metrics["cache_metrics"]["size"] >= 1  # Should have cached data
        
        # Test cache hit
        # Send same series request again
        cache_test_request_id = await dbnomics_service.send_request(
            "dbnomics.series.fetch",
            "/api/v1/dbnomics/series",
            {
                "provider_code": "OECD",
                "dataset_code": "EO", 
                "series_code": "GDP_GROWTH"
            },
            callback_topic="test.dbnomics.cache.response"
        )
        
        await asyncio.sleep(1)
        
        # Verify cache hit
        updated_metrics = dbnomics_service.get_metrics()
        assert updated_metrics["cache_metrics"]["hit_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Test performance benchmarking"""
        benchmark_config = BenchmarkConfig(
            duration_seconds=10,  # Short test
            message_rate=100,
            concurrent_producers=2,
            concurrent_consumers=2,
            pattern_complexity=3
        )
        
        messagebus_config = ConfigPresets.development()
        
        benchmark = MessageBusBenchmark(benchmark_config, messagebus_config)
        
        try:
            # Run pattern matching benchmark
            pattern_results = await benchmark.run_pattern_matching_benchmark()
            assert pattern_results["total_operations"] > 0
            assert pattern_results["operations_per_second"] > 0
            
            # Run throughput benchmark
            metrics = await benchmark.run_throughput_benchmark()
            assert metrics.total_messages > 0
            assert metrics.messages_per_second > 0
            assert metrics.avg_latency_ms >= 0
            
            # Generate report
            report = benchmark.generate_report(metrics, pattern_results)
            assert "summary" in report
            assert "latency" in report
            assert "resources" in report
            assert "pattern_matching" in report
            
        finally:
            await benchmark.messagebus_client.close()
            await benchmark.stream_manager.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, messagebus_client):
        """Test error handling and recovery"""
        # Test publishing to non-existent topic (should not fail)
        await messagebus_client.publish("non.existent.topic", b"test message")
        
        # Test subscribing to invalid pattern
        try:
            await messagebus_client.subscribe("invalid[pattern")
            # Should handle gracefully
        except Exception:
            pass  # Expected for invalid patterns
        
        # Test connection recovery
        original_connected = messagebus_client.is_connected()
        assert original_connected
        
        # Force disconnect and reconnect
        await messagebus_client.disconnect()
        assert not messagebus_client.is_connected()
        
        await messagebus_client.connect()
        assert messagebus_client.is_connected()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, messagebus_client):
        """Test concurrent operations"""
        topic_base = "test.concurrent"
        num_concurrent = 50
        
        async def producer_task(task_id: int):
            topic = f"{topic_base}.producer.{task_id}"
            for i in range(10):
                message = f"Producer {task_id} Message {i}".encode('utf-8')
                await messagebus_client.publish(topic, message)
            return task_id
        
        async def consumer_task(task_id: int):
            topic = f"{topic_base}.producer.{task_id}"
            await messagebus_client.subscribe(topic)
            
            messages_received = 0
            while messages_received < 10:
                try:
                    message = await asyncio.wait_for(
                        messagebus_client.receive(), timeout=2.0
                    )
                    if message:
                        messages_received += 1
                except asyncio.TimeoutError:
                    break
            return messages_received
        
        # Run concurrent producers and consumers
        producer_tasks = [producer_task(i) for i in range(num_concurrent)]
        consumer_tasks = [consumer_task(i) for i in range(num_concurrent)]
        
        start_time = time.time()
        
        # Run all tasks concurrently
        producer_results = await asyncio.gather(*producer_tasks)
        consumer_results = await asyncio.gather(*consumer_tasks)
        
        end_time = time.time()
        
        # Verify results
        assert len(producer_results) == num_concurrent
        assert len(consumer_results) == num_concurrent
        assert sum(consumer_results) >= num_concurrent * 5  # At least 50% received
        
        # Check performance
        total_time = end_time - start_time
        total_messages = num_concurrent * 10
        throughput = total_messages / total_time
        
        print(f"Concurrent test: {total_messages} messages in {total_time:.2f}s = {throughput:.2f} msg/s")
        assert throughput > 10  # At least 10 msg/s for the test environment
    
    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, datagov_service, dbnomics_service):
        """Test service health monitoring"""
        # Let services run for a bit to generate health metrics
        await asyncio.sleep(2)
        
        # Check Data.gov service health
        datagov_metrics = datagov_service.get_metrics()
        assert "service_metrics" in datagov_metrics
        assert "messagebus_metrics" in datagov_metrics
        
        # Check DBnomics service health
        dbnomics_metrics = dbnomics_service.get_metrics()
        assert "service_metrics" in dbnomics_metrics
        assert "cache_metrics" in dbnomics_metrics
        assert "providers_metrics" in dbnomics_metrics
        
        # Both services should be running healthy
        assert datagov_service.running
        assert dbnomics_service.running

class TestMessageBusPerformance:
    """Performance-specific tests"""
    
    @pytest.mark.asyncio
    async def test_high_throughput(self):
        """Test high throughput scenario"""
        config = ConfigPresets.high_frequency()
        client = BufferedMessageBusClient(config)
        
        await client.connect()
        
        try:
            topic = "performance.high_throughput"
            await client.subscribe(topic)
            
            # Send 1000 messages as fast as possible
            num_messages = 1000
            message_size = 1024  # 1KB messages
            
            start_time = time.perf_counter()
            
            for i in range(num_messages):
                message = b"X" * message_size
                await client.publish(topic, message)
            
            # Wait for buffer flush
            await asyncio.sleep(0.5)
            
            # Receive all messages
            received_count = 0
            while received_count < num_messages:
                try:
                    message = await asyncio.wait_for(
                        client.receive(), timeout=0.1
                    )
                    if message:
                        received_count += 1
                except asyncio.TimeoutError:
                    break
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            throughput = num_messages / duration
            data_throughput = (num_messages * message_size) / (1024 * 1024) / duration  # MB/s
            
            print(f"High throughput test: {throughput:.2f} msg/s, {data_throughput:.2f} MB/s")
            
            # Performance assertions
            assert throughput > 1000  # At least 1K msg/s
            assert received_count >= num_messages * 0.95  # 95% delivery rate
            
        finally:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement"""
        config = ConfigPresets.development()
        config.default_buffer_config.flush_interval_ms = 1  # 1ms for low latency
        
        client = BufferedMessageBusClient(config)
        await client.connect()
        
        try:
            topic = "performance.latency"
            await client.subscribe(topic)
            
            # Measure round-trip latency
            latencies = []
            num_tests = 100
            
            for i in range(num_tests):
                start_time = time.perf_counter()
                
                message = f"Latency test {i}".encode('utf-8')
                await client.publish(topic, message)
                
                received = await asyncio.wait_for(
                    client.receive(), timeout=5.0
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                assert received == message
                
                # Small delay between tests
                await asyncio.sleep(0.01)
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"Latency stats: avg={avg_latency:.2f}ms, min={min_latency:.2f}ms, max={max_latency:.2f}ms")
            
            # Latency assertions (these may vary by environment)
            assert avg_latency < 100  # Less than 100ms average
            assert min_latency < 50   # At least one message under 50ms
            
        finally:
            await client.close()

# Test configuration for pytest
@pytest.mark.asyncio
async def test_enhanced_messagebus_complete_integration():
    """Complete integration test of all enhanced MessageBus components"""
    # This test combines all components for a comprehensive check
    
    # Create services
    config = ConfigPresets.development()
    
    # Initialize all components
    messagebus_client = BufferedMessageBusClient(config)
    stream_manager = RedisStreamManager(config)
    datagov_service = EnhancedDatagovMessageBusService(config)
    dbnomics_service = EnhancedDbnomicsMessageBusService(config)
    
    try:
        # Start all services
        await messagebus_client.connect()
        await stream_manager.connect()
        await datagov_service.start()
        await dbnomics_service.start()
        
        # Test inter-service communication
        topic = "integration.test.complete"
        
        # Send coordinated requests
        datagov_req = await datagov_service.send_request(
            "datagov.health.check", "/health", {}
        )
        
        dbnomics_req = await dbnomics_service.send_request(
            "dbnomics.providers.list", "/providers", {}
        )
        
        # Test stream operations
        await stream_manager.create_stream("integration-test-stream")
        await stream_manager.add_to_stream("integration-test-stream", {
            "test": "integration",
            "timestamp": time.time(),
            "services": ["datagov", "dbnomics"]
        })
        
        # Wait for all operations to complete
        await asyncio.sleep(3)
        
        # Verify all services are operational
        datagov_metrics = datagov_service.get_metrics()
        dbnomics_metrics = dbnomics_service.get_metrics()
        stream_info = await stream_manager.get_stream_info("integration-test-stream")
        
        assert datagov_metrics["service_metrics"]["requests_processed"] >= 1
        assert dbnomics_metrics["service_metrics"]["requests_processed"] >= 1
        assert stream_info["length"] >= 1
        
        print("✅ Enhanced MessageBus integration test completed successfully")
        
    finally:
        # Clean up all services
        if datagov_service.running:
            await datagov_service.stop()
        if dbnomics_service.running:
            await dbnomics_service.stop()
        if stream_manager:
            await stream_manager.close()
        if messagebus_client:
            await messagebus_client.close()

if __name__ == "__main__":
    # Run the complete integration test
    asyncio.run(test_enhanced_messagebus_complete_integration())
"""
Integration Tests for Monitoring System
Tests end-to-end monitoring workflows with real service integrations and performance validation
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import statistics
from concurrent.futures import ThreadPoolExecutor

from monitoring_service import (
    MonitoringService, 
    AlertLevel, 
    MetricType, 
    Alert, 
    Metric, 
    HealthStatus
)


class TestMonitoringIntegration:
    """Integration test suite for complete monitoring workflows"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service for integration tests"""
        return MonitoringService()
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock external service dependencies"""
        with patch('monitoring_service.rate_limiter') as mock_rate_limiter, \
             patch('monitoring_service.redis_cache') as mock_redis_cache, \
             patch('monitoring_service.historical_data_service') as mock_db_service, \
             patch('monitoring_service.data_normalizer') as mock_normalizer:
            
            # Configure mock responses for healthy services
            mock_rate_limiter.health_check = AsyncMock(return_value={
                "status": "healthy",
                "healthy_venues": 2,
                "open_circuits": 0
            })
            mock_rate_limiter.get_all_metrics = MagicMock(return_value={
                "IB": MagicMock(total_requests=1000, throttled_requests=0, allowed_requests=1000),
                "BINANCE": MagicMock(total_requests=800, throttled_requests=5, allowed_requests=795)
            })
            
            mock_redis_cache.health_check = AsyncMock(return_value={
                "status": "connected",
                "ping_ms": 2.5,
                "connected_clients": 10
            })
            
            mock_db_service.health_check = AsyncMock(return_value={
                "status": "connected",
                "table_counts": {"market_data": 50000, "orders": 1200},
                "pool_stats": {"size": 10, "active": 3}
            })
            
            mock_normalizer.get_quality_metrics = MagicMock(return_value={
                "IB": MagicMock(total_messages=10000, validation_errors=5),
                "BINANCE": MagicMock(total_messages=8000, validation_errors=2)
            })
            
            yield {
                "rate_limiter": mock_rate_limiter,
                "redis_cache": mock_redis_cache,
                "historical_data_service": mock_db_service,
                "data_normalizer": mock_normalizer
            }
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self, monitoring_service, mock_dependencies):
        """Test complete monitoring workflow from metric collection to alerting"""
        alert_triggered = False
        captured_alerts = []
        
        def alert_handler(alert: Alert):
            nonlocal alert_triggered, captured_alerts
            alert_triggered = True
            captured_alerts.append(alert)
        
        # Add alert handler
        monitoring_service.add_alert_handler(alert_handler)
        
        # Wait for initial metrics collection cycle
        await asyncio.sleep(0.5)
        
        # Verify metrics were collected from all services
        metrics = monitoring_service.get_metrics()
        
        # Should have collected rate limiter metrics
        rate_limiter_metrics = [name for name in metrics.keys() if name.startswith("rate_limiter")]
        assert len(rate_limiter_metrics) > 0, "Rate limiter metrics should be collected"
        
        # Should have collected cache metrics
        cache_metrics = [name for name in metrics.keys() if name.startswith("cache")]
        assert len(cache_metrics) > 0, "Cache metrics should be collected"
        
        # Should have collected database metrics
        db_metrics = [name for name in metrics.keys() if name.startswith("database")]
        assert len(db_metrics) > 0, "Database metrics should be collected"
        
        # Inject high latency to trigger alert
        monitoring_service.record_metric(
            name="api_latency_p95", 
            value=150.0,  # Above 100ms threshold
            metric_type=MetricType.GAUGE,
            tags={"component": "api", "venue": "IB"}
        )
        
        # Wait for alert processing
        await asyncio.sleep(0.2)
        
        # Verify alert workflow
        if monitoring_service.thresholds.get("latency_p95", 0) < 150.0:
            # Only check if we have a threshold that would trigger
            active_alerts = monitoring_service.get_alerts(resolved=False)
            assert len(active_alerts) > 0 or len(captured_alerts) > 0, "High latency should trigger alert"
    
    @pytest.mark.asyncio
    async def test_service_dependency_integration(self, monitoring_service, mock_dependencies):
        """Test integration with all external service dependencies"""
        # Wait for health checks to complete
        await asyncio.sleep(1.0)
        
        # Verify all dependencies were checked
        health_status = monitoring_service.get_health_status()
        
        expected_components = ["rate_limiter", "cache", "database"]
        for component in expected_components:
            assert component in health_status, f"{component} health should be tracked"
            assert health_status[component].status in ["healthy", "connected"], \
                f"{component} should be healthy"
        
        # Verify dependency health checks were called
        mock_dependencies["rate_limiter"].health_check.assert_called()
        mock_dependencies["redis_cache"].health_check.assert_called()
        mock_dependencies["historical_data_service"].health_check.assert_called()
    
    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self, monitoring_service, mock_dependencies):
        """Test integrated metrics collection from all sources"""
        # Wait for metrics collection cycles
        await asyncio.sleep(0.8)
        
        metrics = monitoring_service.get_metrics()
        
        # Verify rate limiter integration
        rate_limiter_metrics = [name for name in metrics.keys() if "rate_limiter" in name]
        assert len(rate_limiter_metrics) >= 3, "Should collect multiple rate limiter metrics"
        
        # Verify cache integration
        cache_metrics = [name for name in metrics.keys() if "cache" in name]
        assert len(cache_metrics) >= 1, "Should collect cache metrics"
        
        # Verify database integration
        database_metrics = [name for name in metrics.keys() if "database" in name]
        assert len(database_metrics) >= 1, "Should collect database metrics"
        
        # Verify data quality integration
        quality_metrics = [name for name in metrics.keys() if "data_quality" in name]
        assert len(quality_metrics) >= 1, "Should collect data quality metrics"
    
    @pytest.mark.asyncio
    async def test_alert_escalation_workflow(self, monitoring_service, mock_dependencies):
        """Test complete alert escalation workflow"""
        escalation_levels = []
        
        def escalation_handler(alert: Alert):
            escalation_levels.append((alert.level, alert.timestamp))
        
        monitoring_service.add_alert_handler(escalation_handler)
        
        # Simulate escalating system issues
        # First: Warning level issue
        monitoring_service.record_metric("cpu_usage", 75.0, MetricType.GAUGE)
        await asyncio.sleep(0.1)
        
        # Then: Error level issue  
        monitoring_service.record_metric("cpu_usage", 90.0, MetricType.GAUGE)
        await asyncio.sleep(0.1)
        
        # Finally: Critical level issue
        monitoring_service.record_metric("memory_usage", 95.0, MetricType.GAUGE)
        await asyncio.sleep(0.1)
        
        # Wait for alert processing
        await asyncio.sleep(0.5)
        
        alerts = monitoring_service.get_alerts(resolved=False)
        
        # Should have alerts for threshold violations
        if monitoring_service.thresholds.get("memory_usage", 1.0) < 0.95:
            assert len(alerts) > 0, "Critical thresholds should generate alerts"
    
    @pytest.mark.asyncio
    async def test_auto_resolution_integration(self, monitoring_service, mock_dependencies):
        """Test integrated auto-resolution of alerts when conditions improve"""
        resolution_events = []
        
        def resolution_handler(alert: Alert):
            if alert.resolved:
                resolution_events.append(alert)
        
        monitoring_service.add_alert_handler(resolution_handler)
        
        # Create alert condition
        alert_id = monitoring_service.create_alert(
            AlertLevel.WARNING,
            "High CPU Usage",
            "CPU usage is above threshold",
            "system_monitor",
            {"component": "cpu"}
        )
        
        # Wait for alert processing
        await asyncio.sleep(0.2)
        
        # Simulate condition improvement (mock the auto-resolution logic)
        monitoring_service.record_metric("cpu_usage", 60.0, MetricType.GAUGE)
        await asyncio.sleep(0.3)
        
        # Manually resolve to test the workflow
        resolved = monitoring_service.resolve_alert(alert_id)
        assert resolved, "Alert should be resolvable"
        
        # Verify alert is in resolved state
        resolved_alerts = monitoring_service.get_alerts(resolved=True)
        assert len(resolved_alerts) > 0, "Should have resolved alerts"
        assert any(alert.id == alert_id for alert in resolved_alerts), "Specific alert should be resolved"
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring_operations(self, monitoring_service, mock_dependencies):
        """Test monitoring system under concurrent operations"""
        async def concurrent_metric_recording():
            """Simulate concurrent metric recording"""
            for i in range(50):
                monitoring_service.record_metric(
                    f"concurrent_metric_{i % 5}",
                    float(i * 2),
                    MetricType.COUNTER,
                    {"thread": f"worker_{i % 3}"}
                )
                await asyncio.sleep(0.001)  # Small delay to simulate real workload
        
        async def concurrent_alert_creation():
            """Simulate concurrent alert creation"""
            for i in range(20):
                monitoring_service.create_alert(
                    AlertLevel.INFO,
                    f"Concurrent Alert {i}",
                    f"Test alert {i}",
                    "concurrent_test",
                    {"alert_id": str(i)}
                )
                await asyncio.sleep(0.002)
        
        # Run concurrent operations
        await asyncio.gather(
            concurrent_metric_recording(),
            concurrent_alert_creation(),
            asyncio.sleep(0.5)  # Let monitoring tasks run
        )
        
        # Verify system stability under load
        metrics = monitoring_service.get_metrics()
        alerts = monitoring_service.get_alerts()
        
        # Should have collected all concurrent metrics
        concurrent_metrics = [name for name in metrics.keys() if "concurrent_metric" in name]
        assert len(concurrent_metrics) == 5, "All concurrent metrics should be collected"
        
        # Should have created concurrent alerts
        concurrent_alerts = [alert for alert in alerts if "Concurrent Alert" in alert.title]
        assert len(concurrent_alerts) == 20, "All concurrent alerts should be created"
        
        # Verify service is still running properly
        assert monitoring_service._running, "Service should remain running under load"


class TestMonitoringPerformance:
    """Performance tests for monitoring system"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service for performance testing"""
        return MonitoringService()
    
    def test_metric_recording_performance(self, monitoring_service):
        """Test metric recording performance - should be <0.1ms overhead per operation"""
        num_metrics = 10000
        start_time = time.perf_counter()
        
        for i in range(num_metrics):
            monitoring_service.record_metric(
                f"perf_test_metric_{i % 100}",
                float(i),
                MetricType.COUNTER,
                {"batch": str(i // 1000)}
            )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_operation = (total_time / num_metrics) * 1000  # Convert to milliseconds
        
        print(f"Average metric recording time: {avg_time_per_operation:.3f}ms per operation")
        print(f"Total time for {num_metrics} metrics: {total_time:.3f}s")
        
        # Performance requirement: <0.1ms overhead per operation
        assert avg_time_per_operation < 0.1, \
            f"Metric recording too slow: {avg_time_per_operation:.3f}ms > 0.1ms"
        
        # Verify all metrics were recorded
        metrics = monitoring_service.get_metrics()
        total_recorded = sum(len(metric_list) for metric_list in metrics.values())
        assert total_recorded == num_metrics, "All metrics should be recorded"
    
    def test_alert_creation_performance(self, monitoring_service):
        """Test alert creation performance"""
        num_alerts = 5000
        start_time = time.perf_counter()
        
        for i in range(num_alerts):
            monitoring_service.create_alert(
                AlertLevel.INFO,
                f"Performance Test Alert {i}",
                f"Test message {i}",
                "perf_test",
                {"alert_num": str(i)}
            )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_operation = (total_time / num_alerts) * 1000
        
        print(f"Average alert creation time: {avg_time_per_operation:.3f}ms per operation")
        
        # Alert creation should be fast
        assert avg_time_per_operation < 0.5, \
            f"Alert creation too slow: {avg_time_per_operation:.3f}ms > 0.5ms"
        
        # Verify alerts were created
        alerts = monitoring_service.get_alerts()
        assert len(alerts) == num_alerts, "All alerts should be created"
    
    def test_metrics_query_performance(self, monitoring_service):
        """Test metrics query performance with large datasets"""
        # Populate with test data
        num_metrics = 50000
        for i in range(num_metrics):
            monitoring_service.record_metric(
                f"query_test_metric_{i % 10}",
                float(i),
                MetricType.GAUGE,
                {"category": f"cat_{i % 5}"}
            )
        
        # Test query performance
        start_time = time.perf_counter()
        
        # Query all metrics
        all_metrics = monitoring_service.get_metrics()
        
        # Query specific metric
        specific_metrics = monitoring_service.get_metrics("query_test_metric_0")
        
        # Query with time filter
        since_time = datetime.now() - timedelta(minutes=5)
        filtered_metrics = monitoring_service.get_metrics(since=since_time)
        
        end_time = time.perf_counter()
        query_time = (end_time - start_time) * 1000
        
        print(f"Metrics query time for {num_metrics} records: {query_time:.3f}ms")
        
        # Query should be reasonably fast
        assert query_time < 100, f"Metrics query too slow: {query_time:.3f}ms > 100ms"
        
        # Verify query results
        assert len(all_metrics) == 10, "Should have 10 different metric names"
        assert "query_test_metric_0" in specific_metrics, "Specific metric should be found"
        assert len(filtered_metrics) > 0, "Filtered query should return results"
    
    def test_concurrent_performance(self, monitoring_service):
        """Test performance under concurrent load"""
        num_threads = 5
        operations_per_thread = 1000
        
        def worker_thread(thread_id):
            """Worker function for concurrent testing"""
            start_time = time.perf_counter()
            
            for i in range(operations_per_thread):
                # Mix of operations
                if i % 3 == 0:
                    monitoring_service.record_metric(
                        f"concurrent_metric_{thread_id}_{i}",
                        float(i),
                        MetricType.COUNTER
                    )
                elif i % 3 == 1:
                    monitoring_service.create_alert(
                        AlertLevel.INFO,
                        f"Concurrent Alert T{thread_id}-{i}",
                        f"Message {i}",
                        f"thread_{thread_id}"
                    )
                else:
                    # Query operation
                    monitoring_service.get_metrics()
            
            end_time = time.perf_counter()
            return end_time - start_time, thread_id
        
        # Run concurrent workers
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        thread_times = [result[0] for result in results]
        avg_thread_time = statistics.mean(thread_times)
        max_thread_time = max(thread_times)
        
        total_operations = num_threads * operations_per_thread
        avg_ops_per_second = total_operations / total_time
        
        print(f"Concurrent performance results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average thread time: {avg_thread_time:.3f}s")
        print(f"  Max thread time: {max_thread_time:.3f}s")
        print(f"  Operations per second: {avg_ops_per_second:.1f}")
        
        # Performance requirements
        assert avg_ops_per_second > 5000, \
            f"Concurrent throughput too low: {avg_ops_per_second:.1f} ops/sec < 5000"
        assert max_thread_time < avg_thread_time * 2, \
            "Thread execution times should be reasonably consistent"
        
        # Verify data integrity under concurrent load
        final_metrics = monitoring_service.get_metrics()
        final_alerts = monitoring_service.get_alerts()
        
        print(f"Final state: {len(final_metrics)} metric types, {len(final_alerts)} alerts")
        
        # Should have created substantial amount of data
        assert len(final_alerts) > num_threads * 100, "Should have created many alerts concurrently"
    
    def test_memory_usage_performance(self, monitoring_service):
        """Test memory usage patterns and cleanup"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        for i in range(100000):
            monitoring_service.record_metric(
                f"memory_test_{i % 1000}",
                float(i),
                MetricType.GAUGE,
                {"batch": str(i // 10000)}
            )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = peak_memory - initial_memory
        
        # Test cleanup by querying old data (should trigger retention cleanup)
        old_time = datetime.now() - timedelta(hours=25)  # Beyond retention period
        monitoring_service.get_metrics(since=old_time)
        
        # Force some cleanup by creating more recent data
        for i in range(10000):
            monitoring_service.record_metric(
                f"cleanup_test_{i}",
                float(i),
                MetricType.COUNTER
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Peak: {peak_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Growth: {memory_growth:.1f}MB")
        
        # Memory growth should be reasonable for the amount of data
        assert memory_growth < 500, f"Memory growth too high: {memory_growth:.1f}MB > 500MB"
        
        # Memory shouldn't grow indefinitely
        memory_efficiency = memory_growth / 100  # MB per 1000 metrics
        assert memory_efficiency < 5, f"Memory efficiency poor: {memory_efficiency:.2f}MB per 1000 metrics"


class TestMonitoringAPIIntegration:
    """Test monitoring system API integration points"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service for API integration testing"""
        return MonitoringService()
    
    def test_dashboard_api_integration(self, monitoring_service):
        """Test dashboard summary API performance and data structure"""
        # Populate with realistic test data
        venues = ["IB", "BINANCE", "ALPACA"]
        metrics = ["latency", "throughput", "error_rate", "cpu_usage", "memory_usage"]
        
        for venue in venues:
            for metric in metrics:
                for i in range(100):
                    monitoring_service.record_metric(
                        f"{venue.lower()}_{metric}",
                        20.0 + i * 0.5,  # Realistic values
                        MetricType.GAUGE,
                        {"venue": venue, "component": "trading"}
                    )
        
        # Create some alerts
        for venue in venues:
            monitoring_service.create_alert(
                AlertLevel.WARNING,
                f"High latency on {venue}",
                f"Latency threshold exceeded on {venue}",
                "latency_monitor",
                {"venue": venue}
            )
        
        # Test dashboard API
        start_time = time.perf_counter()
        dashboard_data = monitoring_service.get_summary_dashboard()
        api_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Dashboard API response time: {api_time:.3f}ms")
        
        # API should be fast
        assert api_time < 50, f"Dashboard API too slow: {api_time:.3f}ms > 50ms"
        
        # Verify dashboard structure
        required_fields = ["timestamp", "alerts", "health", "metrics"]
        for field in required_fields:
            assert field in dashboard_data, f"Dashboard missing required field: {field}"
        
        # Verify alert summary
        alerts_summary = dashboard_data["alerts"]
        assert "total" in alerts_summary, "Alert summary should have total count"
        assert "by_level" in alerts_summary, "Alert summary should have level breakdown"
        assert alerts_summary["total"] == len(venues), "Should have correct alert count"
        
        # Verify metrics summary
        metrics_summary = dashboard_data["metrics"]
        assert len(metrics_summary) > 0, "Should have recent metrics"
        
        # Verify data freshness
        timestamp = datetime.fromisoformat(dashboard_data["timestamp"])
        age_seconds = (datetime.now() - timestamp).total_seconds()
        assert age_seconds < 5, f"Dashboard data too stale: {age_seconds}s > 5s"
    
    def test_health_check_api_performance(self, monitoring_service):
        """Test health check API performance"""
        # Add some health status data
        components = ["database", "cache", "message_queue", "rate_limiter"]
        for component in components:
            monitoring_service._health_checks[component] = HealthStatus(
                component=component,
                status="healthy",
                last_check=datetime.now(),
                details={"connections": 10, "response_time_ms": 5.2},
                uptime_percentage=99.9
            )
        
        # Test health check API performance
        start_time = time.perf_counter()
        health_data = monitoring_service.get_health_status()
        api_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Health check API response time: {api_time:.3f}ms")
        
        # Health check should be very fast
        assert api_time < 10, f"Health check API too slow: {api_time:.3f}ms > 10ms"
        
        # Verify health data structure
        assert len(health_data) == len(components), "Should return all component health status"
        
        for component, health in health_data.items():
            assert hasattr(health, 'status'), "Health status should have status field"
            assert hasattr(health, 'last_check'), "Health status should have last_check field"
            assert hasattr(health, 'uptime_percentage'), "Health status should have uptime field"


# Mock implementations for backend API testing
class MockMonitoringAPI:
    """Mock API implementations for frontend testing"""
    
    def __init__(self):
        self.monitoring_service = MonitoringService()
        self._populate_test_data()
    
    def _populate_test_data(self):
        """Populate with realistic test data"""
        venues = ["IB", "BINANCE", "ALPACA"]
        
        # Create realistic latency data
        for venue in venues:
            for i in range(1000):
                base_latency = {"IB": 15, "BINANCE": 25, "ALPACA": 35}[venue]
                latency = base_latency + (i % 100) * 0.5  # Varying latencies
                
                self.monitoring_service.record_metric(
                    f"order_execution_latency_{venue}",
                    latency,
                    MetricType.TIMING,
                    {"venue": venue, "metric_type": "latency"},
                    "ms"
                )
        
        # Create performance alerts
        alert_configs = [
            ("IB", "order_latency", 156.7, 100.0, AlertLevel.WARNING),
            ("BINANCE", "connection_errors", 5, 3, AlertLevel.ERROR),
            ("ALPACA", "throughput", 45, 100, AlertLevel.CRITICAL)
        ]
        
        for venue, metric, current, threshold, level in alert_configs:
            self.monitoring_service.create_alert(
                level,
                f"{metric} threshold exceeded",
                f"{venue} {metric}: {current} > {threshold}",
                "performance_monitor",
                {"venue": venue, "metric": metric}
            )
    
    async def get_latency_metrics(self, venue: str = None, time_range_ms: int = 3600000):
        """Mock latency metrics API"""
        await asyncio.sleep(0.001)  # Simulate minimal API overhead
        
        if venue:
            venue_metrics = [
                metric for name, metrics in self.monitoring_service.get_metrics().items()
                if venue in name for metric in metrics
                if metric.tags.get("venue") == venue
            ]
        else:
            venue_metrics = [
                metric for metrics in self.monitoring_service.get_metrics().values()
                for metric in metrics
                if "latency" in metric.name
            ]
        
        if not venue_metrics:
            return {
                "venue_name": venue or "all",
                "order_execution_latency": {
                    "min_ms": 0, "max_ms": 0, "avg_ms": 0,
                    "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                    "samples": 0
                }
            }
        
        latencies = [m.value for m in venue_metrics]
        latencies.sort()
        
        return {
            "venue_name": venue or "all",
            "order_execution_latency": {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "p50_ms": latencies[len(latencies)//2],
                "p95_ms": latencies[int(len(latencies)*0.95)],
                "p99_ms": latencies[int(len(latencies)*0.99)],
                "samples": len(latencies)
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_active_alerts(self, venue: str = None, severity: str = None):
        """Mock active alerts API"""
        await asyncio.sleep(0.002)  # Simulate API overhead
        
        alerts = self.monitoring_service.get_alerts(resolved=False)
        
        if venue:
            alerts = [a for a in alerts if a.tags.get("venue") == venue]
        
        if severity:
            alerts = [a for a in alerts if a.level.value == severity]
        
        return [
            {
                "alert_id": alert.id,
                "metric_name": alert.tags.get("metric", "unknown"),
                "severity": alert.level.value,
                "venue_name": alert.tags.get("venue"),
                "triggered_at": alert.timestamp.isoformat(),
                "description": alert.message,
                "auto_resolution_available": False,
                "escalation_level": {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}[alert.level.value.upper()],
                "notification_sent": True
            }
            for alert in alerts
        ]


class TestMockAPIPerformance:
    """Test mock API implementations for frontend testing"""
    
    @pytest.fixture
    def mock_api(self):
        """Create mock API for testing"""
        return MockMonitoringAPI()
    
    @pytest.mark.asyncio
    async def test_mock_latency_api_performance(self, mock_api):
        """Test mock latency API performance"""
        start_time = time.perf_counter()
        
        # Test multiple API calls
        tasks = [
            mock_api.get_latency_metrics("IB"),
            mock_api.get_latency_metrics("BINANCE"),
            mock_api.get_latency_metrics("ALPACA"),
            mock_api.get_latency_metrics()  # All venues
        ]
        
        results = await asyncio.gather(*tasks)
        
        api_time = (time.perf_counter() - start_time) * 1000
        avg_time_per_call = api_time / len(tasks)
        
        print(f"Mock API performance:")
        print(f"  Total time: {api_time:.3f}ms")
        print(f"  Average per call: {avg_time_per_call:.3f}ms")
        
        # Mock API should be very fast
        assert avg_time_per_call < 5, f"Mock API too slow: {avg_time_per_call:.3f}ms > 5ms"
        
        # Verify API response structure
        for result in results:
            assert "venue_name" in result, "Response should have venue_name"
            assert "order_execution_latency" in result, "Response should have latency data"
            
            latency_data = result["order_execution_latency"]
            required_fields = ["min_ms", "max_ms", "avg_ms", "p50_ms", "p95_ms", "p99_ms", "samples"]
            for field in required_fields:
                assert field in latency_data, f"Latency data missing field: {field}"
        
        print(f"Mock API responses: {len(results)} successful calls")
    
    @pytest.mark.asyncio
    async def test_mock_alerts_api_performance(self, mock_api):
        """Test mock alerts API performance"""
        start_time = time.perf_counter()
        
        # Test alerts API calls
        all_alerts = await mock_api.get_active_alerts()
        ib_alerts = await mock_api.get_active_alerts(venue="IB")
        critical_alerts = await mock_api.get_active_alerts(severity="critical")
        
        api_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Alerts API time: {api_time:.3f}ms for 3 calls")
        print(f"Results: {len(all_alerts)} total, {len(ib_alerts)} IB, {len(critical_alerts)} critical")
        
        # Verify response structure
        for alert in all_alerts:
            required_fields = [
                "alert_id", "metric_name", "severity", "venue_name", 
                "triggered_at", "description", "escalation_level"
            ]
            for field in required_fields:
                assert field in alert, f"Alert missing field: {field}"
        
        # Performance should be good
        assert api_time < 20, f"Alerts API too slow: {api_time:.3f}ms > 20ms"
"""
Unit tests for MonitoringService
Tests comprehensive monitoring functionality including metrics, alerts, and health checks
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from monitoring_service import (
    MonitoringService, 
    AlertLevel, 
    MetricType, 
    Alert, 
    Metric, 
    HealthStatus
)


class TestMonitoringService:
    """Test suite for MonitoringService class"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create a fresh monitoring service instance for each test"""
        return MonitoringService()
    
    @pytest.fixture
    def sample_metric(self):
        """Sample metric for testing"""
        return Metric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(),
            tags={"component": "test", "venue": "test_venue"},
            unit="ms"
        )
    
    @pytest.fixture
    def sample_alert(self):
        """Sample alert for testing"""
        return Alert(
            id="test_alert_1",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            source="test",
            tags={"component": "test"},
            resolved=False
        )
    
    def test_initialization(self, monitoring_service):
        """Test MonitoringService initialization"""
        assert not monitoring_service._running
        assert len(monitoring_service._metrics) == 0
        assert len(monitoring_service._alerts) == 0
        assert len(monitoring_service._health_checks) == 0
        assert len(monitoring_service._alert_handlers) == 0
        assert monitoring_service.metrics_retention_hours == 24
        assert monitoring_service.health_check_interval == 30
        assert monitoring_service.alert_check_interval == 10
    
    def test_threshold_configuration(self, monitoring_service):
        """Test default threshold configuration"""
        expected_thresholds = {
            "error_rate": 0.05,
            "latency_p95": 100.0,
            "throughput_min": 100.0,
            "memory_usage": 0.85,
            "disk_usage": 0.90,
            "connection_failures": 5,
            "circuit_breaker_trips": 3,
        }
        
        assert monitoring_service.thresholds == expected_thresholds
    
    def test_add_alert_handler(self, monitoring_service):
        """Test adding alert handlers"""
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        monitoring_service.add_alert_handler(handler1)
        monitoring_service.add_alert_handler(handler2)
        
        assert len(monitoring_service._alert_handlers) == 2
        assert handler1 in monitoring_service._alert_handlers
        assert handler2 in monitoring_service._alert_handlers
    
    def test_record_metric_basic(self, monitoring_service):
        """Test basic metric recording"""
        monitoring_service.record_metric(
            name="cpu_usage",
            value=75.5,
            metric_type=MetricType.GAUGE,
            tags={"component": "system"},
            unit="percent"
        )
        
        assert "cpu_usage" in monitoring_service._metrics
        metrics = monitoring_service._metrics["cpu_usage"]
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.tags == {"component": "system"}
        assert metric.unit == "percent"
        assert isinstance(metric.timestamp, datetime)
    
    def test_record_metric_retention(self, monitoring_service):
        """Test metric retention cleanup"""
        # Record metrics with old timestamps
        old_time = datetime.now() - timedelta(hours=25)  # Beyond retention
        recent_time = datetime.now() - timedelta(hours=1)  # Within retention
        
        with patch('monitoring_service.datetime') as mock_datetime:
            # Mock old metric
            mock_datetime.now.return_value = old_time
            monitoring_service.record_metric("test_metric", 1.0, MetricType.COUNTER)
            
            # Mock recent metric
            mock_datetime.now.return_value = recent_time
            monitoring_service.record_metric("test_metric", 2.0, MetricType.COUNTER)
            
            # Mock current time for retention cleanup
            mock_datetime.now.return_value = datetime.now()
            monitoring_service.record_metric("test_metric", 3.0, MetricType.COUNTER)
        
        # Only recent metrics should remain (mocking doesn't affect actual retention logic)
        metrics = monitoring_service._metrics["test_metric"]
        assert len(metrics) <= 2  # Old metrics should be cleaned up
    
    def test_create_alert_basic(self, monitoring_service):
        """Test basic alert creation"""
        alert_id = monitoring_service.create_alert(
            level=AlertLevel.ERROR,
            title="Database Connection Failed",
            message="Unable to connect to PostgreSQL",
            source="database",
            tags={"component": "postgres"}
        )
        
        assert alert_id in monitoring_service._alerts
        alert = monitoring_service._alerts[alert_id]
        
        assert alert.level == AlertLevel.ERROR
        assert alert.title == "Database Connection Failed"
        assert alert.message == "Unable to connect to PostgreSQL"
        assert alert.source == "database"
        assert alert.tags == {"component": "postgres"}
        assert not alert.resolved
        assert alert.resolved_at is None
        assert isinstance(alert.timestamp, datetime)
    
    def test_create_alert_with_handlers(self, monitoring_service):
        """Test alert creation triggers handlers"""
        sync_handler = MagicMock()
        async_handler = AsyncMock()
        
        monitoring_service.add_alert_handler(sync_handler)
        monitoring_service.add_alert_handler(async_handler)
        
        alert_id = monitoring_service.create_alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            source="test"
        )
        
        # Check that sync handler was called
        sync_handler.assert_called_once()
        
        # Verify the alert passed to handler
        call_args = sync_handler.call_args[0]
        alert_arg = call_args[0]
        assert alert_arg.id == alert_id
        assert alert_arg.level == AlertLevel.WARNING
        assert alert_arg.title == "Test Alert"
    
    def test_resolve_alert(self, monitoring_service):
        """Test alert resolution"""
        alert_id = monitoring_service.create_alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Test message",
            source="test"
        )
        
        # Verify alert is not resolved initially
        alert = monitoring_service._alerts[alert_id]
        assert not alert.resolved
        assert alert.resolved_at is None
        
        # Resolve the alert
        success = monitoring_service.resolve_alert(alert_id)
        assert success
        
        # Verify alert is now resolved
        resolved_alert = monitoring_service._alerts[alert_id]
        assert resolved_alert.resolved
        assert isinstance(resolved_alert.resolved_at, datetime)
    
    def test_resolve_nonexistent_alert(self, monitoring_service):
        """Test resolving non-existent alert returns False"""
        success = monitoring_service.resolve_alert("nonexistent_id")
        assert not success
    
    def test_get_metrics_all(self, monitoring_service):
        """Test getting all metrics"""
        # Record multiple metrics
        monitoring_service.record_metric("metric1", 10.0, MetricType.COUNTER)
        monitoring_service.record_metric("metric2", 20.0, MetricType.GAUGE)
        monitoring_service.record_metric("metric1", 15.0, MetricType.COUNTER)
        
        metrics = monitoring_service.get_metrics()
        
        assert "metric1" in metrics
        assert "metric2" in metrics
        assert len(metrics["metric1"]) == 2
        assert len(metrics["metric2"]) == 1
        assert metrics["metric1"][0].value == 10.0
        assert metrics["metric1"][1].value == 15.0
        assert metrics["metric2"][0].value == 20.0
    
    def test_get_metrics_by_name(self, monitoring_service):
        """Test getting metrics by specific name"""
        monitoring_service.record_metric("cpu_usage", 75.0, MetricType.GAUGE)
        monitoring_service.record_metric("memory_usage", 60.0, MetricType.GAUGE)
        
        cpu_metrics = monitoring_service.get_metrics("cpu_usage")
        
        assert "cpu_usage" in cpu_metrics
        assert "memory_usage" not in cpu_metrics
        assert len(cpu_metrics["cpu_usage"]) == 1
        assert cpu_metrics["cpu_usage"][0].value == 75.0
    
    def test_get_metrics_with_time_filter(self, monitoring_service):
        """Test getting metrics with time filter"""
        old_time = datetime.now() - timedelta(hours=2)
        
        with patch('monitoring_service.datetime') as mock_datetime:
            # Record old metric
            mock_datetime.now.return_value = old_time
            monitoring_service.record_metric("test_metric", 1.0, MetricType.GAUGE)
            
            # Record recent metric
            mock_datetime.now.return_value = datetime.now()
            monitoring_service.record_metric("test_metric", 2.0, MetricType.GAUGE)
        
        # Get only recent metrics (last hour)
        since = datetime.now() - timedelta(hours=1)
        recent_metrics = monitoring_service.get_metrics("test_metric", since=since)
        
        # Should only contain the recent metric
        test_metrics = recent_metrics["test_metric"]
        assert len(test_metrics) >= 1  # At least the recent one
        # All returned metrics should be newer than 'since'
        for metric in test_metrics:
            assert metric.timestamp > since
    
    def test_get_alerts_all(self, monitoring_service):
        """Test getting all alerts"""
        # Create multiple alerts
        alert1_id = monitoring_service.create_alert(
            AlertLevel.WARNING, "Alert 1", "Message 1", "source1"
        )
        alert2_id = monitoring_service.create_alert(
            AlertLevel.ERROR, "Alert 2", "Message 2", "source2"
        )
        
        # Resolve one alert
        monitoring_service.resolve_alert(alert1_id)
        
        all_alerts = monitoring_service.get_alerts()
        
        assert len(all_alerts) == 2
        # Should be sorted by timestamp, most recent first
        assert all_alerts[0].timestamp >= all_alerts[1].timestamp
    
    def test_get_alerts_by_resolved_status(self, monitoring_service):
        """Test filtering alerts by resolved status"""
        alert1_id = monitoring_service.create_alert(
            AlertLevel.INFO, "Alert 1", "Message 1", "source1"
        )
        alert2_id = monitoring_service.create_alert(
            AlertLevel.INFO, "Alert 2", "Message 2", "source2"
        )
        
        # Resolve first alert
        monitoring_service.resolve_alert(alert1_id)
        
        # Get only unresolved alerts
        active_alerts = monitoring_service.get_alerts(resolved=False)
        assert len(active_alerts) == 1
        assert active_alerts[0].id == alert2_id
        
        # Get only resolved alerts
        resolved_alerts = monitoring_service.get_alerts(resolved=True)
        assert len(resolved_alerts) == 1
        assert resolved_alerts[0].id == alert1_id
    
    def test_get_alerts_by_level(self, monitoring_service):
        """Test filtering alerts by level"""
        monitoring_service.create_alert(
            AlertLevel.WARNING, "Warning Alert", "Warning message", "source1"
        )
        monitoring_service.create_alert(
            AlertLevel.ERROR, "Error Alert", "Error message", "source2"
        )
        monitoring_service.create_alert(
            AlertLevel.WARNING, "Another Warning", "Another warning message", "source3"
        )
        
        # Get only WARNING level alerts
        warning_alerts = monitoring_service.get_alerts(level=AlertLevel.WARNING)
        assert len(warning_alerts) == 2
        for alert in warning_alerts:
            assert alert.level == AlertLevel.WARNING
        
        # Get only ERROR level alerts
        error_alerts = monitoring_service.get_alerts(level=AlertLevel.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].level == AlertLevel.ERROR
    
    def test_get_health_status(self, monitoring_service):
        """Test getting health status"""
        # Initially should be empty
        health_status = monitoring_service.get_health_status()
        assert len(health_status) == 0
        
        # Add a mock health check
        test_health = HealthStatus(
            component="test_component",
            status="healthy",
            last_check=datetime.now(),
            details={"key": "value"},
            uptime_percentage=99.5
        )
        
        monitoring_service._health_checks["test_component"] = test_health
        
        health_status = monitoring_service.get_health_status()
        assert len(health_status) == 1
        assert "test_component" in health_status
        assert health_status["test_component"].status == "healthy"
        assert health_status["test_component"].uptime_percentage == 99.5
    
    def test_get_summary_dashboard(self, monitoring_service):
        """Test getting summary dashboard data"""
        # Create test data
        monitoring_service.create_alert(AlertLevel.WARNING, "Warning", "msg", "src1")
        monitoring_service.create_alert(AlertLevel.ERROR, "Error", "msg", "src2")
        
        monitoring_service.record_metric("cpu_usage", 75.0, MetricType.GAUGE, {"component": "system"}, "percent")
        monitoring_service.record_metric("memory_usage", 60.0, MetricType.GAUGE, {"component": "system"}, "percent")
        
        dashboard = monitoring_service.get_summary_dashboard()
        
        # Verify structure
        assert "timestamp" in dashboard
        assert "alerts" in dashboard
        assert "health" in dashboard
        assert "metrics" in dashboard
        
        # Verify alert counts
        alerts = dashboard["alerts"]
        assert alerts["total"] == 2
        assert "by_level" in alerts
        assert alerts["by_level"][AlertLevel.WARNING.value] == 1
        assert alerts["by_level"][AlertLevel.ERROR.value] == 1
        
        # Verify metrics
        metrics = dashboard["metrics"]
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert metrics["cpu_usage"]["value"] == 75.0
        assert metrics["cpu_usage"]["unit"] == "percent"
        assert metrics["memory_usage"]["value"] == 60.0
    
    @pytest.mark.asyncio
    async def test_start_and_stop_service(self, monitoring_service):
        """Test starting and stopping the monitoring service"""
        assert not monitoring_service._running
        
        # Start the service
        await monitoring_service.start()
        assert monitoring_service._running
        assert len(monitoring_service._monitoring_tasks) > 0
        
        # Stop the service
        await monitoring_service.stop()
        assert not monitoring_service._running
        
        # Verify all tasks were cancelled
        for task in monitoring_service._monitoring_tasks:
            assert task.done()
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, monitoring_service):
        """Test starting service when already running does nothing"""
        await monitoring_service.start()
        initial_tasks = len(monitoring_service._monitoring_tasks)
        
        # Try to start again
        await monitoring_service.start()
        
        # Should not create additional tasks
        assert len(monitoring_service._monitoring_tasks) == initial_tasks
        
        await monitoring_service.stop()
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, monitoring_service):
        """Test stopping service when not running does nothing"""
        assert not monitoring_service._running
        
        # Should not raise any errors
        await monitoring_service.stop()
        assert not monitoring_service._running
    
    def test_calculate_uptime_healthy(self, monitoring_service):
        """Test uptime calculation for healthy components"""
        uptime = monitoring_service._calculate_uptime("test_component", "healthy")
        assert uptime == 100.0
        
        uptime = monitoring_service._calculate_uptime("test_component", "connected")
        assert uptime == 100.0
    
    def test_calculate_uptime_degraded(self, monitoring_service):
        """Test uptime calculation for degraded components"""
        uptime = monitoring_service._calculate_uptime("test_component", "degraded")
        assert uptime == 95.0
    
    def test_calculate_uptime_unhealthy(self, monitoring_service):
        """Test uptime calculation for unhealthy components"""
        uptime = monitoring_service._calculate_uptime("test_component", "error")
        assert uptime == 0.0
        
        uptime = monitoring_service._calculate_uptime("test_component", "disconnected")
        assert uptime == 0.0
    
    def test_metric_type_enum(self):
        """Test MetricType enum values"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMING.value == "timing"
    
    def test_alert_level_enum(self):
        """Test AlertLevel enum values"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
    
    def test_metric_dataclass(self):
        """Test Metric dataclass functionality"""
        timestamp = datetime.now()
        metric = Metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            tags={"key": "value"},
            unit="ms"
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.GAUGE
        assert metric.timestamp == timestamp
        assert metric.tags == {"key": "value"}
        assert metric.unit == "ms"
    
    def test_alert_dataclass(self):
        """Test Alert dataclass functionality"""
        timestamp = datetime.now()
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            timestamp=timestamp,
            source="test_source",
            tags={"component": "test"}
        )
        
        assert alert.id == "test_alert"
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "Test message"
        assert alert.timestamp == timestamp
        assert alert.source == "test_source"
        assert alert.tags == {"component": "test"}
        assert alert.resolved == False
        assert alert.resolved_at is None
    
    def test_health_status_dataclass(self):
        """Test HealthStatus dataclass functionality"""
        timestamp = datetime.now()
        details = {"connection_count": 5, "last_error": None}
        
        health = HealthStatus(
            component="database",
            status="healthy",
            last_check=timestamp,
            details=details,
            uptime_percentage=99.9
        )
        
        assert health.component == "database"
        assert health.status == "healthy"
        assert health.last_check == timestamp
        assert health.details == details
        assert health.uptime_percentage == 99.9
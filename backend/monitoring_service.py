"""
Monitoring and Alerting Service
Provides comprehensive monitoring for market data infrastructure with real-time alerting.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

from enums import Venue
from rate_limiter import rate_limiter
from redis_cache import redis_cache
from historical_data_service import historical_data_service
from data_normalizer import data_normalizer


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    tags: Dict[str, str]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str]
    unit: Optional[str] = None


@dataclass
class HealthStatus:
    """System health status"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    details: Dict[str, Any]
    uptime_percentage: float


class MonitoringService:
    """
    Comprehensive monitoring service for market data infrastructure
    with real-time metrics collection, alerting, and health monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._metrics: Dict[str, List[Metric]] = {}
        self._alerts: Dict[str, Alert] = {}
        self._health_checks: Dict[str, HealthStatus] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._running = False
        self._monitoring_tasks: List[asyncio.Task] = []
        
        # Monitoring configuration
        self.metrics_retention_hours = 24
        self.health_check_interval = 30  # seconds
        self.alert_check_interval = 10   # seconds
        
        # Alert thresholds
        self.thresholds = {
            "error_rate": 0.05,           # 5% error rate
            "latency_p95": 100.0,         # 100ms P95 latency
            "throughput_min": 100.0,      # 100 messages/sec minimum
            "memory_usage": 0.85,         # 85% memory usage
            "disk_usage": 0.90,           # 90% disk usage
            "connection_failures": 5,     # 5 connection failures
            "circuit_breaker_trips": 3,   # 3 circuit breaker trips
        }
        
    async def start(self) -> None:
        """Start monitoring service"""
        if self._running:
            return
            
        self.logger.info("Starting monitoring service...")
        self._running = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._collect_metrics(), name="metrics_collector"),
            asyncio.create_task(self._health_monitor(), name="health_monitor"),
            asyncio.create_task(self._alert_processor(), name="alert_processor"),
            asyncio.create_task(self._cleanup_old_data(), name="cleanup_task"),
        ]
        
        self._monitoring_tasks.extend(tasks)
        self.logger.info("Monitoring service started")
        
    async def stop(self) -> None:
        """Stop monitoring service"""
        if not self._running:
            return
            
        self.logger.info("Stopping monitoring service...")
        self._running = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
                
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
        self.logger.info("Monitoring service stopped")
        
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler callback"""
        self._alert_handlers.append(handler)
        
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     tags: Dict[str, str] = None, unit: str = None) -> None:
        """Record a metric value"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        if name not in self._metrics:
            self._metrics[name] = []
            
        self._metrics[name].append(metric)
        
        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        self._metrics[name] = [
            m for m in self._metrics[name] 
            if m.timestamp > cutoff_time
        ]
        
    def create_alert(self, level: AlertLevel, title: str, message: str,
                    source: str, tags: Dict[str, str] = None) -> str:
        """Create a new alert"""
        alert_id = f"{source}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            tags=tags or {}
        )
        
        self._alerts[alert_id] = alert
        
        # Notify alert handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert))
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
                
        self.logger.warning(f"Alert created: {title} - {message}")
        return alert_id
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            self._alerts[alert_id].resolved_at = datetime.now()
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
        
    def get_metrics(self, name: str = None, since: datetime = None) -> Dict[str, List[Metric]]:
        """Get metrics data"""
        if name:
            metrics = self._metrics.get(name, [])
            if since:
                metrics = [m for m in metrics if m.timestamp > since]
            return {name: metrics}
        else:
            result = {}
            for metric_name, metric_list in self._metrics.items():
                if since:
                    metric_list = [m for m in metric_list if m.timestamp > since]
                result[metric_name] = metric_list
            return result
            
    def get_alerts(self, resolved: bool = None, level: AlertLevel = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = list(self._alerts.values())
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
            
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
            
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
        
    def get_health_status(self) -> Dict[str, HealthStatus]:
        """Get current health status of all components"""
        return self._health_checks.copy()
        
    async def _collect_metrics(self) -> None:
        """Collect system metrics periodically"""
        while self._running:
            try:
                await self._collect_rate_limiter_metrics()
                await self._collect_cache_metrics()
                await self._collect_database_metrics()
                await self._collect_data_quality_metrics()
                await self._collect_system_metrics()
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)  # Back off on error
                
    async def _collect_rate_limiter_metrics(self) -> None:
        """Collect rate limiter metrics"""
        try:
            health = await rate_limiter.health_check()
            
            self.record_metric(
                "rate_limiter.healthy_venues",
                health.get("healthy_venues", 0),
                MetricType.GAUGE,
                {"component": "rate_limiter"}
            )
            
            self.record_metric(
                "rate_limiter.open_circuits",
                health.get("open_circuits", 0),
                MetricType.GAUGE,
                {"component": "rate_limiter"}
            )
            
            # Venue-specific metrics
            all_metrics = rate_limiter.get_all_metrics()
            for venue_name, metrics in all_metrics.items():
                venue_tags = {"venue": venue_name, "component": "rate_limiter"}
                
                self.record_metric(
                    "rate_limiter.total_requests",
                    metrics.total_requests,
                    MetricType.COUNTER,
                    venue_tags
                )
                
                self.record_metric(
                    "rate_limiter.throttled_requests",
                    metrics.throttled_requests,
                    MetricType.COUNTER,
                    venue_tags
                )
                
                if metrics.total_requests > 0:
                    success_rate = metrics.allowed_requests / metrics.total_requests
                    self.record_metric(
                        "rate_limiter.success_rate",
                        success_rate,
                        MetricType.GAUGE,
                        venue_tags
                    )
                    
                    # Check for high error rate
                    if success_rate < (1 - self.thresholds["error_rate"]):
                        self.create_alert(
                            AlertLevel.WARNING,
                            f"High error rate for {venue_name}",
                            f"Success rate: {success_rate:.2%}",
                            "rate_limiter",
                            venue_tags
                        )
                        
        except Exception as e:
            self.logger.error(f"Error collecting rate limiter metrics: {e}")
            
    async def _collect_cache_metrics(self) -> None:
        """Collect Redis cache metrics"""
        try:
            health = await redis_cache.health_check()
            
            if health.get("status") == "connected":
                self.record_metric(
                    "cache.ping_ms",
                    health.get("ping_ms", 0),
                    MetricType.TIMING,
                    {"component": "cache"}
                )
                
                self.record_metric(
                    "cache.connected_clients",
                    health.get("connected_clients", 0),
                    MetricType.GAUGE,
                    {"component": "cache"}
                )
                
                # Check ping latency
                ping_ms = health.get("ping_ms", 0)
                if ping_ms > 10:  # > 10ms ping
                    self.create_alert(
                        AlertLevel.WARNING,
                        "High cache latency",
                        f"Redis ping: {ping_ms:.2f}ms",
                        "cache"
                    )
                    
            else:
                self.create_alert(
                    AlertLevel.ERROR,
                    "Cache connection failed",
                    health.get("error", "Unknown error"),
                    "cache"
                )
                
        except Exception as e:
            self.logger.error(f"Error collecting cache metrics: {e}")
            
    async def _collect_database_metrics(self) -> None:
        """Collect PostgreSQL database metrics"""
        try:
            health = await historical_data_service.health_check()
            
            if health.get("status") == "connected":
                table_counts = health.get("table_counts", {})
                
                for table, count in table_counts.items():
                    self.record_metric(
                        f"database.{table}_count",
                        count,
                        MetricType.GAUGE,
                        {"component": "database", "table": table}
                    )
                    
                pool_stats = health.get("pool_stats", {})
                if pool_stats:
                    self.record_metric(
                        "database.pool_size",
                        pool_stats.get("size", 0),
                        MetricType.GAUGE,
                        {"component": "database"}
                    )
                    
            else:
                self.create_alert(
                    AlertLevel.ERROR,
                    "Database connection failed",
                    health.get("error", "Unknown error"),
                    "database"
                )
                
        except Exception as e:
            self.logger.error(f"Error collecting database metrics: {e}")
            
    async def _collect_data_quality_metrics(self) -> None:
        """Collect data quality metrics"""
        try:
            quality_metrics = data_normalizer.get_quality_metrics()
            
            for venue_name, metrics in quality_metrics.items():
                venue_tags = {"venue": venue_name, "component": "data_quality"}
                
                self.record_metric(
                    "data_quality.total_messages",
                    metrics.total_messages,
                    MetricType.COUNTER,
                    venue_tags
                )
                
                self.record_metric(
                    "data_quality.validation_errors",
                    metrics.validation_errors,
                    MetricType.COUNTER,
                    venue_tags
                )
                
                if metrics.total_messages > 0:
                    error_rate = metrics.validation_errors / metrics.total_messages
                    self.record_metric(
                        "data_quality.error_rate",
                        error_rate,
                        MetricType.GAUGE,
                        venue_tags
                    )
                    
                    # Check for high error rate
                    if error_rate > self.thresholds["error_rate"]:
                        self.create_alert(
                            AlertLevel.WARNING,
                            f"High data quality error rate for {venue_name}",
                            f"Error rate: {error_rate:.2%}",
                            "data_quality",
                            venue_tags
                        )
                        
        except Exception as e:
            self.logger.error(f"Error collecting data quality metrics: {e}")
            
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(
                "system.cpu_percent",
                cpu_percent,
                MetricType.GAUGE,
                {"component": "system"},
                "percent"
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric(
                "system.memory_percent",
                memory.percent,
                MetricType.GAUGE,
                {"component": "system"},
                "percent"
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric(
                "system.disk_percent",
                disk_percent,
                MetricType.GAUGE,
                {"component": "system"},
                "percent"
            )
            
            # Check thresholds
            if memory.percent > self.thresholds["memory_usage"] * 100:
                self.create_alert(
                    AlertLevel.WARNING,
                    "High memory usage",
                    f"Memory usage: {memory.percent:.1f}%",
                    "system"
                )
                
            if disk_percent > self.thresholds["disk_usage"] * 100:
                self.create_alert(
                    AlertLevel.WARNING,
                    "High disk usage",
                    f"Disk usage: {disk_percent:.1f}%",
                    "system"
                )
                
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
    async def _health_monitor(self) -> None:
        """Monitor component health"""
        while self._running:
            try:
                await self._check_component_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)  # Back off on error
                
    async def _check_component_health(self) -> None:
        """Check health of all components"""
        components = [
            ("rate_limiter", rate_limiter.health_check),
            ("cache", redis_cache.health_check),
            ("database", historical_data_service.health_check),
        ]
        
        for component_name, health_check_func in components:
            try:
                health_data = await health_check_func()
                status = health_data.get("status", "unknown")
                
                self._health_checks[component_name] = HealthStatus(
                    component=component_name,
                    status=status,
                    last_check=datetime.now(),
                    details=health_data,
                    uptime_percentage=self._calculate_uptime(component_name, status)
                )
                
                # Create alerts for unhealthy components
                if status in ["error", "disconnected"]:
                    self.create_alert(
                        AlertLevel.CRITICAL,
                        f"{component_name} unhealthy",
                        f"Status: {status}",
                        "health_monitor",
                        {"component": component_name}
                    )
                    
            except Exception as e:
                self.logger.error(f"Error checking {component_name} health: {e}")
                
    def _calculate_uptime(self, component: str, current_status: str) -> float:
        """Calculate component uptime percentage"""
        # Simple implementation - in production this would use historical data
        if current_status in ["connected", "healthy"]:
            return 100.0
        elif current_status in ["degraded"]:
            return 95.0
        else:
            return 0.0
            
    async def _alert_processor(self) -> None:
        """Process and manage alerts"""
        while self._running:
            try:
                # Auto-resolve old alerts if component is healthy
                await self._auto_resolve_alerts()
                await asyncio.sleep(self.alert_check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(30)
                
    async def _auto_resolve_alerts(self) -> None:
        """Automatically resolve alerts when conditions are met"""
        for alert in self._alerts.values():
            if alert.resolved:
                continue
                
            # Check if alert should be auto-resolved
            component = alert.tags.get("component")
            if component and component in self._health_checks:
                health = self._health_checks[component]
                if health.status in ["healthy", "connected"]:
                    # Auto-resolve if component is healthy
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    self.logger.info(f"Auto-resolved alert: {alert.id}")
                    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts"""
        while self._running:
            try:
                # Clean up old alerts (keep for 7 days)
                cutoff_time = datetime.now() - timedelta(days=7)
                old_alerts = [
                    alert_id for alert_id, alert in self._alerts.items()
                    if alert.timestamp < cutoff_time and alert.resolved
                ]
                
                for alert_id in old_alerts:
                    del self._alerts[alert_id]
                    
                if old_alerts:
                    self.logger.info(f"Cleaned up {len(old_alerts)} old alerts")
                    
                # Clean up old metrics (done in record_metric)
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes
                
    def get_summary_dashboard(self) -> Dict[str, Any]:
        """Get summary dashboard data"""
        # Active alerts by level
        alerts = self.get_alerts(resolved=False)
        alert_counts = {level.value: 0 for level in AlertLevel}
        for alert in alerts:
            alert_counts[alert.level.value] += 1
            
        # Component health
        health_summary = {}
        for component, status in self._health_checks.items():
            health_summary[component] = {
                "status": status.status,
                "uptime": status.uptime_percentage,
                "last_check": status.last_check.isoformat()
            }
            
        # Recent metrics
        recent_metrics = {}
        since = datetime.now() - timedelta(minutes=5)
        metrics = self.get_metrics(since=since)
        
        for metric_name, metric_list in metrics.items():
            if metric_list:
                latest = metric_list[-1]
                recent_metrics[metric_name] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat()
                }
                
        return {
            "timestamp": datetime.now().isoformat(),
            "alerts": {
                "total": len(alerts),
                "by_level": alert_counts
            },
            "health": health_summary,
            "metrics": recent_metrics
        }


# Global monitoring service instance
monitoring_service = MonitoringService()
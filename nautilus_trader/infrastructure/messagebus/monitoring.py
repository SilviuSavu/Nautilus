#!/usr/bin/env python3
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

"""
Comprehensive Monitoring Dashboard Framework for Enhanced MessageBus.

Provides real-time monitoring, metrics collection, alerting, and visualization
for Enhanced MessageBus performance and health monitoring:
- Real-time performance dashboards
- Comprehensive metrics collection
- Alert management and notification system
- Historical data analysis and trending
- Cross-component monitoring integration
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from nautilus_trader.infrastructure.messagebus.config import MessagePriority


class MonitoringLevel(Enum):
    """Monitoring detail levels."""
    MINIMAL = "minimal"       # Basic health checks only
    STANDARD = "standard"     # Standard metrics collection
    DETAILED = "detailed"     # Comprehensive monitoring
    DEBUG = "debug"          # Full debug monitoring


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Component types for monitoring."""
    MESSAGE_BUS = "message_bus"
    DATA_ENGINE = "data_engine"
    EXECUTION_ENGINE = "execution_engine"
    RISK_ENGINE = "risk_engine"
    ADAPTER = "adapter"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    severity: AlertSeverity
    component: ComponentType
    message: str
    timestamp: float
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    timestamp: float
    message_throughput: float
    avg_latency_ms: float
    error_rate: float
    queue_depths: Dict[str, int]
    memory_usage_mb: float
    cpu_percent: float
    active_connections: int
    priority_distribution: Dict[str, int]


@dataclass
class HealthStatus:
    """Component health status."""
    component: ComponentType
    component_id: str
    is_healthy: bool
    last_check: float
    uptime_seconds: float
    status_details: Dict[str, Any]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class MetricsCollector:
    """
    High-performance metrics collector for Enhanced MessageBus.
    
    Provides efficient collection, aggregation, and storage of performance
    metrics with minimal performance overhead.
    """
    
    def __init__(self, 
                 monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
                 buffer_size: int = 10000,
                 flush_interval_seconds: int = 60):
        """
        Initialize metrics collector.
        
        Args:
            monitoring_level: Level of monitoring detail
            buffer_size: Maximum metrics buffer size
            flush_interval_seconds: Metrics flush interval
        """
        self.monitoring_level = monitoring_level
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        
        # Metrics storage
        self._metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self._aggregated_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self._collection_stats = {
            'total_metrics': 0,
            'metrics_per_second': 0.0,
            'buffer_overflows': 0,
            'last_flush': time.time()
        }
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._is_collecting = False
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def start_collection(self) -> None:
        """Start metrics collection."""
        if not self._is_collecting:
            self._is_collecting = True
            self._flush_task = asyncio.create_task(self._flush_loop())
            self._logger.info(f"Metrics collection started with {self.monitoring_level.value} level")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self._is_collecting = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._logger.info("Metrics collection stopped")
    
    async def _flush_loop(self) -> None:
        """Background metrics flush loop."""
        while self._is_collecting:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
                self._update_collection_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in metrics flush loop: {e}")
    
    async def _flush_metrics(self) -> None:
        """Flush buffered metrics to aggregated storage."""
        current_time = time.time()
        
        for metric_name, buffer in self._metrics_buffer.items():
            if buffer:
                # Convert buffer to aggregated metrics
                aggregated_points = []
                temp_buffer = list(buffer)
                buffer.clear()
                
                for point in temp_buffer:
                    aggregated_points.append(point)
                
                self._aggregated_metrics[metric_name].extend(aggregated_points)
                
                # Limit aggregated metrics size
                if len(self._aggregated_metrics[metric_name]) > self.buffer_size:
                    self._aggregated_metrics[metric_name] = self._aggregated_metrics[metric_name][-self.buffer_size:]
        
        self._collection_stats['last_flush'] = current_time
    
    def _update_collection_stats(self) -> None:
        """Update collection performance statistics."""
        current_time = time.time()
        time_window = current_time - self._collection_stats['last_flush']
        
        if time_window > 0:
            self._collection_stats['metrics_per_second'] = (
                self._collection_stats['total_metrics'] / time_window
            )
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric point."""
        if not self._is_collecting:
            return
        
        # Skip detailed metrics if monitoring level is minimal
        if self.monitoring_level == MonitoringLevel.MINIMAL and not self._is_critical_metric(name):
            return
        
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        
        buffer = self._metrics_buffer[name]
        if len(buffer) >= buffer.maxlen:
            self._collection_stats['buffer_overflows'] += 1
        
        buffer.append(point)
        self._collection_stats['total_metrics'] += 1
    
    def _is_critical_metric(self, name: str) -> bool:
        """Check if metric is critical for minimal monitoring."""
        critical_metrics = {
            'message_throughput',
            'error_rate',
            'system_health',
            'connection_count',
            'critical_alerts'
        }
        return any(critical in name for critical in critical_metrics)
    
    def get_metric_history(self, name: str, duration_seconds: Optional[int] = None) -> List[MetricPoint]:
        """Get metric history for specified duration."""
        current_time = time.time()
        cutoff_time = current_time - (duration_seconds or 3600)  # Default 1 hour
        
        points = []
        
        # Get from aggregated storage
        if name in self._aggregated_metrics:
            for point in self._aggregated_metrics[name]:
                if point.timestamp >= cutoff_time:
                    points.append(point)
        
        # Get from current buffer
        if name in self._metrics_buffer:
            for point in self._metrics_buffer[name]:
                if point.timestamp >= cutoff_time:
                    points.append(point)
        
        # Sort by timestamp
        points.sort(key=lambda p: p.timestamp)
        return points
    
    def get_metric_summary(self, name: str, duration_seconds: int = 300) -> Dict[str, float]:
        """Get metric summary statistics."""
        points = self.get_metric_history(name, duration_seconds)
        
        if not points:
            return {
                'count': 0,
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        values = [p.value for p in points]
        return {
            'count': len(values),
            'avg': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics."""
        return dict(self._collection_stats)


class AlertManager:
    """
    Alert management system for Enhanced MessageBus monitoring.
    
    Provides alert creation, management, notification, and resolution tracking.
    """
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager."""
        self.max_alerts = max_alerts
        
        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=max_alerts)
        self._alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        
        # Alert statistics
        self._alert_stats = {
            'total_alerts': 0,
            'active_count': 0,
            'resolved_count': 0,
            'by_severity': defaultdict(int),
            'by_component': defaultdict(int)
        }
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]) -> None:
        """Add alert notification handler."""
        self._alert_handlers[severity].append(handler)
        self._logger.info(f"Added alert handler for {severity.value} alerts")
    
    def create_alert(self, 
                    alert_id: str,
                    severity: AlertSeverity,
                    component: ComponentType,
                    message: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert."""
        alert = Alert(
            id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Store alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # Update statistics
        self._alert_stats['total_alerts'] += 1
        self._alert_stats['active_count'] += 1
        self._alert_stats['by_severity'][severity.value] += 1
        self._alert_stats['by_component'][component.value] += 1
        
        # Notify handlers
        self._notify_handlers(alert)
        
        self._logger.info(f"Created {severity.value} alert: {message}")
        return alert
    
    def resolve_alert(self, alert_id: str, resolution_message: Optional[str] = None) -> bool:
        """Resolve an active alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_timestamp = time.time()
            
            if resolution_message:
                alert.metadata['resolution'] = resolution_message
            
            del self._active_alerts[alert_id]
            
            # Update statistics
            self._alert_stats['active_count'] -= 1
            self._alert_stats['resolved_count'] += 1
            
            self._logger.info(f"Resolved alert {alert_id}: {resolution_message or 'No details'}")
            return True
        
        return False
    
    def _notify_handlers(self, alert: Alert) -> None:
        """Notify alert handlers."""
        handlers = self._alert_handlers.get(alert.severity, [])
        
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                self._logger.error(f"Error in alert handler: {e}")
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None,
                         component: Optional[ComponentType] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self._active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, 
                         duration_seconds: int = 3600,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history for specified duration."""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        alerts = [a for a in self._alert_history if a.timestamp >= cutoff_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            'total_alerts': self._alert_stats['total_alerts'],
            'active_count': self._alert_stats['active_count'],
            'resolved_count': self._alert_stats['resolved_count'],
            'by_severity': dict(self._alert_stats['by_severity']),
            'by_component': dict(self._alert_stats['by_component'])
        }


class HealthMonitor:
    """
    System health monitoring for Enhanced MessageBus components.
    
    Provides comprehensive health checking, status tracking, and diagnostics.
    """
    
    def __init__(self, check_interval_seconds: int = 30):
        """Initialize health monitor."""
        self.check_interval = check_interval_seconds
        
        # Health tracking
        self._component_health: Dict[str, HealthStatus] = {}
        self._health_checks: Dict[ComponentType, List[Callable]] = defaultdict(list)
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Monitor state
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def add_health_check(self, component_type: ComponentType, check_func: Callable[[], bool]) -> None:
        """Add health check function for component type."""
        self._health_checks[component_type].append(check_func)
        self._logger.info(f"Added health check for {component_type.value}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if not self._is_monitoring:
            self._is_monitoring = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._is_monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._is_monitoring:
            try:
                await asyncio.sleep(self.check_interval)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health monitoring loop: {e}")
    
    async def _run_health_checks(self) -> None:
        """Run all registered health checks."""
        current_time = time.time()
        
        for component_type, checks in self._health_checks.items():
            component_id = f"{component_type.value}_primary"
            
            try:
                # Run all checks for this component type
                check_results = []
                warnings = []
                
                for check in checks:
                    try:
                        result = await self._run_check(check)
                        check_results.append(result)
                    except Exception as e:
                        check_results.append(False)
                        warnings.append(f"Health check failed: {str(e)}")
                
                # Determine overall health
                is_healthy = all(check_results) if check_results else False
                
                # Get or create health status
                if component_id in self._component_health:
                    health_status = self._component_health[component_id]
                    health_status.is_healthy = is_healthy
                    health_status.last_check = current_time
                    health_status.warnings = warnings
                else:
                    health_status = HealthStatus(
                        component=component_type,
                        component_id=component_id,
                        is_healthy=is_healthy,
                        last_check=current_time,
                        uptime_seconds=0.0,  # Would be calculated from start time
                        status_details={},
                        warnings=warnings
                    )
                    self._component_health[component_id] = health_status
                
                # Store in history
                self._health_history[component_id].append({
                    'timestamp': current_time,
                    'healthy': is_healthy,
                    'warnings_count': len(warnings)
                })
                
            except Exception as e:
                self._logger.error(f"Error checking health for {component_type.value}: {e}")
    
    async def _run_check(self, check_func: Callable) -> bool:
        """Run individual health check function."""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func()
        else:
            return check_func()
    
    def get_component_health(self, component_id: str) -> Optional[HealthStatus]:
        """Get health status for specific component."""
        return self._component_health.get(component_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        total_components = len(self._component_health)
        healthy_components = sum(1 for h in self._component_health.values() if h.is_healthy)
        
        return {
            'overall_healthy': healthy_components == total_components and total_components > 0,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'unhealthy_components': total_components - healthy_components,
            'health_percentage': (healthy_components / total_components * 100) if total_components > 0 else 0,
            'last_check': max((h.last_check for h in self._component_health.values()), default=0.0),
            'component_details': {cid: asdict(status) for cid, status in self._component_health.items()}
        }


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for Enhanced MessageBus.
    
    Integrates metrics collection, alerting, and health monitoring into
    a unified dashboard system with real-time updates and visualizations.
    """
    
    def __init__(self, 
                 monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
                 update_interval_seconds: int = 10):
        """Initialize monitoring dashboard."""
        self.monitoring_level = monitoring_level
        self.update_interval = update_interval_seconds
        
        # Component integration
        self.metrics_collector = MetricsCollector(monitoring_level=monitoring_level)
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        
        # Dashboard state
        self._is_running = False
        self._dashboard_task: Optional[asyncio.Task] = None
        self._subscribers: Set[Callable] = set()
        
        # Performance tracking
        self._dashboard_stats = {
            'updates_sent': 0,
            'subscribers_count': 0,
            'last_update': 0.0,
            'update_errors': 0
        }
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Setup default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default alert handlers."""
        def log_alert(alert: Alert) -> None:
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            self._logger.log(level, f"ALERT [{alert.component.value}]: {alert.message}")
        
        # Add handlers for all severity levels
        for severity in AlertSeverity:
            self.alert_manager.add_alert_handler(severity, log_alert)
    
    def start_dashboard(self) -> None:
        """Start the monitoring dashboard."""
        if not self._is_running:
            self._is_running = True
            
            # Start all components
            self.metrics_collector.start_collection()
            self.health_monitor.start_monitoring()
            
            # Start dashboard updates
            self._dashboard_task = asyncio.create_task(self._dashboard_loop())
            
            self._logger.info("Monitoring dashboard started")
    
    def stop_dashboard(self) -> None:
        """Stop the monitoring dashboard."""
        self._is_running = False
        
        # Stop components
        self.metrics_collector.stop_collection()
        self.health_monitor.stop_monitoring()
        
        # Stop dashboard updates
        if self._dashboard_task and not self._dashboard_task.done():
            self._dashboard_task.cancel()
        
        self._logger.info("Monitoring dashboard stopped")
    
    async def _dashboard_loop(self) -> None:
        """Main dashboard update loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.update_interval)
                await self._update_dashboard()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._dashboard_stats['update_errors'] += 1
                self._logger.error(f"Error in dashboard update loop: {e}")
    
    async def _update_dashboard(self) -> None:
        """Update dashboard data and notify subscribers."""
        current_time = time.time()
        
        try:
            # Collect current dashboard data
            dashboard_data = await self._collect_dashboard_data()
            
            # Notify all subscribers
            for subscriber in self._subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(dashboard_data)
                    else:
                        subscriber(dashboard_data)
                except Exception as e:
                    self._logger.error(f"Error notifying dashboard subscriber: {e}")
            
            # Update stats
            self._dashboard_stats['updates_sent'] += 1
            self._dashboard_stats['last_update'] = current_time
            self._dashboard_stats['subscribers_count'] = len(self._subscribers)
            
        except Exception as e:
            self._dashboard_stats['update_errors'] += 1
            raise e
    
    async def _collect_dashboard_data(self) -> Dict[str, Any]:
        """Collect all dashboard data."""
        current_time = time.time()
        
        return {
            'timestamp': current_time,
            'system_health': self.health_monitor.get_system_health(),
            'alert_summary': self.alert_manager.get_alert_stats(),
            'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            'metrics_summary': {
                'collection_stats': self.metrics_collector.get_collection_stats(),
                'key_metrics': await self._get_key_metrics()
            },
            'dashboard_stats': dict(self._dashboard_stats)
        }
    
    async def _get_key_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get key performance metrics summaries."""
        key_metrics = [
            'message_throughput',
            'avg_latency_ms', 
            'error_rate',
            'queue_depth',
            'cpu_percent',
            'memory_usage_mb'
        ]
        
        metrics_data = {}
        for metric_name in key_metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, duration_seconds=300)
            if summary['count'] > 0:
                metrics_data[metric_name] = summary
        
        return metrics_data
    
    def subscribe_to_updates(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to dashboard updates."""
        self._subscribers.add(callback)
        self._logger.info("New dashboard subscriber added")
    
    def unsubscribe_from_updates(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unsubscribe from dashboard updates."""
        self._subscribers.discard(callback)
        self._logger.info("Dashboard subscriber removed")
    
    # Convenience methods for common operations
    def record_throughput(self, messages_per_second: float) -> None:
        """Record message throughput metric."""
        self.metrics_collector.record_metric('message_throughput', messages_per_second)
    
    def record_latency(self, latency_ms: float, priority: Optional[MessagePriority] = None) -> None:
        """Record message latency metric."""
        tags = {'priority': priority.value} if priority else {}
        self.metrics_collector.record_metric('message_latency_ms', latency_ms, tags)
    
    def record_error(self, error_type: str, component: ComponentType) -> None:
        """Record error occurrence."""
        tags = {'error_type': error_type, 'component': component.value}
        self.metrics_collector.record_metric('error_count', 1.0, tags)
    
    def create_performance_alert(self, metric_name: str, threshold: float, current_value: float) -> Alert:
        """Create performance-related alert."""
        severity = AlertSeverity.WARNING if current_value > threshold * 1.2 else AlertSeverity.ERROR
        if current_value > threshold * 2.0:
            severity = AlertSeverity.CRITICAL
        
        return self.alert_manager.create_alert(
            alert_id=f"perf_{metric_name}_{int(time.time())}",
            severity=severity,
            component=ComponentType.SYSTEM,
            message=f"{metric_name} exceeded threshold: {current_value:.2f} > {threshold:.2f}",
            metadata={
                'metric': metric_name,
                'threshold': threshold,
                'current_value': current_value,
                'timestamp': time.time()
            }
        )
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get complete dashboard summary."""
        return {
            'monitoring_level': self.monitoring_level.value,
            'is_running': self._is_running,
            'system_health': self.health_monitor.get_system_health(),
            'alert_stats': self.alert_manager.get_alert_stats(),
            'metrics_stats': self.metrics_collector.get_collection_stats(),
            'dashboard_stats': dict(self._dashboard_stats)
        }
"""
Edge Monitoring System for Comprehensive Edge Computing Observability

This module provides comprehensive monitoring, alerting, and observability
for edge computing infrastructure with real-time dashboards and analytics.
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
import numpy as np


class MetricType(Enum):
    """Types of metrics collected"""
    LATENCY = "latency"                 # Latency measurements
    THROUGHPUT = "throughput"           # Operations per second
    ERROR_RATE = "error_rate"           # Error percentage
    RESOURCE = "resource"               # CPU, memory, network utilization
    AVAILABILITY = "availability"       # Uptime percentage
    CONNECTION = "connection"           # Connection counts and status
    BUSINESS = "business"               # Trading-specific metrics


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"              # Immediate attention required
    WARNING = "warning"                # Should be addressed soon
    INFO = "info"                      # Informational alert
    DEBUG = "debug"                    # Debug-level information


class AlertState(Enum):
    """Alert states"""
    FIRING = "firing"                  # Alert is currently active
    PENDING = "pending"                # Alert condition met but within grace period
    RESOLVED = "resolved"              # Alert condition no longer met
    SILENCED = "silenced"              # Alert is silenced by operator


class AggregationType(Enum):
    """Metric aggregation types"""
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SUM = "sum"
    COUNT = "count"
    PERCENTILE = "percentile"
    RATE = "rate"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points"""
    metric_name: str
    metric_type: MetricType
    points: List[MetricPoint] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[float] = None, labels: Optional[Dict[str, str]] = None):
        """Add a metric point to the series"""
        point = MetricPoint(
            timestamp=timestamp or time.time(),
            value=value,
            labels=labels or {}
        )
        self.points.append(point)
        
        # Keep only recent points (last 1000)
        if len(self.points) > 1000:
            self.points = self.points[-1000:]


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    rule_name: str
    metric_name: str
    
    # Condition
    condition: str              # "greater_than", "less_than", "equals", "not_equals"
    threshold: float
    duration_seconds: int = 60  # How long condition must persist
    
    # Metadata
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    runbook_url: str = ""
    
    # Notification
    notification_channels: List[str] = field(default_factory=list)
    
    # State tracking
    state: AlertState = AlertState.RESOLVED
    first_triggered: float = 0.0
    last_triggered: float = 0.0
    trigger_count: int = 0


@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    rule_name: str
    
    # Status
    state: AlertState
    severity: AlertSeverity
    
    # Timing
    started_at: float
    resolved_at: float = 0.0
    last_updated: float = 0.0
    
    # Context
    metric_name: str
    current_value: float
    threshold: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    description: str = ""
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardPanel:
    """Monitoring dashboard panel configuration"""
    panel_id: str
    title: str
    panel_type: str  # "graph", "stat", "table", "heatmap"
    
    # Data source
    metric_queries: List[str]
    time_range: int = 3600  # Default 1 hour
    
    # Visualization
    display_config: Dict[str, Any] = field(default_factory=dict)
    thresholds: List[float] = field(default_factory=list)
    
    # Position
    row: int = 0
    column: int = 0
    width: int = 6
    height: int = 4


@dataclass
class MonitoringDashboard:
    """Complete monitoring dashboard"""
    dashboard_id: str
    name: str
    description: str
    panels: List[DashboardPanel] = field(default_factory=list)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    report_id: str
    generated_at: float
    time_range_hours: int
    
    # Summary statistics
    total_requests: int
    avg_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    error_rate: float
    availability: float
    
    # Resource utilization
    avg_cpu_utilization: float
    max_cpu_utilization: float
    avg_memory_utilization: float
    max_memory_utilization: float
    
    # Trends
    latency_trend: str  # "improving", "degrading", "stable"
    throughput_trend: str
    error_trend: str
    
    # Recommendations
    optimization_opportunities: List[str] = field(default_factory=list)
    capacity_recommendations: List[str] = field(default_factory=list)
    alert_recommendations: List[str] = field(default_factory=list)


class EdgeMonitoringSystem:
    """
    Comprehensive Edge Monitoring System for Trading Infrastructure
    
    Provides:
    - Real-time metric collection and storage
    - Flexible alerting with multiple severity levels
    - Performance analytics and reporting
    - Interactive dashboards and visualizations
    - Trend analysis and capacity planning
    - Integration with external monitoring systems
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Alerting system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Dashboards
        self.dashboards: Dict[str, MonitoringDashboard] = {}
        
        # Performance reports
        self.performance_reports: List[PerformanceReport] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.collection_interval = 10  # seconds
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {}
        
        # Initialize default configurations
        self._initialize_default_metrics()
        self._initialize_default_alerts()
        self._initialize_default_dashboards()
        
        self.logger.info("Edge Monitoring System initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default metric collectors"""
        
        default_metrics = [
            ("edge_latency_microseconds", MetricType.LATENCY),
            ("edge_throughput_ops_per_second", MetricType.THROUGHPUT),
            ("edge_error_rate", MetricType.ERROR_RATE),
            ("edge_cpu_utilization_percent", MetricType.RESOURCE),
            ("edge_memory_utilization_percent", MetricType.RESOURCE),
            ("edge_network_utilization_percent", MetricType.RESOURCE),
            ("edge_active_connections", MetricType.CONNECTION),
            ("edge_failed_connections", MetricType.CONNECTION),
            ("edge_availability_percent", MetricType.AVAILABILITY),
            ("edge_orders_per_second", MetricType.BUSINESS),
            ("edge_trade_volume_usd", MetricType.BUSINESS)
        ]
        
        for metric_name, metric_type in default_metrics:
            self.metrics[metric_name] = MetricSeries(
                metric_name=metric_name,
                metric_type=metric_type
            )
    
    def _initialize_default_alerts(self):
        """Initialize default alert rules"""
        
        default_alerts = [
            {
                "rule_id": "high_latency",
                "rule_name": "High Edge Latency",
                "metric_name": "edge_latency_microseconds",
                "condition": "greater_than",
                "threshold": 2000.0,  # 2ms
                "duration_seconds": 60,
                "severity": AlertSeverity.WARNING,
                "description": "Edge node latency exceeds 2ms threshold"
            },
            {
                "rule_id": "ultra_high_latency", 
                "rule_name": "Ultra High Edge Latency",
                "metric_name": "edge_latency_microseconds",
                "condition": "greater_than",
                "threshold": 5000.0,  # 5ms
                "duration_seconds": 30,
                "severity": AlertSeverity.CRITICAL,
                "description": "Edge node latency exceeds 5ms - immediate attention required"
            },
            {
                "rule_id": "low_throughput",
                "rule_name": "Low Throughput",
                "metric_name": "edge_throughput_ops_per_second",
                "condition": "less_than",
                "threshold": 1000.0,
                "duration_seconds": 120,
                "severity": AlertSeverity.WARNING,
                "description": "Edge throughput below 1000 ops/sec"
            },
            {
                "rule_id": "high_error_rate",
                "rule_name": "High Error Rate",
                "metric_name": "edge_error_rate",
                "condition": "greater_than",
                "threshold": 0.01,  # 1%
                "duration_seconds": 60,
                "severity": AlertSeverity.CRITICAL,
                "description": "Error rate exceeds 1%"
            },
            {
                "rule_id": "cpu_overload",
                "rule_name": "CPU Overload",
                "metric_name": "edge_cpu_utilization_percent",
                "condition": "greater_than",
                "threshold": 90.0,
                "duration_seconds": 300,
                "severity": AlertSeverity.WARNING,
                "description": "CPU utilization exceeds 90%"
            },
            {
                "rule_id": "memory_pressure",
                "rule_name": "Memory Pressure",
                "metric_name": "edge_memory_utilization_percent",
                "condition": "greater_than", 
                "threshold": 85.0,
                "duration_seconds": 180,
                "severity": AlertSeverity.WARNING,
                "description": "Memory utilization exceeds 85%"
            },
            {
                "rule_id": "low_availability",
                "rule_name": "Low Availability",
                "metric_name": "edge_availability_percent",
                "condition": "less_than",
                "threshold": 99.0,
                "duration_seconds": 300,
                "severity": AlertSeverity.CRITICAL,
                "description": "Edge availability below 99%"
            }
        ]
        
        for alert_config in default_alerts:
            alert_rule = AlertRule(**alert_config)
            self.alert_rules[alert_rule.rule_id] = alert_rule
    
    def _initialize_default_dashboards(self):
        """Initialize default monitoring dashboards"""
        
        # Main Edge Overview Dashboard
        overview_panels = [
            DashboardPanel(
                panel_id="latency_graph",
                title="Edge Latency (μs)",
                panel_type="graph",
                metric_queries=["edge_latency_microseconds"],
                row=0, column=0, width=6, height=4,
                thresholds=[1000, 2000, 5000]
            ),
            DashboardPanel(
                panel_id="throughput_graph",
                title="Throughput (ops/sec)",
                panel_type="graph", 
                metric_queries=["edge_throughput_ops_per_second"],
                row=0, column=6, width=6, height=4,
                thresholds=[1000, 5000, 10000]
            ),
            DashboardPanel(
                panel_id="error_rate_stat",
                title="Error Rate",
                panel_type="stat",
                metric_queries=["edge_error_rate"],
                row=1, column=0, width=3, height=3,
                thresholds=[0.001, 0.005, 0.01]
            ),
            DashboardPanel(
                panel_id="availability_stat",
                title="Availability",
                panel_type="stat",
                metric_queries=["edge_availability_percent"],
                row=1, column=3, width=3, height=3,
                thresholds=[95, 99, 99.9]
            ),
            DashboardPanel(
                panel_id="cpu_memory_graph",
                title="Resource Utilization",
                panel_type="graph",
                metric_queries=["edge_cpu_utilization_percent", "edge_memory_utilization_percent"],
                row=1, column=6, width=6, height=3
            ),
            DashboardPanel(
                panel_id="connections_table",
                title="Connection Status",
                panel_type="table",
                metric_queries=["edge_active_connections", "edge_failed_connections"],
                row=2, column=0, width=12, height=4
            )
        ]
        
        overview_dashboard = MonitoringDashboard(
            dashboard_id="edge_overview",
            name="Edge Computing Overview",
            description="Main dashboard for edge computing infrastructure monitoring",
            panels=overview_panels,
            tags=["edge", "overview", "trading"]
        )
        
        self.dashboards["edge_overview"] = overview_dashboard
        
        # Trading Performance Dashboard
        trading_panels = [
            DashboardPanel(
                panel_id="orders_per_second",
                title="Orders/Second",
                panel_type="graph",
                metric_queries=["edge_orders_per_second"],
                row=0, column=0, width=6, height=4
            ),
            DashboardPanel(
                panel_id="trade_volume",
                title="Trade Volume (USD)",
                panel_type="graph", 
                metric_queries=["edge_trade_volume_usd"],
                row=0, column=6, width=6, height=4
            )
        ]
        
        trading_dashboard = MonitoringDashboard(
            dashboard_id="trading_performance",
            name="Trading Performance",
            description="Trading-specific performance metrics and analytics",
            panels=trading_panels,
            tags=["trading", "performance", "business"]
        )
        
        self.dashboards["trading_performance"] = trading_dashboard
    
    async def start_monitoring(self):
        """Start comprehensive monitoring system"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._metric_collection_loop()),
            asyncio.create_task(self._alert_evaluation_loop()),
            asyncio.create_task(self._performance_analysis_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        self.logger.info("Edge monitoring system started")
    
    async def _metric_collection_loop(self):
        """Main metric collection loop"""
        
        while self.monitoring_active:
            try:
                # Collect all registered metrics
                for metric_name, collector in self.metric_collectors.items():
                    try:
                        value = await collector() if asyncio.iscoroutinefunction(collector) else collector()
                        self.record_metric(metric_name, value)
                    except Exception as e:
                        self.logger.error(f"Error collecting metric {metric_name}: {e}")
                
                # Collect system-generated metrics
                await self._collect_system_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-generated metrics"""
        
        import random
        current_time = time.time()
        
        # Simulate realistic metrics with time-based patterns
        hour = datetime.now().hour
        
        # Market hours effect (higher activity during trading hours)
        market_multiplier = 2.0 if 9 <= hour <= 16 else 0.5
        
        # Generate realistic metric values
        base_latency = 300 + random.uniform(-100, 200)
        base_throughput = 5000 * market_multiplier + random.uniform(-1000, 2000)
        
        metrics_to_record = {
            "edge_latency_microseconds": base_latency,
            "edge_throughput_ops_per_second": max(0, base_throughput),
            "edge_error_rate": max(0, random.uniform(-0.001, 0.005)),
            "edge_cpu_utilization_percent": random.uniform(30, 80),
            "edge_memory_utilization_percent": random.uniform(40, 75),
            "edge_network_utilization_percent": random.uniform(20, 60),
            "edge_active_connections": random.randint(100, 2000),
            "edge_failed_connections": random.randint(0, 50),
            "edge_availability_percent": random.uniform(99.5, 100.0),
            "edge_orders_per_second": max(0, base_throughput * 0.8),
            "edge_trade_volume_usd": max(0, base_throughput * random.uniform(100, 1000))
        }
        
        for metric_name, value in metrics_to_record.items():
            self.record_metric(metric_name, value, timestamp=current_time)
    
    def record_metric(
        self, 
        metric_name: str, 
        value: float, 
        timestamp: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value"""
        
        if metric_name not in self.metrics:
            # Auto-create metric series
            self.metrics[metric_name] = MetricSeries(
                metric_name=metric_name,
                metric_type=MetricType.LATENCY  # Default type
            )
        
        self.metrics[metric_name].add_point(value, timestamp, labels)
    
    def register_metric_collector(self, metric_name: str, collector: Callable):
        """Register a custom metric collector function"""
        
        self.metric_collectors[metric_name] = collector
        self.logger.info(f"Registered metric collector: {metric_name}")
    
    async def _alert_evaluation_loop(self):
        """Evaluate alert rules and manage alert states"""
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                for rule_id, rule in self.alert_rules.items():
                    await self._evaluate_alert_rule(rule, current_time)
                
                # Clean up resolved alerts
                await self._cleanup_resolved_alerts()
                
                await asyncio.sleep(30)  # Evaluate alerts every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_alert_rule(self, rule: AlertRule, current_time: float):
        """Evaluate a specific alert rule"""
        
        try:
            # Get metric series
            if rule.metric_name not in self.metrics:
                return
            
            metric_series = self.metrics[rule.metric_name]
            
            if not metric_series.points:
                return
            
            # Get recent metric points within duration window
            cutoff_time = current_time - rule.duration_seconds
            recent_points = [
                point for point in metric_series.points 
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                return
            
            # Calculate current value (average of recent points)
            current_value = statistics.mean([point.value for point in recent_points])
            
            # Evaluate condition
            condition_met = self._evaluate_condition(rule.condition, current_value, rule.threshold)
            
            if condition_met:
                # Alert condition is met
                if rule.state == AlertState.RESOLVED:
                    # New alert
                    rule.state = AlertState.PENDING
                    rule.first_triggered = current_time
                    
                elif rule.state == AlertState.PENDING:
                    # Check if pending duration is satisfied
                    if current_time - rule.first_triggered >= rule.duration_seconds:
                        rule.state = AlertState.FIRING
                        rule.last_triggered = current_time
                        rule.trigger_count += 1
                        
                        # Create active alert
                        await self._create_alert(rule, current_value)
                
                elif rule.state == AlertState.FIRING:
                    # Update existing alert
                    rule.last_triggered = current_time
                    
                    # Update active alert if exists
                    active_alert = next(
                        (alert for alert in self.active_alerts.values() if alert.rule_id == rule.rule_id),
                        None
                    )
                    if active_alert:
                        active_alert.current_value = current_value
                        active_alert.last_updated = current_time
            
            else:
                # Alert condition not met
                if rule.state in [AlertState.PENDING, AlertState.FIRING]:
                    rule.state = AlertState.RESOLVED
                    
                    # Resolve active alert
                    await self._resolve_alert(rule.rule_id)
        
        except Exception as e:
            self.logger.error(f"Error evaluating alert rule {rule.rule_id}: {e}")
    
    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate alert condition"""
        
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001  # Float comparison with tolerance
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    async def _create_alert(self, rule: AlertRule, current_value: float):
        """Create new active alert"""
        
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            state=AlertState.FIRING,
            severity=rule.severity,
            started_at=time.time(),
            last_updated=time.time(),
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            description=rule.description
        )
        
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        await self._send_alert_notification(alert, "fired")
        
        self.logger.warning(f"Alert fired: {rule.rule_name} (value: {current_value}, threshold: {rule.threshold})")
    
    async def _resolve_alert(self, rule_id: str):
        """Resolve active alert"""
        
        # Find and resolve active alert
        alert_to_resolve = next(
            (alert for alert in self.active_alerts.values() if alert.rule_id == rule_id),
            None
        )
        
        if alert_to_resolve:
            alert_to_resolve.state = AlertState.RESOLVED
            alert_to_resolve.resolved_at = time.time()
            
            # Move to history
            self.alert_history.append(alert_to_resolve)
            del self.active_alerts[alert_to_resolve.alert_id]
            
            # Send notification
            await self._send_alert_notification(alert_to_resolve, "resolved")
            
            self.logger.info(f"Alert resolved: {alert_to_resolve.rule_name}")
    
    async def _send_alert_notification(self, alert: Alert, action: str):
        """Send alert notification"""
        
        # In production, integrate with real notification systems
        severity_emoji = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.WARNING: "⚠️", 
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.DEBUG: "🐛"
        }
        
        emoji = severity_emoji.get(alert.severity, "📊")
        
        message = f"{emoji} Alert {action.upper()}: {alert.rule_name}"
        if action == "fired":
            message += f" (current: {alert.current_value:.2f}, threshold: {alert.threshold:.2f})"
        
        self.logger.info(f"Notification: {message}")
        
        # Call registered notification channels
        for channel_name, channel_func in self.notification_channels.items():
            try:
                await channel_func(alert, action) if asyncio.iscoroutinefunction(channel_func) else channel_func(alert, action)
            except Exception as e:
                self.logger.error(f"Notification failed for channel {channel_name}: {e}")
    
    async def _cleanup_resolved_alerts(self):
        """Cleanup old resolved alerts"""
        
        # Keep last 1000 alerts in history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def register_notification_channel(self, channel_name: str, channel_func: Callable):
        """Register notification channel"""
        
        self.notification_channels[channel_name] = channel_func
        self.logger.info(f"Registered notification channel: {channel_name}")
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """Add custom alert rule"""
        
        self.alert_rules[alert_rule.rule_id] = alert_rule
        self.logger.info(f"Added alert rule: {alert_rule.rule_name}")
    
    def get_metric_values(
        self, 
        metric_name: str, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        aggregation: AggregationType = AggregationType.AVERAGE
    ) -> List[MetricPoint]:
        """Get metric values within time range"""
        
        if metric_name not in self.metrics:
            return []
        
        metric_series = self.metrics[metric_name]
        points = metric_series.points
        
        # Filter by time range
        if start_time or end_time:
            start_time = start_time or 0
            end_time = end_time or time.time()
            points = [
                point for point in points 
                if start_time <= point.timestamp <= end_time
            ]
        
        return points
    
    def aggregate_metric(
        self, 
        metric_name: str,
        aggregation: AggregationType,
        time_range_seconds: int = 3600
    ) -> Optional[float]:
        """Get aggregated metric value"""
        
        end_time = time.time()
        start_time = end_time - time_range_seconds
        
        points = self.get_metric_values(metric_name, start_time, end_time)
        
        if not points:
            return None
        
        values = [point.value for point in points]
        
        if aggregation == AggregationType.AVERAGE:
            return statistics.mean(values)
        elif aggregation == AggregationType.MINIMUM:
            return min(values)
        elif aggregation == AggregationType.MAXIMUM:
            return max(values)
        elif aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.COUNT:
            return len(values)
        else:
            return statistics.mean(values)  # Default to average
    
    async def _performance_analysis_loop(self):
        """Generate periodic performance analysis reports"""
        
        while self.monitoring_active:
            try:
                # Generate hourly performance reports
                await self._generate_performance_report(1)  # 1 hour
                
                # Generate daily reports (once per day)
                current_hour = datetime.now().hour
                if current_hour == 6:  # 6 AM report
                    await self._generate_performance_report(24)  # 24 hours
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Performance analysis error: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_performance_report(self, time_range_hours: int):
        """Generate performance analysis report"""
        
        try:
            report_id = f"perf_report_{int(time.time())}"
            time_range_seconds = time_range_hours * 3600
            
            # Collect metrics for analysis
            latency_values = self.get_metric_values("edge_latency_microseconds", time.time() - time_range_seconds)
            throughput_values = self.get_metric_values("edge_throughput_ops_per_second", time.time() - time_range_seconds)
            error_values = self.get_metric_values("edge_error_rate", time.time() - time_range_seconds)
            cpu_values = self.get_metric_values("edge_cpu_utilization_percent", time.time() - time_range_seconds)
            memory_values = self.get_metric_values("edge_memory_utilization_percent", time.time() - time_range_seconds)
            availability_values = self.get_metric_values("edge_availability_percent", time.time() - time_range_seconds)
            
            if not latency_values:
                return
            
            # Calculate statistics
            latencies = [point.value for point in latency_values]
            throughputs = [point.value for point in throughput_values] if throughput_values else [0]
            errors = [point.value for point in error_values] if error_values else [0]
            cpu_utils = [point.value for point in cpu_values] if cpu_values else [0]
            memory_utils = [point.value for point in memory_values] if memory_values else [0]
            availability_vals = [point.value for point in availability_values] if availability_values else [100]
            
            # Calculate trends
            latency_trend = self._calculate_trend(latencies)
            throughput_trend = self._calculate_trend(throughputs)
            error_trend = self._calculate_trend(errors)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                statistics.mean(latencies),
                statistics.mean(throughputs),
                statistics.mean(errors),
                statistics.mean(cpu_utils),
                statistics.mean(memory_utils)
            )
            
            report = PerformanceReport(
                report_id=report_id,
                generated_at=time.time(),
                time_range_hours=time_range_hours,
                total_requests=len(latency_values),
                avg_latency_us=statistics.mean(latencies),
                p95_latency_us=np.percentile(latencies, 95) if len(latencies) >= 20 else statistics.mean(latencies),
                p99_latency_us=np.percentile(latencies, 99) if len(latencies) >= 100 else statistics.mean(latencies),
                error_rate=statistics.mean(errors),
                availability=statistics.mean(availability_vals),
                avg_cpu_utilization=statistics.mean(cpu_utils),
                max_cpu_utilization=max(cpu_utils) if cpu_utils else 0,
                avg_memory_utilization=statistics.mean(memory_utils),
                max_memory_utilization=max(memory_utils) if memory_utils else 0,
                latency_trend=latency_trend,
                throughput_trend=throughput_trend,
                error_trend=error_trend,
                optimization_opportunities=recommendations["optimizations"],
                capacity_recommendations=recommendations["capacity"],
                alert_recommendations=recommendations["alerts"]
            )
            
            self.performance_reports.append(report)
            
            # Keep only recent reports
            if len(self.performance_reports) > 100:
                self.performance_reports = self.performance_reports[-100:]
            
            self.logger.info(f"Generated performance report: {report_id} (avg latency: {report.avg_latency_us:.1f}μs)")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values"""
        
        if len(values) < 4:
            return "stable"
        
        # Compare first half vs second half
        half_point = len(values) // 2
        first_half_avg = statistics.mean(values[:half_point])
        second_half_avg = statistics.mean(values[half_point:])
        
        change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        if change_percent > 5:
            return "degrading"  # For latency/errors, higher is worse
        elif change_percent < -5:
            return "improving"
        else:
            return "stable"
    
    def _generate_recommendations(
        self, 
        avg_latency: float,
        avg_throughput: float,
        avg_error_rate: float,
        avg_cpu: float,
        avg_memory: float
    ) -> Dict[str, List[str]]:
        """Generate performance recommendations"""
        
        recommendations = {
            "optimizations": [],
            "capacity": [],
            "alerts": []
        }
        
        # Latency recommendations
        if avg_latency > 2000:
            recommendations["optimizations"].append("Consider upgrading to ultra-performance edge nodes")
        elif avg_latency > 1000:
            recommendations["optimizations"].append("Optimize network configuration and enable CPU isolation")
        
        # Throughput recommendations
        if avg_throughput < 5000:
            recommendations["capacity"].append("Consider scaling out edge nodes for higher throughput")
        
        # Error rate recommendations
        if avg_error_rate > 0.005:
            recommendations["optimizations"].append("Investigate error sources and improve error handling")
        
        # Resource recommendations
        if avg_cpu > 80:
            recommendations["capacity"].append("CPU utilization high - consider adding CPU cores")
        
        if avg_memory > 80:
            recommendations["capacity"].append("Memory utilization high - consider increasing memory allocation")
        
        # Alert recommendations
        if avg_latency > 1500:
            recommendations["alerts"].append("Consider adding latency alert at 1500μs threshold")
        
        return recommendations
    
    async def _cleanup_loop(self):
        """Cleanup old data and optimize storage"""
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                cutoff_time = current_time - (24 * 3600)  # 24 hours ago
                
                # Cleanup old metric points
                for metric_series in self.metrics.values():
                    original_count = len(metric_series.points)
                    metric_series.points = [
                        point for point in metric_series.points
                        if point.timestamp > cutoff_time
                    ]
                    
                    if len(metric_series.points) < original_count:
                        self.logger.debug(f"Cleaned up {original_count - len(metric_series.points)} old points from {metric_series.metric_name}")
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status"""
        
        active_alerts_by_severity = {}
        for severity in AlertSeverity:
            active_alerts_by_severity[severity.value] = len([
                alert for alert in self.active_alerts.values() 
                if alert.severity == severity
            ])
        
        return {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "collection_interval_seconds": self.collection_interval,
            "metrics_summary": {
                "total_metrics": len(self.metrics),
                "total_metric_points": sum(len(series.points) for series in self.metrics.values()),
                "registered_collectors": len(self.metric_collectors)
            },
            "alerting_summary": {
                "total_alert_rules": len(self.alert_rules),
                "active_alerts": len(self.active_alerts),
                "alerts_by_severity": active_alerts_by_severity,
                "alert_history_count": len(self.alert_history)
            },
            "dashboards_summary": {
                "total_dashboards": len(self.dashboards),
                "total_panels": sum(len(dashboard.panels) for dashboard in self.dashboards.values())
            },
            "performance_reports": {
                "total_reports": len(self.performance_reports),
                "latest_report_time": self.performance_reports[-1].generated_at if self.performance_reports else None
            },
            "notification_channels": list(self.notification_channels.keys()),
            "recent_metrics": {
                metric_name: series.points[-1].value if series.points else None
                for metric_name, series in self.metrics.items()
            }
        }
    
    def get_dashboard(self, dashboard_id: str) -> Optional[MonitoringDashboard]:
        """Get dashboard configuration"""
        return self.dashboards.get(dashboard_id)
    
    def get_dashboard_data(self, dashboard_id: str, time_range_hours: int = 1) -> Dict[str, Any]:
        """Get dashboard data for visualization"""
        
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        time_range_seconds = time_range_hours * 3600
        end_time = time.time()
        start_time = end_time - time_range_seconds
        
        panel_data = {}
        
        for panel in dashboard.panels:
            panel_data[panel.panel_id] = {
                "title": panel.title,
                "type": panel.panel_type,
                "data": {}
            }
            
            for metric_query in panel.metric_queries:
                metric_points = self.get_metric_values(metric_query, start_time, end_time)
                panel_data[panel.panel_id]["data"][metric_query] = [
                    {"timestamp": point.timestamp, "value": point.value, "labels": point.labels}
                    for point in metric_points
                ]
        
        return {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "generated_at": time.time(),
            "time_range_hours": time_range_hours,
            "panels": panel_data
        }
    
    def get_latest_performance_report(self) -> Optional[PerformanceReport]:
        """Get latest performance report"""
        return self.performance_reports[-1] if self.performance_reports else None
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        
        self.monitoring_active = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        self.monitoring_tasks.clear()
        self.logger.info("Edge monitoring system stopped")
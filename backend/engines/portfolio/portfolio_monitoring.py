#!/usr/bin/env python3
"""
Portfolio Engine Monitoring & Error Handling
Comprehensive monitoring, error handling, and performance tracking for the optimized portfolio engine.
"""

import asyncio
import logging
import time
import json
import traceback
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    PORTFOLIO_METRICS = "portfolio_metrics"
    SME_PERFORMANCE = "sme_performance"
    MESSAGEBUS_LATENCY = "messagebus_latency"

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertEvent:
    """Alert event data structure"""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class ErrorEvent:
    """Error event data structure"""
    error_id: str
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    source: str
    request_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False

class PerformanceTracker:
    """Advanced performance tracking and analysis"""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.rolling_averages: Dict[MetricType, float] = {}
        self.percentiles: Dict[MetricType, Dict[str, float]] = {}
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            MetricType.RESPONSE_TIME: {
                "warning": 5.0,      # 5ms warning
                "critical": 10.0     # 10ms critical
            },
            MetricType.ERROR_RATE: {
                "warning": 0.05,     # 5% warning
                "critical": 0.10     # 10% critical
            },
            MetricType.RESOURCE_USAGE: {
                "warning": 0.80,     # 80% warning
                "critical": 0.90     # 90% critical
            },
            MetricType.MESSAGEBUS_LATENCY: {
                "warning": 2.0,      # 2ms warning
                "critical": 5.0      # 5ms critical
            }
        }
    
    def record_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None) -> None:
        """Record a performance metric"""
        with self.lock:
            metric = PerformanceMetric(
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self.metrics_history[metric_type].append(metric)
            self._update_statistics(metric_type)
    
    def _update_statistics(self, metric_type: MetricType) -> None:
        """Update rolling statistics for a metric type"""
        history = self.metrics_history[metric_type]
        if not history:
            return
        
        values = [m.value for m in history]
        
        # Rolling average
        self.rolling_averages[metric_type] = sum(values) / len(values)
        
        # Percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        self.percentiles[metric_type] = {
            "p50": sorted_values[int(0.5 * n)],
            "p90": sorted_values[int(0.9 * n)],
            "p95": sorted_values[int(0.95 * n)],
            "p99": sorted_values[int(0.99 * n)] if n >= 100 else sorted_values[-1]
        }
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.lock:
            return {
                "rolling_averages": self.rolling_averages.copy(),
                "percentiles": self.percentiles.copy(),
                "total_metrics": {mt.value: len(history) for mt, history in self.metrics_history.items()},
                "timestamp": datetime.now().isoformat()
            }
    
    def check_thresholds(self) -> List[AlertEvent]:
        """Check performance thresholds and generate alerts"""
        alerts = []
        
        with self.lock:
            for metric_type, avg_value in self.rolling_averages.items():
                if metric_type not in self.thresholds:
                    continue
                
                thresholds = self.thresholds[metric_type]
                
                if avg_value >= thresholds["critical"]:
                    alerts.append(AlertEvent(
                        alert_id=f"perf_critical_{metric_type.value}_{int(time.time())}",
                        level=AlertLevel.CRITICAL,
                        title=f"Critical Performance Threshold Exceeded",
                        description=f"{metric_type.value} average ({avg_value:.2f}) exceeds critical threshold ({thresholds['critical']})",
                        timestamp=datetime.now(),
                        source="performance_tracker",
                        metadata={"metric_type": metric_type.value, "value": avg_value, "threshold": thresholds["critical"]}
                    ))
                elif avg_value >= thresholds["warning"]:
                    alerts.append(AlertEvent(
                        alert_id=f"perf_warning_{metric_type.value}_{int(time.time())}",
                        level=AlertLevel.WARNING,
                        title=f"Performance Threshold Warning",
                        description=f"{metric_type.value} average ({avg_value:.2f}) exceeds warning threshold ({thresholds['warning']})",
                        timestamp=datetime.now(),
                        source="performance_tracker",
                        metadata={"metric_type": metric_type.value, "value": avg_value, "threshold": thresholds["warning"]}
                    ))
        
        return alerts

class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self):
        self.error_history: List[ErrorEvent] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.max_error_history = 1000
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def _register_default_recovery_strategies(self) -> None:
        """Register default error recovery strategies"""
        self.recovery_strategies.update({
            "messagebus_connection_error": self._recover_messagebus_connection,
            "sme_acceleration_error": self._recover_sme_acceleration,
            "portfolio_calculation_error": self._recover_portfolio_calculation,
            "database_connection_error": self._recover_database_connection
        })
    
    async def handle_error(self, error: Exception, source: str, 
                          request_data: Dict[str, Any] = None) -> ErrorEvent:
        """Handle an error with automatic recovery attempts"""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Create error event
        error_event = ErrorEvent(
            error_id=f"err_{int(time.time())}_{hash(error_message) % 10000:04d}",
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            timestamp=datetime.now(),
            source=source,
            request_data=request_data or {}
        )
        
        # Record error
        self._record_error(error_event)
        
        # Attempt recovery if strategy exists
        recovery_key = self._get_recovery_key(error_type, source)
        if recovery_key in self.recovery_strategies:
            try:
                error_event.recovery_attempted = True
                recovery_result = await self.recovery_strategies[recovery_key](error, error_event)
                error_event.recovery_successful = recovery_result
                
                if recovery_result:
                    logger.info(f"Successfully recovered from error {error_event.error_id}")
                else:
                    logger.warning(f"Recovery attempt failed for error {error_event.error_id}")
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed for error {error_event.error_id}: {recovery_error}")
                error_event.recovery_successful = False
        
        # Check if circuit breaker should be triggered
        self._check_circuit_breaker(source, error_type)
        
        return error_event
    
    def _record_error(self, error_event: ErrorEvent) -> None:
        """Record error event in history"""
        self.error_history.append(error_event)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Update error counts
        self.error_counts[error_event.error_type] += 1
        
        logger.error(f"Error recorded: {error_event.error_id} - {error_event.error_type}: {error_event.error_message}")
    
    def _get_recovery_key(self, error_type: str, source: str) -> str:
        """Generate recovery strategy key"""
        # Try specific combinations first
        specific_keys = [
            f"{source}_{error_type.lower()}",
            f"{error_type.lower()}",
            f"{source}_error"
        ]
        
        for key in specific_keys:
            if key in self.recovery_strategies:
                return key
        
        return ""
    
    def _check_circuit_breaker(self, source: str, error_type: str) -> None:
        """Check if circuit breaker should be triggered"""
        key = f"{source}_{error_type}"
        
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "count": 0,
                "window_start": datetime.now(),
                "state": "closed"  # closed, open, half-open
            }
        
        cb = self.circuit_breakers[key]
        
        # Reset window if needed (5-minute windows)
        if datetime.now() - cb["window_start"] > timedelta(minutes=5):
            cb["count"] = 0
            cb["window_start"] = datetime.now()
            if cb["state"] == "open":
                cb["state"] = "half-open"
        
        cb["count"] += 1
        
        # Trigger circuit breaker if too many errors (5 errors in 5 minutes)
        if cb["count"] >= 5 and cb["state"] != "open":
            cb["state"] = "open"
            logger.critical(f"Circuit breaker OPEN for {key}: {cb['count']} errors in window")
    
    def is_circuit_breaker_open(self, source: str, error_type: str = None) -> bool:
        """Check if circuit breaker is open"""
        if error_type:
            key = f"{source}_{error_type}"
            return self.circuit_breakers.get(key, {}).get("state") == "open"
        
        # Check any circuit breaker for the source
        return any(
            cb.get("state") == "open" 
            for key, cb in self.circuit_breakers.items() 
            if key.startswith(f"{source}_")
        )
    
    async def _recover_messagebus_connection(self, error: Exception, error_event: ErrorEvent) -> bool:
        """Recover from messagebus connection errors"""
        try:
            logger.info("Attempting messagebus connection recovery...")
            await asyncio.sleep(1)  # Wait before retry
            # Add actual messagebus reconnection logic here
            return True
        except Exception as e:
            logger.error(f"Messagebus recovery failed: {e}")
            return False
    
    async def _recover_sme_acceleration(self, error: Exception, error_event: ErrorEvent) -> bool:
        """Recover from SME acceleration errors"""
        try:
            logger.info("Attempting SME acceleration recovery...")
            # Fallback to CPU processing
            return True
        except Exception as e:
            logger.error(f"SME recovery failed: {e}")
            return False
    
    async def _recover_portfolio_calculation(self, error: Exception, error_event: ErrorEvent) -> bool:
        """Recover from portfolio calculation errors"""
        try:
            logger.info("Attempting portfolio calculation recovery...")
            # Use simplified calculation method
            return True
        except Exception as e:
            logger.error(f"Portfolio calculation recovery failed: {e}")
            return False
    
    async def _recover_database_connection(self, error: Exception, error_event: ErrorEvent) -> bool:
        """Recover from database connection errors"""
        try:
            logger.info("Attempting database connection recovery...")
            await asyncio.sleep(2)  # Wait before retry
            # Add actual database reconnection logic here
            return True
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        recent_errors = [e for e in self.error_history if datetime.now() - e.timestamp < timedelta(hours=24)]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_24h": len(recent_errors),
            "error_types": dict(self.error_counts),
            "recovery_success_rate": self._calculate_recovery_success_rate(),
            "circuit_breakers": {
                key: {"state": cb["state"], "count": cb["count"]}
                for key, cb in self.circuit_breakers.items()
            },
            "most_recent_errors": [
                {
                    "error_id": e.error_id,
                    "error_type": e.error_type,
                    "timestamp": e.timestamp.isoformat(),
                    "source": e.source,
                    "recovered": e.recovery_successful
                }
                for e in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        recovery_attempts = [e for e in self.error_history if e.recovery_attempted]
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
        return len(successful_recoveries) / len(recovery_attempts)

class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self):
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []
        self.notification_handlers: List[Callable[[AlertEvent], None]] = []
        self.max_alert_history = 1000
        
        # Alert suppression (prevent spam)
        self.alert_suppression: Dict[str, datetime] = {}
        self.suppression_window = timedelta(minutes=5)
    
    def add_notification_handler(self, handler: Callable[[AlertEvent], None]) -> None:
        """Add alert notification handler"""
        self.notification_handlers.append(handler)
    
    async def trigger_alert(self, alert: AlertEvent) -> None:
        """Trigger an alert"""
        # Check for alert suppression
        suppression_key = f"{alert.source}_{alert.title}"
        if suppression_key in self.alert_suppression:
            if datetime.now() - self.alert_suppression[suppression_key] < self.suppression_window:
                logger.debug(f"Alert suppressed: {alert.alert_id}")
                return
        
        # Record alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Maintain history size
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]
        
        # Update suppression
        self.alert_suppression[suppression_key] = datetime.now()
        
        # Notify handlers
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert notification handler failed: {e}")
        
        logger.log(
            logging.CRITICAL if alert.level == AlertLevel.CRITICAL else
            logging.ERROR if alert.level == AlertLevel.ERROR else
            logging.WARNING if alert.level == AlertLevel.WARNING else
            logging.INFO,
            f"ALERT [{alert.level.value.upper()}]: {alert.title} - {alert.description}"
        )
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        recent_alerts = [a for a in self.alert_history if datetime.now() - a.timestamp < timedelta(hours=24)]
        
        alert_counts_by_level = defaultdict(int)
        for alert in recent_alerts:
            alert_counts_by_level[alert.level.value] += 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "recent_alerts_24h": len(recent_alerts),
            "alert_counts_by_level": dict(alert_counts_by_level),
            "active_alerts_details": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source
                }
                for alert in self.active_alerts.values()
            ]
        }

class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_task = None
        self.resource_history: deque = deque(maxlen=1000)
        
        # Resource thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 90.0},
            "disk_percent": {"warning": 85.0, "critical": 95.0}
        }
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Resource monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect resource metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                resource_data = {
                    "timestamp": datetime.now(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
                
                self.resource_history.append(resource_data)
                
                # Check thresholds
                alerts = self._check_resource_thresholds(resource_data)
                
                # In a real implementation, you would send these alerts to AlertManager
                for alert_data in alerts:
                    logger.warning(f"Resource alert: {alert_data}")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def _check_resource_thresholds(self, resource_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check resource thresholds and generate alerts"""
        alerts = []
        
        for metric, value in resource_data.items():
            if metric not in self.thresholds or not isinstance(value, (int, float)):
                continue
            
            thresholds = self.thresholds[metric]
            
            if value >= thresholds["critical"]:
                alerts.append({
                    "level": "critical",
                    "metric": metric,
                    "value": value,
                    "threshold": thresholds["critical"],
                    "message": f"{metric} at {value:.1f}% (critical threshold: {thresholds['critical']}%)"
                })
            elif value >= thresholds["warning"]:
                alerts.append({
                    "level": "warning",
                    "metric": metric,
                    "value": value,
                    "threshold": thresholds["warning"],
                    "message": f"{metric} at {value:.1f}% (warning threshold: {thresholds['warning']}%)"
                })
        
        return alerts
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource usage"""
        if not self.resource_history:
            return {"error": "No resource data available"}
        
        latest = self.resource_history[-1]
        return {
            "current_usage": {
                "cpu_percent": latest["cpu_percent"],
                "memory_percent": latest["memory_percent"],
                "memory_available_gb": latest["memory_available_gb"],
                "disk_percent": latest["disk_percent"],
                "disk_free_gb": latest["disk_free_gb"]
            },
            "timestamp": latest["timestamp"].isoformat(),
            "history_points": len(self.resource_history)
        }

class ComprehensiveMonitor:
    """Comprehensive monitoring system combining all monitoring components"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.error_handler = ErrorHandler()
        self.alert_manager = AlertManager()
        self.resource_monitor = ResourceMonitor()
        
        # Setup default alert notification handler
        self.alert_manager.add_notification_handler(self._default_alert_handler)
        
        # Background monitoring task
        self.monitoring_task = None
        self.monitoring_active = False
    
    async def initialize(self) -> None:
        """Initialize comprehensive monitoring"""
        logger.info("🔍 Initializing Comprehensive Portfolio Engine Monitoring...")
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Start performance monitoring loop
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ Comprehensive monitoring initialized")
    
    async def shutdown(self) -> None:
        """Shutdown comprehensive monitoring"""
        logger.info("🔄 Shutting down comprehensive monitoring...")
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.resource_monitor.stop_monitoring()
        
        logger.info("✅ Comprehensive monitoring shutdown complete")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for performance analysis"""
        while self.monitoring_active:
            try:
                # Check performance thresholds
                alerts = self.performance_tracker.check_thresholds()
                for alert in alerts:
                    await self.alert_manager.trigger_alert(alert)
                
                # Auto-resolve old alerts (implement your logic here)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    def _default_alert_handler(self, alert: AlertEvent) -> None:
        """Default alert notification handler"""
        # In production, this could send notifications via email, Slack, etc.
        logger.info(f"Alert notification: {alert.title} ({alert.level.value})")
    
    def record_request_performance(self, endpoint: str, response_time_ms: float, 
                                 success: bool, metadata: Dict[str, Any] = None) -> None:
        """Record request performance metrics"""
        self.performance_tracker.record_metric(
            MetricType.RESPONSE_TIME,
            response_time_ms,
            {"endpoint": endpoint, "success": success, **(metadata or {})}
        )
        
        if not success:
            self.performance_tracker.record_metric(
                MetricType.ERROR_RATE,
                1.0,
                {"endpoint": endpoint, **(metadata or {})}
            )
    
    def record_sme_performance(self, operation_type: str, processing_time_ns: float,
                              sme_used: bool, metadata: Dict[str, Any] = None) -> None:
        """Record SME performance metrics"""
        self.performance_tracker.record_metric(
            MetricType.SME_PERFORMANCE,
            processing_time_ns / 1_000_000,  # Convert to ms
            {
                "operation_type": operation_type,
                "sme_used": sme_used,
                **(metadata or {})
            }
        )
    
    def record_messagebus_latency(self, operation: str, latency_ms: float,
                                 metadata: Dict[str, Any] = None) -> None:
        """Record messagebus latency metrics"""
        self.performance_tracker.record_metric(
            MetricType.MESSAGEBUS_LATENCY,
            latency_ms,
            {"operation": operation, **(metadata or {})}
        )
    
    async def handle_error(self, error: Exception, source: str, 
                          request_data: Dict[str, Any] = None) -> ErrorEvent:
        """Handle error with comprehensive error handling"""
        return await self.error_handler.handle_error(error, source, request_data)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "performance": self.performance_tracker.get_current_performance(),
            "errors": self.error_handler.get_error_summary(),
            "alerts": self.alert_manager.get_alert_summary(),
            "resources": self.resource_monitor.get_current_resources(),
            "timestamp": datetime.now().isoformat()
        }

# Create singleton instance for global use
comprehensive_monitor = ComprehensiveMonitor()

# Decorator for automatic performance tracking
def track_performance(endpoint: str):
    """Decorator to automatically track endpoint performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                success = True
                error = None
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error = e
                    # Record error
                    await comprehensive_monitor.handle_error(e, endpoint)
                    raise
                finally:
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    comprehensive_monitor.record_request_performance(
                        endpoint, response_time_ms, success
                    )
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    # Would need to handle sync error recording differently
                    raise
                finally:
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    comprehensive_monitor.record_request_performance(
                        endpoint, response_time_ms, success
                    )
            
            return sync_wrapper
    
    return decorator
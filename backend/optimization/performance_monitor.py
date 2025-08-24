"""
Performance Monitor for M4 Max Optimization
===========================================

Real-time performance monitoring system for CPU core utilization, latency tracking,
thermal management, and power optimization specifically designed for M4 Max architecture.
"""

import os
import sys
import time
import psutil
import threading
import subprocess
import json
import statistics
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
import sqlite3
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_USAGE = "memory_usage"
    LATENCY = "latency"
    THERMAL = "thermal"
    POWER = "power"
    THROUGHPUT = "throughput"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"

class AlertLevel(Enum):
    """Alert levels for performance issues"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Single performance metric data point"""
    timestamp: float
    metric_type: MetricType
    value: float
    unit: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyMeasurement:
    """Latency measurement for trading operations"""
    operation_type: str
    start_time: float
    end_time: float
    latency_ms: float
    success: bool
    error_msg: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemAlert:
    """System performance alert"""
    alert_id: str
    timestamp: float
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[float] = None

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for M4 Max trading platform
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.monitoring_active = False
        self.monitor_threads: Dict[str, threading.Thread] = {}
        
        # Metric storage (in-memory with rolling window)
        self.metrics_buffer: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=3600)  # 1 hour of data at 1Hz
            for metric_type in MetricType
        }
        
        # Latency tracking
        self.latency_measurements: deque = deque(maxlen=10000)
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        
        # Alert system
        self.alerts: List[SystemAlert] = []
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        
        # Performance thresholds
        self.thresholds = {
            MetricType.CPU_UTILIZATION: {"warning": 80.0, "critical": 95.0},
            MetricType.MEMORY_USAGE: {"warning": 80.0, "critical": 95.0},
            MetricType.LATENCY: {"warning": 10.0, "critical": 50.0},  # milliseconds
            MetricType.THERMAL: {"warning": 80.0, "critical": 90.0},  # Celsius
        }
        
        # Core-specific monitoring
        self.core_metrics: Dict[int, deque] = {}
        self.is_m4_max = self._detect_m4_max()
        
        # Statistics cache
        self.stats_cache = {}
        self.stats_cache_time = 0
        self.stats_cache_ttl = 5.0  # 5 seconds
        
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Start monitoring
        self.start_monitoring()
    
    def _detect_m4_max(self) -> bool:
        """Detect if running on M4 Max"""
        if sys.platform != "darwin":
            return False
        
        try:
            # Check CPU count (M4 Max has 16 cores)
            cpu_count = os.cpu_count()
            if cpu_count == 16:
                # Additional checks for M4 Max
                result = subprocess.run([
                    "sysctl", "-n", "machdep.cpu.brand_string"
                ], capture_output=True, text=True)
                
                if "Apple" in result.stdout:
                    return True
        except Exception:
            pass
        
        return False
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage"""
        try:
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.db_lock = threading.Lock()
            
            # Create tables
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS latency_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_msg TEXT,
                    context TEXT
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at REAL
                )
            """)
            
            # Create indices
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_latency_timestamp ON latency_measurements(start_time)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            
            self.db_conn.commit()
            
            logger.info("Performance monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fallback to in-memory database
            self.db_conn = sqlite3.connect(":memory:", check_same_thread=False)
    
    def start_monitoring(self) -> None:
        """Start all monitoring threads"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # CPU monitoring
        self.monitor_threads["cpu"] = threading.Thread(
            target=self._monitor_cpu,
            daemon=True,
            name="CPUMonitor"
        )
        
        # Memory monitoring
        self.monitor_threads["memory"] = threading.Thread(
            target=self._monitor_memory,
            daemon=True,
            name="MemoryMonitor"
        )
        
        # Thermal monitoring (macOS)
        if self.is_m4_max:
            self.monitor_threads["thermal"] = threading.Thread(
                target=self._monitor_thermal,
                daemon=True,
                name="ThermalMonitor"
            )
        
        # I/O monitoring
        self.monitor_threads["io"] = threading.Thread(
            target=self._monitor_io,
            daemon=True,
            name="IOMonitor"
        )
        
        # Alert processor
        self.monitor_threads["alerts"] = threading.Thread(
            target=self._process_alerts,
            daemon=True,
            name="AlertProcessor"
        )
        
        # Start all threads
        for name, thread in self.monitor_threads.items():
            thread.start()
            logger.info(f"Started {name} monitoring thread")
        
        logger.info("Performance monitoring started")
    
    def _monitor_cpu(self) -> None:
        """Monitor CPU utilization per core"""
        while self.monitoring_active:
            try:
                # Overall CPU usage
                cpu_percent = psutil.cpu_percent(interval=1.0)
                self._add_metric(MetricType.CPU_UTILIZATION, cpu_percent, "%", "system")
                
                # Per-core usage
                per_core = psutil.cpu_percent(interval=0.1, percpu=True)
                
                for core_id, usage in enumerate(per_core):
                    if core_id not in self.core_metrics:
                        self.core_metrics[core_id] = deque(maxlen=3600)
                    
                    self.core_metrics[core_id].append({
                        "timestamp": time.time(),
                        "usage": usage
                    })
                    
                    # Check for core-specific alerts
                    if usage > 95.0:
                        self._generate_alert(
                            AlertLevel.CRITICAL,
                            MetricType.CPU_UTILIZATION,
                            f"Core {core_id} at {usage}% utilization",
                            usage,
                            95.0
                        )
                
                # CPU frequency information
                try:
                    freq_info = psutil.cpu_freq()
                    if freq_info:
                        self._add_metric(MetricType.CPU_UTILIZATION, freq_info.current, "MHz", "frequency")
                except Exception:
                    pass
                
            except Exception as e:
                logger.error(f"Error in CPU monitoring: {e}")
                time.sleep(5)
    
    def _monitor_memory(self) -> None:
        """Monitor system memory usage"""
        while self.monitoring_active:
            try:
                # System memory
                memory = psutil.virtual_memory()
                self._add_metric(MetricType.MEMORY_USAGE, memory.percent, "%", "system")
                self._add_metric(MetricType.MEMORY_USAGE, memory.available / 1024**3, "GB", "available")
                
                # Swap usage
                swap = psutil.swap_memory()
                self._add_metric(MetricType.MEMORY_USAGE, swap.percent, "%", "swap")
                
                # Check memory pressure
                if memory.percent > self.thresholds[MetricType.MEMORY_USAGE]["critical"]:
                    self._generate_alert(
                        AlertLevel.CRITICAL,
                        MetricType.MEMORY_USAGE,
                        f"High memory usage: {memory.percent}%",
                        memory.percent,
                        self.thresholds[MetricType.MEMORY_USAGE]["critical"]
                    )
                
                time.sleep(5)  # Less frequent than CPU
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(10)
    
    def _monitor_thermal(self) -> None:
        """Monitor thermal state (macOS only)"""
        while self.monitoring_active:
            try:
                # Use powermetrics for thermal information
                result = subprocess.run([
                    "powermetrics", "-s", "cpu_power", "-n", "1", "--format", "text"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    # Parse thermal information (simplified)
                    # In production, you'd parse the actual powermetrics output
                    # This is a placeholder for thermal monitoring
                    thermal_estimate = 45.0 + (psutil.cpu_percent() * 0.5)  # Rough estimate
                    self._add_metric(MetricType.THERMAL, thermal_estimate, "°C", "cpu_temperature")
                
                time.sleep(10)  # Thermal changes slowly
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring: {e}")
                time.sleep(30)
    
    def _monitor_io(self) -> None:
        """Monitor I/O statistics"""
        while self.monitoring_active:
            try:
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self._add_metric(MetricType.DISK_IO, disk_io.read_bytes / 1024**2, "MB", "read")
                    self._add_metric(MetricType.DISK_IO, disk_io.write_bytes / 1024**2, "MB", "write")
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self._add_metric(MetricType.NETWORK_IO, net_io.bytes_sent / 1024**2, "MB", "sent")
                    self._add_metric(MetricType.NETWORK_IO, net_io.bytes_recv / 1024**2, "MB", "received")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in I/O monitoring: {e}")
                time.sleep(10)
    
    def _process_alerts(self) -> None:
        """Process and manage alerts"""
        while self.monitoring_active:
            try:
                # Auto-resolve old alerts
                current_time = time.time()
                
                with self._lock:
                    for alert in self.alerts:
                        if not alert.resolved and (current_time - alert.timestamp) > 300:  # 5 minutes
                            # Check if condition still exists
                            if self._check_alert_condition(alert):
                                continue  # Condition still exists
                            
                            # Auto-resolve
                            alert.resolved = True
                            alert.resolved_at = current_time
                            
                            logger.info(f"Auto-resolved alert: {alert.message}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                time.sleep(60)
    
    def _add_metric(self, metric_type: MetricType, value: float, unit: str, source: str, metadata: Optional[Dict] = None) -> None:
        """Add a metric to the monitoring system"""
        timestamp = time.time()
        
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=metric_type,
            value=value,
            unit=unit,
            source=source,
            metadata=metadata or {}
        )
        
        # Add to in-memory buffer
        with self._lock:
            self.metrics_buffer[metric_type].append(metric)
        
        # Persist to database
        self._persist_metric(metric)
        
        # Check thresholds
        self._check_thresholds(metric)
    
    def _persist_metric(self, metric: PerformanceMetric) -> None:
        """Persist metric to database"""
        try:
            with self.db_lock:
                self.db_conn.execute("""
                    INSERT INTO metrics (timestamp, metric_type, value, unit, source, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp,
                    metric.metric_type.value,
                    metric.value,
                    metric.unit,
                    metric.source,
                    json.dumps(metric.metadata)
                ))
                self.db_conn.commit()
                
        except Exception as e:
            logger.error(f"Error persisting metric: {e}")
    
    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric exceeds thresholds"""
        if metric.metric_type not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric.metric_type]
        
        if metric.value >= thresholds["critical"]:
            self._generate_alert(
                AlertLevel.CRITICAL,
                metric.metric_type,
                f"{metric.metric_type.value} critical: {metric.value}{metric.unit}",
                metric.value,
                thresholds["critical"]
            )
        elif metric.value >= thresholds["warning"]:
            self._generate_alert(
                AlertLevel.WARNING,
                metric.metric_type,
                f"{metric.metric_type.value} warning: {metric.value}{metric.unit}",
                metric.value,
                thresholds["warning"]
            )
    
    def _generate_alert(self, level: AlertLevel, metric_type: MetricType, message: str, value: float, threshold: float) -> None:
        """Generate a performance alert"""
        alert_id = f"{metric_type.value}_{level.value}_{int(time.time())}"
        
        alert = SystemAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            level=level,
            metric_type=metric_type,
            message=message,
            value=value,
            threshold=threshold
        )
        
        with self._lock:
            # Check for duplicate recent alerts
            recent_alerts = [
                a for a in self.alerts
                if a.metric_type == metric_type and 
                   a.level == level and
                   not a.resolved and
                   (alert.timestamp - a.timestamp) < 60  # Within 1 minute
            ]
            
            if recent_alerts:
                return  # Don't spam duplicate alerts
            
            self.alerts.append(alert)
        
        # Persist alert
        self._persist_alert(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Generated alert: {message}")
    
    def _persist_alert(self, alert: SystemAlert) -> None:
        """Persist alert to database"""
        try:
            with self.db_lock:
                self.db_conn.execute("""
                    INSERT INTO alerts (alert_id, timestamp, level, metric_type, message, value, threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.timestamp,
                    alert.level.value,
                    alert.metric_type.value,
                    alert.message,
                    alert.value,
                    alert.threshold
                ))
                self.db_conn.commit()
                
        except Exception as e:
            logger.error(f"Error persisting alert: {e}")
    
    def _check_alert_condition(self, alert: SystemAlert) -> bool:
        """Check if alert condition still exists"""
        # Get recent metrics for the alert type
        recent_metrics = self.get_recent_metrics(alert.metric_type, duration=60)
        
        if not recent_metrics:
            return False
        
        # Check if recent values still exceed threshold
        recent_values = [m.value for m in recent_metrics[-5:]]  # Last 5 measurements
        avg_value = statistics.mean(recent_values)
        
        return avg_value >= alert.threshold
    
    # Public API methods
    
    def start_latency_measurement(self, operation_type: str, operation_id: Optional[str] = None) -> str:
        """Start measuring latency for an operation"""
        if operation_id is None:
            operation_id = f"{operation_type}_{int(time.time() * 1000000)}"
        
        self.active_operations[operation_id] = time.time()
        return operation_id
    
    def end_latency_measurement(
        self, 
        operation_id: str, 
        success: bool = True, 
        error_msg: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> float:
        """End latency measurement and record result"""
        end_time = time.time()
        
        if operation_id not in self.active_operations:
            logger.warning(f"No active operation found for ID: {operation_id}")
            return 0.0
        
        start_time = self.active_operations.pop(operation_id)
        latency_ms = (end_time - start_time) * 1000
        
        # Extract operation type from ID
        operation_type = operation_id.split('_')[0] if '_' in operation_id else "unknown"
        
        measurement = LatencyMeasurement(
            operation_type=operation_type,
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency_ms,
            success=success,
            error_msg=error_msg,
            context=context or {}
        )
        
        # Store measurement
        with self._lock:
            self.latency_measurements.append(measurement)
        
        # Persist to database
        self._persist_latency_measurement(measurement)
        
        # Add as metric
        self._add_metric(MetricType.LATENCY, latency_ms, "ms", operation_type)
        
        return latency_ms
    
    def _persist_latency_measurement(self, measurement: LatencyMeasurement) -> None:
        """Persist latency measurement to database"""
        try:
            with self.db_lock:
                self.db_conn.execute("""
                    INSERT INTO latency_measurements 
                    (operation_type, start_time, end_time, latency_ms, success, error_msg, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    measurement.operation_type,
                    measurement.start_time,
                    measurement.end_time,
                    measurement.latency_ms,
                    measurement.success,
                    measurement.error_msg,
                    json.dumps(measurement.context)
                ))
                self.db_conn.commit()
                
        except Exception as e:
            logger.error(f"Error persisting latency measurement: {e}")
    
    def get_recent_metrics(self, metric_type: MetricType, duration: int = 300) -> List[PerformanceMetric]:
        """Get recent metrics of a specific type"""
        cutoff_time = time.time() - duration
        
        with self._lock:
            return [
                metric for metric in self.metrics_buffer[metric_type]
                if metric.timestamp >= cutoff_time
            ]
    
    def get_latency_stats(self, operation_type: Optional[str] = None, duration: int = 300) -> Dict:
        """Get latency statistics"""
        cutoff_time = time.time() - duration
        
        with self._lock:
            measurements = [
                m for m in self.latency_measurements
                if m.start_time >= cutoff_time and
                   (operation_type is None or m.operation_type == operation_type)
            ]
        
        if not measurements:
            return {"count": 0}
        
        latencies = [m.latency_ms for m in measurements]
        successful = [m for m in measurements if m.success]
        
        return {
            "count": len(measurements),
            "success_rate": len(successful) / len(measurements),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies)
        }
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        current_time = time.time()
        
        # Check cache
        if (current_time - self.stats_cache_time) < self.stats_cache_ttl:
            return self.stats_cache
        
        stats = {
            "timestamp": current_time,
            "system_info": {
                "platform": sys.platform,
                "is_m4_max": self.is_m4_max,
                "cpu_count": os.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / 1024**3
            },
            "current_metrics": {},
            "core_utilization": {},
            "alerts": {
                "total": len(self.alerts),
                "active": len([a for a in self.alerts if not a.resolved]),
                "critical": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.resolved])
            }
        }
        
        # Current metrics
        for metric_type in MetricType:
            recent_metrics = self.get_recent_metrics(metric_type, 60)
            if recent_metrics:
                latest = recent_metrics[-1]
                stats["current_metrics"][metric_type.value] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp
                }
        
        # Per-core utilization
        if self.is_m4_max:
            for core_id, core_data in self.core_metrics.items():
                if core_data:
                    recent_usage = [d["usage"] for d in list(core_data)[-10:]]  # Last 10 measurements
                    stats["core_utilization"][f"core_{core_id}"] = {
                        "current": recent_usage[-1] if recent_usage else 0,
                        "avg": statistics.mean(recent_usage) if recent_usage else 0,
                        "max": max(recent_usage) if recent_usage else 0
                    }
        
        # Cache result
        self.stats_cache = stats
        self.stats_cache_time = current_time
        
        return stats
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric_type: MetricType, warning: float, critical: float) -> None:
        """Set custom thresholds for a metric type"""
        self.thresholds[metric_type] = {"warning": warning, "critical": critical}
        logger.info(f"Updated thresholds for {metric_type.value}: warning={warning}, critical={critical}")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring threads"""
        logger.info("Stopping performance monitoring...")
        
        self.monitoring_active = False
        
        # Wait for threads to finish
        for name, thread in self.monitor_threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
                logger.info(f"Stopped {name} monitoring thread")
        
        # Close database
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
        
        logger.info("Performance monitoring stopped")
    
    def export_metrics(self, start_time: float, end_time: float, output_file: str) -> bool:
        """Export metrics to file for analysis"""
        try:
            with self.db_lock:
                cursor = self.db_conn.execute("""
                    SELECT * FROM metrics 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (start_time, end_time))
                
                metrics_data = []
                for row in cursor.fetchall():
                    metrics_data.append({
                        "timestamp": row[1],
                        "metric_type": row[2],
                        "value": row[3],
                        "unit": row[4],
                        "source": row[5],
                        "metadata": json.loads(row[6]) if row[6] else {}
                    })
            
            # Export to JSON
            with open(output_file, 'w') as f:
                json.dump({
                    "export_time": time.time(),
                    "start_time": start_time,
                    "end_time": end_time,
                    "metrics_count": len(metrics_data),
                    "metrics": metrics_data
                }, f, indent=2)
            
            logger.info(f"Exported {len(metrics_data)} metrics to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False
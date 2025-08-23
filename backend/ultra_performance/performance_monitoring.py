"""
Performance Monitoring and Profiling System for Ultra-Performance Trading

Provides:
- Real-time performance profilers
- CPU and GPU utilization monitoring
- Memory allocation tracking
- Performance regression detection
- Ultra-performance metrics collection
"""

import asyncio
import logging
import threading
import time
import psutil
import gc
import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set, AsyncGenerator
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import weakref
import traceback
import json

# Profiling imports
try:
    import cProfile
    import pstats
    import profile
    PROFILING_AVAILABLE = True
except ImportError:
    cProfile = pstats = profile = None
    PROFILING_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    line_profiler = None
    LINE_PROFILER_AVAILABLE = False

# GPU monitoring
try:
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    pynvml = None
    GPU_MONITORING_AVAILABLE = False

# Advanced metrics
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    GC_TIME = "gc_time"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert"""
    metric_name: str
    severity: AlertSeverity
    threshold: float
    current_value: float
    message: str
    timestamp: float
    acknowledged: bool = False

@dataclass
class ProfilingSession:
    """Profiling session information"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    function_stats: Dict[str, Any] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class SystemResourceSnapshot:
    """System resource usage snapshot"""
    timestamp: float
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0

class RealTimeProfiler:
    """
    Real-time performance profiler for ultra-low latency monitoring
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, ProfilingSession] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.profilers: Dict[str, cProfile.Profile] = {}
        self.is_monitoring = False
        self._lock = threading.RLock()
        
    def start_profiling_session(self, session_id: str) -> bool:
        """Start new profiling session"""
        with self._lock:
            if session_id in self.active_sessions:
                logger.warning(f"Profiling session {session_id} already active")
                return False
                
            if not PROFILING_AVAILABLE:
                logger.error("Profiling not available - install cProfile")
                return False
                
            # Create profiling session
            session = ProfilingSession(
                session_id=session_id,
                start_time=time.time()
            )
            
            # Start cProfile profiler
            profiler = cProfile.Profile()
            profiler.enable()
            
            self.active_sessions[session_id] = session
            self.profilers[session_id] = profiler
            
            logger.info(f"Started profiling session: {session_id}")
            return True
            
    def stop_profiling_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Stop profiling session and return results"""
        with self._lock:
            if session_id not in self.active_sessions:
                return None
                
            session = self.active_sessions[session_id]
            profiler = self.profilers[session_id]
            
            # Stop profiler
            profiler.disable()
            session.end_time = time.time()
            
            # Analyze profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # Extract function statistics
            function_stats = {}
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_number, function_name = func_info
                function_stats[f"{filename}:{line_number}({function_name})"] = {
                    "call_count": cc,
                    "recursive_call_count": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "avg_time_per_call": tt / cc if cc > 0 else 0,
                    "callers": len(callers)
                }
                
            session.function_stats = function_stats
            
            # Clean up
            del self.active_sessions[session_id]
            del self.profilers[session_id]
            
            return {
                "session_id": session_id,
                "duration_seconds": session.end_time - session.start_time,
                "function_stats": function_stats,
                "top_functions": self._get_top_functions(function_stats),
                "memory_stats": session.memory_stats
            }
            
    def _get_top_functions(self, function_stats: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top N functions by cumulative time"""
        sorted_functions = sorted(
            function_stats.items(),
            key=lambda x: x[1]["cumulative_time"],
            reverse=True
        )
        
        return [
            {
                "function": func_name,
                "cumulative_time": stats["cumulative_time"],
                "call_count": stats["call_count"],
                "avg_time_per_call": stats["avg_time_per_call"]
            }
            for func_name, stats in sorted_functions[:top_n]
        ]
        
    @contextmanager
    def profile_function(self, function_name: str):
        """Context manager for profiling individual functions"""
        start_time = time.perf_counter_ns()
        
        # Start memory tracking if available
        if TRACEMALLOC_AVAILABLE:
            tracemalloc.start()
            start_snapshot = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            end_time = time.perf_counter_ns()
            execution_time_us = (end_time - start_time) / 1000
            
            # Record execution time
            self.record_metric(
                name=f"function_execution_time.{function_name}",
                metric_type=MetricType.LATENCY,
                value=execution_time_us,
                unit="microseconds"
            )
            
            # Record memory usage if available
            if TRACEMALLOC_AVAILABLE:
                end_snapshot = tracemalloc.take_snapshot()
                top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
                
                if top_stats:
                    memory_delta = sum(stat.size_diff for stat in top_stats)
                    self.record_metric(
                        name=f"function_memory_delta.{function_name}",
                        metric_type=MetricType.MEMORY_USAGE,
                        value=memory_delta,
                        unit="bytes"
                    )
                    
                tracemalloc.stop()
                
    def record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record performance metric"""
        metric = PerformanceMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            unit=unit,
            tags=tags or {},
            context=context or {}
        )
        
        self.metrics_buffer.append(metric)
        
    def get_metrics_summary(self, last_n_seconds: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified time window"""
        current_time = time.time()
        window_start = current_time - last_n_seconds
        
        # Filter metrics within time window
        recent_metrics = [
            metric for metric in self.metrics_buffer
            if metric.timestamp >= window_start
        ]
        
        # Group metrics by name and type
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[(metric.name, metric.metric_type)].append(metric.value)
            
        # Calculate statistics
        summary = {}
        for (name, metric_type), values in grouped_metrics.items():
            if NUMPY_AVAILABLE and values:
                values_array = np.array(values)
                summary[name] = {
                    "type": metric_type.value,
                    "count": len(values),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "mean": float(np.mean(values_array)),
                    "median": float(np.median(values_array)),
                    "std": float(np.std(values_array)),
                    "p95": float(np.percentile(values_array, 95)),
                    "p99": float(np.percentile(values_array, 99))
                }
            else:
                summary[name] = {
                    "type": metric_type.value,
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "mean": sum(values) / len(values) if values else 0
                }
                
        return summary

class GPUUtilizationMonitor:
    """
    GPU utilization and memory monitoring
    """
    
    def __init__(self):
        self.gpu_available = False
        self.device_count = 0
        self.monitoring_enabled = False
        self._initialize_gpu_monitoring()
        
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        if not GPU_MONITORING_AVAILABLE:
            logger.info("GPU monitoring not available - install pynvml")
            return
            
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.gpu_available = True
            self.monitoring_enabled = True
            
            logger.info(f"Initialized GPU monitoring for {self.device_count} devices")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU utilization statistics"""
        if not self.gpu_available:
            return {"error": "GPU monitoring not available"}
            
        try:
            gpu_stats = {}
            
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device name
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                    
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0
                    
                gpu_stats[f"gpu_{i}"] = {
                    "name": name,
                    "gpu_utilization_percent": util.gpu,
                    "memory_utilization_percent": util.memory,
                    "memory_total_mb": mem_info.total / (1024 * 1024),
                    "memory_used_mb": mem_info.used / (1024 * 1024),
                    "memory_free_mb": mem_info.free / (1024 * 1024),
                    "temperature_celsius": temp,
                    "power_usage_watts": power
                }
                
            return gpu_stats
            
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return {"error": str(e)}
            
    async def monitor_gpu_continuously(self, interval_seconds: float = 1.0) -> AsyncGenerator[Dict[str, Any], None]:
        """Continuously monitor GPU usage"""
        while self.monitoring_enabled:
            stats = self.get_gpu_stats()
            yield stats
            await asyncio.sleep(interval_seconds)

class MemoryAllocationTracker:
    """
    Memory allocation tracking and leak detection
    """
    
    def __init__(self):
        self.allocation_history: deque = deque(maxlen=1000)
        self.tracking_enabled = False
        self.baseline_snapshot = None
        self._lock = threading.RLock()
        
    def start_tracking(self):
        """Start memory allocation tracking"""
        if not TRACEMALLOC_AVAILABLE:
            logger.warning("Memory tracking not available - tracemalloc not found")
            return False
            
        tracemalloc.start()
        self.baseline_snapshot = tracemalloc.take_snapshot()
        self.tracking_enabled = True
        
        logger.info("Started memory allocation tracking")
        return True
        
    def stop_tracking(self):
        """Stop memory allocation tracking"""
        if TRACEMALLOC_AVAILABLE and self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            logger.info("Stopped memory allocation tracking")
            
    def take_memory_snapshot(self) -> Dict[str, Any]:
        """Take memory usage snapshot"""
        if not self.tracking_enabled:
            return {"error": "Memory tracking not enabled"}
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:20]
        
        snapshot_data = {
            "timestamp": time.time(),
            "total_traces": len(snapshot.traces),
            "top_allocators": []
        }
        
        for stat in top_stats:
            snapshot_data["top_allocators"].append({
                "filename": stat.traceback.format()[-1],
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count
            })
            
        with self._lock:
            self.allocation_history.append(snapshot_data)
            
        return snapshot_data
        
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks"""
        if not self.tracking_enabled or not self.baseline_snapshot:
            return {"error": "Memory tracking not active"}
            
        current_snapshot = tracemalloc.take_snapshot()
        top_stats = current_snapshot.compare_to(self.baseline_snapshot, 'lineno')
        
        potential_leaks = []
        for stat in top_stats[:10]:
            if stat.size_diff > 1024 * 1024:  # > 1MB difference
                potential_leaks.append({
                    "filename": stat.traceback.format()[-1],
                    "size_diff_mb": stat.size_diff / (1024 * 1024),
                    "count_diff": stat.count_diff
                })
                
        return {
            "timestamp": time.time(),
            "potential_leaks": potential_leaks,
            "total_size_increase_mb": sum(stat.size_diff for stat in top_stats if stat.size_diff > 0) / (1024 * 1024)
        }

class PerformanceRegressionDetector:
    """
    Detect performance regressions using statistical analysis
    """
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity  # 10% change threshold
        self.baseline_metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_metrics: Dict[str, List[float]] = defaultdict(list)
        self.regression_alerts: List[PerformanceAlert] = []
        self._lock = threading.RLock()
        
    def add_baseline_metric(self, metric_name: str, value: float):
        """Add baseline metric value"""
        with self._lock:
            self.baseline_metrics[metric_name].append(value)
            
            # Keep only recent baselines (last 1000 values)
            if len(self.baseline_metrics[metric_name]) > 1000:
                self.baseline_metrics[metric_name] = self.baseline_metrics[metric_name][-1000:]
                
    def add_current_metric(self, metric_name: str, value: float):
        """Add current metric value and check for regression"""
        with self._lock:
            self.current_metrics[metric_name].append(value)
            
            # Keep only recent measurements
            if len(self.current_metrics[metric_name]) > 100:
                self.current_metrics[metric_name] = self.current_metrics[metric_name][-100:]
                
            # Check for regression if we have baseline data
            if metric_name in self.baseline_metrics and len(self.baseline_metrics[metric_name]) >= 10:
                self._check_for_regression(metric_name)
                
    def _check_for_regression(self, metric_name: str):
        """Check if metric shows performance regression"""
        if not NUMPY_AVAILABLE:
            return
            
        baseline_values = np.array(self.baseline_metrics[metric_name])
        current_values = np.array(self.current_metrics[metric_name])
        
        if len(current_values) < 10:
            return
            
        # Calculate statistics
        baseline_mean = np.mean(baseline_values)
        current_mean = np.mean(current_values[-10:])  # Last 10 measurements
        
        # Check for significant change
        if baseline_mean > 0:
            change_ratio = (current_mean - baseline_mean) / baseline_mean
            
            if abs(change_ratio) > self.sensitivity:
                severity = AlertSeverity.WARNING
                if abs(change_ratio) > 0.25:  # 25% change
                    severity = AlertSeverity.CRITICAL
                elif abs(change_ratio) > 0.50:  # 50% change
                    severity = AlertSeverity.EMERGENCY
                    
                alert = PerformanceAlert(
                    metric_name=metric_name,
                    severity=severity,
                    threshold=baseline_mean * (1 + self.sensitivity),
                    current_value=current_mean,
                    message=f"Performance regression detected: {change_ratio*100:.1f}% change",
                    timestamp=time.time()
                )
                
                self.regression_alerts.append(alert)
                logger.warning(f"Performance regression detected for {metric_name}: {change_ratio*100:.1f}% change")
                
    def get_regression_report(self) -> Dict[str, Any]:
        """Get performance regression report"""
        with self._lock:
            recent_alerts = [
                alert for alert in self.regression_alerts
                if time.time() - alert.timestamp < 3600  # Last hour
            ]
            
            return {
                "total_alerts": len(self.regression_alerts),
                "recent_alerts": len(recent_alerts),
                "alerts_by_severity": {
                    severity.value: len([a for a in recent_alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
                "recent_alert_details": [asdict(alert) for alert in recent_alerts[-10:]]
            }

class UltraPerformanceMetrics:
    """
    Ultra-performance metrics collector with microsecond precision
    """
    
    def __init__(self):
        self.profiler = RealTimeProfiler()
        self.gpu_monitor = GPUUtilizationMonitor()
        self.memory_tracker = MemoryAllocationTracker()
        self.regression_detector = PerformanceRegressionDetector()
        self.system_monitor = SystemResourceMonitor()
        
        # Start background monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def start_monitoring(self, interval_seconds: float = 1.0):
        """Start comprehensive performance monitoring"""
        self.monitoring_active = True
        self.memory_tracker.start_tracking()
        
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        logger.info("Started ultra-performance monitoring")
        
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        self.memory_tracker.stop_tracking()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        logger.info("Stopped ultra-performance monitoring")
        
    async def _monitoring_loop(self, interval_seconds: float):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_snapshot = self.system_monitor.get_system_snapshot()
                
                # Record system metrics
                self.profiler.record_metric(
                    "system.cpu_percent",
                    MetricType.CPU_USAGE,
                    system_snapshot.cpu_percent,
                    "percent"
                )
                
                self.profiler.record_metric(
                    "system.memory_usage_mb",
                    MetricType.MEMORY_USAGE,
                    system_snapshot.memory_usage_mb,
                    "megabytes"
                )
                
                # Collect GPU metrics if available
                if self.gpu_monitor.gpu_available:
                    gpu_stats = self.gpu_monitor.get_gpu_stats()
                    for gpu_id, stats in gpu_stats.items():
                        if isinstance(stats, dict) and "gpu_utilization_percent" in stats:
                            self.profiler.record_metric(
                                f"gpu.{gpu_id}.utilization_percent",
                                MetricType.GPU_USAGE,
                                stats["gpu_utilization_percent"],
                                "percent"
                            )
                            
                # Take memory snapshot
                memory_snapshot = self.memory_tracker.take_memory_snapshot()
                if "error" not in memory_snapshot:
                    self.profiler.record_metric(
                        "memory.total_traces",
                        MetricType.MEMORY_USAGE,
                        memory_snapshot["total_traces"],
                        "count"
                    )
                    
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)
                
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "timestamp": time.time(),
            "profiler_metrics": self.profiler.get_metrics_summary(),
            "gpu_stats": self.gpu_monitor.get_gpu_stats(),
            "memory_leaks": self.memory_tracker.detect_memory_leaks(),
            "regression_report": self.regression_detector.get_regression_report(),
            "system_snapshot": asdict(self.system_monitor.get_system_snapshot())
        }

class SystemResourceMonitor:
    """
    System resource monitoring (CPU, memory, disk, network)
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.system_stats_history: deque = deque(maxlen=3600)  # 1 hour at 1 second intervals
        
    def get_system_snapshot(self) -> SystemResourceSnapshot:
        """Get current system resource snapshot"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            try:
                io_counters = self.process.io_counters()
                disk_read_mb = io_counters.read_bytes / (1024 * 1024)
                disk_write_mb = io_counters.write_bytes / (1024 * 1024)
            except (psutil.AccessDenied, AttributeError):
                disk_read_mb = disk_write_mb = 0
                
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                network_sent_mb = net_io.bytes_sent / (1024 * 1024)
                network_recv_mb = net_io.bytes_recv / (1024 * 1024)
            except AttributeError:
                network_sent_mb = network_recv_mb = 0
                
            snapshot = SystemResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_usage_mb=memory_usage_mb,
                memory_percent=memory_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_sent_mb=network_sent_mb,
                network_io_recv_mb=network_recv_mb
            )
            
            self.system_stats_history.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to get system snapshot: {e}")
            return SystemResourceSnapshot(timestamp=time.time(), cpu_percent=0, memory_usage_mb=0, memory_percent=0,
                                        disk_io_read_mb=0, disk_io_write_mb=0, network_io_sent_mb=0, network_io_recv_mb=0)

# Global instances
real_time_profiler = RealTimeProfiler()
gpu_utilization_monitor = GPUUtilizationMonitor()
memory_allocation_tracker = MemoryAllocationTracker()
performance_regression_detector = PerformanceRegressionDetector()
ultra_performance_metrics = UltraPerformanceMetrics()

# Decorators and context managers
def profile_performance(metric_name: str):
    """Decorator for profiling function performance"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with real_time_profiler.profile_function(metric_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def monitor_latency(operation_name: str):
    """Context manager for monitoring operation latency"""
    start_time = time.perf_counter_ns()
    try:
        yield
    finally:
        end_time = time.perf_counter_ns()
        latency_us = (end_time - start_time) / 1000
        
        real_time_profiler.record_metric(
            f"latency.{operation_name}",
            MetricType.LATENCY,
            latency_us,
            "microseconds"
        )
        
        # Check for regression
        performance_regression_detector.add_current_metric(
            f"latency.{operation_name}",
            latency_us
        )

# Utility functions
async def benchmark_function(func: Callable, iterations: int = 1000, *args, **kwargs) -> Dict[str, Any]:
    """Benchmark function performance"""
    latencies = []
    
    for i in range(iterations):
        start_time = time.perf_counter_ns()
        
        if asyncio.iscoroutinefunction(func):
            await func(*args, **kwargs)
        else:
            func(*args, **kwargs)
            
        end_time = time.perf_counter_ns()
        latency_us = (end_time - start_time) / 1000
        latencies.append(latency_us)
        
    if NUMPY_AVAILABLE:
        latencies_array = np.array(latencies)
        return {
            "iterations": iterations,
            "min_latency_us": float(np.min(latencies_array)),
            "max_latency_us": float(np.max(latencies_array)),
            "mean_latency_us": float(np.mean(latencies_array)),
            "median_latency_us": float(np.median(latencies_array)),
            "std_latency_us": float(np.std(latencies_array)),
            "p95_latency_us": float(np.percentile(latencies_array, 95)),
            "p99_latency_us": float(np.percentile(latencies_array, 99)),
            "p999_latency_us": float(np.percentile(latencies_array, 99.9))
        }
    else:
        return {
            "iterations": iterations,
            "min_latency_us": min(latencies),
            "max_latency_us": max(latencies),
            "mean_latency_us": sum(latencies) / len(latencies),
            "total_time_ms": sum(latencies) / 1000
        }
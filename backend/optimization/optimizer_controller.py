"""
CPU Optimizer Controller for M4 Max Trading Platform
====================================================

Main controller that orchestrates CPU core optimization by integrating:
- CPU Affinity Manager
- Process Manager
- GCD Scheduler
- Performance Monitor
- Workload Classifier
"""

import os
import sys
import time
import yaml
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from .cpu_affinity import CPUAffinityManager, WorkloadPriority
from .process_manager import ProcessManager, ProcessClass
from .gcd_scheduler import GCDScheduler, QoSClass
from .performance_monitor import PerformanceMonitor, MetricType, AlertLevel
from .workload_classifier import WorkloadClassifier, WorkloadCategory, WorkloadFeatures
from .container_cpu_optimizer import ContainerCPUOptimizer, ContainerPriority

logger = logging.getLogger(__name__)

class OptimizationMode(Enum):
    """System optimization modes"""
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"  
    POWER_SAVE = "power_save"
    EMERGENCY = "emergency"

@dataclass
class SystemHealth:
    """Overall system health metrics"""
    cpu_utilization: float
    memory_utilization: float
    thermal_state: str
    active_alerts: int
    critical_alerts: int
    optimization_score: float

class OptimizerController:
    """
    Main controller for CPU optimization system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "cpu_config.yml"
        )
        self.config: Dict = {}
        self.optimization_mode = OptimizationMode.BALANCED
        
        # Core components
        self.cpu_affinity_manager: Optional[CPUAffinityManager] = None
        self.process_manager: Optional[ProcessManager] = None
        self.gcd_scheduler: Optional[GCDScheduler] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.workload_classifier: Optional[WorkloadClassifier] = None
        self.container_optimizer: Optional[ContainerCPUOptimizer] = None
        
        # State tracking
        self.is_initialized = False
        self.is_running = False
        self._shutdown_event = threading.Event()
        
        # Management threads
        self._optimization_thread: Optional[threading.Thread] = None
        self._health_monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        self._mode_change_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "initialization_time": 0.0,
            "optimizations_performed": 0,
            "processes_managed": 0,
            "alerts_generated": 0,
            "mode_changes": 0
        }
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> bool:
        """Load optimization configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Loaded optimization configuration from {self.config_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            # Use default configuration
            self._create_default_config()
            return False
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration YAML: {e}")
            self._create_default_config()
            return False
    
    def _create_default_config(self) -> None:
        """Create default configuration"""
        self.config = {
            "architecture": {
                "total_cores": 16,
                "performance_cores": {"count": 12, "ids": list(range(12))},
                "efficiency_cores": {"count": 4, "ids": list(range(12, 16))}
            },
            "performance_targets": {
                "order_execution": {"max_latency_ms": 1.0},
                "market_data_processing": {"max_latency_ms": 5.0},
                "risk_calculation": {"max_latency_ms": 10.0}
            }
        }
        logger.warning("Using default configuration")
    
    def initialize(self) -> bool:
        """Initialize all optimization components"""
        start_time = time.time()
        
        try:
            logger.info("Initializing CPU optimization system...")
            
            # 1. Initialize CPU Affinity Manager
            self.cpu_affinity_manager = CPUAffinityManager()
            logger.info("✓ CPU Affinity Manager initialized")
            
            # 2. Initialize Process Manager
            self.process_manager = ProcessManager(self.cpu_affinity_manager)
            logger.info("✓ Process Manager initialized")
            
            # 3. Initialize GCD Scheduler
            self.gcd_scheduler = GCDScheduler()
            logger.info("✓ GCD Scheduler initialized")
            
            # 4. Initialize Performance Monitor
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.add_alert_callback(self._handle_performance_alert)
            logger.info("✓ Performance Monitor initialized")
            
            # 5. Initialize Workload Classifier
            self.workload_classifier = WorkloadClassifier()
            logger.info("✓ Workload Classifier initialized")
            
            # 6. Initialize Container CPU Optimizer
            self.container_optimizer = ContainerCPUOptimizer(
                self.cpu_affinity_manager,
                self.performance_monitor,
                self.workload_classifier
            )
            logger.info("✓ Container CPU Optimizer initialized")
            
            # 7. Configure system based on loaded configuration
            self._apply_configuration()
            
            # 8. Start management threads
            self._start_management_threads()
            
            self.is_initialized = True
            self.is_running = True
            
            initialization_time = time.time() - start_time
            self.stats["initialization_time"] = initialization_time
            
            logger.info(f"CPU optimization system initialized successfully in {initialization_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization system: {e}")
            return False
    
    def _apply_configuration(self) -> None:
        """Apply loaded configuration to all components"""
        try:
            # Apply performance monitor thresholds
            if "monitoring" in self.config:
                monitoring_config = self.config["monitoring"]
                
                if "metrics" in monitoring_config:
                    metrics_config = monitoring_config["metrics"]
                    
                    # Set CPU utilization thresholds
                    if "cpu_utilization" in metrics_config:
                        cpu_config = metrics_config["cpu_utilization"]
                        if "alert_thresholds" in cpu_config:
                            thresholds = cpu_config["alert_thresholds"]
                            self.performance_monitor.set_threshold(
                                MetricType.CPU_UTILIZATION,
                                thresholds.get("warning", 80.0),
                                thresholds.get("critical", 95.0)
                            )
                    
                    # Set memory thresholds
                    if "memory_usage" in metrics_config:
                        mem_config = metrics_config["memory_usage"]
                        if "alert_thresholds" in mem_config:
                            thresholds = mem_config["alert_thresholds"]
                            self.performance_monitor.set_threshold(
                                MetricType.MEMORY_USAGE,
                                thresholds.get("warning", 80.0),
                                thresholds.get("critical", 95.0)
                            )
                    
                    # Set latency thresholds
                    if "latency" in metrics_config:
                        latency_config = metrics_config["latency"]
                        if "alert_thresholds" in latency_config:
                            thresholds = latency_config["alert_thresholds"]
                            self.performance_monitor.set_threshold(
                                MetricType.LATENCY,
                                thresholds.get("warning", 10.0),
                                thresholds.get("critical", 50.0)
                            )
            
            logger.info("Applied configuration to all components")
            
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
    
    def _start_management_threads(self) -> None:
        """Start background management threads"""
        
        # Optimization thread - performs periodic optimizations
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True,
            name="OptimizationController"
        )
        self._optimization_thread.start()
        
        # Health monitoring thread
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._health_monitor_thread.start()
        
        logger.info("Started management threads")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop"""
        while not self._shutdown_event.is_set():
            try:
                # Perform periodic optimization
                self._perform_optimization()
                
                # Check if mode change is needed
                self._check_mode_change()
                
                # Sleep for optimization interval
                time.sleep(30)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _health_monitoring_loop(self) -> None:
        """Health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Get system health
                health = self.get_system_health()
                
                # Check for emergency conditions
                if health.optimization_score < 0.3 or health.critical_alerts > 0:
                    if self.optimization_mode != OptimizationMode.EMERGENCY:
                        self.set_optimization_mode(OptimizationMode.EMERGENCY)
                
                # Check for high load conditions
                elif health.cpu_utilization > 85.0 and self.optimization_mode == OptimizationMode.BALANCED:
                    self.set_optimization_mode(OptimizationMode.HIGH_PERFORMANCE)
                
                # Check for low load conditions  
                elif health.cpu_utilization < 30.0 and self.optimization_mode == OptimizationMode.HIGH_PERFORMANCE:
                    self.set_optimization_mode(OptimizationMode.BALANCED)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(30)
    
    def _perform_optimization(self) -> None:
        """Perform system optimization"""
        try:
            # Rebalance CPU workloads
            rebalance_stats = self.cpu_affinity_manager.rebalance_workloads()
            
            if rebalance_stats["moved"] > 0:
                logger.info(f"Rebalanced {rebalance_stats['moved']} processes")
                self.stats["optimizations_performed"] += 1
            
        except Exception as e:
            logger.error(f"Error performing optimization: {e}")
    
    def _check_mode_change(self) -> None:
        """Check if optimization mode should be changed"""
        try:
            current_hour = time.localtime().tm_hour
            
            # Market hours logic (simplified - US market 9:30 AM - 4 PM ET)
            if 9 <= current_hour <= 16:  # Market hours
                if self.optimization_mode != OptimizationMode.HIGH_PERFORMANCE:
                    # Don't automatically switch during market hours unless emergency
                    pass
            else:  # After market hours
                if self.optimization_mode == OptimizationMode.HIGH_PERFORMANCE:
                    # Switch to balanced mode after hours
                    self.set_optimization_mode(OptimizationMode.BALANCED)
                    
        except Exception as e:
            logger.error(f"Error checking mode change: {e}")
    
    def _handle_performance_alert(self, alert) -> None:
        """Handle performance alerts from monitor"""
        self.stats["alerts_generated"] += 1
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Take automatic actions based on alert level
        if alert.level == AlertLevel.CRITICAL:
            if alert.metric_type == MetricType.CPU_UTILIZATION:
                # Emergency CPU optimization
                self._emergency_cpu_optimization()
            elif alert.metric_type == MetricType.THERMAL:
                # Emergency thermal management
                self._emergency_thermal_management()
    
    def _emergency_cpu_optimization(self) -> None:
        """Emergency CPU optimization actions"""
        logger.warning("Performing emergency CPU optimization")
        
        # Force workload rebalancing
        self.cpu_affinity_manager.rebalance_workloads()
        
        # Switch to emergency mode if not already
        if self.optimization_mode != OptimizationMode.EMERGENCY:
            self.set_optimization_mode(OptimizationMode.EMERGENCY)
    
    def _emergency_thermal_management(self) -> None:
        """Emergency thermal management actions"""
        logger.warning("Performing emergency thermal management")
        
        # This would implement thermal throttling in a real system
        # For now, just switch to power save mode
        self.set_optimization_mode(OptimizationMode.POWER_SAVE)
    
    # Public API methods
    
    def register_process(
        self,
        pid: int,
        process_class: ProcessClass,
        priority: Optional[WorkloadPriority] = None,
        preferred_cores: Optional[List[int]] = None
    ) -> bool:
        """Register a process for optimization"""
        if not self.is_initialized:
            logger.error("Optimization system not initialized")
            return False
        
        success = self.process_manager.register_process(
            pid, process_class, priority, preferred_cores
        )
        
        if success:
            self.stats["processes_managed"] += 1
        
        return success
    
    def classify_and_optimize_workload(
        self,
        function_name: str,
        module_name: str,
        execution_context: Optional[Dict] = None
    ) -> Tuple[WorkloadCategory, WorkloadPriority]:
        """Classify a workload and return optimization recommendations"""
        if not self.is_initialized:
            logger.error("Optimization system not initialized")
            return WorkloadCategory.ANALYTICS, WorkloadPriority.NORMAL
        
        # Extract features
        features = self.workload_classifier.extract_features(
            function_name=function_name,
            module_name=module_name,
            execution_time_ms=execution_context.get("execution_time_ms", 0.0) if execution_context else 0.0,
            cpu_usage=execution_context.get("cpu_usage", 0.0) if execution_context else 0.0,
            memory_usage_mb=execution_context.get("memory_mb", 0.0) if execution_context else 0.0
        )
        
        # Classify workload
        category, confidence = self.workload_classifier.classify_workload(features)
        
        # Get priority recommendation
        priority = self.workload_classifier.get_workload_priority(category)
        
        logger.debug(f"Classified {function_name} as {category.value} "
                   f"with priority {priority.name} (confidence: {confidence:.2f})")
        
        return category, priority
    
    def dispatch_task(
        self,
        queue_name: str,
        task_func: Callable,
        *args,
        qos_class: Optional[QoSClass] = None,
        **kwargs
    ) -> str:
        """Dispatch a task using GCD scheduler"""
        if not self.is_initialized:
            logger.error("Optimization system not initialized")
            return ""
        
        return self.gcd_scheduler.dispatch_async(queue_name, task_func, *args, **kwargs)
    
    def start_latency_measurement(self, operation_type: str) -> str:
        """Start measuring latency for an operation"""
        if not self.is_initialized:
            logger.error("Optimization system not initialized") 
            return ""
        
        return self.performance_monitor.start_latency_measurement(operation_type)
    
    def end_latency_measurement(
        self,
        operation_id: str,
        success: bool = True,
        error_msg: Optional[str] = None
    ) -> float:
        """End latency measurement"""
        if not self.is_initialized:
            logger.error("Optimization system not initialized")
            return 0.0
        
        return self.performance_monitor.end_latency_measurement(
            operation_id, success, error_msg
        )
    
    def set_optimization_mode(self, mode: OptimizationMode) -> bool:
        """Set system optimization mode"""
        try:
            previous_mode = self.optimization_mode
            self.optimization_mode = mode
            
            logger.info(f"Optimization mode changed: {previous_mode.value} -> {mode.value}")
            
            # Trigger mode change callbacks
            for callback in self._mode_change_callbacks:
                try:
                    callback(previous_mode, mode)
                except Exception as e:
                    logger.error(f"Error in mode change callback: {e}")
            
            self.stats["mode_changes"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error setting optimization mode: {e}")
            return False
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health metrics"""
        if not self.is_initialized:
            return SystemHealth(0, 0, "unknown", 0, 0, 0.0)
        
        try:
            # Get performance stats
            perf_stats = self.performance_monitor.get_system_stats()
            
            # Get CPU and memory utilization
            cpu_util = perf_stats["current_metrics"].get("cpu_utilization", {}).get("value", 0.0)
            mem_util = perf_stats["current_metrics"].get("memory_usage", {}).get("value", 0.0)
            
            # Get alert counts
            active_alerts = perf_stats["alerts"]["active"]
            critical_alerts = perf_stats["alerts"]["critical"]
            
            # Calculate optimization score (0-1, higher is better)
            optimization_score = self._calculate_optimization_score(perf_stats)
            
            # Determine thermal state
            thermal_state = "normal"
            thermal_metric = perf_stats["current_metrics"].get("thermal", {})
            if thermal_metric:
                temp = thermal_metric.get("value", 0.0)
                if temp > 85:
                    thermal_state = "critical"
                elif temp > 80:
                    thermal_state = "warning"
            
            return SystemHealth(
                cpu_utilization=cpu_util,
                memory_utilization=mem_util,
                thermal_state=thermal_state,
                active_alerts=active_alerts,
                critical_alerts=critical_alerts,
                optimization_score=optimization_score
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(0, 0, "error", 0, 0, 0.0)
    
    def _calculate_optimization_score(self, perf_stats: Dict) -> float:
        """Calculate overall optimization score (0-1)"""
        try:
            score = 1.0
            
            # Penalize high CPU utilization
            cpu_util = perf_stats["current_metrics"].get("cpu_utilization", {}).get("value", 0.0)
            if cpu_util > 80:
                score *= (100 - cpu_util) / 20  # Scale from 1.0 to 0.0 as CPU goes from 80% to 100%
            
            # Penalize high memory utilization
            mem_util = perf_stats["current_metrics"].get("memory_usage", {}).get("value", 0.0)
            if mem_util > 80:
                score *= (100 - mem_util) / 20
            
            # Penalize active alerts
            active_alerts = perf_stats["alerts"]["active"]
            if active_alerts > 0:
                score *= 1 / (1 + active_alerts * 0.1)
            
            # Penalize critical alerts heavily
            critical_alerts = perf_stats["alerts"]["critical"]
            if critical_alerts > 0:
                score *= 1 / (1 + critical_alerts * 0.5)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        try:
            stats = {
                "system_info": {
                    "optimization_mode": self.optimization_mode.value,
                    "is_running": self.is_running,
                    "initialization_time": self.stats["initialization_time"],
                    "uptime_seconds": time.time() - (time.time() - self.stats["initialization_time"])
                },
                "controller_stats": dict(self.stats),
                "cpu_affinity": self.cpu_affinity_manager.get_system_info(),
                "process_management": self.process_manager.get_process_stats(),
                "gcd_scheduler": self.gcd_scheduler.get_system_stats(),
                "performance_monitor": self.performance_monitor.get_system_stats(),
                "workload_classifier": self.workload_classifier.get_classification_stats(),
                "container_optimizer": self.container_optimizer.get_container_stats() if self.container_optimizer else {},
                "system_health": self.get_system_health().__dict__
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stats: {e}")
            return {"error": str(e)}
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for performance alerts"""
        self._alert_callbacks.append(callback)
    
    def add_mode_change_callback(self, callback: Callable) -> None:
        """Add callback for optimization mode changes"""
        self._mode_change_callbacks.append(callback)
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the optimization system"""
        logger.info("Shutting down CPU optimization system...")
        
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for management threads
        if self._optimization_thread and self._optimization_thread.is_alive():
            self._optimization_thread.join(timeout=timeout / 2)
        
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=timeout / 2)
        
        # Shutdown components
        try:
            if self.container_optimizer:
                self.container_optimizer.shutdown()
            
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            if self.process_manager:
                self.process_manager.shutdown()
            
            if self.cpu_affinity_manager:
                self.cpu_affinity_manager.shutdown()
            
            if self.gcd_scheduler:
                self.gcd_scheduler.shutdown()
                
        except Exception as e:
            logger.error(f"Error during component shutdown: {e}")
        
        logger.info("CPU optimization system shutdown complete")
    
    def export_performance_data(self, output_dir: str, duration_hours: int = 24) -> bool:
        """Export performance data for analysis"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            end_time = time.time()
            start_time = end_time - (duration_hours * 3600)
            
            # Export metrics
            metrics_file = os.path.join(output_dir, "performance_metrics.json")
            self.performance_monitor.export_metrics(start_time, end_time, metrics_file)
            
            # Export workload classification data
            classifier_file = os.path.join(output_dir, "workload_classifications.json")
            self.workload_classifier.export_training_data(classifier_file)
            
            # Export comprehensive stats
            stats_file = os.path.join(output_dir, "system_stats.json")
            import json
            with open(stats_file, 'w') as f:
                json.dump(self.get_comprehensive_stats(), f, indent=2, default=str)
            
            logger.info(f"Exported performance data to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return False
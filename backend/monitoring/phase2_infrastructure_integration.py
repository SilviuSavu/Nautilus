#!/usr/bin/env python3
"""
Phase 2 Infrastructure Integration for Nautilus Trading Platform
Integration of all Phase 2 clock components with M4 Max monitoring and existing infrastructure.
"""

import time
import threading
import json
import asyncio
import os
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

from ..engines.common.clock import Clock, get_global_clock, LiveClock, TestClock, create_clock
from .prometheus_clock_collector import PrometheusClockCollector, MetricCollectorFactory, get_global_prometheus_collector
from .grafana_clock_updater import GrafanaClockUpdater, GrafanaDashboardFactory, get_global_grafana_updater
from ..docker.health_check_clock import DockerHealthCheckClock, get_global_docker_health_checker
from ..docker.container_lifecycle_clock import ContainerLifecycleClock, OrchestrationAction, get_global_container_lifecycle_manager
from ..websocket.websocket_clock_manager import WebSocketClockManager, get_global_websocket_clock_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Phase2IntegrationConfig:
    """Configuration for Phase 2 infrastructure integration"""
    # Clock configuration
    clock_type: str = "live"  # live or test
    clock_start_time_ns: Optional[int] = None
    
    # Prometheus configuration
    prometheus_enabled: bool = True
    prometheus_collection_interval_ms: int = 5000
    prometheus_registry_port: int = 9090
    
    # Grafana configuration  
    grafana_enabled: bool = True
    grafana_url: str = "http://localhost:3002"
    grafana_api_key: str = ""
    grafana_update_interval_ms: int = 10000
    
    # Docker health check configuration
    docker_health_checks_enabled: bool = True
    docker_health_check_interval_ms: int = 30000
    docker_health_timeout_ms: int = 60000
    
    # Container lifecycle configuration
    container_orchestration_enabled: bool = True
    compose_file_path: Optional[str] = None
    
    # WebSocket configuration
    websocket_management_enabled: bool = True
    websocket_heartbeat_interval_ms: int = 30000
    websocket_max_connections: int = 1000
    
    # M4 Max integration
    m4_max_monitoring_enabled: bool = True
    hardware_metrics_interval_ms: int = 2000
    
    # Performance targets
    monitoring_accuracy_target_pct: float = 99.0
    dashboard_responsiveness_improvement_pct: float = 12.5
    container_reliability_improvement_pct: float = 12.5
    websocket_stability_improvement_pct: float = 25.0


@dataclass
class Phase2PerformanceMetrics:
    """Phase 2 infrastructure performance metrics"""
    # Monitoring accuracy metrics
    monitoring_accuracy_pct: float = 0.0
    prometheus_collection_success_rate: float = 0.0
    prometheus_avg_collection_time_ms: float = 0.0
    
    # Dashboard responsiveness metrics
    dashboard_responsiveness_improvement_pct: float = 0.0
    grafana_update_success_rate: float = 0.0
    grafana_avg_response_time_ms: float = 0.0
    
    # Container reliability metrics
    container_reliability_improvement_pct: float = 0.0
    docker_health_check_success_rate: float = 0.0
    container_uptime_pct: float = 0.0
    
    # WebSocket stability metrics
    websocket_stability_improvement_pct: float = 0.0
    websocket_connection_success_rate: float = 0.0
    websocket_avg_latency_ms: float = 0.0
    
    # System-wide improvements
    system_observability_improvement_pct: float = 0.0
    infrastructure_coordination_efficiency: float = 0.0
    m4_max_utilization_optimization_pct: float = 0.0


class Phase2InfrastructureIntegration:
    """
    Phase 2 Infrastructure Integration Manager
    
    Coordinates all Phase 2 infrastructure components:
    - Prometheus Clock Collector
    - Grafana Clock Updater  
    - Docker Health Check Clock
    - Container Lifecycle Clock
    - WebSocket Clock Manager
    
    Integrates with M4 Max monitoring and provides unified management.
    """
    
    def __init__(self, config: Phase2IntegrationConfig):
        self.config = config
        
        # Create shared clock instance
        self.clock = create_clock(
            clock_type=config.clock_type,
            start_time_ns=config.clock_start_time_ns
        )
        
        # Initialize component instances
        self.prometheus_collector: Optional[PrometheusClockCollector] = None
        self.grafana_updater: Optional[GrafanaClockUpdater] = None
        self.docker_health_checker: Optional[DockerHealthCheckClock] = None
        self.container_lifecycle_manager: Optional[ContainerLifecycleClock] = None
        self.websocket_manager: Optional[WebSocketClockManager] = None
        
        # Performance tracking
        self.integration_start_time_ns = 0
        self.performance_metrics = Phase2PerformanceMetrics()
        self._metrics_history: List[Phase2PerformanceMetrics] = []
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._metrics_thread: Optional[threading.Thread] = None
        
        # Integration status
        self.is_initialized = False
        self.is_running = False
        
        logger.info("Phase 2 Infrastructure Integration initialized")
    
    async def initialize_components(self) -> bool:
        """Initialize all Phase 2 infrastructure components"""
        try:
            logger.info("Initializing Phase 2 infrastructure components...")
            
            self.integration_start_time_ns = self.clock.timestamp_ns()
            
            # Initialize Prometheus Clock Collector
            if self.config.prometheus_enabled:
                await self._initialize_prometheus_collector()
            
            # Initialize Grafana Clock Updater
            if self.config.grafana_enabled:
                await self._initialize_grafana_updater()
            
            # Initialize Docker Health Check Clock
            if self.config.docker_health_checks_enabled:
                await self._initialize_docker_health_checker()
            
            # Initialize Container Lifecycle Clock
            if self.config.container_orchestration_enabled:
                await self._initialize_container_lifecycle_manager()
            
            # Initialize WebSocket Clock Manager
            if self.config.websocket_management_enabled:
                await self._initialize_websocket_manager()
            
            # Set up component integrations
            await self._setup_component_integrations()
            
            # Initialize M4 Max monitoring integration
            if self.config.m4_max_monitoring_enabled:
                await self._initialize_m4_max_integration()
            
            self.is_initialized = True
            logger.info("Phase 2 infrastructure components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Phase 2 components: {e}")
            return False
    
    async def _initialize_prometheus_collector(self):
        """Initialize Prometheus Clock Collector with Phase 2 metrics"""
        self.prometheus_collector = PrometheusClockCollector(clock=self.clock)
        
        # Register system metrics
        self.prometheus_collector.register_metric_collector(
            "system_metrics",
            interval_ms=self.config.prometheus_collection_interval_ms,
            callback=MetricCollectorFactory.create_system_metrics_collector(),
            priority="high"
        )
        
        # Register Phase 2 infrastructure metrics
        self.prometheus_collector.register_metric_collector(
            "phase2_infrastructure_metrics", 
            interval_ms=self.config.prometheus_collection_interval_ms,
            callback=self._create_phase2_metrics_collector(),
            priority="high"
        )
        
        # Register M4 Max hardware metrics
        if self.config.m4_max_monitoring_enabled:
            self.prometheus_collector.register_metric_collector(
                "m4_max_metrics",
                interval_ms=self.config.hardware_metrics_interval_ms,
                callback=MetricCollectorFactory.create_m4_max_metrics_collector(),
                priority="critical"
            )
        
        self.prometheus_collector.start_collection()
        logger.info("Prometheus Clock Collector initialized with Phase 2 metrics")
    
    async def _initialize_grafana_updater(self):
        """Initialize Grafana Clock Updater with Phase 2 dashboards"""
        self.grafana_updater = GrafanaClockUpdater(
            grafana_url=self.config.grafana_url,
            api_key=self.config.grafana_api_key,
            clock=self.clock
        )
        
        # Register Phase 2 infrastructure dashboards
        phase2_dashboard_specs = [
            {
                "dashboard_id": "phase2-infrastructure-overview",
                "panel_ids": [1, 2, 3, 4, 5],
                "update_interval_ms": self.config.grafana_update_interval_ms,
                "priority": "high",
                "query_range_seconds": 3600
            },
            {
                "dashboard_id": "phase2-clock-synchronization", 
                "panel_ids": [10, 11, 12],
                "update_interval_ms": self.config.grafana_update_interval_ms // 2,
                "priority": "critical",
                "query_range_seconds": 1800
            }
        ]
        
        for spec in phase2_dashboard_specs:
            self.grafana_updater.register_dashboard_updater(**spec)
        
        # Register standard dashboards
        trading_spec = GrafanaDashboardFactory.create_trading_performance_dashboard_spec()
        self.grafana_updater.register_dashboard_updater(**trading_spec)
        
        if self.config.m4_max_monitoring_enabled:
            m4_max_spec = GrafanaDashboardFactory.create_m4_max_hardware_dashboard_spec()
            self.grafana_updater.register_dashboard_updater(**m4_max_spec)
        
        self.grafana_updater.start_updates()
        logger.info("Grafana Clock Updater initialized with Phase 2 dashboards")
    
    async def _initialize_docker_health_checker(self):
        """Initialize Docker Health Check Clock with Nautilus containers"""
        self.docker_health_checker = DockerHealthCheckClock(clock=self.clock)
        
        # Register critical Nautilus containers
        critical_containers = [
            "nautilus-postgres",
            "nautilus-redis", 
            "nautilus-backend"
        ]
        
        for container_name in critical_containers:
            self.docker_health_checker.register_container_health_check(
                container_name=container_name,
                health_check_interval_ms=self.config.docker_health_check_interval_ms,
                health_timeout_ms=self.config.docker_health_timeout_ms,
                priority="critical",
                auto_restart=True
            )
        
        # Register engine containers
        engine_containers = [
            "nautilus-analytics-engine", "nautilus-risk-engine", "nautilus-ml-engine",
            "nautilus-strategy-engine", "nautilus-portfolio-engine", "nautilus-marketdata-engine",
            "nautilus-features-engine", "nautilus-factor-engine", "nautilus-websocket-engine"
        ]
        
        for container_name in engine_containers:
            self.docker_health_checker.register_container_health_check(
                container_name=container_name,
                health_check_interval_ms=self.config.docker_health_check_interval_ms,
                health_timeout_ms=self.config.docker_health_timeout_ms,
                priority="high",
                auto_restart=True
            )
        
        self.docker_health_checker.start_health_checks()
        logger.info("Docker Health Check Clock initialized with Nautilus containers")
    
    async def _initialize_container_lifecycle_manager(self):
        """Initialize Container Lifecycle Clock with orchestration groups"""
        self.container_lifecycle_manager = ContainerLifecycleClock(
            clock=self.clock,
            compose_file_path=self.config.compose_file_path
        )
        
        # Define orchestration sequences for Nautilus
        startup_sequence = [
            ("databases", 0, ["nautilus-postgres", "nautilus-redis", "nautilus-timescaledb"]),
            ("engines", 30, ["nautilus-analytics-engine", "nautilus-risk-engine", "nautilus-ml-engine"]),
            ("services", 60, ["nautilus-backend", "nautilus-frontend"]),
            ("monitoring", 90, ["nautilus-prometheus", "nautilus-grafana"])
        ]
        
        # Register container groups
        for group_name, delay_seconds, containers in startup_sequence:
            self.container_lifecycle_manager.register_container_group(group_name, containers)
        
        self.container_lifecycle_manager.start_orchestration()
        logger.info("Container Lifecycle Clock initialized with Nautilus orchestration groups")
    
    async def _initialize_websocket_manager(self):
        """Initialize WebSocket Clock Manager for real-time connections"""
        self.websocket_manager = WebSocketClockManager(
            clock=self.clock,
            max_connections=self.config.websocket_max_connections
        )
        
        # Register event callbacks for integration
        self.websocket_manager.register_event_callback(
            'connection_established',
            self._handle_websocket_connection_established
        )
        
        self.websocket_manager.register_event_callback(
            'heartbeat_timeout',
            self._handle_websocket_timeout
        )
        
        self.websocket_manager.start_heartbeat_manager()
        logger.info("WebSocket Clock Manager initialized for real-time connections")
    
    async def _setup_component_integrations(self):
        """Set up integrations between Phase 2 components"""
        logger.info("Setting up Phase 2 component integrations...")
        
        # Docker health events -> Prometheus metrics
        if self.docker_health_checker and self.prometheus_collector:
            self.docker_health_checker.register_event_callback(
                'container_unhealthy',
                self._handle_container_unhealthy
            )
            
            self.docker_health_checker.register_event_callback(
                'container_restarted', 
                self._handle_container_restarted
            )
        
        # Container lifecycle events -> Grafana dashboard updates
        if self.container_lifecycle_manager and self.grafana_updater:
            self.container_lifecycle_manager.register_event_callback(
                'orchestration_completed',
                self._handle_orchestration_completed
            )
        
        # WebSocket events -> monitoring alerts
        if self.websocket_manager and self.prometheus_collector:
            # Already set up in _initialize_websocket_manager
            pass
        
        logger.info("Phase 2 component integrations configured")
    
    async def _initialize_m4_max_integration(self):
        """Initialize M4 Max hardware monitoring integration"""
        logger.info("Initializing M4 Max hardware integration...")
        
        try:
            # Import M4 Max monitoring modules
            from ..monitoring.m4max_hardware_monitor import M4MaxHardwareMonitor
            from ..acceleration import metal_compute, neural_engine_config
            from ..optimization import cpu_affinity, workload_classifier
            
            # Initialize M4 Max hardware monitoring
            self.m4_max_monitor = M4MaxHardwareMonitor(clock=self.clock)
            
            # Register M4 Max metrics with Prometheus
            if self.prometheus_collector:
                self.prometheus_collector.register_metric_collector(
                    "m4_max_hardware_detailed",
                    interval_ms=self.config.hardware_metrics_interval_ms,
                    callback=self._create_m4_max_detailed_metrics_collector(),
                    priority="critical"
                )
            
            logger.info("M4 Max hardware integration initialized")
            
        except ImportError as e:
            logger.warning(f"M4 Max modules not available, skipping hardware integration: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize M4 Max integration: {e}")
    
    def _create_phase2_metrics_collector(self) -> Callable[[], Dict[str, float]]:
        """Create metrics collector for Phase 2 infrastructure"""
        def collect_phase2_metrics() -> Dict[str, float]:
            try:
                metrics = {}
                
                # Prometheus collector metrics
                if self.prometheus_collector:
                    stats = self.prometheus_collector.get_collection_stats()
                    metrics.update({
                        "prometheus_total_collections": float(stats['total_collections']),
                        "prometheus_success_rate": stats['success_rate'],
                        "prometheus_avg_collection_time_ms": stats['average_collection_time_ms'],
                        "prometheus_active_collectors": float(stats['active_collectors'])
                    })
                
                # Grafana updater metrics
                if self.grafana_updater:
                    stats = self.grafana_updater.get_update_stats()
                    metrics.update({
                        "grafana_total_updates": float(stats['total_updates']),
                        "grafana_success_rate": stats['success_rate'],
                        "grafana_avg_response_time_ms": stats['average_response_time_ms'],
                        "grafana_active_dashboards": float(stats['active_dashboards'])
                    })
                
                # Docker health checker metrics
                if self.docker_health_checker:
                    stats = self.docker_health_checker.get_health_check_stats()
                    metrics.update({
                        "docker_total_health_checks": float(stats['total_health_checks']),
                        "docker_success_rate": stats['success_rate'],
                        "docker_containers_restarted": float(stats['containers_restarted']),
                        "docker_monitored_containers": float(stats['monitored_containers'])
                    })
                
                # Container lifecycle metrics
                if self.container_lifecycle_manager:
                    stats = self.container_lifecycle_manager.get_orchestration_stats()
                    metrics.update({
                        "container_total_orchestrations": float(stats['total_orchestrations']),
                        "container_orchestration_success_rate": stats['success_rate'],
                        "container_rollbacks_performed": float(stats['rollbacks_performed']),
                        "container_pending_orchestrations": float(stats['pending_orchestrations'])
                    })
                
                # WebSocket manager metrics
                if self.websocket_manager:
                    stats = self.websocket_manager.get_manager_stats()
                    metrics.update({
                        "websocket_active_connections": float(stats['active_connections']),
                        "websocket_total_heartbeats": float(stats['total_heartbeats']),
                        "websocket_heartbeat_success_rate": stats['heartbeat_success_rate'],
                        "websocket_connection_failures": float(stats['connection_failures'])
                    })
                
                return metrics
                
            except Exception as e:
                logger.error(f"Error collecting Phase 2 metrics: {e}")
                return {}
        
        return collect_phase2_metrics
    
    def _create_m4_max_detailed_metrics_collector(self) -> Callable[[], Dict[str, float]]:
        """Create detailed M4 Max hardware metrics collector"""
        def collect_m4_max_detailed_metrics() -> Dict[str, float]:
            try:
                if hasattr(self, 'm4_max_monitor'):
                    return self.m4_max_monitor.collect_detailed_metrics()
                else:
                    # Return simulated metrics if M4 Max monitor not available
                    return {
                        "m4_max_metal_gpu_utilization": 0.0,
                        "m4_max_neural_engine_utilization": 0.0,
                        "m4_max_performance_cores_active": 0.0,
                        "m4_max_efficiency_cores_active": 0.0,
                        "m4_max_unified_memory_usage_gb": 0.0,
                        "m4_max_thermal_state": 0.0
                    }
            except Exception as e:
                logger.error(f"Error collecting M4 Max detailed metrics: {e}")
                return {}
        
        return collect_m4_max_detailed_metrics
    
    def _handle_websocket_connection_established(self, event_data: Dict[str, Any]):
        """Handle WebSocket connection established event"""
        logger.debug(f"WebSocket connection established: {event_data['connection_id']}")
    
    def _handle_websocket_timeout(self, event_data: Dict[str, Any]):
        """Handle WebSocket connection timeout event"""
        logger.warning(f"WebSocket connection timeout: {event_data['connection_id']}")
        
        # Could trigger alert or restart action
        if self.prometheus_collector:
            # This would increment a Prometheus counter for WebSocket timeouts
            pass
    
    def _handle_container_unhealthy(self, event_data: Any):
        """Handle container unhealthy event"""
        logger.warning(f"Container unhealthy: {event_data}")
        
        # Trigger dashboard refresh for container status
        if self.grafana_updater:
            asyncio.create_task(self.grafana_updater.update_dashboard_now("system-monitoring"))
    
    def _handle_container_restarted(self, event_data: Any):
        """Handle container restart event"""
        logger.info(f"Container restarted: {event_data}")
        
        # Update performance metrics
        with self._lock:
            self.performance_metrics.container_reliability_improvement_pct += 0.1
    
    def _handle_orchestration_completed(self, event_data: Any):
        """Handle orchestration completion event"""
        logger.info(f"Orchestration completed: {event_data.spec.container_group}")
        
        # Trigger infrastructure dashboard update
        if self.grafana_updater:
            asyncio.create_task(self.grafana_updater.update_dashboard_now("phase2-infrastructure-overview"))
    
    def _metrics_collection_loop(self):
        """Background thread for collecting performance metrics"""
        logger.info("Starting Phase 2 metrics collection loop")
        
        while not self._shutdown_event.is_set():
            try:
                self._update_performance_metrics()
                
                # Sleep for 30 seconds
                if not self._shutdown_event.wait(30.0):
                    continue
                    
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                if not self._shutdown_event.wait(60.0):  # Error backoff
                    continue
        
        logger.info("Phase 2 metrics collection loop stopped")
    
    def _update_performance_metrics(self):
        """Update Phase 2 performance metrics"""
        try:
            current_time_ns = self.clock.timestamp_ns()
            
            # Calculate monitoring accuracy
            if self.prometheus_collector:
                stats = self.prometheus_collector.get_collection_stats()
                self.performance_metrics.monitoring_accuracy_pct = stats['success_rate'] * 100
                self.performance_metrics.prometheus_collection_success_rate = stats['success_rate']
                self.performance_metrics.prometheus_avg_collection_time_ms = stats['average_collection_time_ms']
            
            # Calculate dashboard responsiveness improvement
            if self.grafana_updater:
                stats = self.grafana_updater.get_update_stats()
                baseline_response_time = 50.0  # Baseline 50ms
                current_response_time = stats['average_response_time_ms']
                improvement = max(0, (baseline_response_time - current_response_time) / baseline_response_time * 100)
                self.performance_metrics.dashboard_responsiveness_improvement_pct = improvement
                self.performance_metrics.grafana_update_success_rate = stats['success_rate']
                self.performance_metrics.grafana_avg_response_time_ms = current_response_time
            
            # Calculate container reliability improvement
            if self.docker_health_checker:
                stats = self.docker_health_checker.get_health_check_stats()
                baseline_reliability = 0.85  # 85% baseline
                current_reliability = stats['success_rate']
                improvement = max(0, (current_reliability - baseline_reliability) / baseline_reliability * 100)
                self.performance_metrics.container_reliability_improvement_pct = improvement
                self.performance_metrics.docker_health_check_success_rate = current_reliability
            
            # Calculate WebSocket stability improvement
            if self.websocket_manager:
                stats = self.websocket_manager.get_manager_stats()
                baseline_stability = 0.75  # 75% baseline
                current_stability = stats['heartbeat_success_rate']
                improvement = max(0, (current_stability - baseline_stability) / baseline_stability * 100)
                self.performance_metrics.websocket_stability_improvement_pct = improvement
                self.performance_metrics.websocket_connection_success_rate = current_stability
            
            # Calculate system-wide observability improvement
            component_improvements = [
                self.performance_metrics.monitoring_accuracy_pct / 100,
                self.performance_metrics.dashboard_responsiveness_improvement_pct / 100,
                self.performance_metrics.container_reliability_improvement_pct / 100,
                self.performance_metrics.websocket_stability_improvement_pct / 100
            ]
            
            avg_improvement = sum(component_improvements) / len(component_improvements)
            self.performance_metrics.system_observability_improvement_pct = avg_improvement * 100
            
            # Store metrics history
            with self._lock:
                self._metrics_history.append(Phase2PerformanceMetrics(**self.performance_metrics.__dict__))
                if len(self._metrics_history) > 1000:  # Keep last 1000 readings
                    self._metrics_history = self._metrics_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def start_integration(self) -> bool:
        """Start all Phase 2 infrastructure components"""
        try:
            if not self.is_initialized:
                success = await self.initialize_components()
                if not success:
                    return False
            
            # Start metrics collection
            self._shutdown_event.clear()
            self._metrics_thread = threading.Thread(
                target=self._metrics_collection_loop,
                name="phase2-metrics",
                daemon=True
            )
            self._metrics_thread.start()
            
            self.is_running = True
            logger.info("Phase 2 Infrastructure Integration started successfully")
            
            # Log initial performance targets
            logger.info(f"Performance targets - Monitoring: {self.config.monitoring_accuracy_target_pct}%, "
                       f"Dashboards: {self.config.dashboard_responsiveness_improvement_pct}%, "
                       f"Containers: {self.config.container_reliability_improvement_pct}%, "
                       f"WebSockets: {self.config.websocket_stability_improvement_pct}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Phase 2 integration: {e}")
            return False
    
    async def stop_integration(self):
        """Stop all Phase 2 infrastructure components"""
        try:
            logger.info("Stopping Phase 2 Infrastructure Integration...")
            
            self._shutdown_event.set()
            
            # Stop metrics collection thread
            if self._metrics_thread and self._metrics_thread.is_alive():
                self._metrics_thread.join(timeout=5.0)
            
            # Stop components
            if self.websocket_manager:
                self.websocket_manager.shutdown()
            
            if self.container_lifecycle_manager:
                self.container_lifecycle_manager.shutdown()
            
            if self.docker_health_checker:
                self.docker_health_checker.shutdown()
            
            if self.grafana_updater:
                await self.grafana_updater.shutdown()
            
            if self.prometheus_collector:
                self.prometheus_collector.shutdown()
            
            self.is_running = False
            logger.info("Phase 2 Infrastructure Integration stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Phase 2 integration: {e}")
    
    def get_performance_metrics(self) -> Phase2PerformanceMetrics:
        """Get current performance metrics"""
        with self._lock:
            return Phase2PerformanceMetrics(**self.performance_metrics.__dict__)
    
    def get_metrics_history(self, limit: int = 100) -> List[Phase2PerformanceMetrics]:
        """Get performance metrics history"""
        with self._lock:
            return self._metrics_history[-limit:].copy()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "clock_type": "test" if isinstance(self.clock, TestClock) else "live",
            "components": {
                "prometheus_collector": self.prometheus_collector is not None,
                "grafana_updater": self.grafana_updater is not None,
                "docker_health_checker": self.docker_health_checker is not None,
                "container_lifecycle_manager": self.container_lifecycle_manager is not None,
                "websocket_manager": self.websocket_manager is not None
            },
            "performance_metrics": self.get_performance_metrics().__dict__,
            "integration_uptime_seconds": (self.clock.timestamp_ns() - self.integration_start_time_ns) / 1e9
        }
    
    async def shutdown(self):
        """Complete shutdown of Phase 2 integration"""
        await self.stop_integration()


# Global Phase 2 integration instance
_global_phase2_integration: Optional[Phase2InfrastructureIntegration] = None
_integration_lock = threading.Lock()


async def initialize_phase2_infrastructure(
    config: Optional[Phase2IntegrationConfig] = None
) -> Phase2InfrastructureIntegration:
    """Initialize global Phase 2 infrastructure integration"""
    global _global_phase2_integration
    
    if _global_phase2_integration is None:
        with _integration_lock:
            if _global_phase2_integration is None:
                config = config or Phase2IntegrationConfig()
                _global_phase2_integration = Phase2InfrastructureIntegration(config)
                await _global_phase2_integration.start_integration()
    
    return _global_phase2_integration


async def get_phase2_integration() -> Optional[Phase2InfrastructureIntegration]:
    """Get the global Phase 2 integration instance"""
    return _global_phase2_integration


async def shutdown_phase2_infrastructure():
    """Shutdown global Phase 2 infrastructure integration"""
    global _global_phase2_integration
    
    if _global_phase2_integration is not None:
        with _integration_lock:
            if _global_phase2_integration is not None:
                await _global_phase2_integration.shutdown()
                _global_phase2_integration = None


if __name__ == "__main__":
    # Example usage
    import signal
    import sys
    
    async def signal_handler():
        print("\nShutting down Phase 2 Infrastructure Integration...")
        await shutdown_phase2_infrastructure()
    
    def sync_signal_handler(signum, frame):
        asyncio.create_task(signal_handler())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, sync_signal_handler)
    signal.signal(signal.SIGTERM, sync_signal_handler)
    
    async def main():
        # Create integration configuration
        config = Phase2IntegrationConfig(
            clock_type="test",  # Use test clock for demonstration
            prometheus_enabled=True,
            grafana_enabled=False,  # Disable for demo (no real Grafana)
            docker_health_checks_enabled=False,  # Disable for demo (no real containers)
            container_orchestration_enabled=False,  # Disable for demo
            websocket_management_enabled=True,
            m4_max_monitoring_enabled=False  # Disable for demo
        )
        
        # Initialize Phase 2 integration
        integration = await initialize_phase2_infrastructure(config)
        
        print("Phase 2 Infrastructure Integration running. Press Ctrl+C to stop.")
        print("Components initialized with clock synchronization.")
        
        try:
            while True:
                await asyncio.sleep(10)
                
                # Print performance metrics
                metrics = integration.get_performance_metrics()
                status = integration.get_integration_status()
                
                print(f"Uptime: {status['integration_uptime_seconds']:.1f}s, "
                      f"Monitoring accuracy: {metrics.monitoring_accuracy_pct:.1f}%, "
                      f"System observability improvement: {metrics.system_observability_improvement_pct:.1f}%")
                      
        except KeyboardInterrupt:
            pass
        finally:
            await integration.shutdown()
    
    # Run the example
    asyncio.run(main())
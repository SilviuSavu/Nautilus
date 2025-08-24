#!/usr/bin/env python3
"""
Container Lifecycle Clock for Nautilus Trading Platform
Orchestration timing coordination with clock-synchronized container management for improved system stability.
"""

import time
import threading
import json
import asyncio
import docker
import subprocess
from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import yaml
import os
from pathlib import Path

from ..engines.common.clock import Clock, get_global_clock, LiveClock, TestClock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationAction(Enum):
    """Container orchestration action types"""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    PAUSE = "pause"
    UNPAUSE = "unpause"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ROLLING_UPDATE = "rolling_update"
    HEALTH_CHECK = "health_check"
    

class OrchestrationPhase(Enum):
    """Container orchestration lifecycle phases"""
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation" 
    COMPLETION = "completion"
    ROLLBACK = "rollback"


@dataclass
class ContainerOrchestrationSpec:
    """Specification for container orchestration timing"""
    container_group: str  # Group name (e.g., "engines", "databases", "monitoring")
    container_names: List[str]
    action: OrchestrationAction
    scheduled_time_ns: int
    dependencies: List[str] = field(default_factory=list)  # Other groups that must complete first
    timeout_ns: int = 300_000_000_000  # 5 minutes default
    priority: int = 100  # Lower number = higher priority
    parallel_execution: bool = True
    rollback_on_failure: bool = True
    health_check_after: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result of an orchestration operation"""
    spec: ContainerOrchestrationSpec
    start_time_ns: int
    end_time_ns: int
    phase: OrchestrationPhase
    success: bool
    containers_affected: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    rollback_performed: bool = False
    execution_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerDependency:
    """Container dependency specification"""
    container_name: str
    depends_on: List[str]
    startup_delay_ms: int = 0
    health_check_timeout_ms: int = 30000
    restart_policy: str = "unless-stopped"
    

class ContainerLifecycleClock:
    """
    Clock-synchronized container orchestration manager
    
    Features:
    - Deterministic orchestration timing using shared clock
    - Dependency-aware container startup sequences
    - Parallel and sequential execution modes
    - Rolling update coordination
    - Health check integration
    - Rollback capabilities
    - M4 Max optimized container coordination
    """
    
    def __init__(
        self,
        clock: Optional[Clock] = None,
        docker_client: Optional[docker.DockerClient] = None,
        compose_file_path: Optional[str] = None
    ):
        self.clock = clock or get_global_clock()
        self.docker_client = docker_client or docker.from_env()
        self.compose_file_path = compose_file_path
        
        # Orchestration specifications and results
        self._orchestration_queue: List[ContainerOrchestrationSpec] = []
        self._orchestration_results: List[OrchestrationResult] = []
        self._container_dependencies: Dict[str, ContainerDependency] = {}
        
        # Container group definitions
        self._container_groups: Dict[str, List[str]] = {
            "databases": ["nautilus-postgres", "nautilus-redis", "nautilus-timescaledb"],
            "engines": [
                "nautilus-analytics-engine", "nautilus-risk-engine", "nautilus-ml-engine",
                "nautilus-strategy-engine", "nautilus-portfolio-engine", "nautilus-marketdata-engine",
                "nautilus-features-engine", "nautilus-factor-engine", "nautilus-websocket-engine"
            ],
            "services": ["nautilus-backend", "nautilus-frontend"],
            "monitoring": ["nautilus-prometheus", "nautilus-grafana", "nautilus-exporters"],
            "load-balancing": ["nautilus-nginx", "nautilus-haproxy"]
        }
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._orchestration_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.total_orchestrations = 0
        self.successful_orchestrations = 0
        self.failed_orchestrations = 0
        self.rollbacks_performed = 0
        
        # Event callbacks for orchestration events
        self._event_callbacks: Dict[str, List[Callable]] = {
            'orchestration_started': [],
            'orchestration_completed': [],
            'orchestration_failed': [],
            'rollback_started': [],
            'rollback_completed': [],
            'dependency_satisfied': [],
            'health_check_completed': []
        }
        
        # Load compose file if provided
        if self.compose_file_path:
            self._load_compose_dependencies()
        
        logger.info("Container Lifecycle Clock initialized with deterministic orchestration timing")
    
    def _load_compose_dependencies(self):
        """Load container dependencies from docker-compose file"""
        try:
            if not os.path.exists(self.compose_file_path):
                logger.warning(f"Compose file not found: {self.compose_file_path}")
                return
                
            with open(self.compose_file_path, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            
            for service_name, service_config in services.items():
                depends_on = service_config.get('depends_on', [])
                
                # Handle both list and dict format for depends_on
                if isinstance(depends_on, dict):
                    depends_on = list(depends_on.keys())
                
                restart_policy = service_config.get('restart', 'unless-stopped')
                
                dependency = ContainerDependency(
                    container_name=service_name,
                    depends_on=depends_on,
                    restart_policy=restart_policy
                )
                
                self._container_dependencies[service_name] = dependency
            
            logger.info(f"Loaded {len(self._container_dependencies)} container dependencies from compose file")
            
        except Exception as e:
            logger.error(f"Failed to load compose dependencies: {e}")
    
    def register_container_group(self, group_name: str, container_names: List[str]):
        """Register a container group for orchestration"""
        with self._lock:
            self._container_groups[group_name] = container_names.copy()
        logger.info(f"Registered container group '{group_name}' with {len(container_names)} containers")
    
    def schedule_orchestration(
        self,
        container_group: str,
        action: OrchestrationAction,
        delay_seconds: float = 0.0,
        dependencies: Optional[List[str]] = None,
        timeout_seconds: float = 300.0,
        priority: int = 100,
        parallel_execution: bool = True,
        rollback_on_failure: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Schedule a container orchestration action
        
        Args:
            container_group: Name of container group to orchestrate
            action: Orchestration action to perform
            delay_seconds: Delay before execution (from now)
            dependencies: Other groups that must complete first
            timeout_seconds: Execution timeout
            priority: Execution priority (lower = higher priority)
            parallel_execution: Execute containers in parallel vs sequential
            rollback_on_failure: Perform rollback on failure
            metadata: Additional metadata
            
        Returns:
            True if scheduling successful
        """
        try:
            if container_group not in self._container_groups:
                raise ValueError(f"Unknown container group: {container_group}")
            
            current_time_ns = self.clock.timestamp_ns()
            scheduled_time_ns = current_time_ns + int(delay_seconds * 1_000_000_000)
            timeout_ns = int(timeout_seconds * 1_000_000_000)
            
            spec = ContainerOrchestrationSpec(
                container_group=container_group,
                container_names=self._container_groups[container_group].copy(),
                action=action,
                scheduled_time_ns=scheduled_time_ns,
                dependencies=dependencies or [],
                timeout_ns=timeout_ns,
                priority=priority,
                parallel_execution=parallel_execution,
                rollback_on_failure=rollback_on_failure,
                metadata=metadata or {}
            )
            
            with self._lock:
                # Insert in priority order
                inserted = False
                for i, existing_spec in enumerate(self._orchestration_queue):
                    if spec.priority < existing_spec.priority or \
                       (spec.priority == existing_spec.priority and spec.scheduled_time_ns < existing_spec.scheduled_time_ns):
                        self._orchestration_queue.insert(i, spec)
                        inserted = True
                        break
                
                if not inserted:
                    self._orchestration_queue.append(spec)
            
            logger.info(f"Scheduled {action.value} for group '{container_group}' "
                       f"in {delay_seconds:.1f}s with priority {priority}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule orchestration: {e}")
            return False
    
    def execute_orchestration_now(
        self,
        container_group: str,
        action: OrchestrationAction,
        parallel_execution: bool = True,
        rollback_on_failure: bool = True
    ) -> OrchestrationResult:
        """
        Execute orchestration immediately
        """
        if container_group not in self._container_groups:
            raise ValueError(f"Unknown container group: {container_group}")
        
        current_time_ns = self.clock.timestamp_ns()
        
        spec = ContainerOrchestrationSpec(
            container_group=container_group,
            container_names=self._container_groups[container_group].copy(),
            action=action,
            scheduled_time_ns=current_time_ns,
            parallel_execution=parallel_execution,
            rollback_on_failure=rollback_on_failure
        )
        
        return self._execute_orchestration(spec)
    
    def _execute_orchestration(self, spec: ContainerOrchestrationSpec) -> OrchestrationResult:
        """Execute a container orchestration operation"""
        start_time_ns = self.clock.timestamp_ns()
        
        result = OrchestrationResult(
            spec=spec,
            start_time_ns=start_time_ns,
            end_time_ns=start_time_ns,
            phase=OrchestrationPhase.PREPARATION,
            success=False
        )
        
        try:
            logger.info(f"Starting {spec.action.value} orchestration for group '{spec.container_group}'")
            self._trigger_event('orchestration_started', result)
            
            # Check dependencies
            if not self._check_dependencies(spec):
                raise Exception("Dependencies not satisfied")
            
            # Preparation phase
            result.phase = OrchestrationPhase.PREPARATION
            containers = self._prepare_containers(spec)
            
            # Execution phase
            result.phase = OrchestrationPhase.EXECUTION
            execution_results = self._execute_container_action(spec, containers)
            result.execution_details = execution_results
            result.containers_affected = list(execution_results.keys())
            
            # Check for failures
            failed_containers = [name for name, res in execution_results.items() if not res.get('success', False)]
            
            if failed_containers and spec.rollback_on_failure:
                logger.warning(f"Failures detected in containers: {failed_containers}. Initiating rollback.")
                result.phase = OrchestrationPhase.ROLLBACK
                self._trigger_event('rollback_started', result)
                
                rollback_success = self._perform_rollback(spec, execution_results)
                result.rollback_performed = True
                
                if rollback_success:
                    self._trigger_event('rollback_completed', result)
                    logger.info(f"Rollback completed successfully for group '{spec.container_group}'")
                else:
                    logger.error(f"Rollback failed for group '{spec.container_group}'")
            
            # Validation phase
            result.phase = OrchestrationPhase.VALIDATION
            if spec.health_check_after and not failed_containers:
                self._perform_health_checks(spec, containers)
                self._trigger_event('health_check_completed', result)
            
            # Completion phase
            result.phase = OrchestrationPhase.COMPLETION
            result.success = len(failed_containers) == 0
            result.end_time_ns = self.clock.timestamp_ns()
            
            # Update tracking
            with self._lock:
                self.total_orchestrations += 1
                if result.success:
                    self.successful_orchestrations += 1
                else:
                    self.failed_orchestrations += 1
                
                if result.rollback_performed:
                    self.rollbacks_performed += 1
                
                # Keep results history (last 100)
                self._orchestration_results.append(result)
                if len(self._orchestration_results) > 100:
                    self._orchestration_results = self._orchestration_results[-100:]
            
            duration_ms = (result.end_time_ns - result.start_time_ns) / 1_000_000
            status = "successfully" if result.success else "with failures"
            logger.info(f"Completed {spec.action.value} orchestration for group '{spec.container_group}' "
                       f"{status} in {duration_ms:.2f}ms")
            
            if result.success:
                self._trigger_event('orchestration_completed', result)
            else:
                self._trigger_event('orchestration_failed', result)
            
            return result
            
        except Exception as e:
            result.end_time_ns = self.clock.timestamp_ns()
            result.error_message = str(e)
            result.success = False
            
            with self._lock:
                self.total_orchestrations += 1
                self.failed_orchestrations += 1
                self._orchestration_results.append(result)
            
            logger.error(f"Orchestration failed for group '{spec.container_group}': {e}")
            self._trigger_event('orchestration_failed', result)
            
            return result
    
    def _check_dependencies(self, spec: ContainerOrchestrationSpec) -> bool:
        """Check if all dependencies are satisfied"""
        if not spec.dependencies:
            return True
        
        with self._lock:
            recent_results = self._orchestration_results[-10:]  # Check last 10 results
        
        for dependency in spec.dependencies:
            # Check if dependency group has completed successfully recently
            dependency_satisfied = False
            
            for result in reversed(recent_results):
                if (result.spec.container_group == dependency and
                    result.success and
                    result.end_time_ns <= spec.scheduled_time_ns):
                    dependency_satisfied = True
                    break
            
            if not dependency_satisfied:
                logger.warning(f"Dependency '{dependency}' not satisfied for group '{spec.container_group}'")
                return False
        
        self._trigger_event('dependency_satisfied', spec)
        return True
    
    def _prepare_containers(self, spec: ContainerOrchestrationSpec) -> List[docker.models.containers.Container]:
        """Prepare containers for orchestration"""
        containers = []
        
        for container_name in spec.container_names:
            try:
                container = self.docker_client.containers.get(container_name)
                containers.append(container)
            except docker.errors.NotFound:
                logger.warning(f"Container '{container_name}' not found, skipping")
            except Exception as e:
                logger.error(f"Error getting container '{container_name}': {e}")
        
        return containers
    
    def _execute_container_action(
        self,
        spec: ContainerOrchestrationSpec,
        containers: List[docker.models.containers.Container]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute the specified action on containers"""
        results = {}
        
        if spec.parallel_execution:
            # Execute in parallel using threads
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(containers)) as executor:
                future_to_container = {
                    executor.submit(self._execute_single_container_action, container, spec.action, spec.timeout_ns): container
                    for container in containers
                }
                
                for future in concurrent.futures.as_completed(future_to_container, timeout=spec.timeout_ns / 1e9):
                    container = future_to_container[future]
                    try:
                        result = future.result()
                        results[container.name] = result
                    except Exception as e:
                        results[container.name] = {
                            'success': False,
                            'error': str(e),
                            'action': spec.action.value
                        }
        else:
            # Execute sequentially, respecting dependencies
            ordered_containers = self._order_containers_by_dependencies(containers)
            
            for container in ordered_containers:
                try:
                    result = self._execute_single_container_action(container, spec.action, spec.timeout_ns)
                    results[container.name] = result
                    
                    # Add startup delay if specified
                    if container.name in self._container_dependencies:
                        delay_ms = self._container_dependencies[container.name].startup_delay_ms
                        if delay_ms > 0:
                            time.sleep(delay_ms / 1000.0)
                    
                except Exception as e:
                    results[container.name] = {
                        'success': False,
                        'error': str(e),
                        'action': spec.action.value
                    }
                    
                    # Stop on first failure for sequential execution
                    logger.error(f"Sequential execution stopped due to failure in '{container.name}'")
                    break
        
        return results
    
    def _execute_single_container_action(
        self,
        container: docker.models.containers.Container,
        action: OrchestrationAction,
        timeout_ns: int
    ) -> Dict[str, Any]:
        """Execute action on a single container"""
        start_time = time.perf_counter()
        timeout_seconds = timeout_ns / 1e9
        
        try:
            container.reload()  # Refresh container state
            
            if action == OrchestrationAction.START:
                if container.status != 'running':
                    container.start()
                    # Wait for container to be running
                    self._wait_for_container_status(container, 'running', timeout_seconds)
                
            elif action == OrchestrationAction.STOP:
                if container.status == 'running':
                    container.stop(timeout=int(timeout_seconds))
                    self._wait_for_container_status(container, 'exited', timeout_seconds)
                
            elif action == OrchestrationAction.RESTART:
                container.restart(timeout=int(timeout_seconds))
                self._wait_for_container_status(container, 'running', timeout_seconds)
                
            elif action == OrchestrationAction.PAUSE:
                if container.status == 'running':
                    container.pause()
                    self._wait_for_container_status(container, 'paused', timeout_seconds)
                
            elif action == OrchestrationAction.UNPAUSE:
                if container.status == 'paused':
                    container.unpause()
                    self._wait_for_container_status(container, 'running', timeout_seconds)
                
            else:
                raise ValueError(f"Unsupported action: {action}")
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'action': action.value,
                'duration_ms': duration_ms,
                'final_status': container.status
            }
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return {
                'success': False,
                'action': action.value,
                'error': str(e),
                'duration_ms': duration_ms
            }
    
    def _wait_for_container_status(self, container, expected_status: str, timeout_seconds: float):
        """Wait for container to reach expected status"""
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < timeout_seconds:
            container.reload()
            if container.status == expected_status:
                return
            time.sleep(0.5)  # Check every 500ms
        
        raise TimeoutError(f"Container {container.name} did not reach status '{expected_status}' within {timeout_seconds}s")
    
    def _order_containers_by_dependencies(
        self, 
        containers: List[docker.models.containers.Container]
    ) -> List[docker.models.containers.Container]:
        """Order containers based on their dependencies"""
        # Simple topological sort
        container_names = [c.name for c in containers]
        ordered = []
        remaining = containers.copy()
        
        while remaining:
            # Find containers with no unresolved dependencies
            ready = []
            
            for container in remaining:
                deps = self._container_dependencies.get(container.name, ContainerDependency(container.name, [])).depends_on
                unresolved_deps = [dep for dep in deps if dep in container_names and dep not in [c.name for c in ordered]]
                
                if not unresolved_deps:
                    ready.append(container)
            
            if not ready:
                # Circular dependency or missing dependency, add remaining containers
                logger.warning("Circular or missing dependencies detected, adding remaining containers")
                ordered.extend(remaining)
                break
            
            # Add ready containers to ordered list and remove from remaining
            for container in ready:
                ordered.append(container)
                remaining.remove(container)
        
        return ordered
    
    def _perform_rollback(self, spec: ContainerOrchestrationSpec, execution_results: Dict[str, Dict[str, Any]]) -> bool:
        """Perform rollback operation"""
        try:
            # Define rollback actions
            rollback_actions = {
                OrchestrationAction.START: OrchestrationAction.STOP,
                OrchestrationAction.STOP: OrchestrationAction.START,
                OrchestrationAction.RESTART: OrchestrationAction.STOP,
                OrchestrationAction.PAUSE: OrchestrationAction.UNPAUSE,
                OrchestrationAction.UNPAUSE: OrchestrationAction.PAUSE,
            }
            
            rollback_action = rollback_actions.get(spec.action)
            if not rollback_action:
                logger.warning(f"No rollback action defined for {spec.action.value}")
                return False
            
            # Create rollback spec
            rollback_spec = ContainerOrchestrationSpec(
                container_group=spec.container_group,
                container_names=spec.container_names,
                action=rollback_action,
                scheduled_time_ns=self.clock.timestamp_ns(),
                timeout_ns=spec.timeout_ns,
                parallel_execution=True,  # Rollbacks should be fast
                rollback_on_failure=False  # Don't rollback the rollback
            )
            
            # Execute rollback
            containers = self._prepare_containers(rollback_spec)
            rollback_results = self._execute_container_action(rollback_spec, containers)
            
            # Check if rollback was successful
            failed_rollbacks = [name for name, res in rollback_results.items() if not res.get('success', False)]
            
            if failed_rollbacks:
                logger.error(f"Rollback failed for containers: {failed_rollbacks}")
                return False
            
            logger.info(f"Rollback completed successfully for group '{spec.container_group}'")
            return True
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            return False
    
    def _perform_health_checks(self, spec: ContainerOrchestrationSpec, containers: List):
        """Perform health checks after orchestration"""
        for container in containers:
            try:
                container.reload()
                
                # Basic health check - container should be running
                if container.status != 'running':
                    logger.warning(f"Health check failed for {container.name}: not running")
                    continue
                
                # Try to execute a simple command
                try:
                    result = container.exec_run('echo "health_check"', timeout=10)
                    if result.exit_code == 0:
                        logger.debug(f"Health check passed for {container.name}")
                    else:
                        logger.warning(f"Health check failed for {container.name}: command failed")
                except:
                    logger.warning(f"Health check failed for {container.name}: command execution failed")
                    
            except Exception as e:
                logger.error(f"Health check error for {container.name}: {e}")
    
    def _orchestration_loop(self):
        """Main orchestration loop running in separate thread"""
        logger.info("Starting container lifecycle orchestration loop")
        
        while not self._shutdown_event.is_set():
            try:
                current_time_ns = self.clock.timestamp_ns()
                
                # Get pending orchestrations
                with self._lock:
                    pending_orchestrations = [
                        spec for spec in self._orchestration_queue
                        if spec.scheduled_time_ns <= current_time_ns
                    ]
                    
                    # Remove from queue
                    for spec in pending_orchestrations:
                        self._orchestration_queue.remove(spec)
                
                # Execute pending orchestrations
                for spec in pending_orchestrations:
                    if self._shutdown_event.is_set():
                        break
                        
                    try:
                        self._execute_orchestration(spec)
                    except Exception as e:
                        logger.error(f"Error executing orchestration: {e}")
                
                # Sleep until next check (minimum 1 second)
                sleep_seconds = 1.0
                
                with self._lock:
                    if self._orchestration_queue:
                        next_scheduled = min(spec.scheduled_time_ns for spec in self._orchestration_queue)
                        time_to_next = (next_scheduled - current_time_ns) / 1e9
                        sleep_seconds = max(0.1, min(sleep_seconds, time_to_next))
                
                if not self._shutdown_event.wait(sleep_seconds):
                    continue
                    
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                if not self._shutdown_event.wait(5.0):  # Error backoff
                    continue
        
        logger.info("Container lifecycle orchestration loop stopped")
    
    def _trigger_event(self, event_type: str, event_data: Any):
        """Trigger registered event callbacks"""
        callbacks = self._event_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for orchestration events"""
        if event_type not in self._event_callbacks:
            logger.warning(f"Unknown event type: {event_type}")
            return
            
        self._event_callbacks[event_type].append(callback)
        logger.info(f"Registered callback for event type: {event_type}")
    
    def start_orchestration(self):
        """Start the background orchestration thread"""
        if self._orchestration_thread and self._orchestration_thread.is_alive():
            logger.warning("Orchestration thread already running")
            return
            
        self._shutdown_event.clear()
        self._orchestration_thread = threading.Thread(
            target=self._orchestration_loop,
            name="container-lifecycle-clock",
            daemon=True
        )
        self._orchestration_thread.start()
        logger.info("Started container lifecycle orchestration")
    
    def stop_orchestration(self):
        """Stop the background orchestration thread"""
        if self._orchestration_thread and self._orchestration_thread.is_alive():
            self._shutdown_event.set()
            self._orchestration_thread.join(timeout=10.0)
            
            if self._orchestration_thread.is_alive():
                logger.warning("Orchestration thread did not stop gracefully")
            else:
                logger.info("Stopped container lifecycle orchestration")
    
    def get_pending_orchestrations(self) -> List[ContainerOrchestrationSpec]:
        """Get list of pending orchestrations"""
        with self._lock:
            return self._orchestration_queue.copy()
    
    def get_orchestration_results(self, limit: int = 10) -> List[OrchestrationResult]:
        """Get recent orchestration results"""
        with self._lock:
            return sorted(
                self._orchestration_results,
                key=lambda r: r.end_time_ns,
                reverse=True
            )[:limit]
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        with self._lock:
            return {
                "total_orchestrations": self.total_orchestrations,
                "successful_orchestrations": self.successful_orchestrations,
                "failed_orchestrations": self.failed_orchestrations,
                "success_rate": self.successful_orchestrations / max(self.total_orchestrations, 1),
                "rollbacks_performed": self.rollbacks_performed,
                "pending_orchestrations": len(self._orchestration_queue),
                "container_groups": len(self._container_groups),
                "clock_type": "test" if isinstance(self.clock, TestClock) else "live"
            }
    
    def cancel_orchestration(self, container_group: str, action: OrchestrationAction) -> bool:
        """Cancel a pending orchestration"""
        with self._lock:
            for i, spec in enumerate(self._orchestration_queue):
                if spec.container_group == container_group and spec.action == action:
                    del self._orchestration_queue[i]
                    logger.info(f"Cancelled {action.value} orchestration for group '{container_group}'")
                    return True
        
        logger.warning(f"No pending {action.value} orchestration found for group '{container_group}'")
        return False
    
    def get_container_groups(self) -> Dict[str, List[str]]:
        """Get all registered container groups"""
        with self._lock:
            return self._container_groups.copy()
    
    def shutdown(self):
        """Clean shutdown of orchestration system"""
        logger.info("Shutting down Container Lifecycle Clock")
        self.stop_orchestration()
        logger.info("Container Lifecycle Clock shutdown complete")


# Global lifecycle manager instance
_global_lifecycle_manager: Optional[ContainerLifecycleClock] = None
_lifecycle_manager_lock = threading.Lock()


def get_global_container_lifecycle_manager(
    clock: Optional[Clock] = None,
    docker_client: Optional[docker.DockerClient] = None,
    compose_file_path: Optional[str] = None
) -> ContainerLifecycleClock:
    """Get or create the global container lifecycle manager"""
    global _global_lifecycle_manager
    
    if _global_lifecycle_manager is None:
        with _lifecycle_manager_lock:
            if _global_lifecycle_manager is None:
                _global_lifecycle_manager = ContainerLifecycleClock(
                    clock=clock,
                    docker_client=docker_client,
                    compose_file_path=compose_file_path
                )
                _global_lifecycle_manager.start_orchestration()
    
    return _global_lifecycle_manager


def shutdown_global_container_lifecycle_manager():
    """Shutdown the global container lifecycle manager"""
    global _global_lifecycle_manager
    
    if _global_lifecycle_manager is not None:
        with _lifecycle_manager_lock:
            if _global_lifecycle_manager is not None:
                _global_lifecycle_manager.shutdown()
                _global_lifecycle_manager = None


if __name__ == "__main__":
    # Example usage
    import signal
    import sys
    
    def signal_handler(signum, frame):
        print("\nShutting down Container Lifecycle Clock...")
        shutdown_global_container_lifecycle_manager()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create lifecycle manager with test clock for demonstration
    test_clock = TestClock()
    lifecycle_manager = ContainerLifecycleClock(clock=test_clock)
    
    # Register example container groups
    lifecycle_manager.register_container_group("test-databases", ["test-postgres", "test-redis"])
    lifecycle_manager.register_container_group("test-services", ["test-backend", "test-frontend"])
    
    # Schedule some orchestrations
    lifecycle_manager.schedule_orchestration(
        container_group="test-databases",
        action=OrchestrationAction.START,
        delay_seconds=5.0,
        priority=1
    )
    
    lifecycle_manager.schedule_orchestration(
        container_group="test-services",
        action=OrchestrationAction.START,
        delay_seconds=10.0,
        dependencies=["test-databases"],
        priority=2
    )
    
    # Start orchestration
    lifecycle_manager.start_orchestration()
    
    print("Container Lifecycle Clock running. Press Ctrl+C to stop.")
    print("Note: Example container groups may not exist, so orchestrations will fail.")
    
    try:
        while True:
            time.sleep(1)
            
            # Advance test clock for demonstration
            test_clock.advance_time(1_000_000_000)  # 1 second
            
            # Print stats every 30 seconds
            if test_clock.timestamp() % 30 == 0:
                stats = lifecycle_manager.get_orchestration_stats()
                pending = lifecycle_manager.get_pending_orchestrations()
                
                print(f"Orchestrations: {stats['total_orchestrations']}, "
                      f"Success rate: {stats['success_rate']:.2%}, "
                      f"Pending: {len(pending)}")
                      
    except KeyboardInterrupt:
        pass
    finally:
        lifecycle_manager.shutdown()
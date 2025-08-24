"""
Container CPU Optimization for Docker Environment
=================================================

Docker container-specific CPU optimization that integrates with the core optimization
system to manage CPU resources across 9 containerized engines on M4 Max architecture.
"""

import os
import sys
import time
import json
import logging
import docker
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import psutil

from .cpu_affinity import CPUAffinityManager, WorkloadPriority
from .process_manager import ProcessManager, ProcessClass
from .performance_monitor import PerformanceMonitor, MetricType
from .workload_classifier import WorkloadClassifier, WorkloadCategory

logger = logging.getLogger(__name__)

class ContainerPriority(Enum):
    """Container priority levels for CPU allocation"""
    ULTRA_CRITICAL = 1     # Risk engine, trading core
    CRITICAL = 2           # Market data, analytics engine  
    HIGH = 3               # ML engine, factor engine
    NORMAL = 4             # Features, websocket engines
    LOW = 5                # Portfolio, strategy engines
    BACKGROUND = 6         # Monitoring, utilities

@dataclass
class ContainerInfo:
    """Information about a managed container"""
    container_id: str
    container_name: str
    image_name: str
    priority: ContainerPriority
    cpu_limit: float
    cpu_reservation: float
    memory_limit: str
    assigned_cores: List[int] = field(default_factory=list)
    process_pids: Set[int] = field(default_factory=set)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_optimized: float = 0.0

class ContainerCPUOptimizer:
    """
    Container-specific CPU optimization for M4 Max Docker environment
    """
    
    def __init__(
        self, 
        cpu_affinity_manager: CPUAffinityManager,
        performance_monitor: PerformanceMonitor,
        workload_classifier: WorkloadClassifier
    ):
        self.cpu_affinity_manager = cpu_affinity_manager
        self.performance_monitor = performance_monitor
        self.workload_classifier = workload_classifier
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            self.docker_client = None
            
        # Container tracking
        self.managed_containers: Dict[str, ContainerInfo] = {}
        self.container_priority_mapping = self._initialize_container_priorities()
        
        # Core allocation strategy
        self.core_allocation_strategy = self._initialize_core_allocation()
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "containers_optimized": 0,
            "core_reassignments": 0,
            "optimization_cycles": 0,
            "performance_improvements": 0.0
        }
        
        # Initialize container discovery
        self._discover_containers()
        self._start_monitoring()
    
    def _initialize_container_priorities(self) -> Dict[str, ContainerPriority]:
        """Initialize container priority mappings"""
        return {
            "nautilus-risk-engine": ContainerPriority.ULTRA_CRITICAL,
            "nautilus-backend": ContainerPriority.ULTRA_CRITICAL,
            "nautilus-marketdata-engine": ContainerPriority.CRITICAL,
            "nautilus-analytics-engine": ContainerPriority.CRITICAL,
            "nautilus-ml-engine": ContainerPriority.HIGH,
            "nautilus-factor-engine": ContainerPriority.HIGH,
            "nautilus-features-engine": ContainerPriority.NORMAL,
            "nautilus-websocket-engine": ContainerPriority.NORMAL,
            "nautilus-portfolio-engine": ContainerPriority.LOW,
            "nautilus-strategy-engine": ContainerPriority.LOW,
            "nautilus-frontend": ContainerPriority.BACKGROUND,
            "nautilus-nginx": ContainerPriority.BACKGROUND,
            "nautilus-redis": ContainerPriority.BACKGROUND,
            "nautilus-postgres": ContainerPriority.BACKGROUND,
            "nautilus-prometheus": ContainerPriority.BACKGROUND,
            "nautilus-grafana": ContainerPriority.BACKGROUND
        }
    
    def _initialize_core_allocation(self) -> Dict[ContainerPriority, List[int]]:
        """Initialize core allocation strategy for M4 Max"""
        # M4 Max: 12 P-cores (0-11) + 4 E-cores (12-15)
        return {
            ContainerPriority.ULTRA_CRITICAL: [0, 1, 2, 3],      # First 4 P-cores
            ContainerPriority.CRITICAL: [4, 5, 6, 7],            # Next 4 P-cores  
            ContainerPriority.HIGH: [8, 9, 10, 11],              # Last 4 P-cores
            ContainerPriority.NORMAL: [8, 9, 10, 11, 12, 13],    # Last P-cores + E-cores
            ContainerPriority.LOW: [12, 13, 14, 15],             # E-cores only
            ContainerPriority.BACKGROUND: [14, 15]               # Last 2 E-cores
        }
    
    def _discover_containers(self) -> None:
        """Discover and register Nautilus containers"""
        if not self.docker_client:
            logger.warning("Docker client not available - skipping container discovery")
            return
            
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                container_name = container.name
                
                # Only manage Nautilus containers
                if not container_name.startswith("nautilus-"):
                    continue
                
                # Get container priority
                priority = self.container_priority_mapping.get(
                    container_name, 
                    ContainerPriority.NORMAL
                )
                
                # Get resource limits from Docker
                container_info = self._extract_container_info(container, priority)
                
                if container_info:
                    self.managed_containers[container.id] = container_info
                    logger.info(f"Discovered container: {container_name} "
                              f"(priority: {priority.name})")
            
            logger.info(f"Discovered {len(self.managed_containers)} Nautilus containers")
            
        except Exception as e:
            logger.error(f"Error discovering containers: {e}")
    
    def _extract_container_info(
        self, 
        container, 
        priority: ContainerPriority
    ) -> Optional[ContainerInfo]:
        """Extract container information from Docker container object"""
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            # Extract CPU and memory limits
            cpu_limit = 0.0
            cpu_reservation = 0.0
            memory_limit = "0"
            
            # Get resource configuration
            container_config = container.attrs.get('HostConfig', {})
            
            if 'CpuQuota' in container_config and 'CpuPeriod' in container_config:
                quota = container_config['CpuQuota']
                period = container_config['CpuPeriod']
                if quota > 0 and period > 0:
                    cpu_limit = quota / period
            
            if 'Memory' in container_config:
                memory_limit = str(container_config['Memory'])
                
            # Get current resource usage
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = self._calculate_memory_usage(stats)
            
            return ContainerInfo(
                container_id=container.id,
                container_name=container.name,
                image_name=container.image.tags[0] if container.image.tags else "unknown",
                priority=priority,
                cpu_limit=cpu_limit or 2.0,  # Default limit
                cpu_reservation=cpu_reservation,
                memory_limit=memory_limit,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                last_optimized=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error extracting container info for {container.name}: {e}")
            return None
    
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats"""
        try:
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            
            if not cpu_stats or not precpu_stats:
                return 0.0
            
            cpu_usage = cpu_stats.get('cpu_usage', {})
            precpu_usage = precpu_stats.get('cpu_usage', {})
            
            cpu_delta = cpu_usage.get('total_usage', 0) - precpu_usage.get('total_usage', 0)
            system_cpu_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
            
            if system_cpu_delta > 0:
                number_cpus = len(cpu_usage.get('percpu_usage', []))
                cpu_percent = (cpu_delta / system_cpu_delta) * number_cpus * 100.0
                return min(100.0, max(0.0, cpu_percent))
                
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_memory_usage(self, stats: Dict) -> float:
        """Calculate memory usage percentage from Docker stats"""
        try:
            memory_stats = stats.get('memory_stats', {})
            usage = memory_stats.get('usage', 0)
            limit = memory_stats.get('limit', 0)
            
            if limit > 0:
                return (usage / limit) * 100.0
                
            return 0.0
            
        except Exception:
            return 0.0
    
    def _start_monitoring(self) -> None:
        """Start container monitoring thread"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ContainerCPUMonitor"
        )
        self.monitor_thread.start()
        logger.info("Container CPU monitoring started")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update container stats
                self._update_container_stats()
                
                # Perform optimization
                self._optimize_containers()
                
                # Update metrics
                self._update_performance_metrics()
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in container monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _update_container_stats(self) -> None:
        """Update container resource usage statistics"""
        if not self.docker_client:
            return
            
        try:
            with self._lock:
                for container_id, container_info in self.managed_containers.items():
                    try:
                        container = self.docker_client.containers.get(container_id)
                        
                        if container.status != 'running':
                            continue
                            
                        # Get updated stats
                        stats = container.stats(stream=False)
                        container_info.cpu_usage = self._calculate_cpu_usage(stats)
                        container_info.memory_usage = self._calculate_memory_usage(stats)
                        
                        # Update process PIDs
                        container_info.process_pids = self._get_container_pids(container)
                        
                    except Exception as e:
                        logger.warning(f"Error updating stats for container {container_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error updating container stats: {e}")
    
    def _get_container_pids(self, container) -> Set[int]:
        """Get process PIDs running in container"""
        try:
            # Get container processes
            processes = container.top()
            pids = set()
            
            for process in processes.get('Processes', []):
                try:
                    pid = int(process[1])  # PID is usually in the second column
                    pids.add(pid)
                except (ValueError, IndexError):
                    continue
                    
            return pids
            
        except Exception:
            return set()
    
    def _optimize_containers(self) -> None:
        """Optimize container CPU allocation"""
        try:
            with self._lock:
                optimization_needed = []
                
                # Identify containers that need optimization
                for container_id, container_info in self.managed_containers.items():
                    
                    # Check if optimization is needed
                    if self._needs_optimization(container_info):
                        optimization_needed.append((container_id, container_info))
                
                if not optimization_needed:
                    return
                
                logger.info(f"Optimizing {len(optimization_needed)} containers")
                
                # Sort by priority (ultra critical first)
                optimization_needed.sort(key=lambda x: x[1].priority.value)
                
                # Optimize each container
                for container_id, container_info in optimization_needed:
                    self._optimize_single_container(container_info)
                
                self.stats["optimization_cycles"] += 1
                
        except Exception as e:
            logger.error(f"Error during container optimization: {e}")
    
    def _needs_optimization(self, container_info: ContainerInfo) -> bool:
        """Check if container needs CPU optimization"""
        current_time = time.time()
        
        # Optimize if hasn't been optimized in the last 60 seconds
        if (current_time - container_info.last_optimized) < 60:
            return False
        
        # Optimize if CPU usage is high
        if container_info.cpu_usage > 80.0:
            return True
            
        # Optimize if CPU usage is very low (might be over-allocated)
        if container_info.cpu_usage < 10.0 and len(container_info.assigned_cores) > 1:
            return True
            
        # Optimize critical containers more frequently
        if container_info.priority in [ContainerPriority.ULTRA_CRITICAL, ContainerPriority.CRITICAL]:
            return (current_time - container_info.last_optimized) > 30
        
        return False
    
    def _optimize_single_container(self, container_info: ContainerInfo) -> None:
        """Optimize CPU allocation for a single container"""
        try:
            # Get target cores for this priority level
            target_cores = self._get_target_cores(container_info)
            
            if not target_cores:
                logger.warning(f"No target cores available for {container_info.container_name}")
                return
            
            # Assign cores to container processes
            cores_assigned = self._assign_cores_to_container(container_info, target_cores)
            
            if cores_assigned:
                container_info.assigned_cores = target_cores
                container_info.last_optimized = time.time()
                
                self.stats["containers_optimized"] += 1
                self.stats["core_reassignments"] += len(target_cores)
                
                logger.info(f"Optimized {container_info.container_name}: "
                          f"assigned cores {target_cores} "
                          f"(CPU usage: {container_info.cpu_usage:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error optimizing container {container_info.container_name}: {e}")
    
    def _get_target_cores(self, container_info: ContainerInfo) -> List[int]:
        """Get target CPU cores for container based on priority and load"""
        base_cores = self.core_allocation_strategy.get(container_info.priority, [])
        
        if not base_cores:
            return []
        
        # Adjust core count based on CPU usage
        if container_info.cpu_usage > 80.0:
            # High usage - allocate more cores if available
            if container_info.priority in [ContainerPriority.ULTRA_CRITICAL, ContainerPriority.CRITICAL]:
                # Critical containers get extra P-cores if needed
                extra_cores = [c for c in range(12) if c not in base_cores][:2]
                return base_cores + extra_cores
                
        elif container_info.cpu_usage < 20.0:
            # Low usage - reduce core allocation
            if len(base_cores) > 1:
                return base_cores[:max(1, len(base_cores) // 2)]
        
        return base_cores
    
    def _assign_cores_to_container(
        self, 
        container_info: ContainerInfo, 
        target_cores: List[int]
    ) -> bool:
        """Assign CPU cores to all processes in a container"""
        success_count = 0
        total_processes = len(container_info.process_pids)
        
        if total_processes == 0:
            logger.warning(f"No processes found for container {container_info.container_name}")
            return False
        
        # Determine workload priority for container processes
        workload_priority = self._get_workload_priority(container_info.priority)
        
        # Assign each process to cores
        for pid in container_info.process_pids:
            try:
                success = self.cpu_affinity_manager.assign_process_to_cores(
                    pid,
                    workload_priority,
                    target_cores
                )
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to assign cores to PID {pid}: {e}")
        
        # Consider successful if at least 50% of processes were assigned
        return success_count >= (total_processes * 0.5)
    
    def _get_workload_priority(self, container_priority: ContainerPriority) -> WorkloadPriority:
        """Map container priority to workload priority"""
        mapping = {
            ContainerPriority.ULTRA_CRITICAL: WorkloadPriority.ULTRA_LOW_LATENCY,
            ContainerPriority.CRITICAL: WorkloadPriority.LOW_LATENCY,
            ContainerPriority.HIGH: WorkloadPriority.NORMAL,
            ContainerPriority.NORMAL: WorkloadPriority.NORMAL,
            ContainerPriority.LOW: WorkloadPriority.BACKGROUND,
            ContainerPriority.BACKGROUND: WorkloadPriority.BACKGROUND
        }
        return mapping.get(container_priority, WorkloadPriority.NORMAL)
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for monitoring"""
        try:
            with self._lock:
                # Calculate aggregate metrics
                total_cpu_usage = 0.0
                total_memory_usage = 0.0
                critical_containers = 0
                high_usage_containers = 0
                
                for container_info in self.managed_containers.values():
                    total_cpu_usage += container_info.cpu_usage
                    total_memory_usage += container_info.memory_usage
                    
                    if container_info.priority in [ContainerPriority.ULTRA_CRITICAL, ContainerPriority.CRITICAL]:
                        critical_containers += 1
                        
                    if container_info.cpu_usage > 80.0:
                        high_usage_containers += 1
                
                # Send metrics to performance monitor
                self.performance_monitor._add_metric(
                    MetricType.CPU_UTILIZATION,
                    total_cpu_usage / len(self.managed_containers) if self.managed_containers else 0.0,
                    "%",
                    "container_aggregate"
                )
                
                # Custom container metrics
                self.performance_monitor._add_metric(
                    MetricType.THROUGHPUT,
                    len(self.managed_containers),
                    "containers",
                    "managed_containers"
                )
                
                if high_usage_containers > critical_containers * 0.5:
                    # High resource pressure
                    logger.warning(f"{high_usage_containers} containers showing high CPU usage")
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_container_stats(self) -> Dict[str, Any]:
        """Get comprehensive container statistics"""
        with self._lock:
            container_details = {}
            
            for container_id, container_info in self.managed_containers.items():
                container_details[container_info.container_name] = {
                    "priority": container_info.priority.name,
                    "cpu_usage": container_info.cpu_usage,
                    "memory_usage": container_info.memory_usage,
                    "assigned_cores": container_info.assigned_cores,
                    "process_count": len(container_info.process_pids),
                    "cpu_limit": container_info.cpu_limit,
                    "last_optimized": container_info.last_optimized
                }
            
            # Priority distribution
            priority_counts = {}
            for priority in ContainerPriority:
                count = len([c for c in self.managed_containers.values() 
                           if c.priority == priority])
                priority_counts[priority.name] = count
            
            return {
                "total_containers": len(self.managed_containers),
                "container_details": container_details,
                "priority_distribution": priority_counts,
                "optimization_stats": dict(self.stats),
                "core_allocation_strategy": {
                    priority.name: cores for priority, cores 
                    in self.core_allocation_strategy.items()
                }
            }
    
    def force_container_optimization(self, container_name: Optional[str] = None) -> Dict[str, Any]:
        """Force optimization of specific container or all containers"""
        try:
            with self._lock:
                containers_optimized = 0
                
                for container_info in self.managed_containers.values():
                    if container_name and container_info.container_name != container_name:
                        continue
                        
                    # Reset last optimized time to force optimization
                    container_info.last_optimized = 0
                    self._optimize_single_container(container_info)
                    containers_optimized += 1
                
                return {
                    "success": True,
                    "containers_optimized": containers_optimized,
                    "message": f"Forced optimization of {containers_optimized} containers"
                }
                
        except Exception as e:
            logger.error(f"Error forcing container optimization: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_container_priority(
        self, 
        container_name: str, 
        new_priority: ContainerPriority
    ) -> bool:
        """Update container priority and re-optimize"""
        try:
            with self._lock:
                for container_info in self.managed_containers.values():
                    if container_info.container_name == container_name:
                        old_priority = container_info.priority
                        container_info.priority = new_priority
                        
                        # Force re-optimization
                        container_info.last_optimized = 0
                        self._optimize_single_container(container_info)
                        
                        logger.info(f"Updated {container_name} priority: "
                                  f"{old_priority.name} -> {new_priority.name}")
                        return True
                
                logger.warning(f"Container {container_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error updating container priority: {e}")
            return False
    
    def get_container_performance_analysis(self) -> Dict[str, Any]:
        """Get detailed performance analysis of containers"""
        with self._lock:
            analysis = {
                "timestamp": time.time(),
                "performance_summary": {},
                "optimization_opportunities": [],
                "resource_efficiency": {}
            }
            
            # Performance summary
            cpu_usages = [c.cpu_usage for c in self.managed_containers.values()]
            if cpu_usages:
                analysis["performance_summary"] = {
                    "avg_cpu_usage": sum(cpu_usages) / len(cpu_usages),
                    "max_cpu_usage": max(cpu_usages),
                    "min_cpu_usage": min(cpu_usages),
                    "high_usage_containers": len([u for u in cpu_usages if u > 80.0]),
                    "idle_containers": len([u for u in cpu_usages if u < 5.0])
                }
            
            # Optimization opportunities
            for container_info in self.managed_containers.values():
                if container_info.cpu_usage > 90.0:
                    analysis["optimization_opportunities"].append({
                        "container": container_info.container_name,
                        "issue": "High CPU usage",
                        "recommendation": "Consider scaling out or allocating more cores",
                        "current_usage": container_info.cpu_usage,
                        "assigned_cores": container_info.assigned_cores
                    })
                elif container_info.cpu_usage < 5.0 and len(container_info.assigned_cores) > 1:
                    analysis["optimization_opportunities"].append({
                        "container": container_info.container_name,
                        "issue": "Over-allocated cores",
                        "recommendation": "Reduce core allocation to improve efficiency",
                        "current_usage": container_info.cpu_usage,
                        "assigned_cores": container_info.assigned_cores
                    })
            
            # Resource efficiency
            for priority in ContainerPriority:
                priority_containers = [
                    c for c in self.managed_containers.values() 
                    if c.priority == priority
                ]
                
                if priority_containers:
                    avg_usage = sum(c.cpu_usage for c in priority_containers) / len(priority_containers)
                    total_cores = sum(len(c.assigned_cores) for c in priority_containers)
                    
                    analysis["resource_efficiency"][priority.name] = {
                        "container_count": len(priority_containers),
                        "avg_cpu_usage": avg_usage,
                        "total_cores_assigned": total_cores,
                        "efficiency_score": min(100.0, (avg_usage / max(1, total_cores * 25)) * 100)
                    }
            
            return analysis
    
    def shutdown(self) -> None:
        """Shutdown container CPU optimizer"""
        logger.info("Shutting down container CPU optimizer...")
        
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        if self.docker_client:
            self.docker_client.close()
        
        logger.info("Container CPU optimizer shutdown complete")
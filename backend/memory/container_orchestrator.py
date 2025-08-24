"""
Container Memory Orchestrator for M4 Max Architecture

Manages dynamic memory allocation across 16+ containers with priority-based
allocation, intelligent rebalancing, and emergency memory management for
the Nautilus trading platform.

Key Features:
- Dynamic memory allocation across 16+ containers
- Priority-based memory allocation with trading workload awareness
- Intelligent container memory limits and enforcement
- Emergency memory rebalancing and automatic scaling
- Container lifecycle memory management
- Cross-container memory sharing coordination
"""

import asyncio
import threading
import time
import docker
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, NamedTuple
import logging
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .unified_memory_manager import (
    MemoryWorkloadType,
    MemoryRegion,
    get_unified_memory_manager
)
from .memory_pools import get_memory_pool_manager
from .memory_monitor import get_memory_monitor, MemoryAlertLevel


class ContainerPriority(Enum):
    """Container priority levels for memory allocation"""
    CRITICAL = 1      # Trading engines, risk management
    HIGH = 2          # Market data, analytics engines
    NORMAL = 3        # ML models, strategy engines
    LOW = 4           # Background processing, logging
    MAINTENANCE = 5   # Cleanup, monitoring containers


class ContainerState(Enum):
    """Container lifecycle states"""
    STARTING = "starting"
    RUNNING = "running"
    SCALING = "scaling"
    THROTTLED = "throttled"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AllocationStrategy(Enum):
    """Memory allocation strategies"""
    GUARANTEED = "guaranteed"         # Reserved memory, never reclaimed
    BURSTABLE = "burstable"          # Can burst above limit if available
    BEST_EFFORT = "best_effort"      # No guarantees, may be reclaimed
    ADAPTIVE = "adaptive"            # Adjusts based on usage patterns


@dataclass
class ContainerMemorySpec:
    """Memory specification for a container"""
    container_id: str
    container_name: str
    priority: ContainerPriority
    strategy: AllocationStrategy
    
    # Memory limits
    min_memory: int           # Guaranteed minimum
    max_memory: int           # Maximum allowed
    target_memory: int        # Target allocation
    
    # Workload characteristics
    primary_workload: MemoryWorkloadType
    workload_mix: Dict[MemoryWorkloadType, float]  # Percentage breakdown
    
    # Performance requirements
    latency_sensitive: bool = False
    bandwidth_intensive: bool = False
    
    # Scaling parameters
    scale_up_threshold: float = 0.8    # Scale up at 80% usage
    scale_down_threshold: float = 0.3  # Scale down at 30% usage
    scale_factor: float = 1.5          # Scale by 50%
    
    # Lifecycle
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class ContainerMemoryStatus:
    """Current memory status for a container"""
    container_id: str
    state: ContainerState
    allocated_memory: int
    peak_memory: int
    current_usage: int
    usage_percentage: float
    
    # Performance metrics
    allocation_efficiency: float  # usage / allocated
    memory_pressure: float       # 0.0 to 1.0
    bandwidth_usage: float       # bytes/second
    
    # History tracking
    usage_history: deque = field(default_factory=lambda: deque(maxlen=720))  # 12 hours at 1-minute intervals
    allocation_history: deque = field(default_factory=lambda: deque(maxlen=720))
    
    # Scaling metrics
    last_scaled: float = 0.0
    scale_cooldown_until: float = 0.0
    consecutive_pressure_events: int = 0
    
    def update_usage(self, usage: int):
        """Update usage tracking"""
        timestamp = time.time()
        self.current_usage = usage
        self.usage_percentage = (usage / max(1, self.allocated_memory)) * 100
        self.usage_history.append((timestamp, usage))


@dataclass
class RebalancingEvent:
    """Memory rebalancing event"""
    timestamp: float
    trigger: str
    affected_containers: List[str]
    memory_moved: int
    duration: float
    success: bool
    reason: str


class ContainerOrchestrator:
    """
    Container Memory Orchestrator for M4 Max
    
    Manages memory allocation across all containers with intelligent
    rebalancing and emergency management capabilities.
    """
    
    def __init__(self, total_container_memory: int = 32 * 1024 * 1024 * 1024):  # 32GB for containers
        self.total_memory = total_container_memory
        self.reserved_system_memory = 4 * 1024 * 1024 * 1024  # 4GB for system
        self.available_memory = self.total_memory - self.reserved_system_memory
        
        # Container tracking
        self.container_specs: Dict[str, ContainerMemorySpec] = {}
        self.container_status: Dict[str, ContainerMemoryStatus] = {}
        self.container_docker_info: Dict[str, Any] = {}
        
        # Memory allocation tracking
        self.allocated_memory = 0
        self.guaranteed_memory = 0
        self.allocation_history: deque = deque(maxlen=1440)  # 24 hours
        
        # Rebalancing
        self.rebalancing_events: deque = deque(maxlen=100)
        self.last_rebalance = 0.0
        self.rebalance_cooldown = 60.0  # 1 minute cooldown
        
        # Emergency management
        self.emergency_mode = False
        self.emergency_threshold = 0.95  # 95% memory usage
        self.emergency_actions: List[Callable] = []
        
        # Priority queues for different allocation strategies
        self.priority_queues = {
            priority: [] for priority in ContainerPriority
        }
        
        # External integrations
        self.unified_manager = get_unified_memory_manager()
        self.pool_manager = get_memory_pool_manager()
        self.memory_monitor = get_memory_monitor()
        
        # Docker client for container management
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logging.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
        
        # Threading
        self.is_running = False
        self.orchestrator_thread = None
        self.rebalancer_thread = None
        self.monitor_thread = None
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ContainerOrchestrator")
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        
        # Register alert handler
        self.memory_monitor.register_alert_handler(self._handle_memory_alert)
        
        self.logger.info(f"Initialized ContainerOrchestrator with {total_container_memory/1024/1024/1024:.1f}GB total memory")
    
    def start(self):
        """Start the container orchestrator"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background threads
        self.orchestrator_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.rebalancer_thread = threading.Thread(target=self._rebalancing_loop, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        
        self.orchestrator_thread.start()
        self.rebalancer_thread.start()
        self.monitor_thread.start()
        
        # Initialize default container specs if Docker is available
        if self.docker_client:
            self._discover_containers()
        
        self.logger.info("Started container orchestrator")
    
    def stop(self):
        """Stop the container orchestrator"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in [self.orchestrator_thread, self.rebalancer_thread, self.monitor_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Stopped container orchestrator")
    
    def register_container(self, spec: ContainerMemorySpec) -> bool:
        """Register a container with memory specifications"""
        with self.lock:
            # Validate memory requirements
            if spec.min_memory > spec.max_memory:
                self.logger.error(f"Invalid memory spec for {spec.container_name}: min > max")
                return False
            
            if spec.min_memory + self.guaranteed_memory > self.available_memory:
                self.logger.error(f"Cannot guarantee {spec.min_memory} bytes for {spec.container_name}")
                return False
            
            # Register container
            self.container_specs[spec.container_id] = spec
            
            # Initialize status
            self.container_status[spec.container_id] = ContainerMemoryStatus(
                container_id=spec.container_id,
                state=ContainerState.STARTING,
                allocated_memory=spec.target_memory,
                peak_memory=spec.target_memory,
                current_usage=0,
                usage_percentage=0.0,
                allocation_efficiency=0.0,
                memory_pressure=0.0,
                bandwidth_usage=0.0
            )
            
            # Add to priority queue
            self.priority_queues[spec.priority].append(spec.container_id)
            
            # Update memory tracking
            if spec.strategy == AllocationStrategy.GUARANTEED:
                self.guaranteed_memory += spec.min_memory
            
            self.allocated_memory += spec.target_memory
            
            # Set container limits in unified memory manager
            self.unified_manager.set_container_limit(
                spec.container_id, 
                spec.max_memory,
                priority=spec.priority.value
            )
            
            self.logger.info(f"Registered container {spec.container_name} with {spec.target_memory/1024/1024:.1f}MB target")
            return True
    
    def unregister_container(self, container_id: str) -> bool:
        """Unregister a container"""
        with self.lock:
            if container_id not in self.container_specs:
                return False
            
            spec = self.container_specs[container_id]
            status = self.container_status.get(container_id)
            
            # Update memory tracking
            if spec.strategy == AllocationStrategy.GUARANTEED:
                self.guaranteed_memory -= spec.min_memory
            
            if status:
                self.allocated_memory -= status.allocated_memory
            
            # Remove from tracking
            del self.container_specs[container_id]
            if container_id in self.container_status:
                del self.container_status[container_id]
            
            # Remove from priority queues
            for priority_list in self.priority_queues.values():
                if container_id in priority_list:
                    priority_list.remove(container_id)
            
            self.logger.info(f"Unregistered container {container_id}")
            return True
    
    def allocate_memory(self, container_id: str, additional_bytes: int) -> bool:
        """Allocate additional memory to a container"""
        with self.lock:
            if container_id not in self.container_specs:
                return False
            
            spec = self.container_specs[container_id]
            status = self.container_status[container_id]
            
            new_allocation = status.allocated_memory + additional_bytes
            
            # Check limits
            if new_allocation > spec.max_memory:
                self.logger.warning(f"Allocation would exceed max limit for {container_id}")
                return False
            
            # Check available memory
            if self._get_available_memory() < additional_bytes:
                # Try rebalancing
                if self._emergency_rebalance(additional_bytes):
                    self.logger.info(f"Emergency rebalance successful for {container_id}")
                else:
                    self.logger.error(f"Insufficient memory for {container_id}")
                    return False
            
            # Perform allocation
            status.allocated_memory = new_allocation
            status.last_scaled = time.time()
            status.scale_cooldown_until = time.time() + 30.0  # 30 second cooldown
            
            self.allocated_memory += additional_bytes
            
            # Update container limits
            self._update_container_docker_limits(container_id, new_allocation)
            
            self.logger.info(f"Allocated {additional_bytes/1024/1024:.1f}MB to {container_id}")
            return True
    
    def deallocate_memory(self, container_id: str, bytes_to_free: int) -> bool:
        """Deallocate memory from a container"""
        with self.lock:
            if container_id not in self.container_specs:
                return False
            
            spec = self.container_specs[container_id]
            status = self.container_status[container_id]
            
            new_allocation = status.allocated_memory - bytes_to_free
            
            # Check minimum limits
            if new_allocation < spec.min_memory:
                new_allocation = spec.min_memory
                bytes_to_free = status.allocated_memory - new_allocation
            
            if bytes_to_free <= 0:
                return False
            
            # Perform deallocation
            status.allocated_memory = new_allocation
            self.allocated_memory -= bytes_to_free
            
            # Update container limits
            self._update_container_docker_limits(container_id, new_allocation)
            
            self.logger.info(f"Deallocated {bytes_to_free/1024/1024:.1f}MB from {container_id}")
            return True
    
    def scale_container_memory(self, container_id: str, target_usage: float = 0.7) -> bool:
        """Scale container memory based on usage patterns"""
        with self.lock:
            if container_id not in self.container_specs:
                return False
            
            spec = self.container_specs[container_id]
            status = self.container_status[container_id]
            
            # Check cooldown
            if time.time() < status.scale_cooldown_until:
                return False
            
            # Calculate target allocation
            if status.usage_percentage > spec.scale_up_threshold * 100:
                # Scale up
                scale_factor = min(spec.scale_factor, 2.0)  # Cap at 2x
                additional_memory = int(status.allocated_memory * (scale_factor - 1))
                
                return self.allocate_memory(container_id, additional_memory)
            
            elif status.usage_percentage < spec.scale_down_threshold * 100:
                # Scale down
                target_allocation = int(status.current_usage / target_usage)
                target_allocation = max(target_allocation, spec.min_memory)
                
                if target_allocation < status.allocated_memory:
                    bytes_to_free = status.allocated_memory - target_allocation
                    return self.deallocate_memory(container_id, bytes_to_free)
            
            return False
    
    def force_rebalance(self, reason: str = "Manual trigger") -> bool:
        """Force memory rebalancing across all containers"""
        return self._rebalance_memory(reason)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status"""
        with self.lock:
            container_summaries = {}
            
            for container_id, status in self.container_status.items():
                spec = self.container_specs.get(container_id)
                if spec:
                    container_summaries[container_id] = {
                        'name': spec.container_name,
                        'priority': spec.priority.value,
                        'state': status.state.value,
                        'allocated_mb': status.allocated_memory / 1024 / 1024,
                        'usage_mb': status.current_usage / 1024 / 1024,
                        'usage_percentage': status.usage_percentage,
                        'efficiency': status.allocation_efficiency,
                        'memory_pressure': status.memory_pressure
                    }
            
            return {
                'total_memory_gb': self.total_memory / 1024 / 1024 / 1024,
                'available_memory_gb': self._get_available_memory() / 1024 / 1024 / 1024,
                'allocated_memory_gb': self.allocated_memory / 1024 / 1024 / 1024,
                'guaranteed_memory_gb': self.guaranteed_memory / 1024 / 1024 / 1024,
                'utilization_percentage': (self.allocated_memory / self.total_memory) * 100,
                'emergency_mode': self.emergency_mode,
                'container_count': len(self.container_specs),
                'containers': container_summaries,
                'recent_rebalances': len(self.rebalancing_events)
            }
    
    def get_container_recommendations(self, container_id: str) -> Dict[str, Any]:
        """Get memory optimization recommendations for container"""
        if container_id not in self.container_specs:
            return {}
        
        spec = self.container_specs[container_id]
        status = self.container_status[container_id]
        
        recommendations = []
        
        # Efficiency recommendations
        if status.allocation_efficiency < 0.5:
            recommendations.append({
                'type': 'efficiency',
                'severity': 'warning',
                'message': f"Low memory efficiency ({status.allocation_efficiency:.1%})",
                'suggestion': "Consider reducing allocated memory or optimizing application"
            })
        
        # Usage pattern recommendations
        if status.usage_percentage > 90:
            recommendations.append({
                'type': 'capacity',
                'severity': 'critical',
                'message': f"High memory usage ({status.usage_percentage:.1f}%)",
                'suggestion': "Increase memory allocation or optimize memory usage"
            })
        elif status.usage_percentage < 20:
            recommendations.append({
                'type': 'waste',
                'severity': 'info',
                'message': f"Low memory usage ({status.usage_percentage:.1f}%)",
                'suggestion': "Consider reducing memory allocation"
            })
        
        # Scaling recommendations
        if status.consecutive_pressure_events > 3:
            recommendations.append({
                'type': 'scaling',
                'severity': 'warning',
                'message': "Frequent memory pressure events",
                'suggestion': "Enable auto-scaling or increase base allocation"
            })
        
        return {
            'container_id': container_id,
            'container_name': spec.container_name,
            'current_state': status.state.value,
            'recommendations': recommendations,
            'optimal_allocation_mb': self._calculate_optimal_allocation(container_id) / 1024 / 1024
        }
    
    # Private methods
    
    def _orchestration_loop(self):
        """Main orchestration loop"""
        self.logger.info("Started orchestration loop")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update container statuses
                self._update_container_statuses()
                
                # Check for auto-scaling opportunities
                self._check_autoscaling()
                
                # Update allocation history
                self.allocation_history.append((time.time(), self.allocated_memory, len(self.container_specs)))
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                sleep_time = max(0, 5.0 - elapsed)  # 5-second intervals
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")
                time.sleep(5.0)
        
        self.logger.info("Stopped orchestration loop")
    
    def _rebalancing_loop(self):
        """Memory rebalancing loop"""
        self.logger.info("Started rebalancing loop")
        
        while self.is_running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check if rebalancing is needed
                if self._should_rebalance():
                    self._rebalance_memory("Automatic rebalancing")
                
            except Exception as e:
                self.logger.error(f"Error in rebalancing loop: {e}")
        
        self.logger.info("Stopped rebalancing loop")
    
    def _monitoring_loop(self):
        """Container monitoring loop"""
        self.logger.info("Started monitoring loop")
        
        while self.is_running:
            try:
                time.sleep(10)  # Monitor every 10 seconds
                
                # Check for emergency conditions
                utilization = self.allocated_memory / self.total_memory
                if utilization > self.emergency_threshold:
                    if not self.emergency_mode:
                        self._enter_emergency_mode()
                elif utilization < 0.8 and self.emergency_mode:
                    self._exit_emergency_mode()
                
                # Update Docker container info
                if self.docker_client:
                    self._update_docker_container_info()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
        
        self.logger.info("Stopped monitoring loop")
    
    def _discover_containers(self):
        """Discover running containers and register them"""
        if not self.docker_client:
            return
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                # Skip system containers
                if any(name in container.name for name in ['nautilus-', 'trading-', 'engine-']):
                    container_id = container.id[:12]
                    
                    # Determine container type and priority
                    priority, workload = self._classify_container(container.name)
                    
                    # Create default spec
                    spec = ContainerMemorySpec(
                        container_id=container_id,
                        container_name=container.name,
                        priority=priority,
                        strategy=AllocationStrategy.ADAPTIVE,
                        min_memory=128 * 1024 * 1024,      # 128MB minimum
                        max_memory=4 * 1024 * 1024 * 1024,   # 4GB maximum
                        target_memory=512 * 1024 * 1024,    # 512MB target
                        primary_workload=workload,
                        workload_mix={workload: 1.0},
                        latency_sensitive=priority in {ContainerPriority.CRITICAL, ContainerPriority.HIGH}
                    )
                    
                    self.register_container(spec)
                    self.logger.info(f"Auto-registered container {container.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to discover containers: {e}")
    
    def _classify_container(self, container_name: str) -> Tuple[ContainerPriority, MemoryWorkloadType]:
        """Classify container based on name"""
        name_lower = container_name.lower()
        
        # Trading engines - highest priority
        if any(term in name_lower for term in ['trading', 'execution', 'risk', 'order']):
            return ContainerPriority.CRITICAL, MemoryWorkloadType.TRADING_DATA
        
        # Market data and analytics
        elif any(term in name_lower for term in ['market', 'data', 'analytics', 'factor']):
            return ContainerPriority.HIGH, MemoryWorkloadType.ANALYTICS
        
        # ML and strategy engines
        elif any(term in name_lower for term in ['ml', 'strategy', 'model']):
            return ContainerPriority.NORMAL, MemoryWorkloadType.ML_MODELS
        
        # WebSocket and streaming
        elif any(term in name_lower for term in ['websocket', 'stream', 'ws']):
            return ContainerPriority.HIGH, MemoryWorkloadType.WEBSOCKET_STREAMS
        
        # Background services
        else:
            return ContainerPriority.LOW, MemoryWorkloadType.TEMPORARY_COMPUTE
    
    def _update_container_statuses(self):
        """Update memory status for all containers"""
        with self.lock:
            pressure_metrics = self.unified_manager.get_memory_pressure()
            
            for container_id, spec in self.container_specs.items():
                if container_id in self.container_status:
                    status = self.container_status[container_id]
                    
                    # Get current usage from unified memory manager
                    current_usage = pressure_metrics.container_allocations.get(container_id, 0)
                    status.update_usage(current_usage)
                    
                    # Update efficiency metrics
                    if status.allocated_memory > 0:
                        status.allocation_efficiency = current_usage / status.allocated_memory
                    
                    # Calculate memory pressure
                    status.memory_pressure = min(1.0, current_usage / status.allocated_memory)
                    
                    # Track consecutive pressure events
                    if status.memory_pressure > 0.9:
                        status.consecutive_pressure_events += 1
                    else:
                        status.consecutive_pressure_events = max(0, status.consecutive_pressure_events - 1)
                    
                    # Update peak memory
                    if current_usage > status.peak_memory:
                        status.peak_memory = current_usage
    
    def _check_autoscaling(self):
        """Check for auto-scaling opportunities"""
        for container_id, spec in self.container_specs.items():
            if spec.strategy == AllocationStrategy.ADAPTIVE:
                self.scale_container_memory(container_id)
    
    def _should_rebalance(self) -> bool:
        """Determine if memory rebalancing is needed"""
        # Check cooldown
        if time.time() - self.last_rebalance < self.rebalance_cooldown:
            return False
        
        # Check overall utilization
        utilization = self.allocated_memory / self.total_memory
        if utilization > 0.9:
            return True
        
        # Check for highly imbalanced containers
        high_pressure_containers = 0
        low_efficiency_containers = 0
        
        for status in self.container_status.values():
            if status.memory_pressure > 0.9:
                high_pressure_containers += 1
            if status.allocation_efficiency < 0.3:
                low_efficiency_containers += 1
        
        # Rebalance if many containers are under pressure or inefficient
        return high_pressure_containers > 2 or low_efficiency_containers > 3
    
    def _rebalance_memory(self, reason: str) -> bool:
        """Perform memory rebalancing across containers"""
        start_time = time.time()
        affected_containers = []
        total_memory_moved = 0
        
        try:
            with self.lock:
                # Identify containers that can give up memory (low efficiency)
                donors = []
                for container_id, status in self.container_status.items():
                    spec = self.container_specs[container_id]
                    
                    if (status.allocation_efficiency < 0.5 and 
                        status.allocated_memory > spec.min_memory and
                        spec.strategy != AllocationStrategy.GUARANTEED):
                        
                        potential_donation = status.allocated_memory - max(
                            spec.min_memory,
                            int(status.current_usage * 1.2)  # 20% buffer
                        )
                        
                        if potential_donation > 64 * 1024 * 1024:  # At least 64MB
                            donors.append((container_id, potential_donation))
                
                # Identify containers that need memory (high pressure)
                recipients = []
                for container_id, status in self.container_status.items():
                    spec = self.container_specs[container_id]
                    
                    if (status.memory_pressure > 0.8 and 
                        status.allocated_memory < spec.max_memory):
                        
                        needed_memory = min(
                            int(status.current_usage * 1.5) - status.allocated_memory,
                            spec.max_memory - status.allocated_memory
                        )
                        
                        if needed_memory > 64 * 1024 * 1024:  # At least 64MB
                            recipients.append((container_id, needed_memory))
                
                # Sort by priority
                donors.sort(key=lambda x: self.container_specs[x[0]].priority.value, reverse=True)
                recipients.sort(key=lambda x: self.container_specs[x[0]].priority.value)
                
                # Perform rebalancing
                available_memory = sum(donation for _, donation in donors)
                
                for recipient_id, needed in recipients:
                    if available_memory <= 0:
                        break
                    
                    memory_to_allocate = min(needed, available_memory)
                    
                    # Find memory from donors
                    for donor_id, donation in donors:
                        if memory_to_allocate <= 0:
                            break
                        
                        transfer_amount = min(memory_to_allocate, donation)
                        
                        # Transfer memory
                        if (self.deallocate_memory(donor_id, transfer_amount) and
                            self.allocate_memory(recipient_id, transfer_amount)):
                            
                            affected_containers.extend([donor_id, recipient_id])
                            total_memory_moved += transfer_amount
                            memory_to_allocate -= transfer_amount
                            available_memory -= transfer_amount
                            
                            # Update donor donation amount
                            donors[donors.index((donor_id, donation))] = (donor_id, donation - transfer_amount)
                
                # Record rebalancing event
                event = RebalancingEvent(
                    timestamp=time.time(),
                    trigger=reason,
                    affected_containers=list(set(affected_containers)),
                    memory_moved=total_memory_moved,
                    duration=time.time() - start_time,
                    success=total_memory_moved > 0,
                    reason=f"Moved {total_memory_moved/1024/1024:.1f}MB across {len(set(affected_containers))} containers"
                )
                
                self.rebalancing_events.append(event)
                self.last_rebalance = time.time()
                
                if event.success:
                    self.logger.info(f"Rebalancing successful: {event.reason}")
                else:
                    self.logger.warning(f"Rebalancing had no effect: {reason}")
                
                return event.success
                
        except Exception as e:
            self.logger.error(f"Rebalancing failed: {e}")
            return False
    
    def _emergency_rebalance(self, required_memory: int) -> bool:
        """Emergency memory rebalancing to free up required memory"""
        self.logger.warning(f"Emergency rebalancing to free {required_memory/1024/1024:.1f}MB")
        
        freed_memory = 0
        
        # Force scale down low-priority containers
        for priority in [ContainerPriority.LOW, ContainerPriority.MAINTENANCE, ContainerPriority.NORMAL]:
            if freed_memory >= required_memory:
                break
            
            for container_id in self.priority_queues[priority]:
                if container_id in self.container_status:
                    spec = self.container_specs[container_id]
                    status = self.container_status[container_id]
                    
                    if status.allocated_memory > spec.min_memory:
                        # Scale down to minimum
                        bytes_to_free = status.allocated_memory - spec.min_memory
                        if self.deallocate_memory(container_id, bytes_to_free):
                            freed_memory += bytes_to_free
                            
                            if freed_memory >= required_memory:
                                break
        
        return freed_memory >= required_memory
    
    def _enter_emergency_mode(self):
        """Enter emergency memory management mode"""
        self.emergency_mode = True
        self.logger.critical("Entered emergency memory mode")
        
        # Execute emergency actions
        for action in self.emergency_actions:
            try:
                action()
            except Exception as e:
                self.logger.error(f"Emergency action failed: {e}")
        
        # Force aggressive cleanup
        self.unified_manager.force_garbage_collection()
        self._rebalance_memory("Emergency mode activation")
    
    def _exit_emergency_mode(self):
        """Exit emergency memory management mode"""
        self.emergency_mode = False
        self.logger.info("Exited emergency memory mode")
    
    def _get_available_memory(self) -> int:
        """Get currently available memory"""
        return self.total_memory - self.allocated_memory
    
    def _calculate_optimal_allocation(self, container_id: str) -> int:
        """Calculate optimal memory allocation for container"""
        if container_id not in self.container_status:
            return 0
        
        spec = self.container_specs[container_id]
        status = self.container_status[container_id]
        
        # Use historical usage patterns
        if status.usage_history:
            recent_usage = [usage for _, usage in status.usage_history[-60:]]  # Last hour
            if recent_usage:
                avg_usage = statistics.mean(recent_usage)
                peak_usage = max(recent_usage)
                
                # Calculate optimal as average + buffer, but not exceeding peak * safety factor
                optimal = int(avg_usage * 1.3 + peak_usage * 0.2)  # 30% buffer + 20% of peak
                
                # Respect container limits
                optimal = max(spec.min_memory, min(optimal, spec.max_memory))
                
                return optimal
        
        # Fallback to current allocation
        return status.allocated_memory
    
    def _update_container_docker_limits(self, container_id: str, memory_limit: int):
        """Update Docker container memory limits"""
        if not self.docker_client:
            return
        
        try:
            container = self.docker_client.containers.get(container_id)
            
            # Update container resource limits
            container.update(mem_limit=memory_limit)
            
            self.logger.debug(f"Updated Docker memory limit for {container_id}: {memory_limit/1024/1024:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Failed to update Docker limits for {container_id}: {e}")
    
    def _update_docker_container_info(self):
        """Update Docker container information"""
        if not self.docker_client:
            return
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                container_id = container.id[:12]
                if container_id in self.container_specs:
                    stats = container.stats(stream=False)
                    
                    self.container_docker_info[container_id] = {
                        'name': container.name,
                        'status': container.status,
                        'memory_usage': stats.get('memory', {}).get('usage', 0),
                        'memory_limit': stats.get('memory', {}).get('limit', 0),
                        'cpu_usage': stats.get('cpu_stats', {}),
                        'last_updated': time.time()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to update Docker container info: {e}")
    
    def _handle_memory_alert(self, alert):
        """Handle memory alerts from monitor"""
        if alert.level in {MemoryAlertLevel.CRITICAL, MemoryAlertLevel.EMERGENCY}:
            if alert.component != 'system':
                # Container-specific alert
                if alert.component in self.container_specs:
                    # Try scaling or rebalancing
                    if not self.scale_container_memory(alert.component):
                        self._rebalance_memory(f"Alert: {alert.message}")
            else:
                # System-wide alert
                if not self.emergency_mode:
                    self._enter_emergency_mode()


# Global container orchestrator instance
_container_orchestrator = None
_orchestrator_lock = threading.Lock()


def get_container_orchestrator() -> ContainerOrchestrator:
    """Get singleton instance of container orchestrator"""
    global _container_orchestrator
    
    if _container_orchestrator is None:
        with _orchestrator_lock:
            if _container_orchestrator is None:
                _container_orchestrator = ContainerOrchestrator()
    
    return _container_orchestrator


# Convenience functions

def start_container_orchestration():
    """Start container orchestration"""
    orchestrator = get_container_orchestrator()
    orchestrator.start()


def stop_container_orchestration():
    """Stop container orchestration"""
    orchestrator = get_container_orchestrator()
    orchestrator.stop()


def register_trading_container(
    container_id: str,
    container_name: str,
    min_memory_mb: int = 256,
    max_memory_mb: int = 2048
) -> bool:
    """Register a trading container with appropriate settings"""
    orchestrator = get_container_orchestrator()
    
    spec = ContainerMemorySpec(
        container_id=container_id,
        container_name=container_name,
        priority=ContainerPriority.CRITICAL,
        strategy=AllocationStrategy.GUARANTEED,
        min_memory=min_memory_mb * 1024 * 1024,
        max_memory=max_memory_mb * 1024 * 1024,
        target_memory=min_memory_mb * 1024 * 1024,
        primary_workload=MemoryWorkloadType.TRADING_DATA,
        workload_mix={MemoryWorkloadType.TRADING_DATA: 1.0},
        latency_sensitive=True,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3
    )
    
    return orchestrator.register_container(spec)


def get_memory_status() -> Dict[str, Any]:
    """Get current memory status across all containers"""
    orchestrator = get_container_orchestrator()
    return orchestrator.get_memory_status()
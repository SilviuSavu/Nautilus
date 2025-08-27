"""
Process Priority Manager for Trading Platform
=============================================

Manages process priorities and scheduling for ultra-low latency trading operations.
Integrates with CPU affinity manager for optimal core allocation.
"""

import os
import sys
import time
import psutil
import threading
import subprocess
from typing import Dict, List, Optional, Set, Tuple, Callable
from enum import Enum, IntEnum
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
import signal
import resource

from .cpu_affinity import CPUAffinityManager, WorkloadPriority, CoreType

logger = logging.getLogger(__name__)

class ProcessClass(Enum):
    """Process classification for trading platform"""
    TRADING_CORE = "trading_core"          # Order execution, market data ingestion
    RISK_MANAGEMENT = "risk_management"    # Risk calculations, position monitoring
    ANALYTICS = "analytics"                # Performance analytics, reporting
    DATA_PROCESSING = "data_processing"    # Data feeds, factor calculations
    BACKGROUND = "background"              # Backfill, maintenance, logging

class SchedulingPolicy(Enum):
    """Scheduling policies for different process types"""
    REALTIME = "SCHED_FIFO"       # Real-time FIFO (highest priority)
    BATCH = "SCHED_BATCH"         # Batch processing
    NORMAL = "SCHED_NORMAL"       # Normal time-sharing
    IDLE = "SCHED_IDLE"           # Idle/background tasks

@dataclass
class ProcessInfo:
    """Information about a managed process"""
    pid: int
    process_class: ProcessClass
    priority: WorkloadPriority
    scheduling_policy: SchedulingPolicy
    assigned_cores: List[int] = field(default_factory=list)
    nice_value: int = 0
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    last_stats_update: float = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    is_active: bool = True

class ProcessManager:
    """
    Advanced process management for trading platform with M4 Max optimization
    """
    
    def __init__(self, cpu_affinity_manager: CPUAffinityManager):
        self.cpu_affinity_manager = cpu_affinity_manager
        self.managed_processes: Dict[int, ProcessInfo] = {}
        self.process_groups: Dict[ProcessClass, Set[int]] = {
            cls: set() for cls in ProcessClass
        }
        
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._rebalance_thread: Optional[threading.Thread] = None
        
        # Process limits per class
        self.class_limits = {
            ProcessClass.TRADING_CORE: {"max_processes": 4, "memory_mb": 2048},
            ProcessClass.RISK_MANAGEMENT: {"max_processes": 2, "memory_mb": 1024},
            ProcessClass.ANALYTICS: {"max_processes": 6, "memory_mb": 1024},
            ProcessClass.DATA_PROCESSING: {"max_processes": 8, "memory_mb": 512},
            ProcessClass.BACKGROUND: {"max_processes": 10, "memory_mb": 256}
        }
        
        # Market condition callbacks
        self._market_condition_callbacks: List[Callable] = []
        self._current_market_condition = "normal"
        
        self._initialize_monitoring()
    
    def _initialize_monitoring(self) -> None:
        """Initialize process monitoring and management threads"""
        self._monitoring_active = True
        
        # Stats monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_processes,
            daemon=True,
            name="ProcessMonitor"
        )
        self._monitor_thread.start()
        
        # Rebalancing thread
        self._rebalance_thread = threading.Thread(
            target=self._rebalance_processes,
            daemon=True,
            name="ProcessRebalancer"
        )
        self._rebalance_thread.start()
        
        logger.info("Process monitoring initialized")
    
    def register_process(
        self,
        pid: int,
        process_class: ProcessClass,
        priority: Optional[WorkloadPriority] = None,
        preferred_cores: Optional[List[int]] = None,
        memory_limit_mb: Optional[int] = None
    ) -> bool:
        """
        Register a process for management
        """
        try:
            # Validate process exists
            process = psutil.Process(pid)
            
            # Determine priority if not specified
            if priority is None:
                priority = self._get_default_priority(process_class)
            
            # Determine scheduling policy
            scheduling_policy = self._get_scheduling_policy(process_class, priority)
            
            with self._lock:
                # Check class limits
                if not self._check_class_limits(process_class):
                    logger.warning(f"Class {process_class} at maximum capacity")
                    return False
                
                # Create process info
                process_info = ProcessInfo(
                    pid=pid,
                    process_class=process_class,
                    priority=priority,
                    scheduling_policy=scheduling_policy,
                    memory_limit_mb=memory_limit_mb or self.class_limits[process_class]["memory_mb"]
                )
                
                # Set process priority and scheduling
                success = self._configure_process(process_info, preferred_cores)
                
                if success:
                    self.managed_processes[pid] = process_info
                    self.process_groups[process_class].add(pid)
                    
                    logger.info(f"Registered process {pid} as {process_class.value} "
                              f"with priority {priority.name}")
                    return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Cannot register process {pid}: {e}")
        except Exception as e:
            logger.error(f"Error registering process {pid}: {e}")
            
        return False
    
    def _get_default_priority(self, process_class: ProcessClass) -> WorkloadPriority:
        """Get default priority for process class"""
        priority_map = {
            ProcessClass.TRADING_CORE: WorkloadPriority.ULTRA_LOW_LATENCY,
            ProcessClass.RISK_MANAGEMENT: WorkloadPriority.LOW_LATENCY,
            ProcessClass.ANALYTICS: WorkloadPriority.NORMAL,
            ProcessClass.DATA_PROCESSING: WorkloadPriority.NORMAL,
            ProcessClass.BACKGROUND: WorkloadPriority.BACKGROUND
        }
        return priority_map[process_class]
    
    def _get_scheduling_policy(
        self, 
        process_class: ProcessClass, 
        priority: WorkloadPriority
    ) -> SchedulingPolicy:
        """Determine optimal scheduling policy"""
        if process_class == ProcessClass.TRADING_CORE:
            return SchedulingPolicy.REALTIME
        elif process_class == ProcessClass.RISK_MANAGEMENT:
            return SchedulingPolicy.NORMAL if priority == WorkloadPriority.LOW_LATENCY else SchedulingPolicy.BATCH
        elif process_class in [ProcessClass.ANALYTICS, ProcessClass.DATA_PROCESSING]:
            return SchedulingPolicy.BATCH
        else:  # BACKGROUND
            return SchedulingPolicy.IDLE
    
    def _check_class_limits(self, process_class: ProcessClass) -> bool:
        """Check if process class is within limits"""
        current_count = len(self.process_groups[process_class])
        max_count = self.class_limits[process_class]["max_processes"]
        return current_count < max_count
    
    def _configure_process(
        self,
        process_info: ProcessInfo,
        preferred_cores: Optional[List[int]] = None
    ) -> bool:
        """
        Configure process with optimal settings
        """
        try:
            process = psutil.Process(process_info.pid)
            
            # 1. Set CPU affinity
            cores_assigned = self.cpu_affinity_manager.assign_process_to_cores(
                process_info.pid,
                process_info.priority,
                preferred_cores
            )
            
            if cores_assigned:
                # Get assigned cores (simplified - would need to track this properly)
                process_info.assigned_cores = self.cpu_affinity_manager.get_available_cores(
                    CoreType.PERFORMANCE if process_info.priority in [
                        WorkloadPriority.ULTRA_LOW_LATENCY,
                        WorkloadPriority.LOW_LATENCY
                    ] else None
                )[:2]  # Assign up to 2 cores
            
            # 2. Set process priority (nice value)
            nice_value = self._get_nice_value(process_info.process_class)
            if sys.platform != "win32":
                os.setpriority(os.PRIO_PROCESS, process_info.pid, nice_value)
                process_info.nice_value = nice_value
            
            # 3. Set scheduling policy (Linux/macOS)
            if sys.platform == "linux":
                self._set_linux_scheduling(process_info)
            elif sys.platform == "darwin":
                self._set_macos_scheduling(process_info)
            
            # 4. Set resource limits
            self._set_resource_limits(process_info)
            
            # 5. Configure I/O priority
            if hasattr(process, 'ionice'):
                io_class = self._get_io_class(process_info.process_class)
                process.ionice(io_class)
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring process {process_info.pid}: {e}")
            return False
    
    def _get_nice_value(self, process_class: ProcessClass) -> int:
        """Get nice value for process class"""
        nice_map = {
            ProcessClass.TRADING_CORE: -10,      # High priority
            ProcessClass.RISK_MANAGEMENT: -5,    # Above normal
            ProcessClass.ANALYTICS: 0,           # Normal
            ProcessClass.DATA_PROCESSING: 5,     # Below normal
            ProcessClass.BACKGROUND: 10          # Low priority
        }
        return nice_map[process_class]
    
    def _set_linux_scheduling(self, process_info: ProcessInfo) -> None:
        """Set Linux-specific scheduling parameters"""
        try:
            if process_info.scheduling_policy == SchedulingPolicy.REALTIME:
                # Set real-time FIFO scheduling
                subprocess.run([
                    "chrt", "-f", "-p", "99", str(process_info.pid)
                ], check=True, capture_output=True)
            elif process_info.scheduling_policy == SchedulingPolicy.BATCH:
                subprocess.run([
                    "chrt", "-b", "-p", "0", str(process_info.pid)
                ], check=True, capture_output=True)
            elif process_info.scheduling_policy == SchedulingPolicy.IDLE:
                subprocess.run([
                    "chrt", "-i", "-p", "0", str(process_info.pid)
                ], check=True, capture_output=True)
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not set scheduling policy for {process_info.pid}: {e}")
    
    def _set_macos_scheduling(self, process_info: ProcessInfo) -> None:
        """Set macOS-specific scheduling parameters"""
        try:
            # Use taskpolicy for macOS scheduling hints
            if process_info.process_class == ProcessClass.TRADING_CORE:
                # Set utility QoS for performance
                subprocess.run([
                    "taskpolicy", "-c", "utility", "-p", str(process_info.pid)
                ], capture_output=True)
            elif process_info.process_class == ProcessClass.BACKGROUND:
                # Set background QoS
                subprocess.run([
                    "taskpolicy", "-c", "background", "-p", str(process_info.pid)
                ], capture_output=True)
                
        except Exception as e:
            logger.warning(f"Could not set macOS scheduling for {process_info.pid}: {e}")
    
    def _set_resource_limits(self, process_info: ProcessInfo) -> None:
        """Set resource limits for process"""
        try:
            process = psutil.Process(process_info.pid)
            
            # Memory limit
            if process_info.memory_limit_mb:
                # This is informational - actual enforcement would require cgroups on Linux
                # or similar mechanisms
                pass
            
            # Set CPU limit if supported
            if hasattr(process, 'cpu_limit'):
                if process_info.cpu_limit_percent:
                    process.cpu_limit(process_info.cpu_limit_percent)
            
        except Exception as e:
            logger.warning(f"Could not set resource limits for {process_info.pid}: {e}")
    
    def _get_io_class(self, process_class: ProcessClass) -> int:
        """Get I/O scheduling class"""
        # psutil ionice classes: 0=None, 1=RT, 2=BE, 3=Idle
        io_map = {
            ProcessClass.TRADING_CORE: 1,        # Real-time
            ProcessClass.RISK_MANAGEMENT: 2,     # Best effort
            ProcessClass.ANALYTICS: 2,           # Best effort
            ProcessClass.DATA_PROCESSING: 2,     # Best effort
            ProcessClass.BACKGROUND: 3           # Idle
        }
        return io_map[process_class]
    
    def _monitor_processes(self) -> None:
        """Monitor managed processes"""
        while self._monitoring_active:
            try:
                with self._lock:
                    dead_processes = []
                    
                    for pid, process_info in self.managed_processes.items():
                        try:
                            process = psutil.Process(pid)
                            
                            # Update statistics
                            process_info.cpu_usage = process.cpu_percent()
                            process_info.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                            process_info.last_stats_update = time.time()
                            
                            # Check resource limits
                            self._check_resource_limits(process_info)
                            
                        except psutil.NoSuchProcess:
                            dead_processes.append(pid)
                        except Exception as e:
                            logger.error(f"Error monitoring process {pid}: {e}")
                    
                    # Clean up dead processes
                    for pid in dead_processes:
                        self._cleanup_process(pid)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                time.sleep(5)
    
    def _check_resource_limits(self, process_info: ProcessInfo) -> None:
        """Check and enforce resource limits"""
        # Memory limit check
        if (process_info.memory_limit_mb and 
            process_info.memory_usage > process_info.memory_limit_mb * 1.2):  # 20% buffer
            
            logger.warning(f"Process {process_info.pid} exceeding memory limit: "
                         f"{process_info.memory_usage:.1f}MB > {process_info.memory_limit_mb}MB")
            
            # Could implement memory pressure handling here
    
    def _rebalance_processes(self) -> None:
        """Rebalance processes across cores periodically"""
        while self._monitoring_active:
            try:
                time.sleep(30)  # Rebalance every 30 seconds
                
                if not self._monitoring_active:
                    break
                
                # Trigger CPU affinity rebalancing
                rebalance_stats = self.cpu_affinity_manager.rebalance_workloads()
                
                if rebalance_stats["moved"] > 0:
                    logger.info(f"Rebalanced {rebalance_stats['moved']} processes "
                              f"across {rebalance_stats['optimized']} cores")
                
            except Exception as e:
                logger.error(f"Error in process rebalancing: {e}")
                time.sleep(60)
    
    def _cleanup_process(self, pid: int) -> None:
        """Clean up dead process"""
        try:
            if pid in self.managed_processes:
                process_info = self.managed_processes[pid]
                
                # Remove from groups
                self.process_groups[process_info.process_class].discard(pid)
                
                # Release CPU affinity
                self.cpu_affinity_manager.release_process(pid)
                
                # Remove from managed processes
                del self.managed_processes[pid]
                
                logger.info(f"Cleaned up dead process {pid}")
                
        except Exception as e:
            logger.error(f"Error cleaning up process {pid}: {e}")
    
    def update_market_condition(self, condition: str) -> None:
        """
        Update market condition and adjust process priorities accordingly
        """
        self._current_market_condition = condition
        
        with self._lock:
            if condition == "high_volatility":
                # Boost trading and risk management processes
                self._boost_critical_processes()
            elif condition == "market_close":
                # Allow background processes more resources
                self._relax_background_processes()
            
            # Trigger callbacks
            for callback in self._market_condition_callbacks:
                try:
                    callback(condition)
                except Exception as e:
                    logger.error(f"Error in market condition callback: {e}")
    
    def _boost_critical_processes(self) -> None:
        """Boost critical processes during high volatility"""
        critical_classes = [ProcessClass.TRADING_CORE, ProcessClass.RISK_MANAGEMENT]
        
        for process_class in critical_classes:
            for pid in self.process_groups[process_class]:
                try:
                    # Increase priority temporarily
                    if sys.platform != "win32":
                        current_nice = os.getpriority(os.PRIO_PROCESS, pid)
                        new_nice = max(current_nice - 5, -20)
                        os.setpriority(os.PRIO_PROCESS, pid, new_nice)
                        
                except Exception as e:
                    logger.error(f"Error boosting process {pid}: {e}")
    
    def _relax_background_processes(self) -> None:
        """Allow background processes more resources during quiet periods"""
        for pid in self.process_groups[ProcessClass.BACKGROUND]:
            try:
                # Decrease nice value (higher priority)
                if sys.platform != "win32":
                    current_nice = os.getpriority(os.PRIO_PROCESS, pid)
                    new_nice = min(current_nice - 2, 5)
                    os.setpriority(os.PRIO_PROCESS, pid, new_nice)
                    
            except Exception as e:
                logger.error(f"Error relaxing process {pid}: {e}")
    
    def add_market_condition_callback(self, callback: Callable) -> None:
        """Add callback for market condition changes"""
        self._market_condition_callbacks.append(callback)
    
    def get_process_stats(self) -> Dict:
        """Get comprehensive process statistics"""
        with self._lock:
            stats = {
                "total_managed": len(self.managed_processes),
                "by_class": {},
                "resource_usage": {
                    "total_cpu": 0.0,
                    "total_memory_mb": 0.0
                },
                "current_market_condition": self._current_market_condition
            }
            
            # Group by class
            for process_class in ProcessClass:
                class_processes = self.process_groups[process_class]
                class_stats = {
                    "count": len(class_processes),
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0
                }
                
                for pid in class_processes:
                    if pid in self.managed_processes:
                        info = self.managed_processes[pid]
                        class_stats["cpu_usage"] += info.cpu_usage
                        class_stats["memory_usage"] += info.memory_usage
                
                stats["by_class"][process_class.value] = class_stats
                stats["resource_usage"]["total_cpu"] += class_stats["cpu_usage"]
                stats["resource_usage"]["total_memory_mb"] += class_stats["memory_usage"]
            
            return stats
    
    def unregister_process(self, pid: int) -> bool:
        """Unregister a process from management"""
        try:
            with self._lock:
                if pid not in self.managed_processes:
                    return False
                
                self._cleanup_process(pid)
                logger.info(f"Unregistered process {pid}")
                return True
                
        except Exception as e:
            logger.error(f"Error unregistering process {pid}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the process manager"""
        self._monitoring_active = False
        
        # Wait for threads to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        if self._rebalance_thread and self._rebalance_thread.is_alive():
            self._rebalance_thread.join(timeout=5)
        
        # Clean up all managed processes
        with self._lock:
            for pid in list(self.managed_processes.keys()):
                self.cpu_affinity_manager.release_process(pid)
        
        logger.info("Process Manager shutdown complete")
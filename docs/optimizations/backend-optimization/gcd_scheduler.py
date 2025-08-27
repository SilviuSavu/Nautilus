"""
Grand Central Dispatch (GCD) Scheduler for macOS
================================================

Native macOS integration using Grand Central Dispatch for optimal task scheduling
on M4 Max architecture. Provides QoS-aware dispatch queue management for trading workloads.
"""

import os
import sys
import time
import threading
import subprocess
import ctypes
import ctypes.util
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum, IntEnum
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import weakref

logger = logging.getLogger(__name__)

class QoSClass(IntEnum):
    """Quality of Service classes matching macOS QOS_CLASS_*"""
    USER_INTERACTIVE = 0x21    # 33 - Highest priority, user-facing
    USER_INITIATED = 0x19      # 25 - High priority, user-initiated
    DEFAULT = 0x15             # 21 - Default priority
    UTILITY = 0x11             # 17 - Utility, energy efficient
    BACKGROUND = 0x09          # 9 - Background, lowest priority
    UNSPECIFIED = 0x00         # 0 - No QoS specified

class DispatchQueuePriority(Enum):
    """Dispatch queue priorities"""
    HIGH = "DISPATCH_QUEUE_PRIORITY_HIGH"
    DEFAULT = "DISPATCH_QUEUE_PRIORITY_DEFAULT" 
    LOW = "DISPATCH_QUEUE_PRIORITY_LOW"
    BACKGROUND = "DISPATCH_QUEUE_PRIORITY_BACKGROUND"

@dataclass
class TaskInfo:
    """Information about a scheduled task"""
    task_id: str
    qos_class: QoSClass
    queue_name: str
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None

class GCDScheduler:
    """
    Grand Central Dispatch scheduler for macOS with M4 Max optimization
    """
    
    def __init__(self):
        self.is_macos = sys.platform == "darwin"
        self.task_history: Dict[str, TaskInfo] = {}
        self.active_tasks: Dict[str, Future] = {}
        
        # Thread pools for different QoS classes
        self.thread_pools: Dict[QoSClass, ThreadPoolExecutor] = {}
        
        # Task queues
        self.task_queues: Dict[str, queue.Queue] = {}
        self.queue_threads: Dict[str, threading.Thread] = {}
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Initialize GCD integration
        self._initialize_gcd()
        self._create_standard_queues()
    
    def _initialize_gcd(self) -> None:
        """Initialize GCD integration for macOS"""
        if not self.is_macos:
            logger.warning("GCD scheduler running on non-macOS platform - using fallback")
            return
        
        try:
            # Load system libraries
            self._load_system_libraries()
            
            # Create thread pools with QoS-appropriate sizing
            self._create_thread_pools()
            
            logger.info("GCD scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCD: {e}")
            # Fall back to standard thread pools
            self._create_fallback_pools()
    
    def _load_system_libraries(self) -> None:
        """Load macOS system libraries for QoS control"""
        try:
            # Load libSystem for pthread QoS APIs
            self.libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
            
            # Define pthread_set_qos_class_self_np function
            self.pthread_set_qos_class_self_np = self.libsystem.pthread_set_qos_class_self_np
            self.pthread_set_qos_class_self_np.argtypes = [ctypes.c_int, ctypes.c_int]
            self.pthread_set_qos_class_self_np.restype = ctypes.c_int
            
        except Exception as e:
            logger.warning(f"Could not load system libraries: {e}")
            self.libsystem = None
    
    def _create_thread_pools(self) -> None:
        """Create QoS-aware thread pools"""
        # Pool configurations based on M4 Max architecture
        pool_configs = {
            QoSClass.USER_INTERACTIVE: {"max_workers": 4, "thread_name_prefix": "GCD-Interactive"},
            QoSClass.USER_INITIATED: {"max_workers": 6, "thread_name_prefix": "GCD-Initiated"},
            QoSClass.DEFAULT: {"max_workers": 8, "thread_name_prefix": "GCD-Default"},
            QoSClass.UTILITY: {"max_workers": 4, "thread_name_prefix": "GCD-Utility"},
            QoSClass.BACKGROUND: {"max_workers": 2, "thread_name_prefix": "GCD-Background"}
        }
        
        for qos_class, config in pool_configs.items():
            self.thread_pools[qos_class] = ThreadPoolExecutor(
                max_workers=config["max_workers"],
                thread_name_prefix=config["thread_name_prefix"]
            )
    
    def _create_fallback_pools(self) -> None:
        """Create fallback thread pools for non-macOS systems"""
        for qos_class in QoSClass:
            max_workers = 4 if qos_class in [QoSClass.USER_INTERACTIVE, QoSClass.USER_INITIATED] else 2
            self.thread_pools[qos_class] = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"Fallback-{qos_class.name}"
            )
    
    def _create_standard_queues(self) -> None:
        """Create standard dispatch queues for different workload types"""
        standard_queues = {
            "trading.orders": QoSClass.USER_INTERACTIVE,      # Order execution
            "trading.market_data": QoSClass.USER_INITIATED,   # Market data processing
            "trading.risk": QoSClass.USER_INITIATED,          # Risk calculations
            "trading.analytics": QoSClass.DEFAULT,             # Analytics processing
            "trading.data_processing": QoSClass.UTILITY,       # Data processing
            "trading.background": QoSClass.BACKGROUND          # Background tasks
        }
        
        for queue_name, qos_class in standard_queues.items():
            self.create_queue(queue_name, qos_class)
    
    def create_queue(
        self,
        name: str,
        qos_class: QoSClass = QoSClass.DEFAULT,
        concurrent: bool = True
    ) -> bool:
        """
        Create a new dispatch queue
        """
        try:
            with self._lock:
                if name in self.task_queues:
                    logger.warning(f"Queue {name} already exists")
                    return False
                
                # Create task queue
                self.task_queues[name] = queue.Queue()
                
                # Create worker thread for the queue
                worker_thread = threading.Thread(
                    target=self._queue_worker,
                    args=(name, qos_class),
                    daemon=True,
                    name=f"Queue-{name}"
                )
                
                self.queue_threads[name] = worker_thread
                worker_thread.start()
                
                logger.info(f"Created dispatch queue '{name}' with QoS {qos_class.name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating queue {name}: {e}")
            return False
    
    def _queue_worker(self, queue_name: str, qos_class: QoSClass) -> None:
        """Worker thread for processing queue tasks"""
        # Set QoS for this thread
        self._set_thread_qos(qos_class)
        
        task_queue = self.task_queues[queue_name]
        
        while not self._shutdown:
            try:
                # Get task from queue with timeout
                task_item = task_queue.get(timeout=1.0)
                
                if task_item is None:  # Shutdown signal
                    break
                
                task_func, task_args, task_kwargs, task_info = task_item
                
                # Execute task
                self._execute_task(task_func, task_args, task_kwargs, task_info)
                
                task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in queue worker {queue_name}: {e}")
    
    def _set_thread_qos(self, qos_class: QoSClass) -> None:
        """Set Quality of Service for current thread"""
        if not self.is_macos or not self.libsystem:
            return
        
        try:
            # Set pthread QoS class
            result = self.pthread_set_qos_class_self_np(int(qos_class), 0)
            if result != 0:
                logger.warning(f"Failed to set QoS class {qos_class}: {result}")
                
        except Exception as e:
            logger.warning(f"Error setting thread QoS: {e}")
    
    def dispatch_async(
        self,
        queue_name: str,
        task_func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Dispatch a task asynchronously to a named queue
        """
        if task_id is None:
            task_id = f"{queue_name}_{int(time.time() * 1000000)}"
        
        try:
            with self._lock:
                if queue_name not in self.task_queues:
                    logger.error(f"Queue {queue_name} does not exist")
                    return ""
                
                # Create task info
                task_info = TaskInfo(
                    task_id=task_id,
                    qos_class=self._get_queue_qos(queue_name),
                    queue_name=queue_name,
                    submitted_at=time.time()
                )
                
                # Add to task history
                self.task_history[task_id] = task_info
                
                # Submit to queue
                self.task_queues[queue_name].put((task_func, args, kwargs, task_info))
                
                self.stats["tasks_submitted"] += 1
                
                logger.debug(f"Dispatched task {task_id} to queue {queue_name}")
                return task_id
                
        except Exception as e:
            logger.error(f"Error dispatching task to {queue_name}: {e}")
            return ""
    
    def dispatch_sync(
        self,
        queue_name: str,
        task_func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Dispatch a task synchronously to a named queue
        """
        task_id = self.dispatch_async(queue_name, task_func, *args, **kwargs)
        
        if not task_id:
            raise RuntimeError("Failed to dispatch task")
        
        return self.wait_for_task(task_id, timeout)
    
    def dispatch_after(
        self,
        delay: float,
        queue_name: str,
        task_func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Dispatch a task after a delay
        """
        def delayed_task():
            time.sleep(delay)
            return task_func(*args, **kwargs)
        
        return self.dispatch_async(queue_name, delayed_task)
    
    def dispatch_barrier_async(
        self,
        queue_name: str,
        barrier_func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Dispatch a barrier task that will run after all currently queued tasks
        """
        # Simple implementation - add barrier to queue
        # In real GCD, this would ensure no other tasks run until barrier completes
        return self.dispatch_async(queue_name, barrier_func, *args, **kwargs)
    
    def create_group(self) -> 'DispatchGroup':
        """Create a new dispatch group"""
        return DispatchGroup(self)
    
    def _execute_task(
        self,
        task_func: Callable,
        task_args: tuple,
        task_kwargs: dict,
        task_info: TaskInfo
    ) -> None:
        """Execute a task and update statistics"""
        task_info.started_at = time.time()
        
        try:
            # Execute the task
            result = task_func(*task_args, **task_kwargs)
            task_info.result = result
            task_info.completed_at = time.time()
            task_info.execution_time = task_info.completed_at - task_info.started_at
            
            with self._lock:
                self.stats["tasks_completed"] += 1
                self.stats["total_execution_time"] += task_info.execution_time
            
            logger.debug(f"Task {task_info.task_id} completed in {task_info.execution_time:.4f}s")
            
        except Exception as e:
            task_info.error = e
            task_info.completed_at = time.time()
            
            with self._lock:
                self.stats["tasks_failed"] += 1
            
            logger.error(f"Task {task_info.task_id} failed: {e}")
    
    def _get_queue_qos(self, queue_name: str) -> QoSClass:
        """Get QoS class for a queue (simplified mapping)"""
        qos_mapping = {
            "trading.orders": QoSClass.USER_INTERACTIVE,
            "trading.market_data": QoSClass.USER_INITIATED,
            "trading.risk": QoSClass.USER_INITIATED,
            "trading.analytics": QoSClass.DEFAULT,
            "trading.data_processing": QoSClass.UTILITY,
            "trading.background": QoSClass.BACKGROUND
        }
        return qos_mapping.get(queue_name, QoSClass.DEFAULT)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a task to complete and return its result"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id not in self.task_history:
                    raise ValueError(f"Task {task_id} not found")
                
                task_info = self.task_history[task_id]
                
                if task_info.completed_at is not None:
                    if task_info.error:
                        raise task_info.error
                    return task_info.result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            time.sleep(0.001)  # 1ms polling
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        # Note: This is a simplified implementation
        # Real cancellation would require more sophisticated queue management
        with self._lock:
            if task_id in self.task_history:
                task_info = self.task_history[task_id]
                if task_info.started_at is None:
                    # Task not started yet, mark as cancelled
                    task_info.error = Exception("Task cancelled")
                    task_info.completed_at = time.time()
                    return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get status of a task"""
        with self._lock:
            return self.task_history.get(task_id)
    
    def get_queue_stats(self, queue_name: str) -> Dict:
        """Get statistics for a specific queue"""
        with self._lock:
            if queue_name not in self.task_queues:
                return {}
            
            queue_tasks = [
                task for task in self.task_history.values()
                if task.queue_name == queue_name
            ]
            
            completed_tasks = [t for t in queue_tasks if t.completed_at is not None]
            failed_tasks = [t for t in queue_tasks if t.error is not None]
            
            avg_execution_time = 0.0
            if completed_tasks:
                total_time = sum(t.execution_time or 0 for t in completed_tasks)
                avg_execution_time = total_time / len(completed_tasks)
            
            return {
                "queue_name": queue_name,
                "qos_class": self._get_queue_qos(queue_name).name,
                "queue_size": self.task_queues[queue_name].qsize(),
                "total_tasks": len(queue_tasks),
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "avg_execution_time": avg_execution_time
            }
    
    def get_system_stats(self) -> Dict:
        """Get overall system statistics"""
        with self._lock:
            return {
                "platform": "macOS" if self.is_macos else "Other",
                "gcd_available": self.libsystem is not None,
                "active_queues": len(self.task_queues),
                "thread_pools": len(self.thread_pools),
                **self.stats
            }
    
    def shutdown(self, timeout: float = 10.0) -> None:
        """Shutdown the GCD scheduler"""
        logger.info("Shutting down GCD scheduler...")
        
        self._shutdown = True
        
        # Signal all queue workers to stop
        for queue_name, task_queue in self.task_queues.items():
            task_queue.put(None)  # Shutdown signal
        
        # Wait for queue threads
        for queue_name, thread in self.queue_threads.items():
            thread.join(timeout=timeout / len(self.queue_threads))
        
        # Shutdown thread pools
        for qos_class, pool in self.thread_pools.items():
            pool.shutdown(wait=True, timeout=timeout / len(self.thread_pools))
        
        logger.info("GCD scheduler shutdown complete")


class DispatchGroup:
    """
    Dispatch group for coordinating multiple tasks
    """
    
    def __init__(self, scheduler: GCDScheduler):
        self.scheduler = scheduler
        self.tasks: List[str] = []
        self._lock = threading.Lock()
    
    def dispatch_async(
        self,
        queue_name: str,
        task_func: Callable,
        *args,
        **kwargs
    ) -> str:
        """Dispatch task to group"""
        task_id = self.scheduler.dispatch_async(queue_name, task_func, *args, **kwargs)
        
        with self._lock:
            self.tasks.append(task_id)
        
        return task_id
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks in group to complete"""
        start_time = time.time()
        
        for task_id in self.tasks:
            remaining_timeout = None
            if timeout:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                
                if remaining_timeout <= 0:
                    return False
            
            try:
                self.scheduler.wait_for_task(task_id, remaining_timeout)
            except TimeoutError:
                return False
        
        return True
    
    def notify(self, queue_name: str, notification_func: Callable) -> None:
        """Execute notification function when all tasks complete"""
        def notify_wrapper():
            if self.wait():
                notification_func()
        
        # Dispatch notification task
        self.scheduler.dispatch_async(queue_name, notify_wrapper)
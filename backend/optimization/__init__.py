"""
CPU Core Optimization System for M4 Max
========================================

This module provides intelligent CPU core allocation and workload management
specifically optimized for Apple's M4 Max architecture with:
- 12 Performance cores (P-cores): 0-11
- 4 Efficiency cores (E-cores): 12-15

Key Components:
- CPU Affinity Manager: Core detection and assignment
- Process Manager: Priority-based process scheduling
- GCD Scheduler: macOS Grand Central Dispatch integration
- Performance Monitor: Real-time performance tracking
- Workload Classifier: Intelligent task categorization
"""

from .cpu_affinity import CPUAffinityManager
from .process_manager import ProcessManager
from .gcd_scheduler import GCDScheduler
from .performance_monitor import PerformanceMonitor
from .workload_classifier import WorkloadClassifier

__all__ = [
    'CPUAffinityManager',
    'ProcessManager', 
    'GCDScheduler',
    'PerformanceMonitor',
    'WorkloadClassifier'
]

__version__ = "1.0.0"
"""
Nautilus Trading Platform - Phase 5 Edge Computing Integration

This module provides edge computing capabilities for ultra-low latency trading
operations with intelligent placement optimization and regional performance tuning.

Components:
- Edge node management and deployment
- Intelligent edge placement algorithms  
- Edge caching and data replication
- Regional performance optimization
- Edge failover and consistency mechanisms
- Comprehensive edge monitoring
"""

from .edge_node_manager import EdgeNodeManager
from .edge_placement_optimizer import EdgePlacementOptimizer  
from .edge_cache_manager import EdgeCacheManager
from .regional_performance_optimizer import RegionalPerformanceOptimizer
from .edge_failover_manager import EdgeFailoverManager
from .edge_monitoring_system import EdgeMonitoringSystem

__version__ = "1.0.0"
__author__ = "Nautilus Platform"

__all__ = [
    "EdgeNodeManager",
    "EdgePlacementOptimizer", 
    "EdgeCacheManager",
    "RegionalPerformanceOptimizer",
    "EdgeFailoverManager",
    "EdgeMonitoringSystem"
]
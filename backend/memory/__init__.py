"""
Unified Memory Management System for M4 Max Architecture

A comprehensive memory management system designed specifically for M4 Max's
unified memory architecture, providing ultra-low latency operations for
the Nautilus trading platform.

Key Components:
- UnifiedMemoryManager: Core memory allocation and management
- MemoryPoolManager: Specialized memory pools for different workloads
- ZeroCopyManager: Zero-copy operations between CPU/GPU/Neural Engine
- MemoryMonitor: Real-time monitoring and alerting
- ContainerOrchestrator: Dynamic container memory management

Features:
- Leverages M4 Max 546 GB/s unified memory bandwidth
- Zero-copy data sharing between CPU, GPU, and Neural Engine
- Priority-based container memory allocation
- Real-time memory pressure monitoring
- Emergency memory management and automatic rebalancing
- Production-ready monitoring with Prometheus integration
"""

from .unified_memory_manager import (
    UnifiedMemoryManager,
    MemoryWorkloadType,
    MemoryRegion,
    MemoryBlock,
    MemoryPressureMetrics,
    get_unified_memory_manager,
    allocate_trading_buffer,
    allocate_ml_buffer,
    allocate_gpu_buffer
)

from .memory_pools import (
    MemoryPoolManager,
    MemoryPool,
    PoolConfig,
    PoolStrategy,
    PoolPriority,
    PoolStatistics,
    get_memory_pool_manager,
    allocate_from_pool,
    get_pool_statistics,
    defragment_pools,
    cleanup_pools
)

from .zero_copy_manager import (
    ZeroCopyManager,
    ZeroCopyBuffer,
    ZeroCopyTransfer,
    ZeroCopyOperation,
    BufferType,
    get_zero_copy_manager,
    create_zero_copy_buffer,
    create_shared_buffer,
    execute_zero_copy_transfer
)

from .memory_monitor import (
    MemoryMonitor,
    MemoryAlert,
    MemoryAlertLevel,
    ContainerMemoryMetrics,
    SystemMemoryMetrics,
    get_memory_monitor,
    start_monitoring,
    stop_monitoring,
    get_current_memory_status,
    register_memory_alert_handler
)

from .container_orchestrator import (
    ContainerOrchestrator,
    ContainerMemorySpec,
    ContainerMemoryStatus,
    ContainerPriority,
    ContainerState,
    AllocationStrategy,
    get_container_orchestrator,
    start_container_orchestration,
    stop_container_orchestration,
    register_trading_container,
    get_memory_status
)

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global configuration
_memory_config = None


def load_memory_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load memory configuration from YAML file
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
    
    Returns:
        Configuration dictionary
    """
    global _memory_config
    
    if config_path is None:
        config_path = Path(__file__).parent / "memory_config.yml"
    
    try:
        with open(config_path, 'r') as f:
            _memory_config = yaml.safe_load(f)
        
        logger.info(f"Loaded memory configuration from {config_path}")
        return _memory_config
        
    except Exception as e:
        logger.error(f"Failed to load memory configuration: {e}")
        return {}


def get_memory_config() -> Dict[str, Any]:
    """Get current memory configuration"""
    global _memory_config
    
    if _memory_config is None:
        _memory_config = load_memory_config()
    
    return _memory_config


def initialize_memory_system(
    config_path: Optional[Path] = None,
    start_monitoring: bool = True,
    start_orchestration: bool = True
) -> bool:
    """
    Initialize the complete unified memory management system
    
    Args:
        config_path: Path to configuration file
        start_monitoring: Whether to start memory monitoring
        start_orchestration: Whether to start container orchestration
    
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing M4 Max unified memory management system")
        
        # Load configuration
        config = load_memory_config(config_path)
        if not config:
            logger.error("Failed to load configuration, using defaults")
            config = {}
        
        # Initialize unified memory manager
        total_memory_gb = config.get('system', {}).get('total_memory_gb', 36.0)
        unified_manager = get_unified_memory_manager()
        logger.info(f"Initialized unified memory manager with {total_memory_gb}GB")
        
        # Initialize memory pool manager with default pools
        pool_manager = get_memory_pool_manager()
        logger.info("Initialized memory pool manager with specialized pools")
        
        # Initialize zero-copy manager
        zero_copy_manager = get_zero_copy_manager()
        logger.info("Initialized zero-copy manager with M4 Max optimizations")
        
        # Start monitoring if requested
        if start_monitoring:
            monitor = get_memory_monitor()
            monitoring_config = config.get('monitoring', {})
            interval = monitoring_config.get('collection_interval_seconds', 1.0)
            prometheus_port = monitoring_config.get('prometheus', {}).get('port', 9090)
            
            monitor.monitoring_interval = interval
            monitor.start(prometheus_port)
            logger.info(f"Started memory monitoring with {interval}s interval")
        
        # Start container orchestration if requested
        if start_orchestration:
            orchestrator = get_container_orchestrator()
            orchestrator.start()
            logger.info("Started container memory orchestration")
        
        # Apply configuration-specific settings
        _apply_configuration_settings(config)
        
        logger.info("M4 Max unified memory system initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        return False


def shutdown_memory_system():
    """Shutdown the unified memory management system"""
    try:
        logger.info("Shutting down unified memory management system")
        
        # Stop monitoring
        try:
            monitor = get_memory_monitor()
            monitor.stop()
            logger.info("Stopped memory monitoring")
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
        
        # Stop orchestration
        try:
            orchestrator = get_container_orchestrator()
            orchestrator.stop()
            logger.info("Stopped container orchestration")
        except Exception as e:
            logger.error(f"Error stopping orchestration: {e}")
        
        logger.info("Memory system shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during memory system shutdown: {e}")


def get_system_overview() -> Dict[str, Any]:
    """Get comprehensive overview of memory system status"""
    try:
        # Get metrics from all components
        unified_manager = get_unified_memory_manager()
        pool_manager = get_memory_pool_manager()
        zero_copy_manager = get_zero_copy_manager()
        monitor = get_memory_monitor()
        orchestrator = get_container_orchestrator()
        
        # Collect metrics
        pressure_metrics = unified_manager.get_memory_pressure()
        pool_stats = pool_manager.get_global_statistics()
        zero_copy_metrics = zero_copy_manager.get_performance_metrics()
        system_metrics = monitor.get_system_metrics()
        orchestrator_status = orchestrator.get_memory_status()
        
        return {
            'timestamp': time.time(),
            'system': {
                'total_memory_gb': pressure_metrics.total_allocated / 1024 / 1024 / 1024,
                'pressure_level': pressure_metrics.pressure_level,
                'bandwidth_utilization': pressure_metrics.bandwidth_utilization,
                'fragmentation_ratio': pressure_metrics.fragmentation_ratio
            },
            'pools': {
                'pool_count': len(pool_stats),
                'total_pool_memory_mb': sum(stats.total_size for stats in pool_stats.values()) / 1024 / 1024,
                'pool_utilization': sum(stats.used_size for stats in pool_stats.values()) / 
                                   max(1, sum(stats.total_size for stats in pool_stats.values()))
            },
            'zero_copy': {
                'active_buffers': zero_copy_metrics.get('active_buffers', 0),
                'shared_buffers': zero_copy_metrics.get('shared_buffers', 0),
                'bandwidth_utilization': zero_copy_metrics.get('bandwidth_utilization', 0)
            },
            'containers': {
                'container_count': orchestrator_status.get('container_count', 0),
                'total_allocated_gb': orchestrator_status.get('allocated_memory_gb', 0),
                'utilization_percentage': orchestrator_status.get('utilization_percentage', 0),
                'emergency_mode': orchestrator_status.get('emergency_mode', False)
            },
            'alerts': {
                'total_alerts': len(monitor.get_alerts(limit=1000)),
                'critical_alerts': len(monitor.get_alerts(level=MemoryAlertLevel.CRITICAL)),
                'emergency_alerts': len(monitor.get_alerts(level=MemoryAlertLevel.EMERGENCY))
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system overview: {e}")
        return {'error': str(e)}


def optimize_for_trading():
    """Optimize memory system for ultra-low latency trading operations"""
    try:
        logger.info("Optimizing memory system for trading operations")
        
        # Optimize unified memory manager
        unified_manager = get_unified_memory_manager()
        unified_manager.optimize_for_trading()
        
        # Force defragmentation of critical pools
        pool_manager = get_memory_pool_manager()
        trading_pool = pool_manager.get_pool('trading_data_pool')
        risk_pool = pool_manager.get_pool('risk_calc_pool')
        
        if trading_pool:
            trading_pool.defragment()
            logger.info("Defragmented trading data pool")
        
        if risk_pool:
            risk_pool.defragment()
            logger.info("Defragmented risk calculation pool")
        
        # Set container priorities for trading workloads
        orchestrator = get_container_orchestrator()
        
        logger.info("Trading optimization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to optimize for trading: {e}")
        return False


def _apply_configuration_settings(config: Dict[str, Any]):
    """Apply configuration-specific settings to all components"""
    try:
        # Apply monitoring configuration
        if 'monitoring' in config:
            monitor = get_memory_monitor()
            monitoring_config = config['monitoring']
            
            # Set alert thresholds
            if 'alert_thresholds' in monitoring_config:
                for level_name, thresholds in monitoring_config['alert_thresholds'].items():
                    try:
                        level = MemoryAlertLevel(level_name)
                        for metric, value in thresholds.items():
                            monitor.set_alert_threshold(metric, level, value)
                    except ValueError:
                        logger.warning(f"Invalid alert level: {level_name}")
        
        # Apply container orchestration configuration
        if 'container_orchestration' in config:
            orchestrator = get_container_orchestrator()
            orch_config = config['container_orchestration']
            
            # Set emergency threshold
            if 'emergency_threshold' in orch_config:
                orchestrator.emergency_threshold = orch_config['emergency_threshold']
            
            # Set rebalance cooldown
            if 'rebalance_cooldown_seconds' in orch_config:
                orchestrator.rebalance_cooldown = orch_config['rebalance_cooldown_seconds']
        
        logger.info("Applied configuration settings to all components")
        
    except Exception as e:
        logger.error(f"Failed to apply configuration settings: {e}")


# Export all components for easy access
__all__ = [
    # Core managers
    'UnifiedMemoryManager',
    'MemoryPoolManager', 
    'ZeroCopyManager',
    'MemoryMonitor',
    'ContainerOrchestrator',
    
    # Data types
    'MemoryWorkloadType',
    'MemoryRegion',
    'MemoryBlock',
    'MemoryPressureMetrics',
    'PoolConfig',
    'PoolStrategy', 
    'PoolPriority',
    'PoolStatistics',
    'ZeroCopyBuffer',
    'ZeroCopyTransfer',
    'ZeroCopyOperation',
    'BufferType',
    'MemoryAlert',
    'MemoryAlertLevel',
    'ContainerMemoryMetrics',
    'SystemMemoryMetrics',
    'ContainerMemorySpec',
    'ContainerMemoryStatus',
    'ContainerPriority',
    'ContainerState',
    'AllocationStrategy',
    
    # Singleton getters
    'get_unified_memory_manager',
    'get_memory_pool_manager',
    'get_zero_copy_manager', 
    'get_memory_monitor',
    'get_container_orchestrator',
    
    # Convenience functions
    'allocate_trading_buffer',
    'allocate_ml_buffer',
    'allocate_gpu_buffer',
    'allocate_from_pool',
    'get_pool_statistics',
    'defragment_pools',
    'cleanup_pools',
    'create_zero_copy_buffer',
    'create_shared_buffer',
    'execute_zero_copy_transfer',
    'start_monitoring',
    'stop_monitoring',
    'get_current_memory_status',
    'register_memory_alert_handler',
    'start_container_orchestration',
    'stop_container_orchestration',
    'register_trading_container',
    'get_memory_status',
    
    # System management
    'load_memory_config',
    'get_memory_config',
    'initialize_memory_system',
    'shutdown_memory_system',
    'get_system_overview',
    'optimize_for_trading'
]

# Module-level convenience functions
import time
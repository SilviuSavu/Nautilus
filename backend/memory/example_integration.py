"""
Example Integration of Unified Memory Management System

This example demonstrates how to integrate and use the unified memory management
system in the Nautilus trading platform with M4 Max optimization.
"""

import asyncio
import logging
import time
from pathlib import Path

from . import (
    initialize_memory_system,
    get_unified_memory_manager,
    get_memory_pool_manager,
    get_zero_copy_manager,
    get_memory_monitor,
    get_container_orchestrator,
    register_trading_container,
    allocate_trading_buffer,
    create_zero_copy_buffer,
    MemoryWorkloadType,
    BufferType,
    ZeroCopyOperation,
    ContainerPriority,
    MemoryAlertLevel,
    optimize_for_trading,
    get_system_overview
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_custom_alert_handler():
    """Setup custom alert handler for trading-specific alerts"""
    
    def trading_alert_handler(alert):
        """Custom alert handler for trading operations"""
        if alert.level in {MemoryAlertLevel.CRITICAL, MemoryAlertLevel.EMERGENCY}:
            logger.critical(f"TRADING ALERT: {alert.message}")
            
            # Take immediate action for trading-critical alerts
            if "trading" in alert.component.lower() or alert.component.startswith("trading-"):
                logger.critical("Taking emergency action for trading container")
                # Here you would implement emergency trading-specific actions
                # such as pausing non-critical strategies, scaling up trading containers, etc.
        
        elif alert.level == MemoryAlertLevel.WARNING:
            logger.warning(f"Memory warning: {alert.message}")
            
            # Proactive optimization for warnings
            if alert.metric == "pressure_level":
                logger.info("Triggering proactive memory optimization")
                optimize_for_trading()
    
    # Register the handler
    monitor = get_memory_monitor()
    monitor.register_alert_handler(trading_alert_handler)
    logger.info("Registered custom trading alert handler")


async def demonstrate_trading_memory_operations():
    """Demonstrate memory operations optimized for trading workloads"""
    logger.info("Demonstrating trading memory operations")
    
    # Allocate ultra-low latency trading buffers
    market_data_buffer = allocate_trading_buffer(1024 * 1024, container_id="trading-engine-1")
    order_buffer = allocate_trading_buffer(512 * 1024, container_id="trading-engine-1") 
    risk_buffer = allocate_trading_buffer(256 * 1024, container_id="risk-engine-1")
    
    if market_data_buffer and order_buffer and risk_buffer:
        logger.info(f"Allocated trading buffers: market_data={hex(market_data_buffer)}, "
                   f"order={hex(order_buffer)}, risk={hex(risk_buffer)}")
        
        # Demonstrate zero-copy operations for ultra-low latency
        zero_copy_mgr = get_zero_copy_manager()
        
        # Create zero-copy buffers for market data processing
        with zero_copy_mgr.zero_copy_context(
            size=2 * 1024 * 1024,
            buffer_type=BufferType.UNIFIED_BUFFER,
            workload_type=MemoryWorkloadType.TRADING_DATA
        ) as trading_buffer:
            
            # Create GPU-accelerated analytics buffer
            gpu_analytics_buffer = create_zero_copy_buffer(
                size=4 * 1024 * 1024,
                buffer_type=BufferType.METAL_BUFFER,
                workload_type=MemoryWorkloadType.ANALYTICS
            )
            
            if gpu_analytics_buffer:
                logger.info(f"Created GPU analytics buffer: {hex(gpu_analytics_buffer.address)}")
                
                # Execute zero-copy transfer from trading data to GPU analytics
                transfer = zero_copy_mgr.execute_zero_copy_transfer(
                    src_buffer=trading_buffer,
                    dst_buffer=gpu_analytics_buffer,
                    operation=ZeroCopyOperation.CPU_TO_GPU
                )
                
                if transfer and transfer.success:
                    bandwidth_gbps = (transfer.bandwidth_achieved or 0) / (1024 * 1024 * 1024)
                    logger.info(f"Zero-copy transfer successful: {bandwidth_gbps:.2f} GB/s bandwidth")
                
                # Cleanup
                zero_copy_mgr.release_buffer(gpu_analytics_buffer)
    
    # Demonstrate container memory management
    orchestrator = get_container_orchestrator()
    
    # Register trading containers with appropriate priorities
    success = register_trading_container(
        container_id="trading-engine-primary",
        container_name="Primary Trading Engine",
        min_memory_mb=512,
        max_memory_mb=4096
    )
    
    if success:
        logger.info("Registered primary trading engine with guaranteed memory")


async def demonstrate_ml_memory_operations():
    """Demonstrate memory operations for ML workloads"""
    logger.info("Demonstrating ML memory operations")
    
    zero_copy_mgr = get_zero_copy_manager()
    
    # Create Neural Engine optimized buffer for ML models
    ml_model_buffer = create_zero_copy_buffer(
        size=16 * 1024 * 1024,  # 16MB for model
        buffer_type=BufferType.COREML_BUFFER,
        workload_type=MemoryWorkloadType.ML_MODELS,
        data_type=float
    )
    
    if ml_model_buffer:
        logger.info(f"Created ML model buffer: {hex(ml_model_buffer.address)}")
        
        # Create input data buffer
        input_buffer = create_zero_copy_buffer(
            size=1024 * 1024,  # 1MB input
            buffer_type=BufferType.UNIFIED_BUFFER,
            workload_type=MemoryWorkloadType.ML_MODELS
        )
        
        if input_buffer:
            # Demonstrate CPU to Neural Engine transfer
            transfer = zero_copy_mgr.execute_zero_copy_transfer(
                src_buffer=input_buffer,
                dst_buffer=ml_model_buffer,
                operation=ZeroCopyOperation.CPU_TO_NEURAL
            )
            
            if transfer and transfer.success:
                logger.info("ML data transfer to Neural Engine successful")
            
            # Cleanup
            zero_copy_mgr.release_buffer(input_buffer)
        
        zero_copy_mgr.release_buffer(ml_model_buffer)


async def demonstrate_cross_container_sharing():
    """Demonstrate cross-container memory sharing"""
    logger.info("Demonstrating cross-container memory sharing")
    
    # Create shared buffer for multiple analytics engines
    shared_buffer = create_shared_buffer(
        name="market_data_shared",
        size=8 * 1024 * 1024,  # 8MB shared buffer
        container_ids=["analytics-engine-1", "analytics-engine-2", "factor-engine-1"],
        workload_type=MemoryWorkloadType.ANALYTICS
    )
    
    if shared_buffer:
        logger.info(f"Created shared buffer for analytics engines: {hex(shared_buffer.address)}")
        
        # Simulate different containers accessing the shared buffer
        zero_copy_mgr = get_zero_copy_manager()
        
        # Container 1 gets the shared buffer
        container1_buffer = zero_copy_mgr.get_shared_buffer("market_data_shared")
        if container1_buffer:
            logger.info("Analytics engine 1 accessed shared buffer")
        
        # Container 2 gets the same shared buffer
        container2_buffer = zero_copy_mgr.get_shared_buffer("market_data_shared")  
        if container2_buffer:
            logger.info("Analytics engine 2 accessed shared buffer")
        
        # Cleanup - buffers will be automatically cleaned up when ref count reaches 0
        if container1_buffer:
            zero_copy_mgr.release_buffer(container1_buffer)
        if container2_buffer:
            zero_copy_mgr.release_buffer(container2_buffer)


async def monitor_system_performance():
    """Monitor system performance during operations"""
    logger.info("Starting system performance monitoring")
    
    for i in range(10):  # Monitor for 10 iterations
        await asyncio.sleep(2)  # 2-second intervals
        
        # Get comprehensive system overview
        overview = get_system_overview()
        
        if 'error' not in overview:
            system = overview['system']
            containers = overview['containers'] 
            alerts = overview['alerts']
            
            logger.info(
                f"System Status [{i+1}/10]: "
                f"Memory={system['total_memory_gb']:.1f}GB, "
                f"Pressure={system['pressure_level']:.1%}, "
                f"Bandwidth={system['bandwidth_utilization']:.1%}, "
                f"Containers={containers['container_count']}, "
                f"Alerts={alerts['total_alerts']}"
            )
            
            # Check for high pressure
            if system['pressure_level'] > 0.8:
                logger.warning("High memory pressure detected - triggering optimization")
                optimize_for_trading()
            
            # Check for critical alerts
            if alerts['critical_alerts'] > 0:
                logger.critical(f"Critical alerts detected: {alerts['critical_alerts']}")
        else:
            logger.error(f"Failed to get system overview: {overview['error']}")


async def main():
    """Main demonstration function"""
    logger.info("Starting M4 Max Unified Memory Management System Demo")
    
    # Initialize the memory system
    config_path = Path(__file__).parent / "memory_config.yml"
    success = initialize_memory_system(
        config_path=config_path,
        start_monitoring=True,
        start_orchestration=True
    )
    
    if not success:
        logger.error("Failed to initialize memory system")
        return
    
    # Setup custom alert handling
    setup_custom_alert_handler()
    
    # Wait for system to stabilize
    await asyncio.sleep(2)
    
    # Demonstrate different memory operations
    await demonstrate_trading_memory_operations()
    await asyncio.sleep(1)
    
    await demonstrate_ml_memory_operations()
    await asyncio.sleep(1)
    
    await demonstrate_cross_container_sharing()
    await asyncio.sleep(1)
    
    # Monitor system performance
    await monitor_system_performance()
    
    # Final system overview
    logger.info("\n" + "="*60)
    logger.info("FINAL SYSTEM OVERVIEW")
    logger.info("="*60)
    
    overview = get_system_overview()
    if 'error' not in overview:
        logger.info(f"Total Memory: {overview['system']['total_memory_gb']:.1f} GB")
        logger.info(f"Memory Pressure: {overview['system']['pressure_level']:.1%}")
        logger.info(f"Bandwidth Utilization: {overview['system']['bandwidth_utilization']:.1%}")
        logger.info(f"Fragmentation: {overview['system']['fragmentation_ratio']:.1%}")
        logger.info(f"Active Memory Pools: {overview['pools']['pool_count']}")
        logger.info(f"Pool Utilization: {overview['pools']['pool_utilization']:.1%}")
        logger.info(f"Zero-Copy Buffers: {overview['zero_copy']['active_buffers']}")
        logger.info(f"Container Count: {overview['containers']['container_count']}")
        logger.info(f"Container Memory: {overview['containers']['total_allocated_gb']:.1f} GB")
        logger.info(f"Total Alerts: {overview['alerts']['total_alerts']}")
        logger.info(f"Critical Alerts: {overview['alerts']['critical_alerts']}")
        logger.info(f"Emergency Mode: {'YES' if overview['containers']['emergency_mode'] else 'NO'}")
    
    logger.info("="*60)
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
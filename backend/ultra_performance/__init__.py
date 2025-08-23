"""
Ultra-Performance Optimization Framework for Nautilus Trading Platform

This module provides GPU acceleration, ultra-low latency optimizations,
advanced caching strategies, memory pool optimization, and network optimizations
for microsecond-level trading performance.

Key Features:
- CUDA GPU acceleration for risk calculations and Monte Carlo simulations
- Ultra-low latency optimizations with zero-copy memory techniques
- Intelligent cache warming and distributed caching strategies
- Custom memory allocators and object pooling
- Kernel bypass networking (DPDK) and zero-copy I/O
- Real-time performance monitoring and profiling
"""

from .gpu_acceleration import (
    CUDAManager,
    GPUAcceleratedRiskCalculator,
    GPUMonteCarloSimulator,
    GPUMatrixOperations,
    gpu_memory_manager
)

from .low_latency import (
    UltraLowLatencyOptimizer,
    ZeroCopyMemoryManager,
    CacheFriendlyStructures,
    LockFreeAlgorithms,
    MicrosecondTimer
)

from .advanced_caching import (
    IntelligentCacheWarmer,
    DistributedCacheManager,
    PredictiveCacheLoader,
    CacheCoherencyManager,
    CacheOptimizer
)

from .memory_pool import (
    CustomMemoryAllocator,
    ObjectPoolManager,
    GCOptimizer,
    MemoryMappedFileManager,
    MemoryProfiler
)

from .network_io import (
    DPDKNetworkManager,
    ZeroCopyIOManager,
    OptimizedSerializationProtocols,
    BatchProcessor,
    NetworkLatencyOptimizer
)

from .performance_monitoring import (
    RealTimeProfiler,
    GPUUtilizationMonitor,
    MemoryAllocationTracker,
    PerformanceRegressionDetector,
    UltraPerformanceMetrics
)

# Import global instances from modules
from .gpu_acceleration import (
    cuda_manager,
    gpu_risk_calculator,
    gpu_monte_carlo,
    gpu_matrix_ops
)

__version__ = "1.0.0"
__author__ = "Nautilus Trading Platform Ultra-Performance Team"

__all__ = [
    # GPU Acceleration
    "CUDAManager",
    "GPUAcceleratedRiskCalculator", 
    "GPUMonteCarloSimulator",
    "GPUMatrixOperations",
    "gpu_memory_manager",
    
    # Ultra-Low Latency
    "UltraLowLatencyOptimizer",
    "ZeroCopyMemoryManager",
    "CacheFriendlyStructures", 
    "LockFreeAlgorithms",
    "MicrosecondTimer",
    
    # Advanced Caching
    "IntelligentCacheWarmer",
    "DistributedCacheManager",
    "PredictiveCacheLoader",
    "CacheCoherencyManager",
    "CacheOptimizer",
    
    # Memory Pool Optimization
    "CustomMemoryAllocator",
    "ObjectPoolManager",
    "GCOptimizer",
    "MemoryMappedFileManager",
    "MemoryProfiler",
    
    # Network and I/O
    "DPDKNetworkManager",
    "ZeroCopyIOManager",
    "OptimizedSerializationProtocols",
    "BatchProcessor",
    "NetworkLatencyOptimizer",
    
    # Performance Monitoring
    "RealTimeProfiler",
    "GPUUtilizationMonitor",
    "MemoryAllocationTracker",
    "PerformanceRegressionDetector",
    "UltraPerformanceMetrics"
]
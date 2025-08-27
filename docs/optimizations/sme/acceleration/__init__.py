"""
ARM Scalable Matrix Extension (SME) Acceleration Module

This module provides hardware acceleration for matrix operations using Apple Silicon M4 Max
SME capabilities, delivering 2.9 TFLOPS FP32 peak performance for trading applications.

2025 SME Optimizations:
- JIT-generated kernels outperform vendor BLAS
- FP32 operations prioritized (FP64 is 4x slower)
- Memory bandwidth optimization for 546 GB/s utilization
- Seamless fallback to existing M4 Max optimizations
"""

from .sme_accelerator import SMEAccelerator
from .sme_hardware_router import SMEHardwareRouter
from .sme_memory_optimizer import SMEMemoryOptimizer
from .sme_performance_monitor import SMEPerformanceMonitor
from .sme_jit_kernels import SMEJITKernelGenerator

__version__ = "1.0.0"
__all__ = [
    "SMEAccelerator",
    "SMEHardwareRouter", 
    "SMEMemoryOptimizer",
    "SMEPerformanceMonitor",
    "SMEJITKernelGenerator"
]
"""
Nautilus M4 Max Benchmarking Suite
=================================

Comprehensive benchmarking and validation suite for M4 Max optimizations:
- Performance benchmarking for all M4 Max features
- Hardware validation and capability testing
- Container performance analysis
- Trading-specific latency benchmarks
- ML/AI Neural Engine performance tests
- Stress testing and stability validation
- Comprehensive reporting and regression analysis

Optimized for M4 Max with:
- 12 Performance cores + 4 Efficiency cores
- 40-core GPU with Metal acceleration
- 16-core Neural Engine with Core ML
- 546 GB/s unified memory bandwidth
- Advanced thermal management
"""

from .performance_suite import PerformanceBenchmarkSuite
from .hardware_validation import HardwareValidator
from .container_benchmarks import ContainerBenchmarks
from .trading_benchmarks import TradingBenchmarks
from .ai_benchmarks import AIBenchmarks
from .stress_tests import StressTestSuite
from .benchmark_reporter import BenchmarkReporter

__version__ = "1.0.0"
__author__ = "Nautilus Trading Platform"

__all__ = [
    "PerformanceBenchmarkSuite",
    "HardwareValidator", 
    "ContainerBenchmarks",
    "TradingBenchmarks",
    "AIBenchmarks",
    "StressTestSuite",
    "BenchmarkReporter"
]
#!/usr/bin/env python3
"""
Metal GPU Acceleration Verification Script
Nautilus Trading Platform - M4 Max Optimization

This script verifies that all Metal GPU acceleration dependencies
are properly installed and functioning on Apple Silicon M4 Max.

Usage:
    python3 verify_metal_gpu.py
"""

import sys
import time
import traceback
from typing import Dict, List, Tuple

def print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(test_name: str, success: bool, details: str = "") -> None:
    """Print test result with status indicator."""
    status = "✅" if success else "❌"
    print(f"{status} {test_name}")
    if details:
        print(f"   {details}")

def test_system_compatibility() -> Dict[str, bool]:
    """Test system compatibility for M4 Max."""
    print_header("System Compatibility Check")
    
    results = {}
    
    # Check Python version
    python_version = sys.version
    is_python_compatible = sys.version_info >= (3, 11) and sys.version_info < (3, 14)
    print_result("Python Version", is_python_compatible, f"Version: {python_version.split()[0]}")
    results["python_version"] = is_python_compatible
    
    # Check platform
    import platform
    is_macos = sys.platform == "darwin"
    architecture = platform.machine()
    is_arm64 = architecture == "arm64"
    
    print_result("macOS Platform", is_macos, f"Platform: {sys.platform}")
    print_result("ARM64 Architecture", is_arm64, f"Architecture: {architecture}")
    results["platform"] = is_macos and is_arm64
    
    # Check macOS version
    try:
        import subprocess
        result = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True)
        macos_version = result.stdout.strip()
        version_parts = [int(x) for x in macos_version.split('.')]
        is_sequoia = version_parts[0] >= 15
        print_result("macOS Version", True, f"Version: {macos_version}")
        results["macos_version"] = True
    except Exception as e:
        print_result("macOS Version", False, f"Error: {e}")
        results["macos_version"] = False
    
    # Check M4 Max chip
    try:
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True)
        has_m4_max = "M4 Max" in result.stdout
        if has_m4_max:
            # Extract memory info
            for line in result.stdout.split('\n'):
                if 'Memory:' in line:
                    memory_info = line.strip()
                    break
            else:
                memory_info = "Memory info not found"
        else:
            memory_info = "M4 Max not detected"
        
        print_result("M4 Max Chip", has_m4_max, memory_info)
        results["m4_max"] = has_m4_max
    except Exception as e:
        print_result("M4 Max Detection", False, f"Error: {e}")
        results["m4_max"] = False
    
    return results

def test_core_dependencies() -> Dict[str, bool]:
    """Test core ML framework installations."""
    print_header("Core ML Frameworks")
    
    results = {}
    
    # Test NumPy
    try:
        import numpy as np
        numpy_version = np.__version__
        print_result("NumPy", True, f"Version: {numpy_version}")
        results["numpy"] = True
    except ImportError as e:
        print_result("NumPy", False, f"Import error: {e}")
        results["numpy"] = False
    
    # Test SciPy
    try:
        import scipy
        scipy_version = scipy.__version__
        print_result("SciPy", True, f"Version: {scipy_version}")
        results["scipy"] = True
    except ImportError as e:
        print_result("SciPy", False, f"Import error: {e}")
        results["scipy"] = False
    
    # Test Pandas
    try:
        import pandas as pd
        pandas_version = pd.__version__
        print_result("Pandas", True, f"Version: {pandas_version}")
        results["pandas"] = True
    except ImportError as e:
        print_result("Pandas", False, f"Import error: {e}")
        results["pandas"] = False
    
    return results

def test_pytorch_mps() -> Dict[str, bool]:
    """Test PyTorch Metal Performance Shaders."""
    print_header("PyTorch Metal Performance Shaders")
    
    results = {}
    
    try:
        import torch
        
        # Basic import test
        torch_version = torch.__version__
        print_result("PyTorch Import", True, f"Version: {torch_version}")
        results["pytorch_import"] = True
        
        # MPS availability
        mps_available = torch.backends.mps.is_available()
        print_result("MPS Available", mps_available)
        results["mps_available"] = mps_available
        
        # MPS built
        mps_built = torch.backends.mps.is_built()
        print_result("MPS Built", mps_built)
        results["mps_built"] = mps_built
        
        if mps_available:
            # Test device creation
            try:
                device = torch.device('mps')
                print_result("MPS Device Creation", True, f"Device: {device}")
                results["mps_device"] = True
                
                # Test tensor operations
                try:
                    a = torch.randn(100, 100, device=device)
                    b = torch.randn(100, 100, device=device)
                    c = torch.mm(a, b)
                    torch.mps.synchronize()
                    print_result("MPS Tensor Operations", True, "Matrix multiplication successful")
                    results["mps_operations"] = True
                    
                    # Memory info
                    allocated = torch.mps.current_allocated_memory()
                    cached = torch.mps.driver_allocated_memory()
                    print_result("MPS Memory Info", True, 
                               f"Allocated: {allocated/1024**2:.1f}MB, Cached: {cached/1024**2:.1f}MB")
                    results["mps_memory"] = True
                    
                except Exception as e:
                    print_result("MPS Tensor Operations", False, f"Error: {e}")
                    results["mps_operations"] = False
                    results["mps_memory"] = False
                
            except Exception as e:
                print_result("MPS Device Creation", False, f"Error: {e}")
                results["mps_device"] = False
                results["mps_operations"] = False
                results["mps_memory"] = False
        else:
            results["mps_device"] = False
            results["mps_operations"] = False
            results["mps_memory"] = False
    
    except ImportError as e:
        print_result("PyTorch Import", False, f"Import error: {e}")
        results["pytorch_import"] = False
        results["mps_available"] = False
        results["mps_built"] = False
        results["mps_device"] = False
        results["mps_operations"] = False
        results["mps_memory"] = False
    
    return results

def test_mlx_framework() -> Dict[str, bool]:
    """Test Apple MLX framework."""
    print_header("Apple MLX Framework")
    
    results = {}
    
    try:
        import mlx.core as mx
        
        # Basic import
        mlx_version = getattr(mx, '__version__', 'Unknown')
        print_result("MLX Import", True, f"Version: {mlx_version}")
        results["mlx_import"] = True
        
        # Device detection
        try:
            device = mx.default_device()
            print_result("MLX Device", True, f"Device: {device}")
            results["mlx_device"] = True
            
            # Test operations
            try:
                a = mx.random.normal((100, 100))
                b = mx.random.normal((100, 100))
                c = mx.matmul(a, b)
                mx.eval(c)  # Force evaluation
                print_result("MLX Operations", True, "Matrix multiplication successful")
                results["mlx_operations"] = True
                
                # Memory info
                memory_mb = mx.get_active_memory() / 1024**2
                print_result("MLX Memory", True, f"Active: {memory_mb:.1f}MB")
                results["mlx_memory"] = True
                
            except Exception as e:
                print_result("MLX Operations", False, f"Error: {e}")
                results["mlx_operations"] = False
                results["mlx_memory"] = False
            
        except Exception as e:
            print_result("MLX Device", False, f"Error: {e}")
            results["mlx_device"] = False
            results["mlx_operations"] = False
            results["mlx_memory"] = False
    
    except ImportError as e:
        print_result("MLX Import", False, f"Import error: {e}")
        results["mlx_import"] = False
        results["mlx_device"] = False
        results["mlx_operations"] = False
        results["mlx_memory"] = False
    
    return results

def test_core_ml() -> Dict[str, bool]:
    """Test Core ML tools."""
    print_header("Core ML Tools")
    
    results = {}
    
    try:
        import coremltools as ct
        
        version = ct.__version__
        print_result("Core ML Tools", True, f"Version: {version}")
        results["coremltools"] = True
        
    except ImportError as e:
        print_result("Core ML Tools", False, f"Import error: {e}")
        results["coremltools"] = False
    
    return results

def test_quantitative_finance() -> Dict[str, bool]:
    """Test quantitative finance libraries."""
    print_header("Quantitative Finance Libraries")
    
    results = {}
    
    # Test TA-Lib
    try:
        import talib
        print_result("TA-Lib", True, "Technical analysis library")
        results["talib"] = True
    except ImportError as e:
        print_result("TA-Lib", False, f"Import error: {e}")
        results["talib"] = False
    
    # Test QuantLib
    try:
        import QuantLib as ql
        version = ql.__version__
        print_result("QuantLib", True, f"Version: {version}")
        results["quantlib"] = True
    except ImportError as e:
        print_result("QuantLib", False, f"Import error: {e}")
        results["quantlib"] = False
    
    return results

def test_performance() -> Dict[str, bool]:
    """Test performance benchmarks."""
    print_header("Performance Benchmarks")
    
    results = {}
    
    try:
        import torch
        import torch.nn as nn
        
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            
            # Simple neural network
            class TestNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(1000, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 100)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            # GPU test
            model_gpu = TestNet().to(device)
            input_gpu = torch.randn(100, 1000, device=device)
            
            # Warmup
            for _ in range(5):
                _ = model_gpu(input_gpu)
            torch.mps.synchronize()
            
            # Timing
            start_time = time.time()
            for _ in range(50):
                output = model_gpu(input_gpu)
            torch.mps.synchronize()
            gpu_time = time.time() - start_time
            
            # CPU test
            model_cpu = TestNet()
            input_cpu = torch.randn(100, 1000)
            
            start_time = time.time()
            for _ in range(50):
                output = model_cpu(input_cpu)
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            success = speedup > 1.0
            
            print_result("Neural Network GPU Speedup", success, 
                        f"GPU: {gpu_time:.3f}s, CPU: {cpu_time:.3f}s, Speedup: {speedup:.2f}x")
            results["gpu_speedup"] = success
            
        else:
            print_result("Performance Test", False, "MPS not available")
            results["gpu_speedup"] = False
    
    except Exception as e:
        print_result("Performance Test", False, f"Error: {e}")
        results["gpu_speedup"] = False
    
    return results

def main():
    """Main verification function."""
    print("Metal GPU Acceleration Verification")
    print("Nautilus Trading Platform - M4 Max Optimization")
    print(f"Python {sys.version}")
    print(f"Platform: {sys.platform}")
    
    all_results = {}
    
    # Run all tests
    all_results.update(test_system_compatibility())
    all_results.update(test_core_dependencies())
    all_results.update(test_pytorch_mps())
    all_results.update(test_mlx_framework())
    all_results.update(test_core_ml())
    all_results.update(test_quantitative_finance())
    all_results.update(test_performance())
    
    # Summary
    print_header("Verification Summary")
    
    total_tests = len(all_results)
    passed_tests = sum(all_results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if failed_tests > 0:
        print(f"\nFailed tests:")
        for test_name, success in all_results.items():
            if not success:
                print(f"  ❌ {test_name}")
    
    # Recommendations
    print_header("Recommendations")
    
    critical_tests = ["python_version", "platform", "pytorch_import", "mps_available"]
    critical_failures = [test for test in critical_tests if not all_results.get(test, False)]
    
    if critical_failures:
        print("❌ Critical issues found:")
        for test in critical_failures:
            print(f"  - {test}")
        print("\nPlease resolve critical issues before using GPU acceleration.")
    else:
        print("✅ All critical components verified successfully!")
        print("✅ Metal GPU acceleration is ready for production use.")
        
        if all_results.get("gpu_speedup", False):
            print("✅ Performance benchmarks show GPU acceleration is working optimally.")
        else:
            print("⚠️  GPU performance may be suboptimal. Check workload characteristics.")
    
    return success_rate >= 90.0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
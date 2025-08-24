"""
Hardware Validation for M4 Max Architecture
==========================================

Comprehensive hardware detection and capability validation:
- M4 Max chip detection and verification
- Performance and Efficiency core validation
- GPU core availability and Metal framework testing
- Neural Engine availability and Core ML testing
- Memory architecture validation (546 GB/s bandwidth)
- Thermal management and power efficiency testing
"""

import os
import sys
import subprocess
import platform
import psutil
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import asyncio

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "N/A"

@dataclass
class ValidationResult:
    """Individual hardware validation result"""
    component: str
    test_name: str
    status: ValidationStatus
    value: Optional[Any] = None
    expected: Optional[Any] = None
    message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class HardwareValidationReport:
    """Complete hardware validation report"""
    system_info: Dict[str, Any]
    validation_results: List[ValidationResult]
    m4_max_detected: bool
    optimization_compatibility: Dict[str, bool]
    performance_score: float
    recommendations: List[str]
    validation_time_ms: float

class HardwareValidator:
    """
    Comprehensive hardware validation for M4 Max optimizations
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.system_info = {}
        self.recommendations = []
        
        # Expected M4 Max specifications
        self.m4_max_specs = {
            "cpu_cores": 16,
            "performance_cores": 12,
            "efficiency_cores": 4,
            "gpu_cores": 40,
            "neural_engine_cores": 16,
            "memory_bandwidth_gbps": 546,
            "base_frequency_ghz": 3.2,
            "max_frequency_ghz": 4.05,
            "architecture": "arm64"
        }
    
    async def validate_hardware(self) -> HardwareValidationReport:
        """
        Run complete hardware validation suite
        """
        logger.info("Starting M4 Max hardware validation")
        start_time = time.time()
        
        try:
            # Collect system information
            self.system_info = await self._collect_system_info()
            
            # Core validation tests
            await self._validate_cpu_architecture()
            await self._validate_cpu_cores()
            await self._validate_gpu_capabilities()
            await self._validate_neural_engine()
            await self._validate_memory_architecture()
            await self._validate_metal_framework()
            await self._validate_coreml_support()
            await self._validate_thermal_management()
            await self._validate_power_management()
            await self._validate_unified_memory()
            await self._validate_optimization_frameworks()
            
            # Calculate performance score
            performance_score = self._calculate_performance_score()
            
            # Determine M4 Max detection
            m4_max_detected = self._is_m4_max_detected()
            
            # Generate optimization compatibility matrix
            optimization_compatibility = self._get_optimization_compatibility()
            
            # Generate recommendations
            self._generate_recommendations()
            
            validation_time = (time.time() - start_time) * 1000
            
            report = HardwareValidationReport(
                system_info=self.system_info,
                validation_results=self.results,
                m4_max_detected=m4_max_detected,
                optimization_compatibility=optimization_compatibility,
                performance_score=performance_score,
                recommendations=self.recommendations,
                validation_time_ms=validation_time
            )
            
            logger.info(f"Hardware validation completed in {validation_time:.2f}ms")
            return report
            
        except Exception as e:
            logger.error(f"Hardware validation failed: {e}")
            raise
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        
        info = {
            "timestamp": time.time(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": psutil.virtual_memory().total,
            "memory_available_bytes": psutil.virtual_memory().available,
            "disk_usage": dict(psutil.disk_usage('/'))
        }
        
        # Add macOS specific information
        if platform.system() == "Darwin":
            try:
                # Get system profiler information
                result = subprocess.run([
                    "system_profiler", "SPHardwareDataType", "-json"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    hardware_info = json.loads(result.stdout)
                    if "SPHardwareDataType" in hardware_info:
                        hw_data = hardware_info["SPHardwareDataType"][0]
                        info.update({
                            "chip_type": hw_data.get("chip_type", "Unknown"),
                            "number_processors": hw_data.get("number_processors", "Unknown"),
                            "total_number_cores": hw_data.get("total_number_cores", "Unknown"),
                            "memory": hw_data.get("physical_memory", "Unknown"),
                            "serial_number": hw_data.get("serial_number", "Unknown"),
                            "model_name": hw_data.get("machine_name", "Unknown")
                        })
            except Exception as e:
                logger.warning(f"Could not get macOS system info: {e}")
        
        # Add CPU frequency information
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info.update({
                    "cpu_freq_current": cpu_freq.current,
                    "cpu_freq_min": cpu_freq.min,
                    "cpu_freq_max": cpu_freq.max
                })
        except Exception as e:
            logger.warning(f"Could not get CPU frequency: {e}")
        
        return info
    
    async def _validate_cpu_architecture(self):
        """Validate CPU architecture compatibility"""
        
        machine = platform.machine().lower()
        processor = platform.processor().lower()
        
        # Check for Apple Silicon
        is_apple_silicon = (
            machine in ["arm64", "aarch64"] or 
            "apple" in processor or
            "m1" in processor or "m2" in processor or 
            "m3" in processor or "m4" in processor
        )
        
        if is_apple_silicon:
            status = ValidationStatus.PASS
            message = f"Apple Silicon detected: {machine}"
            
            # Try to identify specific chip
            chip_type = self.system_info.get("chip_type", "").lower()
            if "m4" in chip_type:
                message += f" - Chip: {chip_type}"
        else:
            status = ValidationStatus.WARNING
            message = f"Non-Apple Silicon architecture: {machine}"
            self.recommendations.append("M4 Max optimizations require Apple Silicon hardware")
        
        self.results.append(ValidationResult(
            component="CPU",
            test_name="Architecture Compatibility",
            status=status,
            value=machine,
            expected="arm64 (Apple Silicon)",
            message=message,
            details={
                "processor": processor,
                "chip_type": self.system_info.get("chip_type", "Unknown")
            }
        ))
    
    async def _validate_cpu_cores(self):
        """Validate CPU core configuration"""
        
        total_cores = os.cpu_count()
        expected_total = self.m4_max_specs["cpu_cores"]
        
        # Validate total core count
        if total_cores == expected_total:
            status = ValidationStatus.PASS
            message = f"Correct total core count: {total_cores}"
        elif total_cores and total_cores > 8:
            status = ValidationStatus.WARNING
            message = f"High core count detected: {total_cores} (expected {expected_total})"
        else:
            status = ValidationStatus.FAIL
            message = f"Low core count: {total_cores} (expected {expected_total})"
        
        self.results.append(ValidationResult(
            component="CPU",
            test_name="Total Core Count",
            status=status,
            value=total_cores,
            expected=expected_total,
            message=message
        ))
        
        # Try to detect P and E cores
        await self._detect_core_types()
        
        # Test core performance characteristics
        await self._test_core_performance()
    
    async def _detect_core_types(self):
        """Attempt to detect Performance and Efficiency cores"""
        
        try:
            # On macOS, try to use powermetrics to detect core types
            if platform.system() == "Darwin":
                result = subprocess.run([
                    "powermetrics", "-n", "1", "-s", "cpu_power", "--format", "text"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    output = result.stdout.lower()
                    
                    # Look for P-core and E-core indicators
                    p_core_count = output.count("p-core") or output.count("performance")
                    e_core_count = output.count("e-core") or output.count("efficiency")
                    
                    if p_core_count > 0 and e_core_count > 0:
                        status = ValidationStatus.PASS
                        message = f"Detected P-cores: {p_core_count}, E-cores: {e_core_count}"
                    else:
                        status = ValidationStatus.WARNING
                        message = "Could not detect distinct P and E cores"
                else:
                    status = ValidationStatus.WARNING
                    message = "powermetrics not available for core detection"
            else:
                status = ValidationStatus.NOT_APPLICABLE
                message = "Core type detection not available on this platform"
                
        except Exception as e:
            status = ValidationStatus.WARNING
            message = f"Core detection failed: {str(e)}"
        
        self.results.append(ValidationResult(
            component="CPU",
            test_name="Core Type Detection",
            status=status,
            message=message,
            details={
                "method": "powermetrics" if platform.system() == "Darwin" else "not_available"
            }
        ))
    
    async def _test_core_performance(self):
        """Test core performance characteristics"""
        
        # Simple CPU benchmark across all cores
        def cpu_test():
            """CPU intensive task"""
            result = 0
            for i in range(1000000):
                result += i ** 0.5
            return result
        
        # Time the CPU test
        start_time = time.perf_counter()
        result = cpu_test()
        duration = time.perf_counter() - start_time
        
        # Calculate performance score (operations per second)
        ops_per_second = 1000000 / duration
        
        # Compare against expected performance
        # M4 Max should handle 1M operations in < 50ms
        expected_min_ops = 20000000  # 20M ops/sec minimum
        
        if ops_per_second >= expected_min_ops:
            status = ValidationStatus.PASS
            message = f"CPU performance: {ops_per_second:.0f} ops/sec"
        elif ops_per_second >= expected_min_ops * 0.7:
            status = ValidationStatus.WARNING
            message = f"CPU performance below optimal: {ops_per_second:.0f} ops/sec"
        else:
            status = ValidationStatus.FAIL
            message = f"CPU performance too low: {ops_per_second:.0f} ops/sec"
        
        self.results.append(ValidationResult(
            component="CPU",
            test_name="Core Performance",
            status=status,
            value=ops_per_second,
            expected=expected_min_ops,
            message=message,
            details={
                "test_duration_ms": duration * 1000,
                "operations": 1000000
            }
        ))
    
    async def _validate_gpu_capabilities(self):
        """Validate GPU capabilities"""
        
        try:
            # Check for Metal availability
            if platform.system() == "Darwin":
                # Try to import Metal-related libraries
                metal_available = False
                torch_mps_available = False
                
                try:
                    import torch
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch_mps_available = True
                        metal_available = True
                except ImportError:
                    pass
                
                try:
                    import mlx.core as mx
                    metal_available = True
                except ImportError:
                    pass
                
                if metal_available:
                    status = ValidationStatus.PASS
                    message = "Metal acceleration available"
                    
                    # Test basic Metal operation
                    if torch_mps_available:
                        await self._test_metal_performance()
                else:
                    status = ValidationStatus.WARNING
                    message = "Metal acceleration libraries not found"
                    self.recommendations.append("Install PyTorch with Metal support or MLX")
            else:
                status = ValidationStatus.NOT_APPLICABLE
                message = "Metal only available on macOS"
        
        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"GPU validation failed: {str(e)}"
        
        self.results.append(ValidationResult(
            component="GPU",
            test_name="Metal Availability",
            status=status,
            message=message,
            details={
                "torch_mps": torch_mps_available if 'torch_mps_available' in locals() else False,
                "mlx_available": 'mlx' in sys.modules
            }
        ))
    
    async def _test_metal_performance(self):
        """Test Metal GPU performance"""
        
        try:
            import torch
            
            if not torch.backends.mps.is_available():
                return
            
            device = torch.device("mps")
            
            # Simple matrix multiplication test
            size = 1024
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            for _ in range(3):
                _ = torch.matmul(a, b)
                torch.mps.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            result = torch.matmul(a, b)
            torch.mps.synchronize()
            duration = time.perf_counter() - start_time
            
            # Calculate GFLOPS
            operations = 2 * size ** 3  # Matrix multiplication FLOPs
            gflops = (operations / 1e9) / duration
            
            # M4 Max should achieve > 500 GFLOPS for this test
            expected_gflops = 500
            
            if gflops >= expected_gflops:
                status = ValidationStatus.PASS
                message = f"Metal performance: {gflops:.1f} GFLOPS"
            elif gflops >= expected_gflops * 0.7:
                status = ValidationStatus.WARNING
                message = f"Metal performance below optimal: {gflops:.1f} GFLOPS"
            else:
                status = ValidationStatus.FAIL
                message = f"Metal performance too low: {gflops:.1f} GFLOPS"
            
            self.results.append(ValidationResult(
                component="GPU",
                test_name="Metal Performance",
                status=status,
                value=gflops,
                expected=expected_gflops,
                message=message,
                details={
                    "test_duration_ms": duration * 1000,
                    "matrix_size": f"{size}x{size}",
                    "operations": operations
                }
            ))
            
        except Exception as e:
            logger.warning(f"Metal performance test failed: {e}")
    
    async def _validate_neural_engine(self):
        """Validate Neural Engine availability"""
        
        try:
            # Check for Core ML availability
            coreml_available = False
            ane_available = False
            
            try:
                import coremltools
                coreml_available = True
                
                # Try to detect ANE
                import platform
                if platform.system() == "Darwin":
                    # Check system information for Neural Engine
                    system_info = self.system_info.get("chip_type", "").lower()
                    if any(chip in system_info for chip in ["m1", "m2", "m3", "m4"]):
                        ane_available = True
                        
            except ImportError:
                pass
            
            if coreml_available and ane_available:
                status = ValidationStatus.PASS
                message = "Neural Engine and Core ML available"
            elif ane_available:
                status = ValidationStatus.WARNING
                message = "Neural Engine detected but Core ML not available"
                self.recommendations.append("Install coremltools for Neural Engine support")
            else:
                status = ValidationStatus.FAIL
                message = "Neural Engine not detected"
        
        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"Neural Engine validation failed: {str(e)}"
        
        self.results.append(ValidationResult(
            component="Neural Engine",
            test_name="ANE Availability",
            status=status,
            message=message,
            details={
                "coreml_available": coreml_available if 'coreml_available' in locals() else False,
                "expected_cores": self.m4_max_specs["neural_engine_cores"]
            }
        ))
    
    async def _validate_memory_architecture(self):
        """Validate memory architecture"""
        
        total_memory_gb = self.system_info["memory_total_bytes"] / (1024**3)
        
        # M4 Max typically comes with 36GB or 128GB unified memory
        expected_memory_options = [32, 36, 64, 128]  # GB
        closest_expected = min(expected_memory_options, 
                              key=lambda x: abs(x - total_memory_gb))
        
        if abs(total_memory_gb - closest_expected) < 4:  # Within 4GB
            status = ValidationStatus.PASS
            message = f"Memory configuration: {total_memory_gb:.1f}GB"
        else:
            status = ValidationStatus.WARNING
            message = f"Unusual memory configuration: {total_memory_gb:.1f}GB"
        
        self.results.append(ValidationResult(
            component="Memory",
            test_name="Total Memory",
            status=status,
            value=total_memory_gb,
            expected=closest_expected,
            message=message
        ))
        
        # Test memory bandwidth
        await self._test_memory_bandwidth()
    
    async def _test_memory_bandwidth(self):
        """Test memory bandwidth performance"""
        
        try:
            import numpy as np
            
            # Test memory bandwidth with large array operations
            size = 100 * 1024 * 1024  # 100MB
            data = np.random.random(size // 8).astype(np.float64)
            
            # Memory copy test
            start_time = time.perf_counter()
            data_copy = data.copy()
            duration = time.perf_counter() - start_time
            
            # Calculate bandwidth in GB/s
            bytes_copied = data.nbytes
            bandwidth_gbps = (bytes_copied / 1e9) / duration
            
            # M4 Max theoretical bandwidth is 546 GB/s
            # Practical achievable bandwidth is typically 20-50% of theoretical
            expected_min_bandwidth = 100  # GB/s (conservative estimate)
            
            if bandwidth_gbps >= expected_min_bandwidth:
                status = ValidationStatus.PASS
                message = f"Memory bandwidth: {bandwidth_gbps:.1f} GB/s"
            elif bandwidth_gbps >= expected_min_bandwidth * 0.5:
                status = ValidationStatus.WARNING
                message = f"Memory bandwidth below optimal: {bandwidth_gbps:.1f} GB/s"
            else:
                status = ValidationStatus.FAIL
                message = f"Memory bandwidth too low: {bandwidth_gbps:.1f} GB/s"
            
            self.results.append(ValidationResult(
                component="Memory",
                test_name="Memory Bandwidth",
                status=status,
                value=bandwidth_gbps,
                expected=expected_min_bandwidth,
                message=message,
                details={
                    "test_size_mb": bytes_copied / (1024*1024),
                    "test_duration_ms": duration * 1000,
                    "theoretical_max_gbps": self.m4_max_specs["memory_bandwidth_gbps"]
                }
            ))
            
        except Exception as e:
            logger.warning(f"Memory bandwidth test failed: {e}")
    
    async def _validate_metal_framework(self):
        """Validate Metal framework installation"""
        
        metal_components = {
            "Metal Performance Shaders": False,
            "PyTorch MPS": False,
            "MLX": False
        }
        
        # Check PyTorch MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                metal_components["PyTorch MPS"] = True
        except ImportError:
            pass
        
        # Check MLX
        try:
            import mlx.core
            metal_components["MLX"] = True
        except ImportError:
            pass
        
        # Metal Performance Shaders is part of macOS
        if platform.system() == "Darwin":
            metal_components["Metal Performance Shaders"] = True
        
        available_count = sum(metal_components.values())
        
        if available_count >= 2:
            status = ValidationStatus.PASS
            message = f"Metal frameworks available: {available_count}/3"
        elif available_count >= 1:
            status = ValidationStatus.WARNING
            message = f"Limited Metal support: {available_count}/3"
            self.recommendations.append("Install additional Metal frameworks (PyTorch MPS, MLX)")
        else:
            status = ValidationStatus.FAIL
            message = "No Metal frameworks available"
        
        self.results.append(ValidationResult(
            component="Metal",
            test_name="Framework Availability",
            status=status,
            value=available_count,
            expected=3,
            message=message,
            details=metal_components
        ))
    
    async def _validate_coreml_support(self):
        """Validate Core ML support"""
        
        try:
            import coremltools
            version = coremltools.__version__
            status = ValidationStatus.PASS
            message = f"Core ML Tools version: {version}"
            
            # Test basic Core ML functionality
            try:
                # This is a basic test - in production you'd have a real model
                from coremltools.models import MLModel
                status = ValidationStatus.PASS
            except Exception:
                status = ValidationStatus.WARNING
                message += " (limited functionality)"
                
        except ImportError:
            status = ValidationStatus.FAIL
            message = "Core ML Tools not installed"
            self.recommendations.append("Install coremltools: pip install coremltools")
        
        self.results.append(ValidationResult(
            component="Core ML",
            test_name="Framework Support",
            status=status,
            message=message
        ))
    
    async def _validate_thermal_management(self):
        """Validate thermal management capabilities"""
        
        try:
            # Get current CPU temperatures if available
            temperatures = psutil.sensors_temperatures()
            
            if temperatures:
                cpu_temps = []
                for name, entries in temperatures.items():
                    if 'cpu' in name.lower():
                        for entry in entries:
                            cpu_temps.append(entry.current)
                
                if cpu_temps:
                    avg_temp = sum(cpu_temps) / len(cpu_temps)
                    max_temp = max(cpu_temps)
                    
                    if max_temp < 85:  # Below thermal throttling
                        status = ValidationStatus.PASS
                        message = f"CPU temperature: {avg_temp:.1f}°C (max: {max_temp:.1f}°C)"
                    elif max_temp < 95:
                        status = ValidationStatus.WARNING
                        message = f"CPU running warm: {avg_temp:.1f}°C (max: {max_temp:.1f}°C)"
                    else:
                        status = ValidationStatus.FAIL
                        message = f"CPU overheating: {avg_temp:.1f}°C (max: {max_temp:.1f}°C)"
                        self.recommendations.append("Check cooling and reduce system load")
                else:
                    status = ValidationStatus.WARNING
                    message = "CPU temperature sensors not found"
            else:
                status = ValidationStatus.NOT_APPLICABLE
                message = "Temperature monitoring not available"
                
        except Exception as e:
            status = ValidationStatus.WARNING
            message = f"Thermal validation failed: {str(e)}"
        
        self.results.append(ValidationResult(
            component="Thermal",
            test_name="Temperature Management",
            status=status,
            message=message
        ))
    
    async def _validate_power_management(self):
        """Validate power management capabilities"""
        
        try:
            # Check battery information if available
            battery = psutil.sensors_battery()
            
            if battery:
                power_plugged = battery.power_plugged
                percent = battery.percent
                
                if power_plugged:
                    status = ValidationStatus.PASS
                    message = f"Power adapter connected, battery: {percent}%"
                elif percent > 50:
                    status = ValidationStatus.WARNING
                    message = f"Running on battery: {percent}%"
                    self.recommendations.append("Connect power adapter for optimal performance")
                else:
                    status = ValidationStatus.FAIL
                    message = f"Low battery: {percent}%"
                    self.recommendations.append("Charge battery or connect power adapter")
            else:
                status = ValidationStatus.NOT_APPLICABLE
                message = "Battery information not available (desktop system)"
                
        except Exception as e:
            status = ValidationStatus.WARNING
            message = f"Power validation failed: {str(e)}"
        
        self.results.append(ValidationResult(
            component="Power",
            test_name="Power Management",
            status=status,
            message=message
        ))
    
    async def _validate_unified_memory(self):
        """Validate unified memory architecture"""
        
        # On Apple Silicon, all memory is unified
        if platform.machine().lower() in ["arm64", "aarch64"]:
            # Test memory sharing between CPU and GPU
            try:
                import torch
                
                if torch.backends.mps.is_available():
                    # Create data on CPU
                    cpu_data = torch.randn(1000, 1000)
                    
                    # Move to GPU
                    start_time = time.perf_counter()
                    gpu_data = cpu_data.to("mps")
                    transfer_time = time.perf_counter() - start_time
                    
                    # Unified memory should have very fast transfers
                    if transfer_time < 0.001:  # < 1ms
                        status = ValidationStatus.PASS
                        message = f"Unified memory verified (transfer: {transfer_time*1000:.2f}ms)"
                    else:
                        status = ValidationStatus.WARNING
                        message = f"Memory transfer slower than expected: {transfer_time*1000:.2f}ms"
                else:
                    status = ValidationStatus.WARNING
                    message = "Cannot test unified memory without Metal support"
            except ImportError:
                status = ValidationStatus.WARNING
                message = "Cannot test unified memory without PyTorch"
        else:
            status = ValidationStatus.NOT_APPLICABLE
            message = "Unified memory only available on Apple Silicon"
        
        self.results.append(ValidationResult(
            component="Memory",
            test_name="Unified Memory Architecture",
            status=status,
            message=message
        ))
    
    async def _validate_optimization_frameworks(self):
        """Validate optimization framework availability"""
        
        frameworks = {
            "numpy": False,
            "scipy": False,
            "pandas": False,
            "torch": False,
            "coremltools": False,
            "mlx": False,
            "psutil": True,  # We're using it already
        }
        
        # Check each framework
        for framework in frameworks:
            if framework == "psutil":
                continue
                
            try:
                __import__(framework)
                frameworks[framework] = True
            except ImportError:
                pass
        
        available_count = sum(frameworks.values())
        total_count = len(frameworks)
        
        if available_count >= total_count * 0.8:
            status = ValidationStatus.PASS
            message = f"Optimization frameworks: {available_count}/{total_count}"
        elif available_count >= total_count * 0.6:
            status = ValidationStatus.WARNING
            message = f"Some optimization frameworks missing: {available_count}/{total_count}"
        else:
            status = ValidationStatus.FAIL
            message = f"Many optimization frameworks missing: {available_count}/{total_count}"
            self.recommendations.append("Install missing optimization frameworks")
        
        self.results.append(ValidationResult(
            component="Software",
            test_name="Optimization Frameworks",
            status=status,
            value=available_count,
            expected=total_count,
            message=message,
            details=frameworks
        ))
    
    def _is_m4_max_detected(self) -> bool:
        """Determine if M4 Max is detected"""
        
        # Check system information
        chip_type = self.system_info.get("chip_type", "").lower()
        if "m4" in chip_type and "max" in chip_type:
            return True
        
        # Check architecture and core count
        is_apple_silicon = platform.machine().lower() in ["arm64", "aarch64"]
        has_correct_cores = os.cpu_count() == self.m4_max_specs["cpu_cores"]
        
        # Check if most validations pass
        passed_tests = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        total_tests = len(self.results)
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return is_apple_silicon and has_correct_cores and success_rate > 0.7
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        
        if not self.results:
            return 0.0
        
        weights = {
            ValidationStatus.PASS: 100,
            ValidationStatus.WARNING: 60,
            ValidationStatus.FAIL: 0,
            ValidationStatus.NOT_APPLICABLE: 80  # Don't penalize for N/A tests
        }
        
        total_score = 0
        total_weight = 0
        
        for result in self.results:
            score = weights[result.status]
            total_score += score
            total_weight += 100
        
        return total_score / total_weight * 100 if total_weight > 0 else 0.0
    
    def _get_optimization_compatibility(self) -> Dict[str, bool]:
        """Get optimization feature compatibility matrix"""
        
        compatibility = {
            "metal_acceleration": False,
            "neural_engine": False,
            "cpu_optimization": False,
            "unified_memory": False,
            "thermal_management": False,
            "coreml_support": False
        }
        
        for result in self.results:
            if result.status in [ValidationStatus.PASS, ValidationStatus.WARNING]:
                if "Metal" in result.component:
                    compatibility["metal_acceleration"] = True
                elif "Neural Engine" in result.component:
                    compatibility["neural_engine"] = True
                elif "CPU" in result.component:
                    compatibility["cpu_optimization"] = True
                elif "Memory" in result.component and "Unified" in result.test_name:
                    compatibility["unified_memory"] = True
                elif "Thermal" in result.component:
                    compatibility["thermal_management"] = True
                elif "Core ML" in result.component:
                    compatibility["coreml_support"] = True
        
        return compatibility
    
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        
        # Add general recommendations based on results
        failed_tests = [r for r in self.results if r.status == ValidationStatus.FAIL]
        warning_tests = [r for r in self.results if r.status == ValidationStatus.WARNING]
        
        if len(failed_tests) > len(self.results) * 0.3:
            self.recommendations.append("Hardware may not be suitable for M4 Max optimizations")
        
        if any("Metal" in r.component for r in failed_tests + warning_tests):
            self.recommendations.append("Consider installing/updating Metal acceleration frameworks")
        
        if any("Neural Engine" in r.component for r in failed_tests + warning_tests):
            self.recommendations.append("Install Core ML tools for Neural Engine support")
        
        if any("Memory" in r.component for r in failed_tests):
            self.recommendations.append("Consider upgrading system memory for optimal performance")
        
        # Remove duplicates
        self.recommendations = list(set(self.recommendations))
    
    def export_report(self, filename: str, format: str = "json") -> None:
        """Export validation report to file"""
        
        # Create a simple report for now
        report_data = {
            "system_info": self.system_info,
            "validation_results": [asdict(r) for r in self.results],
            "m4_max_detected": self._is_m4_max_detected(),
            "performance_score": self._calculate_performance_score(),
            "recommendations": self.recommendations
        }
        
        if format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Hardware validation report exported to {filename}")
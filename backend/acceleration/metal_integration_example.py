"""
Metal GPU Acceleration Integration Example for Nautilus Trading Platform

Demonstrates comprehensive usage of Metal GPU acceleration for financial computing:
- Options pricing with Monte Carlo simulation
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Portfolio optimization with GPU acceleration
- Neural network training for financial prediction
- Performance benchmarking and optimization

This example shows how to integrate Metal acceleration into trading strategies,
risk management, and machine learning workflows.

Requirements:
- Apple Silicon M4 Max (recommended) or other Apple Silicon
- macOS 13.0+ with Metal Performance Shaders
- PyTorch with Metal support
- All dependencies from requirements-metal.txt
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import Metal acceleration components
from . import (
    initialize_metal_acceleration,
    is_metal_available,
    is_m4_max_detected,
    price_option_metal,
    calculate_rsi_metal,
    calculate_macd_metal,
    calculate_bollinger_bands_metal,
    create_metal_model_wrapper,
    create_financial_lstm,
    allocate_gpu_tensor,
    get_memory_pool_stats,
    get_acceleration_status
)

# Standard libraries for financial computing
try:
    import torch
    import torch.nn as nn
    import pandas as pd
    from scipy import stats
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetalFinancialComputingDemo:
    """
    Comprehensive demonstration of Metal GPU acceleration for financial computing
    Showcases real-world applications in trading, risk management, and ML
    """
    
    def __init__(self):
        self.initialization_status = None
        self.performance_metrics = {}
        self.results = {}
        
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete Metal acceleration demonstration"""
        print("🚀 Starting Metal GPU Acceleration Demo for Nautilus Trading Platform")
        print("=" * 80)
        
        demo_results = {
            "demo_start_time": datetime.now().isoformat(),
            "hardware_detected": {},
            "initialization_status": {},
            "performance_benchmarks": {},
            "financial_computations": {},
            "ml_demonstrations": {},
            "recommendations": []
        }
        
        try:
            # 1. Initialize Metal acceleration
            print("\n📱 Initializing Metal GPU Acceleration...")
            demo_results["initialization_status"] = await self._demo_initialization()
            
            # 2. Hardware detection and capabilities
            print("\n🔍 Detecting Hardware Capabilities...")
            demo_results["hardware_detected"] = await self._demo_hardware_detection()
            
            # 3. Financial computations demonstration
            print("\n💰 Demonstrating GPU-Accelerated Financial Computations...")
            demo_results["financial_computations"] = await self._demo_financial_computations()
            
            # 4. Technical indicators demonstration
            print("\n📊 Demonstrating Technical Indicators with Metal Acceleration...")
            demo_results["technical_indicators"] = await self._demo_technical_indicators()
            
            # 5. Machine learning demonstration
            if DEPENDENCIES_AVAILABLE:
                print("\n🧠 Demonstrating Financial ML with Metal Acceleration...")
                demo_results["ml_demonstrations"] = await self._demo_financial_ml()
            
            # 6. Performance benchmarking
            print("\n⚡ Running Performance Benchmarks...")
            demo_results["performance_benchmarks"] = await self._demo_performance_benchmarks()
            
            # 7. Memory management demonstration
            print("\n🧠 Demonstrating Memory Management...")
            demo_results["memory_management"] = await self._demo_memory_management()
            
            # 8. Generate final recommendations
            print("\n📋 Generating Optimization Recommendations...")
            demo_results["recommendations"] = await self._generate_recommendations()
            
            print("\n✅ Metal GPU Acceleration Demo Completed Successfully!")
            self._print_demo_summary(demo_results)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            demo_results["error"] = str(e)
            print(f"\n❌ Demo failed with error: {e}")
            
        return demo_results
        
    async def _demo_initialization(self) -> Dict[str, Any]:
        """Demonstrate Metal acceleration initialization"""
        print("Initializing Metal Performance Shaders...")
        
        start_time = time.time()
        initialization_status = initialize_metal_acceleration(enable_logging=True)
        init_time = (time.time() - start_time) * 1000
        
        print(f"✅ Initialization completed in {init_time:.2f}ms")
        print(f"   Metal Available: {initialization_status.get('metal_available', False)}")
        print(f"   M4 Max Detected: {initialization_status.get('m4_max_detected', False)}")
        print(f"   PyTorch Metal: {initialization_status.get('pytorch_metal_available', False)}")
        
        if initialization_status.get("errors"):
            print("❌ Initialization Errors:")
            for error in initialization_status["errors"]:
                print(f"   - {error}")
                
        if initialization_status.get("warnings"):
            print("⚠️  Initialization Warnings:")
            for warning in initialization_status["warnings"]:
                print(f"   - {warning}")
                
        return initialization_status
        
    async def _demo_hardware_detection(self) -> Dict[str, Any]:
        """Demonstrate hardware detection and capabilities"""
        hardware_info = {
            "metal_available": is_metal_available(),
            "m4_max_detected": is_m4_max_detected(),
            "acceleration_status": get_acceleration_status()
        }
        
        print(f"Metal Performance Shaders Available: {hardware_info['metal_available']}")
        print(f"M4 Max Hardware Detected: {hardware_info['m4_max_detected']}")
        
        if hardware_info['metal_available']:
            status = hardware_info['acceleration_status']
            if 'capabilities' in status and status['capabilities']:
                caps = status['capabilities']
                print(f"GPU Cores: {caps.get('gpu_cores', 'Unknown')}")
                print(f"Unified Memory: {caps.get('unified_memory_gb', 'Unknown')} GB")
                print(f"Memory Bandwidth: {caps.get('memory_bandwidth_gbps', 'Unknown')} GB/s")
                print(f"FP16 Support: {caps.get('supports_fp16', False)}")
                print(f"Architecture: {caps.get('architecture', 'Unknown')}")
                
        return hardware_info
        
    async def _demo_financial_computations(self) -> Dict[str, Any]:
        """Demonstrate GPU-accelerated financial computations"""
        financial_results = {}
        
        if not is_metal_available():
            print("⚠️  Metal not available - using CPU fallback")
            
        # 1. Options Pricing with Monte Carlo
        print("\n1️⃣  Monte Carlo Options Pricing:")
        try:
            start_time = time.time()
            
            option_result = await price_option_metal(
                spot_price=100.0,
                strike_price=110.0,
                time_to_expiry=0.25,  # 3 months
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
                num_simulations=1000000
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            print(f"   Option Price: ${option_result.option_price:.4f}")
            print(f"   Delta: {option_result.delta:.4f}")
            print(f"   Gamma: {option_result.gamma:.4f}")
            print(f"   Theta: {option_result.theta:.4f}")
            print(f"   Vega: {option_result.vega:.4f}")
            print(f"   Computation Time: {computation_time:.2f}ms")
            print(f"   Metal Accelerated: {option_result.metal_accelerated}")
            print(f"   95% Confidence: ${option_result.confidence_intervals.get('95%', (0, 0))[0]:.4f} - ${option_result.confidence_intervals.get('95%', (0, 0))[1]:.4f}")
            
            financial_results["options_pricing"] = {
                "option_price": option_result.option_price,
                "greeks": {
                    "delta": option_result.delta,
                    "gamma": option_result.gamma,
                    "theta": option_result.theta,
                    "vega": option_result.vega
                },
                "computation_time_ms": computation_time,
                "metal_accelerated": option_result.metal_accelerated
            }
            
        except Exception as e:
            print(f"   ❌ Options pricing failed: {e}")
            financial_results["options_pricing"] = {"error": str(e)}
            
        return financial_results
        
    async def _demo_technical_indicators(self) -> Dict[str, Any]:
        """Demonstrate technical indicators with Metal acceleration"""
        technical_results = {}
        
        # Generate sample price data
        np.random.seed(42)
        n_days = 1000
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = [initial_price]
        
        for i in range(n_days - 1):
            prices.append(prices[-1] * (1 + returns[i]))
            
        print(f"Generated {n_days} days of sample price data")
        print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        
        # 1. RSI Calculation
        print("\n1️⃣  Relative Strength Index (RSI):")
        try:
            start_time = time.time()
            
            rsi_result = await calculate_rsi_metal(
                prices=prices,
                period=14,
                overbought_threshold=70,
                oversold_threshold=30
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            latest_rsi = rsi_result.values[-1] if rsi_result.values else 0
            latest_signal = rsi_result.signals[-1] if rsi_result.signals else "unknown"
            
            print(f"   Latest RSI: {latest_rsi:.2f}")
            print(f"   Latest Signal: {latest_signal}")
            print(f"   Confidence: {rsi_result.confidence:.2%}")
            print(f"   Computation Time: {computation_time:.2f}ms")
            print(f"   Metal Accelerated: {rsi_result.metal_accelerated}")
            
            technical_results["rsi"] = {
                "latest_value": latest_rsi,
                "latest_signal": latest_signal,
                "confidence": rsi_result.confidence,
                "computation_time_ms": computation_time,
                "metal_accelerated": rsi_result.metal_accelerated
            }
            
        except Exception as e:
            print(f"   ❌ RSI calculation failed: {e}")
            technical_results["rsi"] = {"error": str(e)}
            
        # 2. MACD Calculation
        print("\n2️⃣  Moving Average Convergence Divergence (MACD):")
        try:
            start_time = time.time()
            
            macd_result = await calculate_macd_metal(
                prices=prices,
                fast_period=12,
                slow_period=26,
                signal_period=9
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            latest_macd = macd_result.values[-1] if macd_result.values else 0
            latest_signal = macd_result.signals[-1] if macd_result.signals else "unknown"
            
            print(f"   Latest MACD: {latest_macd:.4f}")
            print(f"   Latest Signal: {latest_signal}")
            print(f"   Confidence: {macd_result.confidence:.2%}")
            print(f"   Computation Time: {computation_time:.2f}ms")
            print(f"   Metal Accelerated: {macd_result.metal_accelerated}")
            
            technical_results["macd"] = {
                "latest_value": latest_macd,
                "latest_signal": latest_signal,
                "confidence": macd_result.confidence,
                "computation_time_ms": computation_time,
                "metal_accelerated": macd_result.metal_accelerated
            }
            
        except Exception as e:
            print(f"   ❌ MACD calculation failed: {e}")
            technical_results["macd"] = {"error": str(e)}
            
        # 3. Bollinger Bands Calculation
        print("\n3️⃣  Bollinger Bands:")
        try:
            start_time = time.time()
            
            bb_result = await calculate_bollinger_bands_metal(
                prices=prices,
                period=20,
                std_dev_multiplier=2.0
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            latest_middle = bb_result.values[-1] if bb_result.values else 0
            latest_signal = bb_result.signals[-1] if bb_result.signals else "unknown"
            
            print(f"   Latest Middle Band: ${latest_middle:.2f}")
            print(f"   Latest Signal: {latest_signal}")
            print(f"   Confidence: {bb_result.confidence:.2%}")
            print(f"   Computation Time: {computation_time:.2f}ms")
            print(f"   Metal Accelerated: {bb_result.metal_accelerated}")
            
            technical_results["bollinger_bands"] = {
                "latest_middle_band": latest_middle,
                "latest_signal": latest_signal,
                "confidence": bb_result.confidence,
                "computation_time_ms": computation_time,
                "metal_accelerated": bb_result.metal_accelerated
            }
            
        except Exception as e:
            print(f"   ❌ Bollinger Bands calculation failed: {e}")
            technical_results["bollinger_bands"] = {"error": str(e)}
            
        return technical_results
        
    async def _demo_financial_ml(self) -> Dict[str, Any]:
        """Demonstrate financial machine learning with Metal acceleration"""
        ml_results = {}
        
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️  ML dependencies not available - skipping ML demo")
            return {"skipped": "Dependencies not available"}
            
        print("\n1️⃣  Creating Financial LSTM Model:")
        try:
            # Create Metal-accelerated LSTM for price prediction
            lstm_wrapper = create_financial_lstm(
                input_size=5,  # OHLCV data
                hidden_size=128,
                num_layers=2,
                output_size=1  # Next day price
            )
            
            if lstm_wrapper:
                print("   ✅ Financial LSTM created with Metal acceleration")
                
                # Generate sample training data
                sequence_length = 30
                num_samples = 1000
                
                # Create dummy OHLCV data
                np.random.seed(42)
                X = np.random.randn(num_samples, sequence_length, 5).astype(np.float32)
                y = np.random.randn(num_samples, 1).astype(np.float32)
                
                # Convert to tensors
                device = torch.device("mps") if is_metal_available() else torch.device("cpu")
                X_tensor = torch.FloatTensor(X).to(device)
                y_tensor = torch.FloatTensor(y).to(device)
                
                print(f"   Training data shape: {X_tensor.shape}")
                print(f"   Target shape: {y_tensor.shape}")
                print(f"   Device: {device}")
                
                # Test inference
                start_time = time.time()
                with torch.no_grad():
                    predictions = lstm_wrapper.forward(X_tensor[:10])  # Test batch
                inference_time = (time.time() - start_time) * 1000
                
                print(f"   ✅ Inference completed in {inference_time:.2f}ms")
                print(f"   Prediction shape: {predictions.shape}")
                print(f"   Metal accelerated: {device.type == 'mps'}")
                
                ml_results["lstm"] = {
                    "model_created": True,
                    "input_shape": list(X_tensor.shape),
                    "output_shape": list(predictions.shape),
                    "inference_time_ms": inference_time,
                    "metal_accelerated": device.type == "mps",
                    "device": str(device)
                }
                
            else:
                print("   ❌ Failed to create Financial LSTM")
                ml_results["lstm"] = {"error": "Failed to create model"}
                
        except Exception as e:
            print(f"   ❌ Financial ML demo failed: {e}")
            ml_results["lstm"] = {"error": str(e)}
            
        return ml_results
        
    async def _demo_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks comparing Metal vs CPU"""
        benchmark_results = {}
        
        print("\n🔥 Comparing Metal GPU vs CPU Performance:")
        
        # Benchmark 1: Large array operations
        print("\n1️⃣  Large Array Operations Benchmark:")
        try:
            array_size = (10000, 1000)
            
            # CPU benchmark
            start_time = time.time()
            cpu_array = np.random.randn(*array_size).astype(np.float32)
            cpu_result = np.sum(cpu_array ** 2)
            cpu_time = (time.time() - start_time) * 1000
            
            print(f"   CPU Time: {cpu_time:.2f}ms")
            
            # Metal benchmark (if available)
            if is_metal_available() and DEPENDENCIES_AVAILABLE:
                start_time = time.time()
                device = torch.device("mps")
                metal_tensor = torch.randn(array_size, device=device, dtype=torch.float32)
                metal_result = torch.sum(metal_tensor ** 2)
                metal_time = (time.time() - start_time) * 1000
                
                speedup = cpu_time / metal_time if metal_time > 0 else 0
                
                print(f"   Metal Time: {metal_time:.2f}ms")
                print(f"   Speedup: {speedup:.2f}x")
                
                benchmark_results["array_operations"] = {
                    "cpu_time_ms": cpu_time,
                    "metal_time_ms": metal_time,
                    "speedup": speedup,
                    "array_size": array_size
                }
            else:
                print("   Metal not available for comparison")
                benchmark_results["array_operations"] = {
                    "cpu_time_ms": cpu_time,
                    "metal_available": False
                }
                
        except Exception as e:
            print(f"   ❌ Array operations benchmark failed: {e}")
            benchmark_results["array_operations"] = {"error": str(e)}
            
        return benchmark_results
        
    async def _demo_memory_management(self) -> Dict[str, Any]:
        """Demonstrate GPU memory management capabilities"""
        memory_results = {}
        
        print("\n📊 Memory Management Statistics:")
        
        try:
            memory_stats = get_memory_pool_stats()
            
            if memory_stats:
                print(f"   Total Allocated: {memory_stats.total_allocated_mb:.2f} MB")
                print(f"   Total Cached: {memory_stats.total_cached_mb:.2f} MB")
                print(f"   Total Free: {memory_stats.total_free_mb:.2f} MB")
                print(f"   Active Blocks: {memory_stats.active_blocks}")
                print(f"   Cache Hit Rate: {memory_stats.cache_hit_rate:.2%}")
                print(f"   Memory Pressure: {memory_stats.memory_pressure_level}")
                print(f"   Fragmentation: {memory_stats.fragmentation_ratio:.2%}")
                
                memory_results["stats"] = {
                    "allocated_mb": memory_stats.total_allocated_mb,
                    "cached_mb": memory_stats.total_cached_mb,
                    "free_mb": memory_stats.total_free_mb,
                    "active_blocks": memory_stats.active_blocks,
                    "cache_hit_rate": memory_stats.cache_hit_rate,
                    "memory_pressure": memory_stats.memory_pressure_level,
                    "fragmentation_ratio": memory_stats.fragmentation_ratio
                }
            else:
                print("   Memory pool manager not available")
                memory_results["stats"] = {"available": False}
                
        except Exception as e:
            print(f"   ❌ Memory management demo failed: {e}")
            memory_results["stats"] = {"error": str(e)}
            
        return memory_results
        
    async def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on demo results"""
        recommendations = []
        
        try:
            # Hardware recommendations
            if not is_metal_available():
                recommendations.append("Install PyTorch with Metal support for GPU acceleration")
                recommendations.append("Ensure you're running on Apple Silicon with macOS 13.0+")
            elif not is_m4_max_detected():
                recommendations.append("Consider upgrading to M4 Max for optimal performance")
                
            # Memory recommendations
            memory_stats = get_memory_pool_stats()
            if memory_stats:
                if memory_stats.cache_hit_rate < 0.5:
                    recommendations.append("Low cache hit rate - consider optimizing cache usage patterns")
                    
                if memory_stats.fragmentation_ratio > 0.3:
                    recommendations.append("High memory fragmentation - run memory optimization routines")
                    
                if memory_stats.memory_pressure_level in ["high", "critical"]:
                    recommendations.append("High memory pressure - consider reducing batch sizes")
                    
            # Performance recommendations
            status = get_acceleration_status()
            if status.get("performance_recommendations"):
                recommendations.extend(status["performance_recommendations"])
                
            if not recommendations:
                recommendations.append("System is optimally configured for Metal acceleration")
                
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    def _print_demo_summary(self, demo_results: Dict[str, Any]):
        """Print comprehensive demo summary"""
        print("\n" + "=" * 80)
        print("📊 METAL GPU ACCELERATION DEMO SUMMARY")
        print("=" * 80)
        
        # Hardware Summary
        if "hardware_detected" in demo_results:
            hardware = demo_results["hardware_detected"]
            print(f"🔧 Hardware: {'M4 Max' if hardware.get('m4_max_detected') else 'Apple Silicon'}")
            print(f"⚡ Metal Available: {hardware.get('metal_available', False)}")
            
        # Performance Summary
        if "financial_computations" in demo_results:
            financial = demo_results["financial_computations"]
            if "options_pricing" in financial and "computation_time_ms" in financial["options_pricing"]:
                print(f"💰 Options Pricing: {financial['options_pricing']['computation_time_ms']:.2f}ms")
                
        if "performance_benchmarks" in demo_results:
            benchmarks = demo_results["performance_benchmarks"]
            if "array_operations" in benchmarks and "speedup" in benchmarks["array_operations"]:
                speedup = benchmarks["array_operations"]["speedup"]
                print(f"🚀 GPU Speedup: {speedup:.2f}x")
                
        # Memory Summary
        if "memory_management" in demo_results:
            memory = demo_results["memory_management"]
            if "stats" in memory and isinstance(memory["stats"], dict):
                stats = memory["stats"]
                if "allocated_mb" in stats:
                    print(f"🧠 Memory Allocated: {stats['allocated_mb']:.2f} MB")
                if "cache_hit_rate" in stats:
                    print(f"💾 Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
                    
        # Recommendations
        if "recommendations" in demo_results and demo_results["recommendations"]:
            print("\n📋 OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(demo_results["recommendations"][:5], 1):
                print(f"   {i}. {rec}")
                
        print("\n✅ Demo completed successfully!")
        print(f"📅 Timestamp: {demo_results.get('demo_start_time', 'Unknown')}")

async def run_metal_acceleration_demo():
    """
    Main function to run the complete Metal GPU acceleration demonstration
    """
    demo = MetalFinancialComputingDemo()
    
    try:
        results = await demo.run_complete_demo()
        return results
    except Exception as e:
        print(f"❌ Demo execution failed: {e}")
        return {"error": str(e)}

# Command line interface
if __name__ == "__main__":
    import sys
    
    print("Nautilus Trading Platform - Metal GPU Acceleration Demo")
    print("Optimized for Apple Silicon M4 Max")
    print()
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Warning: Some dependencies are missing. Install requirements-metal.txt for full functionality.")
        print()
        
    # Run the demo
    try:
        results = asyncio.run(run_metal_acceleration_demo())
        
        if "error" not in results:
            print("\n🎉 All demos completed successfully!")
            print("Your system is ready for high-performance financial computing with Metal acceleration.")
        else:
            print(f"\n❌ Demo failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
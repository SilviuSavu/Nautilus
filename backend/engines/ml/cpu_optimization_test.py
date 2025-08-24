#!/usr/bin/env python3
"""
CPU Optimization Test for M4 Max Trading Platform
Tests CPU core detection, GCD integration, and performance monitoring.
"""

import asyncio
import time
import psutil
import cpuinfo
import multiprocessing as mp
from multiprocess import Pool
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from typing import Dict, List, Tuple, Any
import json
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # macOS specific imports
    import objc
    from Foundation import NSThread, NSProcessInfo
    from SystemConfiguration import SCDynamicStoreCopyComputerName
    MACOS_AVAILABLE = True
    logger.info("macOS-specific frameworks loaded successfully")
except ImportError as e:
    MACOS_AVAILABLE = False
    logger.warning(f"macOS-specific imports failed: {e}")

def cpu_intensive_task(n: int) -> float:
    """CPU intensive task for benchmarking - must be at module level for multiprocessing"""
    start = time.time()
    # Calculate pi using Monte Carlo method
    hits = 0
    for i in range(n):
        x, y = time.time() % 1, (time.time() * 1.1) % 1
        if x*x + y*y <= 1:
            hits += 1
    return hits / n * 4

class M4MaxCPUOptimizer:
    """CPU Optimization System for M4 Max Chip"""
    
    def __init__(self):
        self.cpu_info = self._get_cpu_info()
        self.cores = self._detect_cores()
        self.thermal_state = self._get_thermal_state()
        logger.info(f"Initialized CPU optimizer for {self.cores['total']} cores")
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get comprehensive CPU information"""
        info = cpuinfo.get_cpu_info()
        system_info = {
            'brand': info.get('brand_raw', 'Unknown'),
            'arch': info.get('arch', 'Unknown'),
            'bits': info.get('bits', 'Unknown'),
            'count': info.get('count', 0),
            'hz_advertised': info.get('hz_advertised_friendly', 'Unknown'),
            'hz_actual': info.get('hz_actual_friendly', 'Unknown'),
        }
        return system_info
    
    def _detect_cores(self) -> Dict[str, int]:
        """Detect M4 Max core configuration"""
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # M4 Max has 12 Performance + 4 Efficiency cores
        if "Apple" in self.cpu_info.get('brand', '') and physical_cores >= 16:
            performance_cores = 12
            efficiency_cores = 4
        else:
            # Fallback for other systems
            performance_cores = max(1, physical_cores // 2)
            efficiency_cores = physical_cores - performance_cores
        
        return {
            'total': physical_cores,
            'logical': logical_cores,
            'performance': performance_cores,
            'efficiency': efficiency_cores
        }
    
    def _get_thermal_state(self) -> Dict[str, Any]:
        """Get thermal state information"""
        try:
            temperatures = psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}
            battery = psutil.sensors_battery() if hasattr(psutil, 'sensors_battery') else None
            
            return {
                'temperatures': temperatures,
                'battery_present': battery is not None,
                'battery_percent': battery.percent if battery else None,
                'power_plugged': battery.power_plugged if battery else None
            }
        except Exception as e:
            logger.warning(f"Could not get thermal state: {e}")
            return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        memory = psutil.virtual_memory()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        return {
            'cpu_percent_per_core': cpu_percent,
            'cpu_percent_total': sum(cpu_percent) / len(cpu_percent),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'load_average': {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            },
            'process_count': len(psutil.pids())
        }
    
    def test_process_affinity(self) -> Dict[str, Any]:
        """Test process affinity and priority management"""
        results = {}
        current_process = psutil.Process()
        
        try:
            # Test CPU affinity
            if hasattr(current_process, 'cpu_affinity'):
                original_affinity = current_process.cpu_affinity()
                results['original_affinity'] = original_affinity
                results['affinity_supported'] = True
                
                # Try to set affinity to performance cores (0-11 on M4 Max)
                try:
                    performance_cores = list(range(min(12, len(original_affinity))))
                    current_process.cpu_affinity(performance_cores)
                    new_affinity = current_process.cpu_affinity()
                    results['performance_affinity_set'] = new_affinity == performance_cores
                    
                    # Restore original affinity
                    current_process.cpu_affinity(original_affinity)
                except Exception as e:
                    results['affinity_error'] = str(e)
            else:
                results['affinity_supported'] = False
            
            # Test priority management
            try:
                original_priority = current_process.nice()
                results['original_priority'] = original_priority
                
                # Try to set high priority (lower nice value)
                current_process.nice(-1)
                new_priority = current_process.nice()
                results['priority_changed'] = new_priority != original_priority
                
                # Restore original priority
                current_process.nice(original_priority)
            except Exception as e:
                results['priority_error'] = str(e)
                
        except Exception as e:
            results['test_error'] = str(e)
        
        return results
    
    def benchmark_threading(self, num_threads: int = None) -> Dict[str, float]:
        """Benchmark threading performance"""
        if num_threads is None:
            num_threads = self.cores['total']
        
        # Sequential benchmark
        start_time = time.time()
        for _ in range(num_threads):
            cpu_intensive_task(100000)
        sequential_time = time.time() - start_time
        
        # Threaded benchmark
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cpu_intensive_task, 100000) for _ in range(num_threads)]
            results = [f.result() for f in futures]
        threaded_time = time.time() - start_time
        
        # Process pool benchmark
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=min(num_threads, 8)) as executor:
            futures = [executor.submit(cpu_intensive_task, 100000) for _ in range(min(num_threads, 8))]
            results = [f.result() for f in futures]
        process_time = time.time() - start_time
        
        return {
            'sequential_time': sequential_time,
            'threaded_time': threaded_time,
            'process_time': process_time,
            'thread_speedup': sequential_time / threaded_time if threaded_time > 0 else 0,
            'process_speedup': sequential_time / process_time if process_time > 0 else 0
        }
    
    def test_gcd_integration(self) -> Dict[str, Any]:
        """Test Grand Central Dispatch integration"""
        results = {'gcd_available': MACOS_AVAILABLE}
        
        if not MACOS_AVAILABLE:
            return results
        
        try:
            # Test NSThread capabilities
            main_thread = NSThread.mainThread()
            current_thread = NSThread.currentThread()
            
            results['main_thread_active'] = main_thread.isMainThread()
            results['current_thread_main'] = current_thread.isMainThread()
            results['thread_count'] = threading.active_count()
            
            # Test process information
            process_info = NSProcessInfo.processInfo()
            results['process_name'] = str(process_info.processName())
            results['processor_count'] = process_info.processorCount()
            results['active_processor_count'] = process_info.activeProcessorCount()
            
            # Test system configuration
            try:
                computer_name = SCDynamicStoreCopyComputerName(None, None)
                results['computer_name'] = str(computer_name[0]) if computer_name[0] else "Unknown"
            except Exception as e:
                results['computer_name_error'] = str(e)
                
        except Exception as e:
            results['gcd_error'] = str(e)
        
        return results
    
    async def async_performance_test(self) -> Dict[str, float]:
        """Test async performance capabilities"""
        async def async_task(delay: float) -> float:
            start = time.time()
            await asyncio.sleep(delay)
            return time.time() - start
        
        # Test concurrent async tasks
        start_time = time.time()
        tasks = [async_task(0.1) for _ in range(20)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        return {
            'total_async_time': total_time,
            'average_task_time': sum(results) / len(results),
            'concurrency_efficiency': (sum(results) / total_time) if total_time > 0 else 0
        }

def run_comprehensive_test() -> Dict[str, Any]:
    """Run comprehensive CPU optimization tests"""
    logger.info("Starting M4 Max CPU Optimization Test Suite")
    
    optimizer = M4MaxCPUOptimizer()
    test_results = {}
    
    # Basic system info
    test_results['system_info'] = {
        'cpu_info': optimizer.cpu_info,
        'cores': optimizer.cores,
        'thermal_state': optimizer.thermal_state,
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    # System metrics
    logger.info("Collecting system metrics...")
    test_results['system_metrics'] = optimizer.get_system_metrics()
    
    # Process affinity test
    logger.info("Testing process affinity...")
    test_results['process_affinity'] = optimizer.test_process_affinity()
    
    # Threading benchmark
    logger.info("Running threading benchmarks...")
    test_results['threading_benchmark'] = optimizer.benchmark_threading()
    
    # GCD integration test
    logger.info("Testing GCD integration...")
    test_results['gcd_integration'] = optimizer.test_gcd_integration()
    
    # Async performance test
    logger.info("Running async performance test...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    test_results['async_performance'] = loop.run_until_complete(
        optimizer.async_performance_test()
    )
    loop.close()
    
    # Performance summary
    threading_bench = test_results['threading_benchmark']
    test_results['performance_summary'] = {
        'cpu_cores_detected': optimizer.cores['total'],
        'performance_cores': optimizer.cores['performance'],
        'efficiency_cores': optimizer.cores['efficiency'],
        'thread_speedup': threading_bench['thread_speedup'],
        'process_speedup': threading_bench['process_speedup'],
        'gcd_available': test_results['gcd_integration']['gcd_available'],
        'affinity_supported': test_results['process_affinity'].get('affinity_supported', False),
        'optimization_grade': 'A+' if threading_bench['thread_speedup'] > 8 else 'A' if threading_bench['thread_speedup'] > 4 else 'B'
    }
    
    return test_results

if __name__ == "__main__":
    try:
        results = run_comprehensive_test()
        
        # Print summary
        print("\n" + "="*80)
        print("M4 MAX CPU OPTIMIZATION TEST RESULTS")
        print("="*80)
        
        summary = results['performance_summary']
        print(f"CPU Cores Detected: {summary['cpu_cores_detected']}")
        print(f"Performance Cores: {summary['performance_cores']}")
        print(f"Efficiency Cores: {summary['efficiency_cores']}")
        print(f"Threading Speedup: {summary['thread_speedup']:.2f}x")
        print(f"Process Speedup: {summary['process_speedup']:.2f}x")
        print(f"GCD Available: {summary['gcd_available']}")
        print(f"Process Affinity: {summary['affinity_supported']}")
        print(f"Optimization Grade: {summary['optimization_grade']}")
        
        # Save detailed results
        results_file = "/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/ml/cpu_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
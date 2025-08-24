"""
Trading Core Performance Benchmark
===================================

Comprehensive benchmarking suite for trading core optimizations.
Measures latency, throughput, and memory efficiency improvements.

Benchmark Categories:
- Order Processing Latency
- Memory Pool Efficiency  
- Venue Selection Performance
- End-to-End Trading Flow
- Concurrent Load Testing
"""

import asyncio
import time
import statistics
import gc
import psutil
import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

from .execution_engine import ExecutionEngine, SmartOrderRouter
from .optimized_execution_engine import OptimizedExecutionEngine
from .order_management import Order, OrderManagementSystem, OrderType, OrderSide, TimeInForce
from .poolable_objects import (
    create_pooled_order, release_pooled_order, trading_pools, 
    PooledOrder, OrderType, OrderSide, TimeInForce
)
from .memory_pool import pool_manager, get_global_pool_status
from .execution_engine import VenueStatus

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    test_name: str
    implementation: str  # "baseline" or "optimized"
    latency_stats: Dict[str, float]  # min, max, avg, p50, p95, p99
    throughput: float  # operations per second
    memory_stats: Dict[str, Any]
    cpu_usage: float
    error_count: int
    test_duration_seconds: float
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark for trading core optimizations.
    
    Features:
    - Latency distribution analysis
    - Memory usage tracking
    - Throughput measurement
    - Comparative analysis (baseline vs optimized)
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_engine = ExecutionEngine()
        self.optimized_engine = OptimizedExecutionEngine()
        self.baseline_oms = OrderManagementSystem()
        
        # Mock venue for testing
        self.mock_venue = MockVenueForBenchmark("TEST_VENUE")
        
        # Setup engines
        self.baseline_engine.add_venue(self.mock_venue)
        self.optimized_engine.add_venue(self.mock_venue)
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting Trading Core Performance Benchmark")
        
        # Warm up
        await self._warmup()
        
        benchmark_results = {
            "benchmark_timestamp": time.time(),
            "system_info": self._get_system_info(),
            "test_results": {},
            "summary": {}
        }
        
        # Run individual benchmarks
        tests = [
            ("order_creation", self._benchmark_order_creation),
            ("order_processing", self._benchmark_order_processing),
            ("venue_selection", self._benchmark_venue_selection),
            ("memory_efficiency", self._benchmark_memory_efficiency),
            ("concurrent_load", self._benchmark_concurrent_load),
            ("end_to_end_latency", self._benchmark_end_to_end_latency)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"Running benchmark: {test_name}")
            try:
                result = await test_func()
                benchmark_results["test_results"][test_name] = result
            except Exception as e:
                logger.error(f"Benchmark {test_name} failed: {e}")
                benchmark_results["test_results"][test_name] = {"error": str(e)}
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary(benchmark_results["test_results"])
        
        logger.info("Trading Core Performance Benchmark Complete")
        return benchmark_results
    
    async def _warmup(self):
        """Warm up engines and pools."""
        logger.info("Warming up engines and memory pools...")
        
        # Create and release objects to populate pools
        for _ in range(100):
            order = create_pooled_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            release_pooled_order(order)
        
        # Warm up engines with test orders
        for _ in range(10):
            test_order = create_pooled_order(
                symbol="WARM",
                side=OrderSide.BUY, 
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            
            await self.optimized_engine.submit_order_optimized(test_order)
            release_pooled_order(test_order)
        
        # Force GC to clear warmup allocations
        gc.collect()
        
        logger.info("Warmup complete")
    
    async def _benchmark_order_creation(self) -> Dict[str, Any]:
        """Benchmark order creation performance."""
        iterations = 10000
        
        # Baseline: Traditional order creation
        baseline_times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100.0,
                price=150.0
            )
            
            end = time.perf_counter_ns()
            baseline_times.append(end - start)
        
        # Optimized: Pool-based order creation
        optimized_times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            
            order = create_pooled_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100.0,
                price=150.0
            )
            release_pooled_order(order)
            
            end = time.perf_counter_ns()
            optimized_times.append(end - start)
        
        return {
            "baseline": self._calculate_stats(baseline_times, "Order Creation Baseline"),
            "optimized": self._calculate_stats(optimized_times, "Order Creation Optimized"),
            "improvement": {
                "latency_reduction_percent": self._calculate_improvement(baseline_times, optimized_times),
                "p99_improvement_us": (np.percentile(baseline_times, 99) - np.percentile(optimized_times, 99)) / 1000
            }
        }
    
    async def _benchmark_order_processing(self) -> Dict[str, Any]:
        """Benchmark order processing performance."""
        iterations = 1000
        
        # Baseline processing
        baseline_times = []
        for i in range(iterations):
            order = Order(
                symbol=f"TEST{i % 10}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            
            start = time.perf_counter_ns()
            await self.baseline_engine.submit_order(order)
            end = time.perf_counter_ns()
            
            baseline_times.append(end - start)
        
        # Optimized processing
        optimized_times = []
        for i in range(iterations):
            order = create_pooled_order(
                symbol=f"TEST{i % 10}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            
            start = time.perf_counter_ns()
            await self.optimized_engine.submit_order_optimized(order)
            end = time.perf_counter_ns()
            
            optimized_times.append(end - start)
            release_pooled_order(order)
        
        return {
            "baseline": self._calculate_stats(baseline_times, "Order Processing Baseline"),
            "optimized": self._calculate_stats(optimized_times, "Order Processing Optimized"),
            "improvement": {
                "latency_reduction_percent": self._calculate_improvement(baseline_times, optimized_times),
                "p99_improvement_us": (np.percentile(baseline_times, 99) - np.percentile(optimized_times, 99)) / 1000
            }
        }
    
    async def _benchmark_venue_selection(self) -> Dict[str, Any]:
        """Benchmark venue selection performance."""
        iterations = 5000
        
        # Add multiple mock venues for selection testing
        venues = []
        for i in range(5):
            venue = MockVenueForBenchmark(f"VENUE_{i}")
            venues.append(venue)
            self.optimized_engine.add_venue(venue)
        
        # Baseline router
        baseline_router = SmartOrderRouter()
        for venue in venues:
            baseline_router.add_venue(venue)
        
        # Baseline venue selection
        baseline_times = []
        for i in range(iterations):
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            
            start = time.perf_counter_ns()
            await baseline_router.route_order(order)
            end = time.perf_counter_ns()
            
            baseline_times.append(end - start)
        
        # Optimized venue selection
        optimized_times = []
        for i in range(iterations):
            order = create_pooled_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            
            start = time.perf_counter_ns()
            await self.optimized_engine.router.route_order_optimized(order)
            end = time.perf_counter_ns()
            
            optimized_times.append(end - start)
            release_pooled_order(order)
        
        return {
            "baseline": self._calculate_stats(baseline_times, "Venue Selection Baseline"),
            "optimized": self._calculate_stats(optimized_times, "Venue Selection Optimized"),
            "improvement": {
                "latency_reduction_percent": self._calculate_improvement(baseline_times, optimized_times),
                "p99_improvement_us": (np.percentile(baseline_times, 99) - np.percentile(optimized_times, 99)) / 1000
            },
            "venue_cache_stats": self.optimized_engine.router.venue_cache.get_venue_metrics()
        }
    
    async def _benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory usage and pool efficiency."""
        iterations = 10000
        
        # Measure baseline memory usage
        gc.collect()
        baseline_start_memory = self.process.memory_info().rss
        
        baseline_objects = []
        for _ in range(iterations):
            order = Order(
                symbol="MEMORY_TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            baseline_objects.append(order)
        
        baseline_peak_memory = self.process.memory_info().rss
        baseline_memory_used = baseline_peak_memory - baseline_start_memory
        
        # Clear baseline objects
        del baseline_objects
        gc.collect()
        
        # Measure optimized memory usage
        optimized_start_memory = self.process.memory_info().rss
        
        for _ in range(iterations):
            order = create_pooled_order(
                symbol="MEMORY_TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            release_pooled_order(order)  # Immediate release
        
        optimized_peak_memory = self.process.memory_info().rss
        optimized_memory_used = optimized_peak_memory - optimized_start_memory
        
        # Pool statistics
        pool_stats = trading_pools.get_pool_statistics()
        global_pool_stats = pool_manager.get_global_metrics()
        
        return {
            "baseline_memory_mb": baseline_memory_used / (1024 * 1024),
            "optimized_memory_mb": optimized_memory_used / (1024 * 1024),
            "memory_reduction_percent": ((baseline_memory_used - optimized_memory_used) / baseline_memory_used) * 100,
            "memory_savings_mb": (baseline_memory_used - optimized_memory_used) / (1024 * 1024),
            "pool_statistics": pool_stats,
            "global_pool_metrics": global_pool_stats,
            "pool_efficiency": {
                "avg_hit_rate": global_pool_stats.get("average_hit_rate", 0),
                "total_active_objects": global_pool_stats.get("total_objects_active", 0),
                "total_pooled_objects": global_pool_stats.get("total_objects_available", 0)
            }
        }
    
    async def _benchmark_concurrent_load(self) -> Dict[str, Any]:
        """Benchmark concurrent order processing."""
        concurrent_orders = 1000
        batch_size = 100
        
        async def process_order_batch_baseline():
            orders = []
            start_time = time.perf_counter()
            
            for i in range(batch_size):
                order = Order(
                    symbol=f"LOAD{i % 10}",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100.0
                )
                orders.append(order)
                await self.baseline_engine.submit_order(order)
            
            end_time = time.perf_counter()
            return end_time - start_time, len(orders)
        
        async def process_order_batch_optimized():
            orders = []
            start_time = time.perf_counter()
            
            for i in range(batch_size):
                order = create_pooled_order(
                    symbol=f"LOAD{i % 10}",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100.0
                )
                orders.append(order)
                await self.optimized_engine.submit_order_optimized(order)
            
            end_time = time.perf_counter()
            
            # Release all orders
            for order in orders:
                release_pooled_order(order)
            
            return end_time - start_time, len(orders)
        
        # Baseline concurrent processing
        baseline_tasks = []
        for _ in range(concurrent_orders // batch_size):
            task = asyncio.create_task(process_order_batch_baseline())
            baseline_tasks.append(task)
        
        baseline_start = time.perf_counter()
        baseline_results = await asyncio.gather(*baseline_tasks)
        baseline_total_time = time.perf_counter() - baseline_start
        
        baseline_total_orders = sum(result[1] for result in baseline_results)
        baseline_throughput = baseline_total_orders / baseline_total_time
        
        # Optimized concurrent processing
        optimized_tasks = []
        for _ in range(concurrent_orders // batch_size):
            task = asyncio.create_task(process_order_batch_optimized())
            optimized_tasks.append(task)
        
        optimized_start = time.perf_counter()
        optimized_results = await asyncio.gather(*optimized_tasks)
        optimized_total_time = time.perf_counter() - optimized_start
        
        optimized_total_orders = sum(result[1] for result in optimized_results)
        optimized_throughput = optimized_total_orders / optimized_total_time
        
        return {
            "baseline": {
                "total_orders": baseline_total_orders,
                "total_time_seconds": baseline_total_time,
                "throughput_orders_per_second": baseline_throughput
            },
            "optimized": {
                "total_orders": optimized_total_orders,
                "total_time_seconds": optimized_total_time,
                "throughput_orders_per_second": optimized_throughput
            },
            "improvement": {
                "throughput_improvement_percent": ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100,
                "time_reduction_percent": ((baseline_total_time - optimized_total_time) / baseline_total_time) * 100
            }
        }
    
    async def _benchmark_end_to_end_latency(self) -> Dict[str, Any]:
        """Benchmark complete end-to-end trading flow."""
        iterations = 500
        
        # End-to-end baseline flow
        baseline_times = []
        for i in range(iterations):
            start = time.perf_counter_ns()
            
            # Create order
            order = Order(
                symbol=f"E2E{i % 10}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100.0,
                price=150.0
            )
            
            # Submit through OMS
            await self.baseline_oms.submit_order(order)
            
            # Submit to execution engine
            await self.baseline_engine.submit_order(order)
            
            end = time.perf_counter_ns()
            baseline_times.append(end - start)
        
        # End-to-end optimized flow
        optimized_times = []
        for i in range(iterations):
            start = time.perf_counter_ns()
            
            # Create order from pool
            order = create_pooled_order(
                symbol=f"E2E{i % 10}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100.0,
                price=150.0
            )
            
            # Submit to optimized execution engine
            await self.optimized_engine.submit_order_optimized(order)
            
            # Release order
            release_pooled_order(order)
            
            end = time.perf_counter_ns()
            optimized_times.append(end - start)
        
        return {
            "baseline": self._calculate_stats(baseline_times, "End-to-End Baseline"),
            "optimized": self._calculate_stats(optimized_times, "End-to-End Optimized"),
            "improvement": {
                "latency_reduction_percent": self._calculate_improvement(baseline_times, optimized_times),
                "p99_improvement_us": (np.percentile(baseline_times, 99) - np.percentile(optimized_times, 99)) / 1000,
                "target_achievement": {
                    "target_p99_us": 2800,  # <2.8ms target
                    "actual_p99_us": np.percentile(optimized_times, 99) / 1000,
                    "target_met": np.percentile(optimized_times, 99) / 1000 < 2800
                }
            }
        }
    
    def _calculate_stats(self, times: List[float], name: str) -> Dict[str, Any]:
        """Calculate timing statistics."""
        if not times:
            return {"error": "No timing data"}
        
        times_us = [t / 1000 for t in times]  # Convert to microseconds
        
        return {
            "name": name,
            "sample_count": len(times),
            "min_us": min(times_us),
            "max_us": max(times_us),
            "avg_us": statistics.mean(times_us),
            "median_us": statistics.median(times_us),
            "p95_us": np.percentile(times_us, 95),
            "p99_us": np.percentile(times_us, 99),
            "std_dev_us": statistics.stdev(times_us) if len(times_us) > 1 else 0
        }
    
    def _calculate_improvement(self, baseline_times: List[float], optimized_times: List[float]) -> float:
        """Calculate percentage improvement."""
        if not baseline_times or not optimized_times:
            return 0.0
        
        baseline_avg = statistics.mean(baseline_times)
        optimized_avg = statistics.mean(optimized_times)
        
        if baseline_avg == 0:
            return 0.0
        
        return ((baseline_avg - optimized_avg) / baseline_avg) * 100
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{psutil.version_info[0]}.{psutil.version_info[1]}.{psutil.version_info[2]}",
            "process_info": {
                "pid": self.process.pid,
                "threads": self.process.num_threads(),
                "memory_mb": self.process.memory_info().rss / (1024*1024)
            }
        }
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "overall_performance": {},
            "key_improvements": {},
            "target_achievement": {},
            "recommendations": []
        }
        
        # Aggregate improvements
        improvements = []
        p99_improvements = []
        
        for test_name, result in test_results.items():
            if "improvement" in result and "latency_reduction_percent" in result["improvement"]:
                improvements.append(result["improvement"]["latency_reduction_percent"])
            
            if "improvement" in result and "p99_improvement_us" in result["improvement"]:
                p99_improvements.append(result["improvement"]["p99_improvement_us"])
        
        if improvements:
            summary["overall_performance"]["avg_latency_improvement_percent"] = statistics.mean(improvements)
            summary["overall_performance"]["max_latency_improvement_percent"] = max(improvements)
        
        if p99_improvements:
            summary["overall_performance"]["total_p99_improvement_us"] = sum(p99_improvements)
        
        # Memory efficiency
        if "memory_efficiency" in test_results:
            mem_result = test_results["memory_efficiency"]
            summary["key_improvements"]["memory_reduction_percent"] = mem_result.get("memory_reduction_percent", 0)
            summary["key_improvements"]["pool_hit_rate"] = mem_result.get("pool_efficiency", {}).get("avg_hit_rate", 0)
        
        # Throughput improvements
        if "concurrent_load" in test_results:
            load_result = test_results["concurrent_load"]
            summary["key_improvements"]["throughput_improvement_percent"] = load_result.get("improvement", {}).get("throughput_improvement_percent", 0)
        
        # Target achievement
        if "end_to_end_latency" in test_results:
            e2e_result = test_results["end_to_end_latency"]
            target_info = e2e_result.get("improvement", {}).get("target_achievement", {})
            summary["target_achievement"] = target_info
        
        # Recommendations
        avg_improvement = summary["overall_performance"].get("avg_latency_improvement_percent", 0)
        if avg_improvement > 90:
            summary["recommendations"].append("Excellent performance improvements achieved. Ready for production deployment.")
        elif avg_improvement > 70:
            summary["recommendations"].append("Good performance improvements. Consider additional optimizations for maximum benefit.")
        else:
            summary["recommendations"].append("Performance improvements detected. Additional optimization work recommended.")
        
        return summary
    
    def save_results(self, filename: str):
        """Save benchmark results to file."""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filename}")


class MockVenueForBenchmark:
    """Mock venue for performance testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = VenueStatus.CONNECTED
        self.metrics = type('obj', (object,), {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'average_fill_time_ms': 5.0,
            'fill_rate_percentage': 95.0,
            'uptime_percentage': 99.5
        })()
        self.callbacks = []
    
    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)
    
    async def submit_order(self, order) -> bool:
        self.metrics.total_orders += 1
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        return True
    
    async def get_quote(self, symbol: str):
        # Simulate quote retrieval with small delay
        await asyncio.sleep(0.001)  # 1ms simulated latency
        return type('obj', (object,), {
            'symbol': symbol,
            'bid_price': 100.0,
            'ask_price': 100.05,
            'bid_size': 1000,
            'ask_size': 1000,
            'spread': 0.05,
            'mid_price': 100.025
        })()


# Convenience function to run benchmark
async def run_performance_benchmark() -> Dict[str, Any]:
    """Run comprehensive performance benchmark."""
    benchmark = PerformanceBenchmark()
    return await benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    async def main():
        results = await run_performance_benchmark()
        
        # Print summary
        print("\n" + "="*80)
        print("TRADING CORE PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        summary = results.get("summary", {})
        overall = summary.get("overall_performance", {})
        
        print(f"Average Latency Improvement: {overall.get('avg_latency_improvement_percent', 0):.1f}%")
        print(f"Maximum Latency Improvement: {overall.get('max_latency_improvement_percent', 0):.1f}%")
        print(f"Total P99 Improvement: {overall.get('total_p99_improvement_us', 0):.2f}μs")
        
        key_improvements = summary.get("key_improvements", {})
        print(f"Memory Reduction: {key_improvements.get('memory_reduction_percent', 0):.1f}%")
        print(f"Pool Hit Rate: {key_improvements.get('pool_hit_rate', 0):.1f}%")
        print(f"Throughput Improvement: {key_improvements.get('throughput_improvement_percent', 0):.1f}%")
        
        target = summary.get("target_achievement", {})
        if target:
            print(f"\nTarget Achievement:")
            print(f"  Target P99: {target.get('target_p99_us', 0):.0f}μs")
            print(f"  Actual P99: {target.get('actual_p99_us', 0):.0f}μs")
            print(f"  Target Met: {'YES' if target.get('target_met', False) else 'NO'}")
        
        print("\nRecommendations:")
        for rec in summary.get("recommendations", []):
            print(f"  • {rec}")
        
        print("\n" + "="*80)
    
    asyncio.run(main())
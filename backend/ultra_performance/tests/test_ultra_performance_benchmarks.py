"""
Ultra-Performance Benchmarking and Validation Tests

Comprehensive benchmarking suite to validate:
- GPU acceleration performance gains
- Ultra-low latency optimizations
- Memory pool efficiency
- Network I/O optimizations  
- Cache performance improvements
- Overall system performance regression testing
"""

import asyncio
import pytest
import time
import numpy as np
import threading
from typing import Dict, List, Any
import logging

# Import ultra-performance modules
try:
    from ultra_performance.gpu_acceleration import (
        cuda_manager, gpu_risk_calculator, gpu_monte_carlo, gpu_matrix_ops
    )
    from ultra_performance.low_latency import (
        ultra_low_latency_optimizer, microsecond_timer, UltraFastOrderBook
    )
    from ultra_performance.advanced_caching import (
        distributed_cache_manager, intelligent_cache_warmer
    )
    from ultra_performance.memory_pool import (
        custom_memory_allocator, object_pool_manager, gc_optimizer
    )
    from ultra_performance.network_io import (
        dpdk_network_manager, zero_copy_io_manager, optimized_serialization
    )
    from ultra_performance.performance_monitoring import (
        ultra_performance_metrics, benchmark_function
    )
    ULTRA_PERFORMANCE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ultra-performance modules not available: {e}")
    ULTRA_PERFORMANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Test fixtures
@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for testing"""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 1000)  # Daily returns with 2% volatility

@pytest.fixture
def sample_correlation_data():
    """Generate sample correlation matrix data"""
    np.random.seed(42)
    n_assets = 10
    returns = np.random.normal(0, 0.02, (252, n_assets))
    return returns

class TestGPUAcceleration:
    """Test GPU acceleration performance and correctness"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_gpu_var_calculation_performance(self, sample_returns_data):
        """Test GPU VaR calculation performance vs CPU"""
        # GPU calculation
        start_time = time.perf_counter()
        gpu_result = await gpu_risk_calculator.calculate_var_gpu(
            sample_returns_data, confidence_level=0.05
        )
        gpu_time = time.perf_counter() - start_time
        
        # CPU calculation (fallback)
        start_time = time.perf_counter()
        cpu_result = await gpu_risk_calculator._calculate_var_cpu(
            sample_returns_data, confidence_level=0.05, lookback_days=252
        )
        cpu_time = time.perf_counter() - start_time
        
        # Validate results are similar (within 5% tolerance)
        assert abs(gpu_result["var_95"] - cpu_result["var_95"]) < 0.05 * abs(cpu_result["var_95"])
        
        # Log performance comparison
        performance_gain = cpu_time / max(gpu_time, 0.001)
        logger.info(f"GPU VaR Performance: {performance_gain:.2f}x faster than CPU")
        
        # Validate GPU acceleration was used if available
        if cuda_manager.devices:
            assert gpu_result.get("gpu_accelerated", False)
            
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_gpu_monte_carlo_performance(self):
        """Test GPU Monte Carlo simulation performance"""
        params = {
            "initial_price": 100.0,
            "volatility": 0.2,
            "risk_free_rate": 0.02,
            "time_horizon": 1.0,
            "num_simulations": 10000,
            "num_steps": 252
        }
        
        # GPU simulation
        start_time = time.perf_counter()
        gpu_result = await gpu_monte_carlo.simulate_price_paths_gpu(**params)
        gpu_time = time.perf_counter() - start_time
        
        # CPU simulation
        start_time = time.perf_counter()
        cpu_result = await gpu_monte_carlo._simulate_price_paths_cpu(**params)
        cpu_time = time.perf_counter() - start_time
        
        # Validate statistical properties
        price_diff = abs(gpu_result["mean_final_price"] - cpu_result["mean_final_price"])
        assert price_diff < 5.0  # Mean prices should be within $5
        
        # Performance comparison
        if gpu_result.get("gpu_accelerated", False):
            performance_gain = cpu_time / max(gpu_time, 0.001)
            logger.info(f"GPU Monte Carlo Performance: {performance_gain:.2f}x faster than CPU")
            assert performance_gain > 1.0  # GPU should be faster
            
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_gpu_correlation_matrix_performance(self, sample_correlation_data):
        """Test GPU correlation matrix calculation performance"""
        # GPU calculation
        start_time = time.perf_counter()
        gpu_result = await gpu_matrix_ops.calculate_correlation_matrix_gpu(sample_correlation_data)
        gpu_time = time.perf_counter() - start_time
        
        # CPU calculation
        start_time = time.perf_counter()
        cpu_result = await gpu_matrix_ops._calculate_correlation_matrix_cpu(sample_correlation_data)
        cpu_time = time.perf_counter() - start_time
        
        # Validate correlation matrices are similar
        gpu_matrix = np.array(gpu_result["correlation_matrix"])
        cpu_matrix = np.array(cpu_result["correlation_matrix"])
        
        matrix_diff = np.abs(gpu_matrix - cpu_matrix).mean()
        assert matrix_diff < 0.01  # Mean difference < 1%
        
        # Performance validation
        if gpu_result.get("gpu_accelerated", False):
            performance_gain = cpu_time / max(gpu_time, 0.001)
            logger.info(f"GPU Correlation Matrix Performance: {performance_gain:.2f}x faster than CPU")

class TestUltraLowLatency:
    """Test ultra-low latency optimizations"""
    
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_microsecond_timer_precision(self):
        """Test microsecond timer precision and accuracy"""
        timer = microsecond_timer
        timer.reset()
        
        # Test multiple measurements
        for i in range(100):
            timer.start()
            time.sleep(0.001)  # 1ms sleep
            latency = timer.stop()
            
            # Should measure approximately 1000 microseconds (±200μs tolerance)
            assert 800 <= latency <= 1200, f"Timer precision issue: {latency}μs"
            
        metrics = timer.get_metrics()
        assert metrics.total_operations == 100
        assert 800 <= metrics.avg_latency_us <= 1200
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_ultra_fast_orderbook_performance(self):
        """Test ultra-fast order book performance"""
        orderbook = UltraFastOrderBook("AAPL", None, None)  # Simplified for testing
        
        latencies = []
        
        # Benchmark order book updates
        for i in range(1000):
            price = 100.0 + (i % 100) * 0.01
            quantity = 1000 + (i % 500)
            
            if i % 2 == 0:
                latency = orderbook.update_bid(price, quantity)
            else:
                latency = orderbook.update_ask(price + 0.05, quantity)
                
            latencies.append(latency)
            
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Ultra-low latency requirements
        assert avg_latency < 50.0, f"Average latency too high: {avg_latency}μs"
        assert max_latency < 200.0, f"Max latency too high: {max_latency}μs"
        
        logger.info(f"Ultra-fast OrderBook - Avg: {avg_latency:.2f}μs, Max: {max_latency:.2f}μs")
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_zero_copy_memory_operations(self):
        """Test zero-copy memory operations performance"""
        from ultra_performance.low_latency import zero_copy_memory_manager
        
        buffer_sizes = [1024, 4096, 16384, 65536]
        
        for size in buffer_sizes:
            # Create zero-copy buffer
            start_time = time.perf_counter_ns()
            buffer = zero_copy_memory_manager.get_zero_copy_buffer(size, f"test_buffer_{size}")
            creation_time = (time.perf_counter_ns() - start_time) / 1000  # microseconds
            
            # Test memory operations
            start_time = time.perf_counter_ns()
            buffer[:1000] = b'x' * 1000  # Write operation
            read_data = bytes(buffer[:1000])  # Read operation
            operation_time = (time.perf_counter_ns() - start_time) / 1000
            
            # Return buffer
            zero_copy_memory_manager.return_zero_copy_buffer(buffer, f"test_buffer_{size}")
            
            # Validate performance
            assert creation_time < 100.0, f"Buffer creation too slow: {creation_time}μs"
            assert operation_time < 50.0, f"Memory operations too slow: {operation_time}μs"
            
        logger.info("Zero-copy memory operations performance validated")

class TestAdvancedCaching:
    """Test advanced caching strategies"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_multi_level_cache_performance(self):
        """Test multi-level cache hierarchy performance"""
        cache_manager = distributed_cache_manager
        
        # Test data
        test_keys = [f"test_key_{i}" for i in range(100)]
        test_values = [f"test_value_{i}" * 100 for i in range(100)]  # Larger values
        
        # Warm up cache
        for key, value in zip(test_keys, test_values):
            await cache_manager.set(key, value)
            
        # Benchmark cache hits
        start_time = time.perf_counter()
        
        for _ in range(1000):  # 1000 cache lookups
            key = test_keys[_ % len(test_keys)]
            value = await cache_manager.get(key)
            assert value is not None, f"Cache miss for key: {key}"
            
        total_time = time.perf_counter() - start_time
        avg_time_per_lookup = (total_time * 1_000_000) / 1000  # microseconds
        
        # Cache performance requirements
        assert avg_time_per_lookup < 100.0, f"Cache lookup too slow: {avg_time_per_lookup}μs"
        
        logger.info(f"Multi-level cache performance: {avg_time_per_lookup:.2f}μs per lookup")
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_intelligent_cache_warming(self):
        """Test intelligent cache warming effectiveness"""
        warmer = intelligent_cache_warmer
        
        # Simulate access patterns
        keys_to_access = ["hot_key_1", "hot_key_2", "hot_key_3", "cold_key_1"]
        access_frequencies = [100, 80, 60, 5]  # hot_key_1 accessed most frequently
        
        # Record access patterns
        for key, frequency in zip(keys_to_access, access_frequencies):
            for _ in range(frequency):
                warmer.record_access(key, time.time() + (_ * 0.01))
                
        # Analyze patterns
        patterns = warmer.analyze_access_patterns()
        
        # Validate that hot keys are identified
        assert "hot_key_1" in patterns
        assert patterns["hot_key_1"]["warming_priority"] > patterns["cold_key_1"]["warming_priority"]
        
        logger.info(f"Intelligent cache warming identified {len(patterns)} access patterns")
        
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_cache_coherency_performance(self):
        """Test cache coherency mechanisms"""
        # Test basic coherency operations
        from ultra_performance.advanced_caching import cache_coherency_manager
        
        test_keys = [f"coherency_test_{i}" for i in range(50)]
        
        start_time = time.perf_counter()
        
        # Subscribe to invalidations
        for key in test_keys:
            asyncio.run(cache_coherency_manager.subscribe_to_invalidations(
                key, lambda k: logger.info(f"Key {k} invalidated")
            ))
            
        subscription_time = time.perf_counter() - start_time
        
        # Test invalidation performance
        start_time = time.perf_counter()
        
        for key in test_keys:
            asyncio.run(cache_coherency_manager.invalidate_key(key, broadcast=False))
            
        invalidation_time = time.perf_counter() - start_time
        
        avg_subscription_time = (subscription_time * 1_000_000) / len(test_keys)
        avg_invalidation_time = (invalidation_time * 1_000_000) / len(test_keys)
        
        # Performance requirements
        assert avg_subscription_time < 1000.0, f"Cache subscription too slow: {avg_subscription_time}μs"
        assert avg_invalidation_time < 500.0, f"Cache invalidation too slow: {avg_invalidation_time}μs"

class TestMemoryPoolOptimization:
    """Test memory pool optimization performance"""
    
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_custom_allocator_performance(self):
        """Test custom memory allocator performance"""
        allocator = custom_memory_allocator
        
        allocation_times = []
        deallocation_times = []
        allocated_blocks = []
        
        # Benchmark allocations
        for size in [64, 128, 256, 512, 1024, 2048]:
            for _ in range(100):
                start_time = time.perf_counter_ns()
                result = allocator.allocate(size)
                allocation_time = (time.perf_counter_ns() - start_time) / 1000
                
                if result is not None:
                    block, block_id = result
                    allocation_times.append(allocation_time)
                    allocated_blocks.append(block_id)
                    
        # Benchmark deallocations
        for block_id in allocated_blocks:
            start_time = time.perf_counter_ns()
            success = allocator.deallocate(block_id)
            deallocation_time = (time.perf_counter_ns() - start_time) / 1000
            
            if success:
                deallocation_times.append(deallocation_time)
                
        # Performance validation
        avg_allocation_time = sum(allocation_times) / len(allocation_times)
        avg_deallocation_time = sum(deallocation_times) / len(deallocation_times)
        
        assert avg_allocation_time < 50.0, f"Allocation too slow: {avg_allocation_time}μs"
        assert avg_deallocation_time < 20.0, f"Deallocation too slow: {avg_deallocation_time}μs"
        
        logger.info(f"Memory allocator - Alloc: {avg_allocation_time:.2f}μs, Dealloc: {avg_deallocation_time:.2f}μs")
        
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_object_pool_performance(self):
        """Test object pool performance"""
        # Create pool for dictionary objects
        pool = object_pool_manager.create_pool(dict, pool_size=1000)
        
        allocation_times = []
        return_times = []
        objects = []
        
        # Benchmark object allocations from pool
        for _ in range(1000):
            start_time = time.perf_counter_ns()
            obj = object_pool_manager.get_object(dict)
            allocation_time = (time.perf_counter_ns() - start_time) / 1000
            
            if obj is not None:
                allocation_times.append(allocation_time)
                objects.append(obj)
                
        # Benchmark object returns to pool
        for obj in objects:
            start_time = time.perf_counter_ns()
            success = object_pool_manager.return_object(obj)
            return_time = (time.perf_counter_ns() - start_time) / 1000
            
            if success:
                return_times.append(return_time)
                
        # Validate performance
        avg_allocation_time = sum(allocation_times) / len(allocation_times)
        avg_return_time = sum(return_times) / len(return_times)
        
        assert avg_allocation_time < 10.0, f"Object allocation too slow: {avg_allocation_time}μs"
        assert avg_return_time < 5.0, f"Object return too slow: {avg_return_time}μs"
        
        logger.info(f"Object pool - Get: {avg_allocation_time:.2f}μs, Return: {avg_return_time:.2f}μs")
        
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_gc_optimization_effectiveness(self):
        """Test garbage collection optimization effectiveness"""
        gc_optimizer.optimize_gc_settings()
        
        # Create objects to trigger GC
        large_objects = []
        for _ in range(1000):
            large_objects.append([0] * 1000)  # Create large lists
            
        # Measure manual GC performance
        start_time = time.perf_counter()
        gc_results = gc_optimizer.manual_gc_cycle()
        gc_time = time.perf_counter() - start_time
        
        # Validate GC performance
        assert gc_time < 0.1, f"GC cycle too slow: {gc_time:.3f}s"
        assert gc_results["total_time_ms"] < 100.0, f"GC time too high: {gc_results['total_time_ms']:.1f}ms"
        
        # Clean up
        large_objects.clear()
        
        logger.info(f"GC optimization - Total time: {gc_results['total_time_ms']:.1f}ms")

class TestNetworkIOOptimization:
    """Test network I/O optimization performance"""
    
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_serialization_performance(self):
        """Test optimized serialization performance"""
        from ultra_performance.network_io import SerializationProtocol
        
        serializer = optimized_serialization
        
        # Test data of various sizes
        test_data = [
            {"price": 100.50, "quantity": 1000, "timestamp": time.time()},
            {"symbol": "AAPL", "bid": 150.25, "ask": 150.30},
            [100.0, 200.0, 300.0, 400.0, 500.0],
            "Simple string message"
        ]
        
        protocols_to_test = [
            SerializationProtocol.MSGPACK,
            SerializationProtocol.ORJSON,
            SerializationProtocol.BINARY_STRUCT,
            SerializationProtocol.NATIVE_BYTES
        ]
        
        for protocol in protocols_to_test:
            total_serialize_time = 0
            total_deserialize_time = 0
            successful_tests = 0
            
            for data in test_data:
                # Serialize
                start_time = time.perf_counter_ns()
                serialized = serializer.serialize_optimized(data, protocol)
                serialize_time = (time.perf_counter_ns() - start_time) / 1000
                
                if serialized is not None:
                    # Deserialize
                    start_time = time.perf_counter_ns()
                    deserialized = serializer.deserialize_optimized(serialized, protocol)
                    deserialize_time = (time.perf_counter_ns() - start_time) / 1000
                    
                    if deserialized is not None:
                        total_serialize_time += serialize_time
                        total_deserialize_time += deserialize_time
                        successful_tests += 1
                        
            if successful_tests > 0:
                avg_serialize_time = total_serialize_time / successful_tests
                avg_deserialize_time = total_deserialize_time / successful_tests
                
                # Performance requirements (should be sub-millisecond)
                assert avg_serialize_time < 1000.0, f"{protocol.value} serialization too slow: {avg_serialize_time:.1f}μs"
                assert avg_deserialize_time < 1000.0, f"{protocol.value} deserialization too slow: {avg_deserialize_time:.1f}μs"
                
                logger.info(f"{protocol.value} - Serialize: {avg_serialize_time:.1f}μs, Deserialize: {avg_deserialize_time:.1f}μs")
                
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_zero_copy_io_performance(self):
        """Test zero-copy I/O operations"""
        io_manager = zero_copy_io_manager
        
        buffer_sizes = [1024, 4096, 16384]
        
        for size in buffer_sizes:
            # Test buffer allocation/return performance
            allocation_times = []
            operation_times = []
            
            for _ in range(100):
                # Allocate buffer
                start_time = time.perf_counter_ns()
                buffer = io_manager.get_zero_copy_buffer(size)
                allocation_time = (time.perf_counter_ns() - start_time) / 1000
                allocation_times.append(allocation_time)
                
                # Perform operations on buffer
                start_time = time.perf_counter_ns()
                buffer[:min(1000, size)] = b'x' * min(1000, size)
                data = bytes(buffer[:min(1000, size)])
                operation_time = (time.perf_counter_ns() - start_time) / 1000
                operation_times.append(operation_time)
                
                # Return buffer
                io_manager.return_zero_copy_buffer(buffer)
                
            avg_allocation_time = sum(allocation_times) / len(allocation_times)
            avg_operation_time = sum(operation_times) / len(operation_times)
            
            assert avg_allocation_time < 100.0, f"Buffer allocation too slow: {avg_allocation_time}μs"
            assert avg_operation_time < 50.0, f"Buffer operations too slow: {avg_operation_time}μs"
            
        logger.info("Zero-copy I/O performance validated")

class TestPerformanceMonitoring:
    """Test performance monitoring and regression detection"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_performance_monitoring_overhead(self):
        """Test that performance monitoring has minimal overhead"""
        # Baseline - function without monitoring
        def test_function():
            time.sleep(0.001)
            return sum(range(1000))
            
        # Measure baseline performance
        baseline_times = []
        for _ in range(100):
            start_time = time.perf_counter_ns()
            test_function()
            execution_time = (time.perf_counter_ns() - start_time) / 1000
            baseline_times.append(execution_time)
            
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Measure with performance monitoring
        from ultra_performance.performance_monitoring import monitor_latency
        
        monitored_times = []
        for _ in range(100):
            start_time = time.perf_counter_ns()
            with monitor_latency("test_function"):
                test_function()
            execution_time = (time.perf_counter_ns() - start_time) / 1000
            monitored_times.append(execution_time)
            
        monitored_avg = sum(monitored_times) / len(monitored_times)
        
        # Monitoring overhead should be minimal (<10%)
        overhead_ratio = monitored_avg / baseline_avg
        assert overhead_ratio < 1.1, f"Monitoring overhead too high: {(overhead_ratio - 1) * 100:.1f}%"
        
        logger.info(f"Performance monitoring overhead: {(overhead_ratio - 1) * 100:.1f}%")
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_benchmark_function_accuracy(self):
        """Test benchmark function accuracy and consistency"""
        def predictable_function(n: int = 1000):
            """Function with predictable performance characteristics"""
            return sum(i * i for i in range(n))
            
        # Benchmark the function
        results = await benchmark_function(predictable_function, iterations=100, n=1000)
        
        # Validate benchmark results
        assert results["iterations"] == 100
        assert results["min_latency_us"] > 0
        assert results["max_latency_us"] >= results["min_latency_us"]
        assert results["mean_latency_us"] >= results["min_latency_us"]
        
        # Performance should be consistent (coefficient of variation < 50%)
        if "std_latency_us" in results:
            cv = results["std_latency_us"] / results["mean_latency_us"]
            assert cv < 0.5, f"Performance too inconsistent: CV = {cv:.2f}"
            
        logger.info(f"Benchmark accuracy validated - Mean: {results['mean_latency_us']:.1f}μs")

class TestIntegratedPerformance:
    """Test integrated ultra-performance optimizations"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    async def test_end_to_end_trading_pipeline_performance(self):
        """Test complete trading pipeline with all optimizations enabled"""
        
        async def mock_trading_pipeline():
            """Mock trading pipeline with key operations"""
            # 1. Receive market data (network I/O)
            market_data = {"symbol": "AAPL", "price": 150.25, "quantity": 1000, "timestamp": time.time()}
            
            # 2. Serialize/deserialize (network optimization)
            serialized = optimized_serialization.serialize_optimized(market_data)
            deserialized = optimized_serialization.deserialize_optimized(serialized)
            
            # 3. Cache lookup (advanced caching)
            cache_key = f"price_{market_data['symbol']}"
            await distributed_cache_manager.set(cache_key, market_data["price"])
            cached_price = await distributed_cache_manager.get(cache_key)
            
            # 4. Risk calculation (GPU acceleration if available)
            sample_returns = np.random.normal(0.001, 0.02, 100)
            risk_result = await gpu_risk_calculator.calculate_var_gpu(sample_returns)
            
            # 5. Order book update (ultra-low latency)
            orderbook = UltraFastOrderBook("AAPL", None, None)
            bid_latency = orderbook.update_bid(market_data["price"] - 0.05, 1000)
            ask_latency = orderbook.update_ask(market_data["price"] + 0.05, 1000)
            
            return {
                "market_data_processed": deserialized is not None,
                "cache_hit": cached_price is not None,
                "risk_calculated": risk_result is not None,
                "orderbook_updated": bid_latency > 0 and ask_latency > 0,
                "orderbook_latency_us": max(bid_latency, ask_latency)
            }
            
        # Benchmark the complete pipeline
        pipeline_results = await benchmark_function(mock_trading_pipeline, iterations=50)
        
        # Validate end-to-end performance
        assert pipeline_results["mean_latency_us"] < 10000.0, f"Pipeline too slow: {pipeline_results['mean_latency_us']:.1f}μs"
        
        # Run actual pipeline to check functionality
        result = await mock_trading_pipeline()
        assert result["market_data_processed"]
        assert result["cache_hit"]  
        assert result["risk_calculated"]
        assert result["orderbook_updated"]
        assert result["orderbook_latency_us"] < 100.0  # Ultra-low latency requirement
        
        logger.info(f"End-to-end trading pipeline performance: {pipeline_results['mean_latency_us']:.1f}μs average")
        
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_system_resource_optimization(self):
        """Test system resource optimization effectiveness"""
        from ultra_performance.performance_monitoring import ultra_performance_metrics
        
        # Get initial system snapshot
        initial_metrics = ultra_performance_metrics.get_comprehensive_report()
        
        # Simulate high-load operations
        operations = []
        for _ in range(1000):
            # Memory allocation
            data = [0] * 1000
            operations.append(data)
            
            # CPU computation  
            result = sum(i * i for i in range(100))
            
            # Cache operations
            asyncio.run(distributed_cache_manager.set(f"test_{_}", result))
            
        # Get final system snapshot
        final_metrics = ultra_performance_metrics.get_comprehensive_report()
        
        # Validate resource usage is reasonable
        if "system_snapshot" in final_metrics:
            cpu_usage = final_metrics["system_snapshot"]["cpu_percent"]
            memory_usage_mb = final_metrics["system_snapshot"]["memory_usage_mb"]
            
            # Resource usage should be reasonable under load
            assert cpu_usage < 90.0, f"CPU usage too high: {cpu_usage}%"
            assert memory_usage_mb < 2048.0, f"Memory usage too high: {memory_usage_mb}MB"
            
        # Clean up
        operations.clear()
        
        logger.info("System resource optimization validated under high load")

# Performance regression tests
class TestPerformanceRegression:
    """Test performance regression detection"""
    
    @pytest.mark.skipif(not ULTRA_PERFORMANCE_AVAILABLE, reason="Ultra-performance modules not available")
    def test_regression_detection_sensitivity(self):
        """Test performance regression detection sensitivity"""
        from ultra_performance.performance_monitoring import performance_regression_detector
        
        metric_name = "test_operation_latency"
        
        # Establish baseline (stable performance)
        baseline_latencies = [100.0 + np.random.normal(0, 5) for _ in range(100)]
        for latency in baseline_latencies:
            performance_regression_detector.add_baseline_metric(metric_name, latency)
            
        # Add normal current metrics (no regression)
        normal_latencies = [102.0 + np.random.normal(0, 5) for _ in range(20)]
        for latency in normal_latencies:
            performance_regression_detector.add_current_metric(metric_name, latency)
            
        report = performance_regression_detector.get_regression_report()
        
        # Should not detect regression for small change
        recent_alerts = [alert for alert in report["recent_alert_details"] if alert["metric_name"] == metric_name]
        assert len(recent_alerts) == 0, "False positive regression detected"
        
        # Introduce performance regression (25% increase)
        regression_latencies = [125.0 + np.random.normal(0, 5) for _ in range(20)]
        for latency in regression_latencies:
            performance_regression_detector.add_current_metric(metric_name, latency)
            
        report = performance_regression_detector.get_regression_report()
        
        # Should detect regression
        recent_alerts = [alert for alert in report["recent_alert_details"] if alert["metric_name"] == metric_name]
        assert len(recent_alerts) > 0, "Failed to detect performance regression"
        
        logger.info(f"Performance regression detection validated - {len(recent_alerts)} alerts generated")

# Utility functions for testing
def generate_realistic_trading_data(n_points: int = 1000) -> Dict[str, np.ndarray]:
    """Generate realistic trading data for testing"""
    np.random.seed(42)
    
    # Generate price series with realistic characteristics
    dt = 1.0 / 252  # Daily data
    drift = 0.05    # 5% annual return
    volatility = 0.2  # 20% annual volatility
    
    returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_points)
    prices = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian Motion
    
    volumes = np.random.lognormal(mean=10, sigma=1, size=n_points)
    timestamps = np.arange(n_points) * dt
    
    return {
        "prices": prices,
        "returns": returns,
        "volumes": volumes,
        "timestamps": timestamps
    }

if __name__ == "__main__":
    # Run all benchmarks
    pytest.main([__file__, "-v", "--tb=short"])
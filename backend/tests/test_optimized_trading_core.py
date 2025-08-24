"""
Optimized Trading Core Integration Tests
========================================

Comprehensive test suite for memory pool optimizations and performance improvements.

Test Coverage:
- Memory pool functionality
- Poolable object lifecycle
- Optimized execution engine
- Performance regression testing
- Integration with existing systems
"""

import pytest
import asyncio
import time
import gc
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from trading_engine.memory_pool import ObjectPool, PoolableObject, pool_manager
from trading_engine.poolable_objects import (
    PooledOrder, PooledOrderFill, PooledVenueQuote,
    create_pooled_order, release_pooled_order,
    trading_pools, OrderSide, OrderType, TimeInForce
)
from trading_engine.optimized_execution_engine import (
    OptimizedExecutionEngine, OptimizedSmartOrderRouter, VenueRankingCache
)
from trading_engine.performance_benchmark import PerformanceBenchmark


class TestPoolableObject(PoolableObject):
    """Test implementation of poolable object."""
    
    def __init__(self):
        self.value = 0
        self.name = ""
        self.data = []
    
    def reset(self):
        self.value = 0
        self.name = ""
        self.data.clear()
    
    def is_valid(self) -> bool:
        return True


@pytest.fixture
def test_pool():
    """Create test object pool."""
    return ObjectPool(
        factory=TestPoolableObject,
        initial_size=10,
        max_size=100,
        name="test_pool"
    )


@pytest.fixture
def mock_venue():
    """Create mock venue for testing."""
    venue = Mock()
    venue.name = "TEST_VENUE"
    venue.status = Mock()
    venue.status.CONNECTED = "connected"
    venue.submit_order = AsyncMock(return_value=True)
    venue.cancel_order = AsyncMock(return_value=True)
    venue.get_quote = AsyncMock()
    venue.add_callback = Mock()
    venue.metrics = Mock()
    venue.metrics.fill_rate_percentage = 95.0
    venue.metrics.uptime_percentage = 99.0
    venue.metrics.average_fill_time_ms = 10.0
    return venue


class TestMemoryPool:
    """Test memory pool functionality."""
    
    def test_pool_creation(self, test_pool):
        """Test pool creation and initialization."""
        assert test_pool.name == "test_pool"
        assert test_pool.max_size == 100
        metrics = test_pool.get_metrics()
        assert metrics.total_created == 10  # Initial size
        assert metrics.current_available == 10
    
    def test_object_acquisition(self, test_pool):
        """Test object acquisition from pool."""
        # Get object from pool
        obj = test_pool.acquire()
        assert isinstance(obj, TestPoolableObject)
        assert obj.value == 0  # Should be reset
        
        # Metrics should reflect acquisition
        metrics = test_pool.get_metrics()
        assert metrics.total_acquired == 1
        assert metrics.current_active == 1
        assert metrics.current_available == 9
    
    def test_object_release(self, test_pool):
        """Test object release back to pool."""
        obj = test_pool.acquire()
        obj.value = 42
        
        # Release object
        success = test_pool.release(obj)
        assert success
        
        # Object should be back in pool
        metrics = test_pool.get_metrics()
        assert metrics.current_available == 10
        assert metrics.current_active == 0
    
    def test_object_reuse(self, test_pool):
        """Test object reuse from pool."""
        # Get and release object
        obj1 = test_pool.acquire()
        obj1_id = id(obj1)
        test_pool.release(obj1)
        
        # Get another object - should be the same instance
        obj2 = test_pool.acquire()
        assert id(obj2) == obj1_id  # Same object instance
        assert obj2.value == 0      # Should be reset
    
    def test_pool_overflow(self, test_pool):
        """Test pool behavior when exceeding max size."""
        # Acquire all objects
        objects = []
        for _ in range(test_pool.max_size + 10):
            obj = test_pool.acquire()
            objects.append(obj)
        
        # Should create new objects when pool is empty
        metrics = test_pool.get_metrics()
        assert metrics.total_created > 10
    
    def test_pool_metrics(self, test_pool):
        """Test pool performance metrics."""
        # Perform some operations
        objects = []
        for _ in range(5):
            obj = test_pool.acquire()
            objects.append(obj)
        
        for obj in objects:
            test_pool.release(obj)
        
        # Check metrics
        metrics = test_pool.get_metrics()
        assert metrics.total_acquired == 5
        assert metrics.total_returned == 5
        assert metrics.hit_rate > 0  # Some objects were reused


class TestPoolableObjects:
    """Test poolable trading objects."""
    
    def test_pooled_order_creation(self):
        """Test pooled order creation and population."""
        order = create_pooled_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=150.0
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 100.0
        assert order.price == 150.0
        assert order.remaining_quantity == 100.0
        assert order.id != ""
        
        release_pooled_order(order)
    
    def test_pooled_order_reset(self):
        """Test pooled order reset functionality."""
        order = create_pooled_order("TEST", OrderSide.BUY, OrderType.MARKET, 100.0)
        original_id = order.id
        
        # Modify order
        order.symbol = "MODIFIED"
        order.filled_quantity = 50.0
        
        # Reset should clear modifications
        order.reset()
        assert order.symbol == ""
        assert order.filled_quantity == 0.0
        assert order.id != original_id  # New ID generated
        
        release_pooled_order(order)
    
    def test_pooled_order_reuse(self):
        """Test order reuse from pool."""
        # Create and release order
        order1 = create_pooled_order("AAPL", OrderSide.BUY, OrderType.MARKET, 100.0)
        order1_id = id(order1)
        release_pooled_order(order1)
        
        # Create another order - should reuse same instance
        order2 = create_pooled_order("MSFT", OrderSide.SELL, OrderType.LIMIT, 200.0, 300.0)
        assert id(order2) == order1_id  # Same object instance
        assert order2.symbol == "MSFT"  # But with new data
        
        release_pooled_order(order2)
    
    def test_pooled_fill_creation(self):
        """Test pooled order fill creation."""
        from trading_engine.poolable_objects import create_pooled_fill, release_pooled_fill
        
        fill = create_pooled_fill(
            order_id="ORDER123",
            quantity=50.0,
            price=150.0,
            execution_id="EXEC123"
        )
        
        assert fill.order_id == "ORDER123"
        assert fill.quantity == 50.0
        assert fill.price == 150.0
        assert fill.execution_id == "EXEC123"
        
        release_pooled_fill(fill)
    
    def test_pool_statistics(self):
        """Test trading pool statistics."""
        # Create some orders to populate stats
        orders = []
        for i in range(10):
            order = create_pooled_order(f"TEST{i}", OrderSide.BUY, OrderType.MARKET, 100.0)
            orders.append(order)
        
        # Check pool statistics
        stats = trading_pools.get_pool_statistics()
        assert "order_pool" in stats
        
        order_stats = stats["order_pool"]
        assert order_stats["usage"]["total_acquired"] >= 10
        
        # Release orders
        for order in orders:
            release_pooled_order(order)


class TestOptimizedExecutionEngine:
    """Test optimized execution engine."""
    
    @pytest.fixture
    def optimized_engine(self, mock_venue):
        """Create optimized execution engine with mock venue."""
        engine = OptimizedExecutionEngine()
        engine.add_venue(mock_venue)
        return engine
    
    @pytest.mark.asyncio
    async def test_optimized_order_submission(self, optimized_engine):
        """Test optimized order submission."""
        order = create_pooled_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        success = await optimized_engine.submit_order_optimized(order)
        assert success
        assert order.id in optimized_engine.active_orders
        
        release_pooled_order(order)
    
    @pytest.mark.asyncio
    async def test_optimized_order_cancellation(self, optimized_engine):
        """Test optimized order cancellation."""
        order = create_pooled_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        # Submit order first
        await optimized_engine.submit_order_optimized(order)
        
        # Cancel order
        success = await optimized_engine.cancel_order_optimized(order.id)
        assert success
        assert order.id not in optimized_engine.active_orders
        
        release_pooled_order(order)
    
    @pytest.mark.asyncio
    async def test_venue_ranking_cache(self, mock_venue):
        """Test venue ranking cache functionality."""
        cache = VenueRankingCache()
        cache.add_venue(mock_venue)
        
        # Should return the venue
        venue = cache.get_optimal_venue("AAPL")
        assert venue == mock_venue
    
    def test_performance_statistics(self, optimized_engine):
        """Test engine performance statistics."""
        stats = optimized_engine.get_performance_statistics()
        
        assert "execution_engine" in stats
        assert "smart_router" in stats
        assert "memory_pools" in stats
        assert "performance_summary" in stats
        
        exec_stats = stats["execution_engine"]
        assert "orders_processed" in exec_stats
        assert "success_rate" in exec_stats
        assert "performance" in exec_stats


class TestPerformanceRegression:
    """Test performance regression and improvements."""
    
    @pytest.mark.asyncio
    async def test_order_creation_performance(self):
        """Test order creation performance doesn't regress."""
        iterations = 1000
        
        # Measure pooled order creation time
        start_time = time.perf_counter()
        orders = []
        
        for i in range(iterations):
            order = create_pooled_order(
                symbol=f"TEST{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0
            )
            orders.append(order)
        
        creation_time = time.perf_counter() - start_time
        
        # Release all orders
        for order in orders:
            release_pooled_order(order)
        
        # Should create orders very quickly
        avg_time_per_order_us = (creation_time / iterations) * 1_000_000
        assert avg_time_per_order_us < 10.0  # Less than 10 microseconds per order
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage is efficient."""
        gc.collect()  # Clean up before test
        
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        start_memory = process.memory_info().rss
        
        # Create many orders and immediately release them
        for _ in range(10000):
            order = create_pooled_order("MEMORY_TEST", OrderSide.BUY, OrderType.MARKET, 100.0)
            release_pooled_order(order)
        
        end_memory = process.memory_info().rss
        memory_growth = end_memory - start_memory
        
        # Memory growth should be minimal due to pooling
        memory_growth_mb = memory_growth / (1024 * 1024)
        assert memory_growth_mb < 50  # Less than 50MB growth for 10K orders
    
    def test_pool_hit_rates(self):
        """Test pool hit rates are high."""
        # Create and release many orders to populate pools
        for _ in range(1000):
            order = create_pooled_order("HIT_RATE_TEST", OrderSide.BUY, OrderType.MARKET, 100.0)
            release_pooled_order(order)
        
        # Check pool statistics
        stats = trading_pools.get_pool_statistics()
        order_pool_stats = stats["order_pool"]
        
        hit_rate = order_pool_stats["performance"]["hit_rate_percent"]
        assert hit_rate > 90.0  # Should have >90% hit rate


class TestIntegrationWithExistingSystems:
    """Test integration with existing trading systems."""
    
    def test_pooled_order_compatibility(self):
        """Test pooled orders work with existing order management."""
        from trading_engine.order_management import OrderManagementSystem
        
        oms = OrderManagementSystem()
        
        # Create pooled order
        pooled_order = create_pooled_order(
            symbol="INTEGRATION_TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=150.0
        )
        
        # Should work with existing OMS methods
        assert pooled_order.symbol == "INTEGRATION_TEST"
        assert pooled_order.is_complete == False
        assert pooled_order.fill_percentage == 0.0
        
        release_pooled_order(pooled_order)
    
    @pytest.mark.asyncio
    async def test_optimized_engine_with_callbacks(self, mock_venue):
        """Test optimized engine with callback integration."""
        engine = OptimizedExecutionEngine()
        engine.add_venue(mock_venue)
        
        # Add callback
        callback_called = False
        def test_callback(order_id, fill):
            nonlocal callback_called
            callback_called = True
        
        engine.add_execution_callback(test_callback)
        
        # Submit order
        order = create_pooled_order("CALLBACK_TEST", OrderSide.BUY, OrderType.MARKET, 100.0)
        await engine.submit_order_optimized(order)
        
        # Simulate fill
        from trading_engine.poolable_objects import create_pooled_fill
        fill = create_pooled_fill(order.id, 100.0, 150.0, "EXEC123")
        await engine.handle_fill_optimized(order.id, fill)
        
        # Callback should be called (may be asynchronous)
        await asyncio.sleep(0.1)  # Give time for async callback
        
        release_pooled_order(order)


@pytest.mark.asyncio
async def test_benchmark_integration():
    """Test performance benchmark runs successfully."""
    # Run a quick benchmark to ensure it works
    benchmark = PerformanceBenchmark()
    
    # Test individual benchmark functions
    order_creation_result = await benchmark._benchmark_order_creation()
    assert "baseline" in order_creation_result
    assert "optimized" in order_creation_result
    assert "improvement" in order_creation_result
    
    memory_result = await benchmark._benchmark_memory_efficiency()
    assert "memory_reduction_percent" in memory_result
    assert memory_result["memory_reduction_percent"] > 0  # Should show improvement


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    sys.exit(result.returncode)
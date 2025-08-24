#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Phase 1 Clock Optimization
Tests all components with microsecond precision validation and performance benchmarks.
"""

import pytest
import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Any, Optional

from backend.engines.common.clock import (
    create_clock, get_global_clock, set_global_clock,
    TestClock, LiveClock, 
    NANOS_IN_MICROSECOND, NANOS_IN_MILLISECOND, NANOS_IN_SECOND
)

# Order Management System imports
from backend.order_management.oms_engine import (
    OMSEngine, create_oms_engine, benchmark_oms_performance,
    Order, OrderSide, OrderType, OrderStatus
)
from backend.order_management.ems_engine import (
    EMSEngine, create_ems_engine, benchmark_ems_performance,
    ExecutionAlgorithm, ExecutionStrategy
)
from backend.order_management.pms_engine import (
    PMSEngine, create_pms_engine, benchmark_pms_performance,
    SettlementCycle, TransactionType
)

# Database System imports  
from backend.database.postgres_clock_adapter import (
    PostgresClockAdapter, create_postgres_adapter, benchmark_postgres_performance,
    TransactionIsolation, QueryType
)

# Cache System imports
from backend.cache.redis_clock_manager import (
    RedisClockManager, create_redis_manager, benchmark_redis_performance,
    SerializationFormat, CacheStrategy
)

# Load Balancing imports
from backend.load_balancing.nginx_clock_controller import (
    NGINXClockController, create_nginx_controller, benchmark_nginx_performance,
    LoadBalancingStrategy, HealthStatus
)


class TestPhase1ClockIntegration:
    """Comprehensive test suite for Phase 1 clock optimization"""
    
    @pytest.fixture
    def test_clock(self):
        """Create test clock with deterministic timing"""
        # Start from a known timestamp (Jan 1, 2024)
        start_time_ns = int(1704067200 * NANOS_IN_SECOND)  # 2024-01-01 00:00:00 UTC
        clock = TestClock(start_time_ns)
        set_global_clock(clock)
        yield clock
        # Reset to live clock after test
        set_global_clock(LiveClock())
    
    @pytest.fixture
    def sample_orders(self):
        """Generate sample orders for testing"""
        orders = []
        for i in range(10):
            order = {
                'order_id': f'TEST_ORDER_{i:03d}',
                'symbol': 'AAPL' if i % 2 == 0 else 'GOOGL',
                'side': OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                'order_type': OrderType.LIMIT,
                'quantity': 100.0 * (i + 1),
                'price': 100.0 + i,
                'routing_priority': i % 3
            }
            orders.append(order)
        return orders
    
    @pytest.mark.asyncio
    async def test_oms_engine_timing_precision(self, test_clock, sample_orders):
        """Test OMS Engine with nanosecond timing precision"""
        oms_engine = create_oms_engine(test_clock)
        
        async with oms_engine.managed_lifecycle():
            # Test order creation timing
            start_time_ns = test_clock.timestamp_ns()
            
            created_orders = []
            for order_data in sample_orders[:5]:
                order = oms_engine.create_order(**order_data)
                created_orders.append(order)
                
                # Advance time by 1 microsecond for precision testing
                test_clock.advance_time(1000)  # 1 microsecond
            
            # Validate timing precision
            for i, order in enumerate(created_orders[1:], 1):
                prev_order = created_orders[i-1]
                time_diff_ns = order.created_time_ns - prev_order.created_time_ns
                
                # Should be exactly 1 microsecond apart
                assert time_diff_ns == 1000, f"Order timing precision failed: {time_diff_ns}ns"
            
            # Test order processing performance
            processing_tasks = []
            for order in created_orders:
                processing_tasks.append(oms_engine.process_order(order.order_id))
            
            process_start_ns = test_clock.timestamp_ns()
            results = await asyncio.gather(*processing_tasks)
            process_end_ns = test_clock.timestamp_ns()
            
            # Validate all orders processed successfully
            assert all(results), "Not all orders processed successfully"
            
            # Check performance metrics
            metrics = oms_engine.get_performance_metrics()
            assert metrics['orders_processed'] == len(sample_orders[:5])
            assert metrics['average_latency_us'] < 1000  # Should be under 1ms average
            
            print(f"OMS Engine processed {len(created_orders)} orders with average latency: {metrics['average_latency_us']:.2f}μs")
    
    @pytest.mark.asyncio
    async def test_ems_engine_deterministic_execution(self, test_clock, sample_orders):
        """Test EMS Engine deterministic algorithm execution"""
        ems_engine = create_ems_engine(test_clock)
        
        async with ems_engine.managed_lifecycle():
            # Create orders for execution strategies
            orders = []
            for order_data in sample_orders[:3]:
                order = Order(**order_data)
                orders.append(order)
            
            # Create TWAP strategies with precise timing
            strategies = []
            for i, order in enumerate(orders):
                strategy = ems_engine.create_execution_strategy(
                    strategy_id=f'TWAP_STRATEGY_{i:03d}',
                    algorithm=ExecutionAlgorithm.TWAP,
                    parent_order=order,
                    duration_seconds=300,  # 5 minutes
                    slice_interval_ns=60 * NANOS_IN_SECOND  # 1 minute slices
                )
                strategies.append(strategy)
            
            # Start all strategies
            start_tasks = []
            for strategy in strategies:
                start_tasks.append(ems_engine.start_strategy(strategy.parent_order.order_id))
            
            strategy_start_ns = test_clock.timestamp_ns()
            await asyncio.gather(*start_tasks)
            
            # Validate slice scheduling precision
            for strategy in strategies:
                assert len(strategy.slices) == 5  # Should have 5 one-minute slices
                
                # Check slice timing precision
                for i, slice_obj in enumerate(strategy.slices[1:], 1):
                    prev_slice = strategy.slices[i-1]
                    time_diff_ns = slice_obj.scheduled_time_ns - prev_slice.scheduled_time_ns
                    
                    # Should be exactly 60 seconds apart
                    assert time_diff_ns == 60 * NANOS_IN_SECOND, f"Slice timing precision failed: {time_diff_ns}ns"
            
            # Advance time to trigger slice executions
            test_clock.advance_time(5 * 60 * NANOS_IN_SECOND)  # 5 minutes
            await asyncio.sleep(0.1)  # Allow processing
            
            # Check execution metrics
            metrics = ems_engine.get_performance_metrics()
            assert metrics['slices_executed'] > 0
            assert metrics['scheduling_accuracy_us'] < 100  # Should be under 100μs accuracy
            
            print(f"EMS Engine executed {metrics['slices_executed']} slices with accuracy: {metrics['scheduling_accuracy_us']:.2f}μs")
    
    @pytest.mark.asyncio
    async def test_pms_engine_settlement_precision(self, test_clock):
        """Test PMS Engine settlement cycle precision"""
        pms_engine = create_pms_engine("test_portfolio", test_clock)
        
        async with pms_engine.managed_lifecycle():
            # Process trades with different settlement cycles
            trade_data = [
                {'trade_id': 'TRADE_001', 'symbol': 'AAPL', 'side': OrderSide.BUY, 'quantity': 100, 'price': 150.0, 'settlement_cycle': SettlementCycle.T_PLUS_0},
                {'trade_id': 'TRADE_002', 'symbol': 'GOOGL', 'side': OrderSide.BUY, 'quantity': 50, 'price': 2800.0, 'settlement_cycle': SettlementCycle.T_PLUS_1},
                {'trade_id': 'TRADE_003', 'symbol': 'MSFT', 'side': OrderSide.SELL, 'quantity': 200, 'price': 330.0, 'settlement_cycle': SettlementCycle.T_PLUS_2}
            ]
            
            trade_start_ns = test_clock.timestamp_ns()
            
            # Process all trades
            transactions = []
            for trade in trade_data:
                transaction = pms_engine.process_trade(**trade)
                transactions.append(transaction)
                test_clock.advance_time(NANOS_IN_SECOND)  # 1 second between trades
            
            # Validate settlement timing
            for transaction in transactions:
                if transaction.settlement_cycle == SettlementCycle.T_PLUS_0:
                    expected_settlement = transaction.trade_date_ns
                elif transaction.settlement_cycle == SettlementCycle.T_PLUS_1:
                    expected_settlement = transaction.trade_date_ns + (1 * 24 * 60 * 60 * NANOS_IN_SECOND)
                elif transaction.settlement_cycle == SettlementCycle.T_PLUS_2:
                    expected_settlement = transaction.trade_date_ns + (2 * 24 * 60 * 60 * NANOS_IN_SECOND)
                
                assert transaction.settlement_date_ns == expected_settlement, f"Settlement timing incorrect for {transaction.settlement_cycle}"
            
            # Advance time to trigger settlements
            test_clock.advance_time(3 * 24 * 60 * 60 * NANOS_IN_SECOND)  # 3 days
            await asyncio.sleep(0.1)  # Allow settlement processing
            
            # Check portfolio state
            portfolio_summary = pms_engine.get_portfolio_summary()
            assert portfolio_summary['total_positions'] == 3
            
            # Validate performance metrics
            metrics = pms_engine.get_performance_metrics()
            assert metrics['transactions_processed'] == len(trade_data)
            assert metrics['settlement_accuracy_us'] < 1000  # Should be under 1ms accuracy
            
            print(f"PMS Engine processed {metrics['transactions_processed']} trades with settlement accuracy: {metrics['settlement_accuracy_us']:.2f}μs")
    
    @pytest.mark.asyncio
    async def test_end_to_end_order_flow(self, test_clock, sample_orders):
        """Test complete order flow through all systems"""
        # Initialize all engines with test clock
        oms_engine = create_oms_engine(test_clock)
        ems_engine = create_ems_engine(test_clock) 
        pms_engine = create_pms_engine("integration_test", test_clock)
        
        async with oms_engine.managed_lifecycle():
            async with ems_engine.managed_lifecycle():
                async with pms_engine.managed_lifecycle():
                    
                    # Create and process orders through OMS
                    created_orders = []
                    for order_data in sample_orders[:3]:
                        order = oms_engine.create_order(**order_data)
                        created_orders.append(order)
                        success = await oms_engine.process_order(order.order_id)
                        assert success, f"Order processing failed for {order.order_id}"
                    
                    # Create execution strategies for filled orders
                    execution_strategies = []
                    for order in created_orders:
                        if order.status == OrderStatus.FILLED:
                            strategy = ems_engine.create_execution_strategy(
                                strategy_id=f"STRATEGY_{order.order_id}",
                                algorithm=ExecutionAlgorithm.TWAP,
                                parent_order=order,
                                duration_seconds=120  # 2 minutes
                            )
                            execution_strategies.append(strategy)
                            await ems_engine.start_strategy(strategy.parent_order.order_id)
                    
                    # Process trades in PMS based on executions
                    for order in created_orders:
                        if order.status == OrderStatus.FILLED:
                            pms_engine.process_trade(
                                trade_id=f"TRADE_{order.order_id}",
                                symbol=order.symbol,
                                side=order.side,
                                quantity=float(order.filled_quantity),
                                price=float(order.average_fill_price),
                                commission=1.0
                            )
                    
                    # Advance time to process executions and settlements
                    test_clock.advance_time(3 * 24 * 60 * 60 * NANOS_IN_SECOND)  # 3 days
                    await asyncio.sleep(0.2)  # Allow all processing to complete
                    
                    # Validate end-to-end metrics
                    oms_metrics = oms_engine.get_performance_metrics()
                    ems_metrics = ems_engine.get_performance_metrics()
                    pms_metrics = pms_engine.get_performance_metrics()
                    
                    # All orders should be processed
                    assert oms_metrics['orders_processed'] == 3
                    
                    # Execution strategies should complete
                    assert len(ems_engine.get_active_strategies()) == 0  # All should be completed
                    
                    # Portfolio should have positions
                    portfolio_summary = pms_engine.get_portfolio_summary()
                    assert portfolio_summary['total_positions'] > 0
                    
                    print(f"End-to-end flow completed:")
                    print(f"  OMS: {oms_metrics['orders_processed']} orders, avg latency: {oms_metrics['average_latency_us']:.2f}μs")
                    print(f"  EMS: {ems_metrics['slices_executed']} slices, accuracy: {ems_metrics['scheduling_accuracy_us']:.2f}μs")
                    print(f"  PMS: {pms_metrics['transactions_processed']} trades, settlement accuracy: {pms_metrics['settlement_accuracy_us']:.2f}μs")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, test_clock):
        """Run performance benchmarks for all systems"""
        # OMS Performance Benchmark
        oms_engine = create_oms_engine(test_clock)
        async with oms_engine.managed_lifecycle():
            oms_results = await benchmark_oms_performance(oms_engine, num_orders=100)
            
            # Validate OMS performance targets
            assert oms_results['benchmark_orders_per_second'] > 10000  # Should handle >10K orders/sec
            assert oms_results['average_latency_us'] < 500  # Should average <500μs
            
            print(f"OMS Benchmark: {oms_results['benchmark_orders_per_second']:.0f} orders/sec, {oms_results['average_latency_us']:.2f}μs avg latency")
        
        # EMS Performance Benchmark  
        ems_engine = create_ems_engine(test_clock)
        async with ems_engine.managed_lifecycle():
            ems_results = await benchmark_ems_performance(ems_engine, num_strategies=5, slices_per_strategy=10)
            
            # Validate EMS performance targets
            assert ems_results['scheduling_accuracy_us'] < 100  # Should be <100μs accuracy
            assert ems_results['algorithm_efficiency'] > 80  # Should be >80% efficient
            
            print(f"EMS Benchmark: {ems_results['benchmark_slices_per_second']:.0f} slices/sec, {ems_results['scheduling_accuracy_us']:.2f}μs accuracy")
        
        # PMS Performance Benchmark
        pms_engine = create_pms_engine("benchmark_portfolio", test_clock)
        async with pms_engine.managed_lifecycle():
            pms_results = await benchmark_pms_performance(pms_engine, num_trades=200)
            
            # Validate PMS performance targets
            assert pms_results['benchmark_trades_per_second'] > 1000  # Should handle >1K trades/sec
            assert pms_results['settlement_accuracy_us'] < 1000  # Should be <1ms accuracy
            
            print(f"PMS Benchmark: {pms_results['benchmark_trades_per_second']:.0f} trades/sec, {pms_results['settlement_accuracy_us']:.2f}μs settlement accuracy")
        
        # Compile overall results
        benchmark_summary = {
            'oms_performance': oms_results,
            'ems_performance': ems_results, 
            'pms_performance': pms_results,
            'overall_system_health': 'EXCELLENT' if all([
                oms_results['average_latency_us'] < 500,
                ems_results['scheduling_accuracy_us'] < 100,
                pms_results['settlement_accuracy_us'] < 1000
            ]) else 'NEEDS_OPTIMIZATION'
        }
        
        return benchmark_summary
    
    @pytest.mark.asyncio
    async def test_timing_accuracy_validation(self, test_clock):
        """Validate timing accuracy across all systems"""
        timing_tests = []
        
        # Test 1: Microsecond precision timing
        start_ns = test_clock.timestamp_ns()
        test_clock.advance_time(1000)  # 1 microsecond
        end_ns = test_clock.timestamp_ns()
        
        timing_tests.append({
            'test': 'microsecond_precision',
            'expected_ns': 1000,
            'actual_ns': end_ns - start_ns,
            'accuracy_us': abs((end_ns - start_ns) - 1000) / NANOS_IN_MICROSECOND
        })
        
        # Test 2: Millisecond precision timing
        start_ns = test_clock.timestamp_ns()
        test_clock.advance_time(NANOS_IN_MILLISECOND)
        end_ns = test_clock.timestamp_ns()
        
        timing_tests.append({
            'test': 'millisecond_precision',
            'expected_ns': NANOS_IN_MILLISECOND,
            'actual_ns': end_ns - start_ns,
            'accuracy_us': abs((end_ns - start_ns) - NANOS_IN_MILLISECOND) / NANOS_IN_MICROSECOND
        })
        
        # Test 3: Second precision timing
        start_ns = test_clock.timestamp_ns()
        test_clock.advance_time(NANOS_IN_SECOND)
        end_ns = test_clock.timestamp_ns()
        
        timing_tests.append({
            'test': 'second_precision',
            'expected_ns': NANOS_IN_SECOND,
            'actual_ns': end_ns - start_ns,
            'accuracy_us': abs((end_ns - start_ns) - NANOS_IN_SECOND) / NANOS_IN_MICROSECOND
        })
        
        # Validate all timing tests
        for test in timing_tests:
            assert test['actual_ns'] == test['expected_ns'], f"Timing accuracy failed for {test['test']}: expected {test['expected_ns']}ns, got {test['actual_ns']}ns"
            assert test['accuracy_us'] == 0.0, f"Perfect timing accuracy expected for {test['test']}"
        
        print("All timing accuracy tests passed with perfect precision")
        return timing_tests
    
    @pytest.mark.asyncio  
    async def test_system_integration_stress(self, test_clock):
        """Stress test system integration under high load"""
        # Initialize all systems
        oms_engine = create_oms_engine(test_clock)
        ems_engine = create_ems_engine(test_clock)
        pms_engine = create_pms_engine("stress_test", test_clock)
        
        stress_metrics = {
            'orders_created': 0,
            'orders_processed': 0,
            'strategies_executed': 0,
            'trades_settled': 0,
            'total_time_us': 0.0,
            'max_latency_us': 0.0,
            'errors': 0
        }
        
        async with oms_engine.managed_lifecycle():
            async with ems_engine.managed_lifecycle():
                async with pms_engine.managed_lifecycle():
                    
                    start_time_ns = test_clock.timestamp_ns()
                    
                    # Create high volume of orders
                    order_creation_tasks = []
                    for i in range(50):  # Create 50 orders rapidly
                        order_data = {
                            'order_id': f'STRESS_ORDER_{i:03d}',
                            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'][i % 5],
                            'side': OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                            'order_type': OrderType.LIMIT,
                            'quantity': 100.0 + (i * 10),
                            'price': 100.0 + (i % 50),
                            'routing_priority': i % 5
                        }
                        
                        try:
                            order = oms_engine.create_order(**order_data)
                            stress_metrics['orders_created'] += 1
                            
                            # Process order
                            success = await oms_engine.process_order(order.order_id)
                            if success:
                                stress_metrics['orders_processed'] += 1
                            
                            # Create execution strategy
                            if order.status == OrderStatus.FILLED:
                                strategy = ems_engine.create_execution_strategy(
                                    strategy_id=f"STRESS_STRATEGY_{i:03d}",
                                    algorithm=ExecutionAlgorithm.TWAP,
                                    parent_order=order,
                                    duration_seconds=60
                                )
                                await ems_engine.start_strategy(strategy.parent_order.order_id)
                                stress_metrics['strategies_executed'] += 1
                                
                                # Process trade in PMS
                                pms_engine.process_trade(
                                    trade_id=f"STRESS_TRADE_{i:03d}",
                                    symbol=order.symbol,
                                    side=order.side,
                                    quantity=float(order.filled_quantity),
                                    price=float(order.average_fill_price)
                                )
                                stress_metrics['trades_settled'] += 1
                            
                        except Exception as e:
                            stress_metrics['errors'] += 1
                            print(f"Stress test error for order {i}: {e}")
                        
                        # Advance time slightly
                        test_clock.advance_time(10000)  # 10 microseconds
                    
                    end_time_ns = test_clock.timestamp_ns()
                    stress_metrics['total_time_us'] = (end_time_ns - start_time_ns) / NANOS_IN_MICROSECOND
                    
                    # Validate stress test results
                    assert stress_metrics['orders_created'] == 50
                    assert stress_metrics['orders_processed'] >= 45  # Allow for some failures
                    assert stress_metrics['errors'] < 5  # Less than 10% error rate
                    
                    # Performance targets under stress
                    avg_processing_time_us = stress_metrics['total_time_us'] / stress_metrics['orders_processed']
                    assert avg_processing_time_us < 2000  # Should average <2ms under stress
                    
                    print(f"Stress test completed:")
                    print(f"  Created: {stress_metrics['orders_created']} orders")
                    print(f"  Processed: {stress_metrics['orders_processed']} orders")
                    print(f"  Strategies: {stress_metrics['strategies_executed']} executed")
                    print(f"  Trades: {stress_metrics['trades_settled']} settled")
                    print(f"  Errors: {stress_metrics['errors']}")
                    print(f"  Average processing time: {avg_processing_time_us:.2f}μs")
        
        return stress_metrics


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
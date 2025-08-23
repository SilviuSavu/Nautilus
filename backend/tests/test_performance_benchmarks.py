"""
Performance benchmark tests for Sprint 3 components.
Tests system performance under various load conditions and validates
performance requirements for production deployment.
"""

import asyncio
import time
import psutil
import pytest
import statistics
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import json
import numpy as np

from websocket.websocket_manager import WebSocketManager
from websocket.subscription_manager import SubscriptionManager
from analytics.performance_calculator import PerformanceCalculator
from analytics.risk_analytics import RiskAnalytics
from risk_management.limit_engine import LimitEngine
from risk_management.risk_monitor import RiskMonitor
from strategy_pipeline.deployment_manager import DeploymentManager


class TestPerformanceBenchmarks:
    """Performance benchmark tests for system components."""

    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitoring utilities."""
        class PerformanceMonitor:
            def __init__(self):
                self.start_time = None
                self.end_time = None
                self.memory_start = None
                self.memory_peak = None
                self.cpu_samples = []

            def start_monitoring(self):
                self.start_time = time.time()
                self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.cpu_samples = []

            def sample_cpu(self):
                self.cpu_samples.append(psutil.cpu_percent(interval=None))

            def stop_monitoring(self):
                self.end_time = time.time()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_peak = max(self.memory_start, current_memory)

            @property
            def duration(self):
                return self.end_time - self.start_time if self.end_time and self.start_time else 0

            @property
            def memory_usage(self):
                return self.memory_peak - self.memory_start if self.memory_peak and self.memory_start else 0

            @property
            def avg_cpu(self):
                return statistics.mean(self.cpu_samples) if self.cpu_samples else 0

        return PerformanceMonitor()

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_websocket_message_throughput(self, performance_monitor):
        """Test WebSocket message throughput under high load."""
        with patch('websocket.websocket_manager.WebSocket') as mock_ws:
            # Setup
            manager = WebSocketManager()
            mock_connections = []
            
            # Create 100 mock connections
            for i in range(100):
                mock_conn = Mock()
                mock_conn.send = AsyncMock()
                mock_connections.append(mock_conn)
                manager.connections[f"client_{i}"] = mock_conn

            # Prepare test messages
            messages = [
                {"type": "market_data", "symbol": f"SYMBOL_{i}", "price": 100 + i}
                for i in range(1000)
            ]

            # Start performance monitoring
            performance_monitor.start_monitoring()
            
            # Benchmark message broadcasting
            tasks = []
            for message in messages:
                for client_id in manager.connections:
                    task = manager.send_to_client(client_id, message)
                    tasks.append(task)
            
            # Sample CPU during processing
            performance_monitor.sample_cpu()
            await asyncio.gather(*tasks, return_exceptions=True)
            performance_monitor.sample_cpu()
            
            performance_monitor.stop_monitoring()

            # Performance assertions
            messages_per_second = len(messages) * len(mock_connections) / performance_monitor.duration
            
            # Benchmark targets
            assert messages_per_second > 50000, f"Throughput too low: {messages_per_second} msg/s"
            assert performance_monitor.duration < 5.0, f"Processing too slow: {performance_monitor.duration}s"
            assert performance_monitor.memory_usage < 100, f"Memory usage too high: {performance_monitor.memory_usage}MB"
            assert performance_monitor.avg_cpu < 80, f"CPU usage too high: {performance_monitor.avg_cpu}%"

            print(f"WebSocket Throughput Benchmark:")
            print(f"  Messages/second: {messages_per_second:,.0f}")
            print(f"  Duration: {performance_monitor.duration:.3f}s")
            print(f"  Memory usage: {performance_monitor.memory_usage:.1f}MB")
            print(f"  Average CPU: {performance_monitor.avg_cpu:.1f}%")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_analytics_calculation_performance(self, performance_monitor):
        """Test analytics calculation performance with large datasets."""
        with patch('analytics.performance_calculator.DatabaseService') as mock_db:
            # Setup large dataset
            calculator = PerformanceCalculator()
            
            # Generate 10,000 trades
            trades = []
            for i in range(10000):
                trades.append({
                    'symbol': f'STOCK_{i % 100}',
                    'quantity': 100 + (i % 500),
                    'price': 50.0 + (i % 100),
                    'timestamp': time.time() - (i * 60),  # 1 trade per minute
                    'side': 'BUY' if i % 2 == 0 else 'SELL'
                })

            mock_db.return_value.get_trades.return_value = trades
            
            # Generate 1,000 positions
            positions = []
            for i in range(1000):
                positions.append({
                    'symbol': f'STOCK_{i % 100}',
                    'quantity': 1000 + (i * 10),
                    'avg_price': 45.0 + (i % 50),
                    'market_price': 50.0 + (i % 100),
                    'unrealized_pnl': (5.0 + (i % 50)) * (1000 + (i * 10))
                })

            mock_db.return_value.get_positions.return_value = positions

            # Start performance monitoring
            performance_monitor.start_monitoring()
            
            # Benchmark calculations
            performance_monitor.sample_cpu()
            
            # Calculate portfolio performance
            portfolio_pnl = await calculator.calculate_portfolio_pnl('portfolio_1')
            performance_monitor.sample_cpu()
            
            # Calculate performance attribution
            attribution = await calculator.calculate_performance_attribution('portfolio_1', '1D')
            performance_monitor.sample_cpu()
            
            # Calculate risk metrics
            risk_analytics = RiskAnalytics()
            risk_metrics = await risk_analytics.calculate_portfolio_risk('portfolio_1')
            performance_monitor.sample_cpu()
            
            performance_monitor.stop_monitoring()

            # Performance assertions
            calculations_per_second = 3 / performance_monitor.duration
            
            # Benchmark targets
            assert performance_monitor.duration < 2.0, f"Calculations too slow: {performance_monitor.duration}s"
            assert calculations_per_second > 1.0, f"Calculation rate too low: {calculations_per_second} calc/s"
            assert performance_monitor.memory_usage < 200, f"Memory usage too high: {performance_monitor.memory_usage}MB"

            print(f"Analytics Performance Benchmark:")
            print(f"  Calculations/second: {calculations_per_second:.2f}")
            print(f"  Duration: {performance_monitor.duration:.3f}s")
            print(f"  Memory usage: {performance_monitor.memory_usage:.1f}MB")
            print(f"  Dataset size: {len(trades)} trades, {len(positions)} positions")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_risk_monitoring_performance(self, performance_monitor):
        """Test risk monitoring performance with high-frequency updates."""
        with patch('risk_management.risk_monitor.DatabaseService') as mock_db:
            # Setup
            risk_monitor = RiskMonitor()
            limit_engine = LimitEngine()
            
            # Generate high-frequency position updates
            position_updates = []
            for i in range(5000):
                position_updates.append({
                    'portfolio_id': f'portfolio_{i % 10}',
                    'symbol': f'STOCK_{i % 500}',
                    'quantity': 1000 + (i % 2000),
                    'market_value': (50.0 + (i % 100)) * (1000 + (i % 2000)),
                    'timestamp': time.time() - (i * 0.1)  # 10 updates per second
                })

            # Mock database responses
            mock_db.return_value.get_portfolio_positions.return_value = position_updates[:100]
            mock_db.return_value.get_risk_limits.return_value = [
                {'limit_type': 'max_position', 'value': 1000000},
                {'limit_type': 'max_portfolio_risk', 'value': 500000},
                {'limit_type': 'max_concentration', 'value': 0.1}
            ]

            # Start performance monitoring
            performance_monitor.start_monitoring()
            
            # Benchmark risk calculations
            risk_calculations = []
            for i in range(0, len(position_updates), 100):  # Process in batches
                batch = position_updates[i:i+100]
                performance_monitor.sample_cpu()
                
                # Calculate portfolio risk for each update batch
                for update in batch:
                    risk_calc = await risk_monitor.calculate_portfolio_risk(
                        update['portfolio_id']
                    )
                    risk_calculations.append(risk_calc)
                
                # Check limits for each update
                for update in batch:
                    limit_check = await limit_engine.check_limits(
                        update['portfolio_id'], 
                        {'position_update': update}
                    )
                    risk_calculations.append(limit_check)
            
            performance_monitor.stop_monitoring()

            # Performance assertions
            calculations_per_second = len(risk_calculations) / performance_monitor.duration
            
            # Benchmark targets
            assert performance_monitor.duration < 10.0, f"Risk calculations too slow: {performance_monitor.duration}s"
            assert calculations_per_second > 500, f"Calculation rate too low: {calculations_per_second} calc/s"
            assert performance_monitor.memory_usage < 150, f"Memory usage too high: {performance_monitor.memory_usage}MB"

            print(f"Risk Monitoring Performance Benchmark:")
            print(f"  Risk calculations/second: {calculations_per_second:,.0f}")
            print(f"  Duration: {performance_monitor.duration:.3f}s")
            print(f"  Memory usage: {performance_monitor.memory_usage:.1f}MB")
            print(f"  Total calculations: {len(risk_calculations)}")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_strategy_deployment_performance(self, performance_monitor):
        """Test strategy deployment performance with multiple concurrent deployments."""
        with patch('strategy_pipeline.deployment_manager.DockerService') as mock_docker, \
             patch('strategy_pipeline.deployment_manager.GitService') as mock_git:
            
            # Setup
            deployment_manager = DeploymentManager()
            
            # Mock services
            mock_docker.return_value.build_image = AsyncMock(return_value={'id': 'image_123'})
            mock_docker.return_value.run_container = AsyncMock(return_value={'id': 'container_123'})
            mock_docker.return_value.stop_container = AsyncMock()
            mock_git.return_value.clone_repository = AsyncMock()
            mock_git.return_value.checkout_version = AsyncMock()

            # Create multiple strategy deployments
            strategies = []
            for i in range(20):
                strategies.append({
                    'strategy_id': f'strategy_{i}',
                    'version': f'v1.{i}',
                    'config': {
                        'symbols': [f'STOCK_{j}' for j in range(i, i+10)],
                        'parameters': {'param1': i, 'param2': i*2}
                    }
                })

            # Start performance monitoring
            performance_monitor.start_monitoring()
            
            # Benchmark concurrent deployments
            deployment_tasks = []
            for strategy in strategies:
                performance_monitor.sample_cpu()
                task = deployment_manager.deploy_strategy(
                    strategy['strategy_id'],
                    strategy['version'],
                    strategy['config']
                )
                deployment_tasks.append(task)
            
            # Wait for all deployments
            deployment_results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            performance_monitor.sample_cpu()
            
            performance_monitor.stop_monitoring()

            # Count successful deployments
            successful_deployments = sum(
                1 for result in deployment_results 
                if not isinstance(result, Exception)
            )

            # Performance assertions
            deployments_per_second = successful_deployments / performance_monitor.duration
            
            # Benchmark targets
            assert performance_monitor.duration < 30.0, f"Deployments too slow: {performance_monitor.duration}s"
            assert deployments_per_second > 0.5, f"Deployment rate too low: {deployments_per_second} dep/s"
            assert successful_deployments >= len(strategies) * 0.9, f"Too many failed deployments"
            assert performance_monitor.memory_usage < 300, f"Memory usage too high: {performance_monitor.memory_usage}MB"

            print(f"Strategy Deployment Performance Benchmark:")
            print(f"  Deployments/second: {deployments_per_second:.2f}")
            print(f"  Duration: {performance_monitor.duration:.3f}s")
            print(f"  Successful deployments: {successful_deployments}/{len(strategies)}")
            print(f"  Memory usage: {performance_monitor.memory_usage:.1f}MB")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_system_integration_performance(self, performance_monitor):
        """Test integrated system performance with all components active."""
        with patch('websocket.websocket_manager.WebSocket'), \
             patch('analytics.performance_calculator.DatabaseService'), \
             patch('risk_management.risk_monitor.DatabaseService'), \
             patch('strategy_pipeline.deployment_manager.DockerService'):
            
            # Setup all components
            websocket_manager = WebSocketManager()
            performance_calculator = PerformanceCalculator()
            risk_monitor = RiskMonitor()
            deployment_manager = DeploymentManager()
            
            # Create mock connections
            for i in range(50):
                mock_conn = Mock()
                mock_conn.send = AsyncMock()
                websocket_manager.connections[f"client_{i}"] = mock_conn

            # Start performance monitoring
            performance_monitor.start_monitoring()
            
            # Simulate integrated system load
            tasks = []
            
            # WebSocket broadcasting task
            async def websocket_load():
                for _ in range(100):
                    message = {"type": "market_update", "data": {"price": 100.0}}
                    await websocket_manager.broadcast(message)
                    await asyncio.sleep(0.01)  # 100 messages per second
            
            # Analytics calculation task
            async def analytics_load():
                for _ in range(20):
                    await performance_calculator.calculate_portfolio_pnl('portfolio_test')
                    await asyncio.sleep(0.1)  # 10 calculations per second
            
            # Risk monitoring task
            async def risk_load():
                for _ in range(50):
                    await risk_monitor.calculate_portfolio_risk('portfolio_test')
                    await asyncio.sleep(0.05)  # 20 calculations per second
            
            # Strategy management task
            async def strategy_load():
                for i in range(5):
                    await deployment_manager.deploy_strategy(f'strategy_{i}', 'v1.0', {})
                    await asyncio.sleep(0.5)  # 2 deployments per second

            # Run all tasks concurrently
            tasks = [
                websocket_load(),
                analytics_load(),
                risk_load(),
                strategy_load()
            ]
            
            performance_monitor.sample_cpu()
            await asyncio.gather(*tasks, return_exceptions=True)
            performance_monitor.sample_cpu()
            
            performance_monitor.stop_monitoring()

            # Performance assertions for integrated system
            assert performance_monitor.duration < 15.0, f"Integrated system too slow: {performance_monitor.duration}s"
            assert performance_monitor.memory_usage < 500, f"Memory usage too high: {performance_monitor.memory_usage}MB"
            assert performance_monitor.avg_cpu < 85, f"CPU usage too high: {performance_monitor.avg_cpu}%"

            print(f"System Integration Performance Benchmark:")
            print(f"  Duration: {performance_monitor.duration:.3f}s")
            print(f"  Memory usage: {performance_monitor.memory_usage:.1f}MB")
            print(f"  Average CPU: {performance_monitor.avg_cpu:.1f}%")
            print(f"  Components: WebSocket (50 clients), Analytics (20 calcs), Risk (50 calcs), Strategy (5 deploys)")


class TestMemoryLeakDetection:
    """Memory leak detection tests for long-running operations."""

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_websocket_memory_stability(self):
        """Test for memory leaks in WebSocket operations over time."""
        with patch('websocket.websocket_manager.WebSocket'):
            manager = WebSocketManager()
            
            # Create connections
            for i in range(100):
                mock_conn = Mock()
                mock_conn.send = AsyncMock()
                manager.connections[f"client_{i}"] = mock_conn

            # Track memory usage over time
            memory_samples = []
            
            for iteration in range(10):
                # Record memory before operations
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Perform operations
                for _ in range(100):
                    message = {"type": "test", "data": f"iteration_{iteration}"}
                    await manager.broadcast(message)
                
                # Record memory after operations
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(memory_after - memory_before)
                
                # Small delay between iterations
                await asyncio.sleep(0.1)

            # Check for memory leaks
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            
            # Assert no significant memory growth trend
            assert memory_trend < 1.0, f"Memory leak detected: {memory_trend}MB/iteration growth"
            
            # Assert memory usage is reasonable
            final_memory = memory_samples[-1]
            assert final_memory < 50, f"Memory usage too high: {final_memory}MB"

            print(f"WebSocket Memory Stability Test:")
            print(f"  Memory trend: {memory_trend:.3f}MB/iteration")
            print(f"  Final memory usage: {final_memory:.1f}MB")
            print(f"  Iterations: {len(memory_samples)}")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_analytics_memory_stability(self):
        """Test for memory leaks in analytics calculations over time."""
        with patch('analytics.performance_calculator.DatabaseService') as mock_db:
            calculator = PerformanceCalculator()
            
            # Mock large dataset
            large_dataset = [
                {'symbol': f'STOCK_{i}', 'price': 100 + i, 'volume': 1000 + i}
                for i in range(1000)
            ]
            mock_db.return_value.get_market_data.return_value = large_dataset

            # Track memory usage over time
            memory_samples = []
            
            for iteration in range(20):
                # Record memory before calculation
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Perform calculation
                result = await calculator.calculate_portfolio_pnl(f'portfolio_{iteration}')
                
                # Record memory after calculation
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(memory_after - memory_before)
                
                # Small delay between iterations
                await asyncio.sleep(0.05)

            # Check for memory leaks
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            
            # Assert no significant memory growth trend
            assert memory_trend < 0.5, f"Memory leak detected: {memory_trend}MB/iteration growth"

            print(f"Analytics Memory Stability Test:")
            print(f"  Memory trend: {memory_trend:.3f}MB/iteration")
            print(f"  Average memory per calculation: {statistics.mean(memory_samples):.1f}MB")
            print(f"  Iterations: {len(memory_samples)}")
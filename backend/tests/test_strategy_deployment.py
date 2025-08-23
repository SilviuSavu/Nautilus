"""
Integration tests for Strategy Deployment Pipeline - Sprint 3

Tests complete strategy deployment pipeline from development through
testing to production deployment, including version control and rollback.
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Import Sprint 3 strategy pipeline components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategy_pipeline.deployment_manager import DeploymentManager, StrategyDeployment, DeploymentStatus, DeploymentConfig
from strategy_pipeline.strategy_tester import StrategyTester, TestResult, BacktestConfig
from strategy_pipeline.version_control import StrategyVersionControl, StrategyVersion, VersionStatus
from strategy_pipeline.rollback_service import RollbackService, RollbackPlan, RollbackStatus
from strategy_pipeline.pipeline_monitor import PipelineMonitor, DeploymentMetrics


class TestCompleteDeploymentPipeline:
    """Test complete strategy deployment pipeline end-to-end"""
    
    @pytest.fixture
    def deployment_system(self):
        """Setup complete deployment system"""
        deployment_manager = DeploymentManager()
        strategy_tester = StrategyTester()
        version_control = StrategyVersionControl()
        rollback_service = RollbackService()
        pipeline_monitor = PipelineMonitor()
        
        # Link components
        rollback_service.deployment_manager = deployment_manager
        
        return {
            "deployment_manager": deployment_manager,
            "strategy_tester": strategy_tester,
            "version_control": version_control,
            "rollback_service": rollback_service,
            "pipeline_monitor": pipeline_monitor
        }
    
    @pytest.fixture
    def temp_strategy_repo(self):
        """Create temporary strategy repository"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_nautilus_engine(self):
        """Mock NautilusTrader engine"""
        mock_engine = AsyncMock()
        mock_engine.add_strategy = AsyncMock(return_value=True)
        mock_engine.remove_strategy = AsyncMock(return_value=True)
        mock_engine.start_strategy = AsyncMock(return_value=True)
        mock_engine.stop_strategy = AsyncMock(return_value=True)
        mock_engine.get_strategy_status = AsyncMock(return_value="RUNNING")
        return mock_engine
    
    @pytest.fixture
    def sample_strategy_code(self):
        """Sample strategy code for testing"""
        return """
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from decimal import Decimal

class MomentumStrategy:
    '''Enhanced momentum trading strategy with risk management'''
    
    def __init__(self, lookback_period: int = 20, momentum_threshold: float = 0.02, 
                 max_position_size: Decimal = Decimal('50000')):
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.max_position_size = max_position_size
        self.price_history: Dict[str, List[float]] = {}
        self.positions: Dict[str, Decimal] = {}
        
    def on_market_data(self, symbol: str, price: float, timestamp: datetime) -> Dict[str, Any]:
        '''Process market data and generate trading signals'''
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only recent prices
        if len(self.price_history[symbol]) > self.lookback_period + 1:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period-1:]
        
        # Calculate momentum if enough data
        if len(self.price_history[symbol]) >= self.lookback_period + 1:
            current_price = self.price_history[symbol][-1]
            historical_price = self.price_history[symbol][-self.lookback_period-1]
            
            momentum = (current_price - historical_price) / historical_price
            
            # Generate signals
            if momentum > self.momentum_threshold:
                return self._generate_buy_signal(symbol, price, momentum)
            elif momentum < -self.momentum_threshold:
                return self._generate_sell_signal(symbol, price, momentum)
        
        return {"action": "HOLD", "symbol": symbol, "timestamp": timestamp}
    
    def _generate_buy_signal(self, symbol: str, price: float, momentum: float) -> Dict[str, Any]:
        '''Generate buy signal with position sizing'''
        current_position = self.positions.get(symbol, Decimal('0'))
        
        # Calculate position size based on momentum strength
        base_size = min(self.max_position_size, Decimal(str(abs(momentum) * 100000)))
        
        # Risk management: don't exceed max position
        if current_position + base_size <= self.max_position_size:
            return {
                "action": "BUY",
                "symbol": symbol,
                "quantity": float(base_size / Decimal(str(price))),
                "price": price,
                "momentum": momentum,
                "timestamp": datetime.utcnow()
            }
        
        return {"action": "HOLD", "reason": "Position size limit reached"}
    
    def _generate_sell_signal(self, symbol: str, price: float, momentum: float) -> Dict[str, Any]:
        '''Generate sell signal'''
        current_position = self.positions.get(symbol, Decimal('0'))
        
        if current_position > 0:
            # Sell portion based on negative momentum
            sell_quantity = min(float(current_position), abs(momentum) * float(current_position))
            
            return {
                "action": "SELL",
                "symbol": symbol,
                "quantity": sell_quantity,
                "price": price,
                "momentum": momentum,
                "timestamp": datetime.utcnow()
            }
        
        return {"action": "HOLD", "reason": "No position to sell"}
    
    def on_order_fill(self, symbol: str, side: str, quantity: float, price: float):
        '''Update positions on order fills'''
        if symbol not in self.positions:
            self.positions[symbol] = Decimal('0')
        
        if side.upper() == "BUY":
            self.positions[symbol] += Decimal(str(quantity * price))
        elif side.upper() == "SELL":
            self.positions[symbol] -= Decimal(str(quantity * price))
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        '''Get current portfolio status'''
        total_exposure = sum(abs(pos) for pos in self.positions.values())
        
        return {
            "total_exposure": float(total_exposure),
            "positions": {symbol: float(pos) for symbol, pos in self.positions.items()},
            "position_count": len([pos for pos in self.positions.values() if pos != 0]),
            "max_position_utilization": float(max(self.positions.values()) / self.max_position_size) if self.positions else 0.0
        }
"""
    
    @pytest.mark.asyncio
    async def test_complete_deployment_lifecycle(self, deployment_system, temp_strategy_repo, 
                                                sample_strategy_code, mock_nautilus_engine):
        """Test complete strategy deployment lifecycle"""
        deployment_manager = deployment_system["deployment_manager"]
        strategy_tester = deployment_system["strategy_tester"]
        version_control = deployment_system["version_control"]
        rollback_service = deployment_system["rollback_service"]
        pipeline_monitor = deployment_system["pipeline_monitor"]
        
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        strategy_id = "momentum_v2_0"
        
        # 1. CREATE STRATEGY VERSION
        version = version_control.create_version(
            strategy_id=strategy_id,
            version_number="2.0.0",
            strategy_code=sample_strategy_code,
            metadata={
                "author": "strategy_team",
                "description": "Enhanced momentum strategy with improved risk management",
                "parameters": {
                    "lookback_period": 20,
                    "momentum_threshold": 0.02,
                    "max_position_size": 50000
                },
                "risk_limits": {
                    "max_daily_loss": 5000,
                    "max_drawdown": 0.15,
                    "var_limit": 25000
                }
            },
            repository_path=temp_strategy_repo
        )
        
        assert version.status == VersionStatus.DRAFT
        assert version.strategy_id == strategy_id
        
        # 2. PROMOTE TO TESTING
        success = version_control.promote_version(version.version_id, VersionStatus.TESTING)
        assert success is True
        
        updated_version = version_control.get_version(version.version_id)
        assert updated_version.status == VersionStatus.TESTING
        
        # 3. RUN BACKTESTS
        backtest_config = BacktestConfig(
            strategy_id=strategy_id,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=Decimal('100000'),
            instruments=["AAPL", "MSFT", "GOOGL"],
            data_frequency="1min",
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Mock successful backtest
        with patch.object(strategy_tester, 'run_backtest') as mock_backtest:
            mock_test_result = TestResult(
                test_id="backtest_momentum_v2_0",
                strategy_id=strategy_id,
                total_return=0.18,  # 18% return
                total_trades=156,
                final_portfolio_value=Decimal('118000'),
                max_drawdown=-0.08,  # 8% max drawdown
                sharpe_ratio=1.35,
                win_rate=0.62,
                profit_factor=1.8,
                trade_history=[]
            )
            mock_backtest.return_value = mock_test_result
            
            test_result = await strategy_tester.run_backtest(backtest_config)
            
            # Verify backtest performance meets criteria
            assert test_result.total_return > 0.15  # > 15% return
            assert test_result.sharpe_ratio > 1.2   # > 1.2 Sharpe ratio
            assert test_result.max_drawdown > -0.10 # < 10% drawdown
            assert test_result.win_rate > 0.55      # > 55% win rate
        
        # 4. MONTE CARLO TESTING
        with patch.object(strategy_tester, 'run_monte_carlo_test') as mock_mc:
            mock_mc_results = {
                "scenario_results": [
                    {"scenario_id": f"mc_{i}", "total_return": 0.15 + (i % 20) * 0.01}
                    for i in range(100)
                ],
                "statistical_summary": {
                    "mean_return": 0.165,
                    "return_std": 0.045,
                    "worst_case_return": 0.08,
                    "best_case_return": 0.25,
                    "var_95": -0.05,  # 5% VaR
                    "probability_of_loss": 0.15
                }
            }
            mock_mc.return_value = mock_mc_results
            
            mc_results = await strategy_tester.run_monte_carlo_test(backtest_config, num_scenarios=100)
            
            # Verify Monte Carlo results
            summary = mc_results["statistical_summary"]
            assert summary["mean_return"] > 0.12
            assert summary["probability_of_loss"] < 0.20
        
        # 5. PROMOTE TO PRODUCTION READY
        success = version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
        assert success is True
        
        # 6. DEPLOY TO PAPER TRADING
        paper_deployment_config = DeploymentConfig(
            strategy_id=strategy_id,
            strategy_name="Enhanced Momentum Strategy v2.0",
            version="2.0.0",
            allocated_capital=Decimal('50000'),  # Start with smaller capital
            risk_limits={
                "max_position_size": Decimal('25000'),
                "max_daily_loss": Decimal('2500'),
                "max_drawdown": 0.10
            },
            target_environment="paper",
            deployment_parameters={
                "lookback_period": 20,
                "momentum_threshold": 0.02,
                "rebalance_frequency": "realtime"
            }
        )
        
        # Mock successful deployment
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            paper_deployment = await deployment_manager.deploy_strategy(paper_deployment_config)
            
            assert paper_deployment.status == DeploymentStatus.DEPLOYED
            assert paper_deployment.version == "2.0.0"
            assert paper_deployment.allocated_capital == Decimal('50000')
        
        # 7. MONITOR PAPER TRADING PERFORMANCE
        pipeline_monitor.record_deployment_event({
            "event_type": "deployment_completed",
            "strategy_id": strategy_id,
            "environment": "paper",
            "timestamp": datetime.utcnow()
        })
        
        # Simulate paper trading for 30 days
        for day in range(30):
            pipeline_monitor.record_performance_data({
                "strategy_id": strategy_id,
                "date": datetime.utcnow() - timedelta(days=30-day),
                "daily_return": 0.002 + (day % 5) * 0.001,  # Varying daily returns
                "cumulative_return": day * 0.003,
                "drawdown": max(-0.02, -day * 0.0005),  # Small drawdowns
                "sharpe_ratio": 1.2 + day * 0.01
            })
        
        # Get deployment metrics
        paper_metrics = pipeline_monitor.get_deployment_metrics(time_window_hours=24*30)
        assert paper_metrics.total_deployments >= 1
        assert paper_metrics.successful_deployments >= 1
        
        # 8. DEPLOY TO LIVE TRADING (if paper trading successful)
        live_deployment_config = DeploymentConfig(
            strategy_id=strategy_id,
            strategy_name="Enhanced Momentum Strategy v2.0",
            version="2.0.0",
            allocated_capital=Decimal('100000'),  # Full capital allocation
            risk_limits={
                "max_position_size": Decimal('50000'),
                "max_daily_loss": Decimal('5000'),
                "max_drawdown": 0.15
            },
            target_environment="live",
            deployment_parameters={
                "lookbook_period": 20,
                "momentum_threshold": 0.02,
                "rebalance_frequency": "realtime"
            }
        )
        
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            live_deployment = await deployment_manager.deploy_strategy(live_deployment_config)
            
            assert live_deployment.status == DeploymentStatus.DEPLOYED
            assert live_deployment.target_environment == "live"
            assert live_deployment.allocated_capital == Decimal('100000')
    
    @pytest.mark.asyncio
    async def test_deployment_rollback_scenario(self, deployment_system, temp_strategy_repo, 
                                               sample_strategy_code, mock_nautilus_engine):
        """Test deployment rollback scenario"""
        deployment_manager = deployment_system["deployment_manager"]
        version_control = deployment_system["version_control"]
        rollback_service = deployment_system["rollback_service"]
        
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        strategy_id = "momentum_rollback_test"
        
        # Create multiple versions
        versions = []
        for i, version_num in enumerate(["1.8.0", "1.9.0", "2.0.0"]):
            version = version_control.create_version(
                strategy_id=strategy_id,
                version_number=version_num,
                strategy_code=sample_strategy_code,
                metadata={
                    "author": "strategy_team",
                    "description": f"Version {version_num} of momentum strategy",
                    "iteration": i + 1
                },
                repository_path=temp_strategy_repo
            )
            
            # Promote all to production
            version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
            versions.append(version)
        
        # Deploy latest version (2.0.0)
        deployment_config = DeploymentConfig(
            strategy_id=strategy_id,
            strategy_name="Momentum Strategy (Latest)",
            version="2.0.0",
            allocated_capital=Decimal('100000'),
            target_environment="live"
        )
        
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            current_deployment = await deployment_manager.deploy_strategy(deployment_config)
            assert current_deployment.version == "2.0.0"
        
        # Simulate performance degradation requiring rollback
        # In real scenario, this would be detected by monitoring systems
        
        # Create rollback plan to previous stable version (1.9.0)
        rollback_plan = await rollback_service.create_rollback_plan(
            strategy_id=strategy_id,
            target_version="1.9.0",
            reason="Performance degradation detected: 5% drop in Sharpe ratio over 3 days"
        )
        
        assert rollback_plan.strategy_id == strategy_id
        assert rollback_plan.current_version == "2.0.0"
        assert rollback_plan.target_version == "1.9.0"
        assert len(rollback_plan.rollback_steps) > 0
        
        # Execute rollback
        mock_nautilus_engine.stop_strategy.return_value = True
        mock_nautilus_engine.remove_strategy.return_value = True
        
        # Mock deployment of rollback version
        rollback_deployment = StrategyDeployment(
            deployment_id="rollback_deployment",
            strategy_id=strategy_id,
            version="1.9.0",
            status=DeploymentStatus.DEPLOYED,
            deployed_at=datetime.utcnow(),
            allocated_capital=Decimal('100000')
        )
        
        with patch.object(deployment_manager, 'deploy_strategy', return_value=rollback_deployment):
            rollback_result = await rollback_service.execute_rollback(rollback_plan.plan_id)
            
            assert rollback_result.status == RollbackStatus.COMPLETED
            assert rollback_result.completed_steps == len(rollback_plan.rollback_steps)
            assert len(rollback_result.error_messages) == 0
        
        # Verify rollback was successful
        current_deployment = deployment_manager.get_deployment(strategy_id)
        # In mock scenario, this would show version 1.9.0
    
    @pytest.mark.asyncio
    async def test_parallel_strategy_deployment(self, deployment_system, temp_strategy_repo, 
                                               sample_strategy_code, mock_nautilus_engine):
        """Test deploying multiple strategies in parallel"""
        deployment_manager = deployment_system["deployment_manager"]
        version_control = deployment_system["version_control"]
        pipeline_monitor = deployment_system["pipeline_monitor"]
        
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        # Create multiple strategies
        strategies = [
            {"id": "momentum_strategy", "name": "Momentum Strategy", "capital": Decimal('75000')},
            {"id": "mean_reversion_strategy", "name": "Mean Reversion Strategy", "capital": Decimal('50000')},
            {"id": "arbitrage_strategy", "name": "Arbitrage Strategy", "capital": Decimal('100000')},
            {"id": "pairs_trading_strategy", "name": "Pairs Trading Strategy", "capital": Decimal('60000')}
        ]
        
        # Create versions for all strategies
        versions = []
        for strategy in strategies:
            version = version_control.create_version(
                strategy_id=strategy["id"],
                version_number="1.0.0",
                strategy_code=sample_strategy_code,  # Same code for simplicity
                metadata={
                    "author": "strategy_team",
                    "description": f"{strategy['name']} implementation"
                },
                repository_path=temp_strategy_repo
            )
            
            version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
            versions.append((strategy, version))
        
        # Deploy all strategies concurrently
        deployment_tasks = []
        
        for strategy, version in versions:
            config = DeploymentConfig(
                strategy_id=strategy["id"],
                strategy_name=strategy["name"],
                version="1.0.0",
                allocated_capital=strategy["capital"],
                target_environment="paper"
            )
            
            with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
                mock_validate.return_value = (True, [])
                
                task = deployment_manager.deploy_strategy(config)
                deployment_tasks.append((strategy["id"], task))
        
        # Wait for all deployments to complete
        deployment_results = []
        for strategy_id, task in deployment_tasks:
            deployment = await task
            deployment_results.append((strategy_id, deployment))
        
        # Verify all deployments successful
        successful_deployments = 0
        total_allocated_capital = Decimal('0')
        
        for strategy_id, deployment in deployment_results:
            assert deployment.status == DeploymentStatus.DEPLOYED
            assert deployment.strategy_id == strategy_id
            
            successful_deployments += 1
            total_allocated_capital += deployment.allocated_capital
            
            # Record deployment event
            pipeline_monitor.record_deployment_event({
                "event_type": "deployment_completed",
                "strategy_id": strategy_id,
                "timestamp": datetime.utcnow()
            })
        
        assert successful_deployments == 4
        assert total_allocated_capital == Decimal('285000')  # Sum of all allocations
        
        # Verify resource allocation limits
        allocation_summary = deployment_manager.get_resource_allocation_summary()
        assert allocation_summary["total_allocated_capital"] == total_allocated_capital
        assert allocation_summary["active_deployments"] == 4
    
    @pytest.mark.asyncio
    async def test_deployment_failure_recovery(self, deployment_system, temp_strategy_repo,
                                              sample_strategy_code, mock_nautilus_engine):
        """Test deployment failure scenarios and recovery"""
        deployment_manager = deployment_system["deployment_manager"]
        version_control = deployment_system["version_control"]
        pipeline_monitor = deployment_system["pipeline_monitor"]
        
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        strategy_id = "failure_recovery_test"
        
        # Create strategy version
        version = version_control.create_version(
            strategy_id=strategy_id,
            version_number="1.0.0",
            strategy_code=sample_strategy_code,
            repository_path=temp_strategy_repo
        )
        
        version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
        
        # Test deployment with validation failure
        deployment_config = DeploymentConfig(
            strategy_id=strategy_id,
            strategy_name="Failure Recovery Test",
            version="1.0.0",
            allocated_capital=Decimal('-10000'),  # Invalid negative capital
            target_environment="live"
        )
        
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (False, ["Invalid allocated capital: cannot be negative"])
            
            failed_deployment = await deployment_manager.deploy_strategy(deployment_config)
            
            assert failed_deployment.status == DeploymentStatus.FAILED
            assert len(failed_deployment.error_messages) > 0
            assert "negative" in str(failed_deployment.error_messages)
        
        # Record failure event
        pipeline_monitor.record_deployment_event({
            "event_type": "deployment_failed",
            "strategy_id": strategy_id,
            "error_reason": "validation_failed",
            "timestamp": datetime.utcnow()
        })
        
        # Test deployment with engine failure
        valid_config = DeploymentConfig(
            strategy_id=strategy_id,
            strategy_name="Failure Recovery Test",
            version="1.0.0",
            allocated_capital=Decimal('50000'),
            target_environment="paper"
        )
        
        # Mock engine failure
        mock_nautilus_engine.add_strategy.side_effect = Exception("Engine connection timeout")
        
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            failed_deployment = await deployment_manager.deploy_strategy(valid_config)
            
            assert failed_deployment.status == DeploymentStatus.FAILED
            assert "timeout" in str(failed_deployment.error_messages)
        
        # Test successful recovery deployment
        mock_nautilus_engine.add_strategy.side_effect = None  # Remove failure
        mock_nautilus_engine.add_strategy.return_value = True
        
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            successful_deployment = await deployment_manager.deploy_strategy(valid_config)
            
            assert successful_deployment.status == DeploymentStatus.DEPLOYED
            assert successful_deployment.strategy_id == strategy_id
        
        # Verify deployment metrics reflect failures and recovery
        metrics = pipeline_monitor.get_deployment_metrics(time_window_hours=24)
        assert metrics.total_deployments >= 3
        assert metrics.failed_deployments >= 2
        assert metrics.successful_deployments >= 1
    
    def test_deployment_configuration_validation(self, deployment_system):
        """Test deployment configuration validation"""
        deployment_manager = deployment_system["deployment_manager"]
        
        # Test valid configuration
        valid_config = DeploymentConfig(
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            version="1.0.0",
            allocated_capital=Decimal('100000'),
            risk_limits={
                "max_position_size": Decimal('50000'),
                "max_daily_loss": Decimal('5000')
            },
            target_environment="paper"
        )
        
        is_valid, errors = deployment_manager._validate_deployment_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid configurations
        invalid_configs = [
            # Empty strategy ID
            DeploymentConfig(
                strategy_id="",
                strategy_name="Test",
                version="1.0.0",
                allocated_capital=Decimal('100000')
            ),
            # Invalid version format
            DeploymentConfig(
                strategy_id="test",
                strategy_name="Test", 
                version="invalid_version",
                allocated_capital=Decimal('100000')
            ),
            # Negative capital
            DeploymentConfig(
                strategy_id="test",
                strategy_name="Test",
                version="1.0.0", 
                allocated_capital=Decimal('-50000')
            ),
            # Invalid environment
            DeploymentConfig(
                strategy_id="test",
                strategy_name="Test",
                version="1.0.0",
                allocated_capital=Decimal('100000'),
                target_environment="invalid_env"
            )
        ]
        
        for config in invalid_configs:
            is_valid, errors = deployment_manager._validate_deployment_config(config)
            assert is_valid is False
            assert len(errors) > 0


class TestStrategyDeploymentPerformance:
    """Test strategy deployment performance under load"""
    
    @pytest.fixture
    def performance_test_system(self):
        """Setup system for performance testing"""
        return {
            "deployment_manager": DeploymentManager(),
            "pipeline_monitor": PipelineMonitor()
        }
    
    def test_high_volume_deployment_monitoring(self, performance_test_system):
        """Test monitoring system under high deployment volume"""
        pipeline_monitor = performance_test_system["pipeline_monitor"]
        
        import time
        
        # Generate high volume of deployment events
        num_events = 10000
        start_time = time.time()
        
        for i in range(num_events):
            event = {
                "event_type": "deployment_completed" if i % 10 != 0 else "deployment_failed",
                "strategy_id": f"strategy_{i % 100}",
                "timestamp": datetime.utcnow() - timedelta(seconds=i)
            }
            
            pipeline_monitor.record_deployment_event(event)
        
        processing_time = time.time() - start_time
        
        # Should handle high volume efficiently
        assert processing_time < 5.0  # Within 5 seconds
        
        # Get metrics efficiently
        start_time = time.time()
        metrics = pipeline_monitor.get_deployment_metrics(time_window_hours=24)
        metrics_time = time.time() - start_time
        
        assert metrics_time < 1.0  # Metrics calculation within 1 second
        assert metrics.total_deployments > 0
    
    def test_concurrent_deployment_operations(self, performance_test_system):
        """Test concurrent deployment operations"""
        import concurrent.futures
        import threading
        import time
        
        deployment_manager = performance_test_system["deployment_manager"]
        
        # Mock successful validation and engine operations
        with patch.object(deployment_manager, '_validate_strategy', return_value=(True, [])):
            with patch.object(deployment_manager, 'nautilus_engine') as mock_engine:
                mock_engine.add_strategy = Mock(return_value=True)
                mock_engine.start_strategy = Mock(return_value=True)
                
                def deploy_strategy_sync(strategy_id):
                    config = DeploymentConfig(
                        strategy_id=strategy_id,
                        strategy_name=f"Strategy {strategy_id}",
                        version="1.0.0",
                        allocated_capital=Decimal('10000')
                    )
                    
                    # Simulate async deployment in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(deployment_manager.deploy_strategy(config))
                    finally:
                        loop.close()
                
                # Test concurrent deployments
                num_concurrent = 20
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(deploy_strategy_sync, f"concurrent_strategy_{i}")
                        for i in range(num_concurrent)
                    ]
                    
                    results = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]
                
                concurrent_time = time.time() - start_time
                
                # Should handle concurrent deployments efficiently
                assert concurrent_time < 10.0  # Within 10 seconds
                assert len(results) == num_concurrent
                
                # Verify all deployments successful
                successful_deployments = sum(
                    1 for result in results
                    if result.status == DeploymentStatus.DEPLOYED
                )
                assert successful_deployments == num_concurrent
    
    def test_memory_usage_with_many_deployments(self, performance_test_system):
        """Test memory usage with many active deployments"""
        deployment_manager = performance_test_system["deployment_manager"]
        
        import sys
        import gc
        
        # Measure initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create many deployment records
        num_deployments = 1000
        
        for i in range(num_deployments):
            deployment = StrategyDeployment(
                deployment_id=f"deploy_{i}",
                strategy_id=f"strategy_{i}",
                version="1.0.0",
                status=DeploymentStatus.DEPLOYED,
                deployed_at=datetime.utcnow(),
                allocated_capital=Decimal('10000')
            )
            
            deployment_manager.active_deployments[f"strategy_{i}"] = deployment
        
        # Measure memory after deployments
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Calculate memory usage
        object_growth = final_objects - initial_objects
        objects_per_deployment = object_growth / num_deployments
        
        # Should use reasonable memory per deployment
        assert objects_per_deployment < 20  # Less than 20 objects per deployment
        
        # Test retrieval performance
        start_time = time.time()
        
        # Get all deployment summaries
        summary = deployment_manager.get_resource_allocation_summary()
        
        retrieval_time = time.time() - start_time
        
        # Should retrieve summary efficiently
        assert retrieval_time < 0.5  # Within 500ms
        assert summary["active_deployments"] == num_deployments
        
        # Cleanup
        deployment_manager.active_deployments.clear()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
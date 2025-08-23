"""
Unit tests for Strategy Framework Components - Sprint 3 Priority 4

Tests strategy deployment, testing, version control, and rollback functionality
with comprehensive coverage including deployment pipelines and error scenarios.
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

# Import Sprint 3 strategy framework components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategies.deployment_manager import (
    DeploymentManager, Deployment, DeploymentStatus, DeploymentConfig
)
from strategies.strategy_tester import (
    StrategyTester, TestResult, TestSuite, BacktestResult
)
from strategies.version_control import (
    VersionControl, VersionInfo, VersionStatus
)
from strategies.rollback_service import (
    RollbackService, RollbackPlan, RollbackStatus
)
from strategies.pipeline_monitor import (
    PipelineMonitor, PipelineStatus, PipelineMetrics
)


class TestDeploymentManager:
    """Test strategy deployment management functionality"""
    
    @pytest.fixture
    def deployment_manager(self):
        """Create deployment manager for testing"""
        return DeploymentManager()
    
    @pytest.fixture
    def sample_deployment_config(self):
        """Sample deployment configuration"""
        return DeploymentConfig(
            strategy_id="momentum_v1_2",
            strategy_name="Enhanced Momentum Strategy",
            version="1.2.0",
            allocated_capital=Decimal('100000'),
            risk_limits={
                "max_position_size": Decimal('50000'),
                "max_daily_loss": Decimal('5000'),
                "max_drawdown": 0.15
            },
            target_environment="paper",
            deployment_parameters={
                "lookback_period": 20,
                "momentum_threshold": 0.02,
                "rebalance_frequency": "daily"
            }
        )
    
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
    
    @pytest.mark.asyncio
    async def test_strategy_deployment_success(self, deployment_manager, sample_deployment_config, mock_nautilus_engine):
        """Test successful strategy deployment"""
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        # Mock strategy validation
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Test deployment
            deployment = await deployment_manager.deploy_strategy(sample_deployment_config)
            
            assert isinstance(deployment, StrategyDeployment)
            assert deployment.strategy_id == "momentum_v1_2"
            assert deployment.status == DeploymentStatus.DEPLOYED
            assert deployment.version == "1.2.0"
            assert deployment.allocated_capital == Decimal('100000')
            
            # Verify engine interactions
            mock_nautilus_engine.add_strategy.assert_called_once()
            mock_nautilus_engine.start_strategy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_strategy_deployment_validation_failure(self, deployment_manager, sample_deployment_config):
        """Test deployment failure due to validation errors"""
        # Mock validation failure
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (False, ["Invalid parameter: lookback_period must be > 0"])
            
            # Test deployment
            deployment = await deployment_manager.deploy_strategy(sample_deployment_config)
            
            assert deployment.status == DeploymentStatus.FAILED
            assert len(deployment.error_messages) == 1
            assert "Invalid parameter" in deployment.error_messages[0]
    
    @pytest.mark.asyncio
    async def test_strategy_deployment_engine_failure(self, deployment_manager, sample_deployment_config, mock_nautilus_engine):
        """Test deployment failure due to engine errors"""
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        # Mock engine failure
        mock_nautilus_engine.add_strategy.side_effect = Exception("Engine connection failed")
        
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Test deployment
            deployment = await deployment_manager.deploy_strategy(sample_deployment_config)
            
            assert deployment.status == DeploymentStatus.FAILED
            assert "Engine connection failed" in str(deployment.error_messages)
    
    @pytest.mark.asyncio
    async def test_strategy_update_deployment(self, deployment_manager, sample_deployment_config, mock_nautilus_engine):
        """Test updating existing strategy deployment"""
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        # First deployment
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            
            original_deployment = await deployment_manager.deploy_strategy(sample_deployment_config)
            
            # Update configuration
            updated_config = sample_deployment_config.copy()
            updated_config.version = "1.2.1"
            updated_config.allocated_capital = Decimal('150000')
            
            # Test update
            updated_deployment = await deployment_manager.update_strategy_deployment(
                sample_deployment_config.strategy_id, updated_config
            )
            
            assert updated_deployment.version == "1.2.1"
            assert updated_deployment.allocated_capital == Decimal('150000')
            assert updated_deployment.status == DeploymentStatus.DEPLOYED
            
            # Verify engine calls for update
            assert mock_nautilus_engine.stop_strategy.called
            assert mock_nautilus_engine.start_strategy.called
    
    @pytest.mark.asyncio
    async def test_strategy_undeploy(self, deployment_manager, sample_deployment_config, mock_nautilus_engine):
        """Test strategy undeployment"""
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        # First deploy strategy
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (True, [])
            deployment = await deployment_manager.deploy_strategy(sample_deployment_config)
        
        # Test undeploy
        success = await deployment_manager.undeploy_strategy(deployment.strategy_id)
        
        assert success is True
        mock_nautilus_engine.stop_strategy.assert_called()
        mock_nautilus_engine.remove_strategy.assert_called()
        
        # Verify deployment status updated
        updated_deployment = deployment_manager.get_deployment(deployment.strategy_id)
        assert updated_deployment.status == DeploymentStatus.STOPPED
    
    def test_deployment_status_monitoring(self, deployment_manager, mock_nautilus_engine):
        """Test deployment status monitoring"""
        deployment_manager.nautilus_engine = mock_nautilus_engine
        
        # Mock active deployment
        mock_deployment = StrategyDeployment(
            deployment_id="deploy_123",
            strategy_id="momentum_v1_2",
            version="1.2.0",
            status=DeploymentStatus.DEPLOYED,
            deployed_at=datetime.utcnow(),
            allocated_capital=Decimal('100000')
        )
        deployment_manager.active_deployments["momentum_v1_2"] = mock_deployment
        
        # Test status monitoring
        status = deployment_manager.get_deployment_status("momentum_v1_2")
        
        assert status["deployment_id"] == "deploy_123"
        assert status["status"] == DeploymentStatus.DEPLOYED.value
        assert status["version"] == "1.2.0"
    
    def test_deployment_resource_allocation(self, deployment_manager):
        """Test deployment resource allocation and limits"""
        # Mock multiple deployments with capital allocation
        deployments = [
            {"strategy_id": "strategy_1", "allocated_capital": Decimal('100000')},
            {"strategy_id": "strategy_2", "allocated_capital": Decimal('150000')},
            {"strategy_id": "strategy_3", "allocated_capital": Decimal('75000')}
        ]
        
        for deploy in deployments:
            mock_deployment = StrategyDeployment(
                deployment_id=f"deploy_{deploy['strategy_id']}",
                strategy_id=deploy['strategy_id'],
                version="1.0.0",
                status=DeploymentStatus.DEPLOYED,
                allocated_capital=deploy['allocated_capital']
            )
            deployment_manager.active_deployments[deploy['strategy_id']] = mock_deployment
        
        # Test resource allocation summary
        allocation_summary = deployment_manager.get_resource_allocation_summary()
        
        assert allocation_summary["total_allocated_capital"] == Decimal('325000')
        assert allocation_summary["active_deployments"] == 3
        assert len(allocation_summary["deployments"]) == 3


class TestStrategyTester:
    """Test strategy testing functionality"""
    
    @pytest.fixture
    def strategy_tester(self):
        """Create strategy tester for testing"""
        return StrategyTester()
    
    @pytest.fixture
    def sample_backtest_config(self):
        """Sample backtest configuration"""
        return BacktestConfig(
            strategy_id="momentum_test",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal('100000'),
            instruments=["AAPL", "MSFT", "GOOGL"],
            data_frequency="1min",
            commission_rate=0.001,
            slippage_rate=0.0005
        )
    
    @pytest.fixture
    def mock_historical_data(self):
        """Mock historical market data"""
        return {
            "AAPL": [
                {"timestamp": "2023-01-01T09:30:00Z", "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000},
                {"timestamp": "2023-01-01T09:31:00Z", "open": 151.0, "high": 153.0, "low": 150.5, "close": 152.5, "volume": 800000},
                # More data points...
            ],
            "MSFT": [
                {"timestamp": "2023-01-01T09:30:00Z", "open": 300.0, "high": 302.0, "low": 299.0, "close": 301.0, "volume": 500000},
                {"timestamp": "2023-01-01T09:31:00Z", "open": 301.0, "high": 303.0, "low": 300.5, "close": 302.5, "volume": 600000},
                # More data points...
            ]
        }
    
    @pytest.mark.asyncio
    async def test_backtest_execution(self, strategy_tester, sample_backtest_config, mock_historical_data):
        """Test strategy backtesting execution"""
        # Mock data provider
        async def mock_get_historical_data(instruments, start_date, end_date, frequency):
            return mock_historical_data
        
        strategy_tester.get_historical_data = mock_get_historical_data
        
        # Mock strategy logic
        def mock_strategy_logic(data_point):
            # Simple momentum strategy logic
            if data_point["close"] > data_point["open"] * 1.01:
                return {"action": "BUY", "quantity": 100}
            elif data_point["close"] < data_point["open"] * 0.99:
                return {"action": "SELL", "quantity": 100}
            return {"action": "HOLD"}
        
        strategy_tester.strategy_logic = mock_strategy_logic
        
        # Run backtest
        result = await strategy_tester.run_backtest(sample_backtest_config)
        
        assert isinstance(result, TestResult)
        assert result.strategy_id == "momentum_test"
        assert result.total_trades >= 0
        assert result.final_portfolio_value > 0
        assert result.max_drawdown <= 0
        assert len(result.trade_history) >= 0
    
    def test_performance_metrics_calculation(self, strategy_tester):
        """Test calculation of strategy performance metrics"""
        # Mock trade history
        trade_history = [
            {"timestamp": datetime(2023, 1, 2), "action": "BUY", "symbol": "AAPL", "quantity": 100, "price": 151.0, "pnl": 0},
            {"timestamp": datetime(2023, 1, 3), "action": "SELL", "symbol": "AAPL", "quantity": 100, "price": 153.0, "pnl": 200},
            {"timestamp": datetime(2023, 1, 5), "action": "BUY", "symbol": "MSFT", "quantity": 50, "price": 302.0, "pnl": 0},
            {"timestamp": datetime(2023, 1, 6), "action": "SELL", "symbol": "MSFT", "quantity": 50, "price": 298.0, "pnl": -200},
        ]
        
        # Mock portfolio values
        portfolio_values = [100000, 100200, 100200, 100000]  # Daily portfolio values
        
        metrics = strategy_tester.calculate_performance_metrics(
            trade_history, portfolio_values, initial_capital=100000
        )
        
        assert metrics["total_return"] == 0.0  # Break-even
        assert metrics["total_trades"] == 2
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 0.5
        assert metrics["profit_factor"] == 1.0  # Equal wins and losses
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
    
    def test_risk_metrics_calculation(self, strategy_tester):
        """Test risk metrics calculation during backtesting"""
        # Mock daily returns
        daily_returns = [0.01, -0.005, 0.02, -0.015, 0.008, -0.012, 0.025, -0.008]
        
        risk_metrics = strategy_tester.calculate_risk_metrics(daily_returns)
        
        assert "volatility" in risk_metrics
        assert "var_95" in risk_metrics
        assert "expected_shortfall" in risk_metrics
        assert "maximum_drawdown" in risk_metrics
        assert "calmar_ratio" in risk_metrics
        
        # Verify reasonable values
        assert risk_metrics["volatility"] > 0
        assert risk_metrics["var_95"] < 0  # VaR should be negative
        assert risk_metrics["maximum_drawdown"] <= 0
    
    @pytest.mark.asyncio
    async def test_monte_carlo_testing(self, strategy_tester, sample_backtest_config):
        """Test Monte Carlo strategy testing"""
        # Mock random scenarios
        def mock_generate_scenarios(num_scenarios=100):
            scenarios = []
            for i in range(num_scenarios):
                scenario = {
                    "scenario_id": f"scenario_{i}",
                    "market_regime": "normal" if i < 80 else "volatile",
                    "parameter_variations": {
                        "lookback_period": 20 + (i % 10),
                        "momentum_threshold": 0.02 + (i % 5) * 0.001
                    }
                }
                scenarios.append(scenario)
            return scenarios
        
        strategy_tester.generate_scenarios = mock_generate_scenarios
        
        # Mock backtest execution for each scenario
        async def mock_run_scenario_backtest(config, scenario):
            return TestResult(
                test_id=f"test_{scenario['scenario_id']}",
                strategy_id=config.strategy_id,
                total_return=0.05 + (hash(scenario['scenario_id']) % 100) / 1000,  # Random return
                total_trades=50 + (hash(scenario['scenario_id']) % 20),
                final_portfolio_value=Decimal('105000'),
                max_drawdown=-0.08 - (hash(scenario['scenario_id']) % 10) / 1000
            )
        
        strategy_tester.run_scenario_backtest = mock_run_scenario_backtest
        
        # Run Monte Carlo test
        mc_results = await strategy_tester.run_monte_carlo_test(sample_backtest_config, num_scenarios=100)
        
        assert "scenario_results" in mc_results
        assert "statistical_summary" in mc_results
        assert len(mc_results["scenario_results"]) == 100
        
        # Verify statistical summary
        summary = mc_results["statistical_summary"]
        assert "mean_return" in summary
        assert "return_std" in summary
        assert "worst_case_return" in summary
        assert "best_case_return" in summary
    
    def test_strategy_optimization(self, strategy_tester):
        """Test strategy parameter optimization"""
        # Define parameter ranges
        parameter_space = {
            "lookback_period": [10, 15, 20, 25, 30],
            "momentum_threshold": [0.01, 0.015, 0.02, 0.025, 0.03],
            "rebalance_frequency": ["daily", "weekly", "monthly"]
        }
        
        # Mock optimization objective function
        def mock_objective_function(parameters):
            # Simulate performance based on parameters
            base_return = 0.05
            lookback_penalty = (parameters["lookback_period"] - 20) * 0.001
            threshold_penalty = abs(parameters["momentum_threshold"] - 0.02) * 10
            
            return base_return - lookback_penalty - threshold_penalty
        
        strategy_tester.objective_function = mock_objective_function
        
        # Run optimization
        optimal_params = strategy_tester.optimize_parameters(parameter_space, max_evaluations=50)
        
        assert "best_parameters" in optimal_params
        assert "best_performance" in optimal_params
        assert "optimization_history" in optimal_params
        
        # Verify optimization found reasonable parameters
        best_params = optimal_params["best_parameters"]
        assert 10 <= best_params["lookback_period"] <= 30
        assert 0.01 <= best_params["momentum_threshold"] <= 0.03
    
    def test_strategy_sensitivity_analysis(self, strategy_tester):
        """Test sensitivity analysis of strategy parameters"""
        base_parameters = {
            "lookback_period": 20,
            "momentum_threshold": 0.02,
            "stop_loss": 0.05
        }
        
        # Mock performance function
        def mock_calculate_performance(params):
            # Simulate how performance changes with parameters
            performance = 0.05  # Base performance
            performance += (params["lookback_period"] - 20) * 0.001
            performance -= abs(params["momentum_threshold"] - 0.02) * 10
            performance += params["stop_loss"] * 2
            return performance
        
        strategy_tester.calculate_performance = mock_calculate_performance
        
        # Run sensitivity analysis
        sensitivity_results = strategy_tester.analyze_parameter_sensitivity(
            base_parameters, variation_pct=0.2
        )
        
        assert len(sensitivity_results) == len(base_parameters)
        
        for param_name, sensitivity in sensitivity_results.items():
            assert "base_value" in sensitivity
            assert "sensitivity_score" in sensitivity
            assert "parameter_impact" in sensitivity
            assert param_name in base_parameters


class TestStrategyVersionControl:
    """Test strategy version control functionality"""
    
    @pytest.fixture
    def version_control(self):
        """Create version control system for testing"""
        return StrategyVersionControl()
    
    @pytest.fixture
    def temp_strategy_repo(self):
        """Create temporary strategy repository"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_strategy_version_creation(self, version_control, temp_strategy_repo):
        """Test creating new strategy version"""
        strategy_code = """
import numpy as np
from nautilus_trader.model.data import Bar

class MomentumStrategy:
    def __init__(self, lookback_period=20):
        self.lookback_period = lookback_period
        self.prices = []
    
    def on_bar(self, bar: Bar):
        self.prices.append(bar.close)
        if len(self.prices) > self.lookback_period:
            momentum = self.prices[-1] / self.prices[-self.lookback_period]
            if momentum > 1.02:
                return "BUY"
            elif momentum < 0.98:
                return "SELL"
        return "HOLD"
        """
        
        version = version_control.create_version(
            strategy_id="momentum_v1",
            version_number="1.0.0",
            strategy_code=strategy_code,
            metadata={
                "author": "test_user",
                "description": "Basic momentum strategy",
                "parameters": {"lookback_period": 20}
            },
            repository_path=temp_strategy_repo
        )
        
        assert isinstance(version, StrategyVersion)
        assert version.strategy_id == "momentum_v1"
        assert version.version_number == "1.0.0"
        assert version.status == VersionStatus.DRAFT
        
        # Verify files created
        strategy_file = temp_strategy_repo / "momentum_v1" / "v1.0.0" / "strategy.py"
        assert strategy_file.exists()
        
        metadata_file = temp_strategy_repo / "momentum_v1" / "v1.0.0" / "metadata.json"
        assert metadata_file.exists()
    
    def test_version_promotion_workflow(self, version_control, temp_strategy_repo):
        """Test version promotion through workflow stages"""
        # Create initial version
        version = version_control.create_version(
            strategy_id="test_strategy",
            version_number="1.0.0",
            strategy_code="# Test strategy code",
            repository_path=temp_strategy_repo
        )
        
        # Test promotion to testing
        success = version_control.promote_version(version.version_id, VersionStatus.TESTING)
        assert success is True
        
        updated_version = version_control.get_version(version.version_id)
        assert updated_version.status == VersionStatus.TESTING
        
        # Test promotion to production
        success = version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
        assert success is True
        
        updated_version = version_control.get_version(version.version_id)
        assert updated_version.status == VersionStatus.PRODUCTION
    
    def test_version_comparison(self, version_control, temp_strategy_repo):
        """Test comparing different strategy versions"""
        # Create version 1.0.0
        code_v1 = "def strategy(): return 'v1'"
        version_1 = version_control.create_version(
            strategy_id="compare_test",
            version_number="1.0.0",
            strategy_code=code_v1,
            repository_path=temp_strategy_repo
        )
        
        # Create version 1.1.0
        code_v2 = "def strategy(): return 'v2'"
        version_2 = version_control.create_version(
            strategy_id="compare_test",
            version_number="1.1.0",
            strategy_code=code_v2,
            repository_path=temp_strategy_repo
        )
        
        # Compare versions
        comparison = version_control.compare_versions(version_1.version_id, version_2.version_id)
        
        assert "code_diff" in comparison
        assert "metadata_changes" in comparison
        assert "version_1" in comparison
        assert "version_2" in comparison
    
    def test_version_rollback_preparation(self, version_control, temp_strategy_repo):
        """Test preparing version for rollback"""
        # Create and promote version
        version = version_control.create_version(
            strategy_id="rollback_test",
            version_number="1.0.0",
            strategy_code="# Original code",
            repository_path=temp_strategy_repo
        )
        
        version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
        
        # Create backup for rollback
        backup = version_control.create_rollback_backup(version.version_id)
        
        assert backup is not None
        assert "backup_id" in backup
        assert "version_snapshot" in backup
        assert backup["version_snapshot"]["version_id"] == version.version_id
    
    def test_version_history_tracking(self, version_control, temp_strategy_repo):
        """Test version history and audit trail"""
        strategy_id = "history_test"
        
        # Create multiple versions
        versions = []
        for i in range(3):
            version = version_control.create_version(
                strategy_id=strategy_id,
                version_number=f"1.{i}.0",
                strategy_code=f"# Version 1.{i}.0 code",
                repository_path=temp_strategy_repo
            )
            versions.append(version)
        
        # Get version history
        history = version_control.get_version_history(strategy_id)
        
        assert len(history) == 3
        assert all(v.strategy_id == strategy_id for v in history)
        
        # Verify chronological order
        version_numbers = [v.version_number for v in history]
        assert version_numbers == ["1.0.0", "1.1.0", "1.2.0"]


class TestRollbackService:
    """Test rollback service functionality"""
    
    @pytest.fixture
    def rollback_service(self):
        """Create rollback service for testing"""
        return RollbackService()
    
    @pytest.fixture
    def mock_deployment_manager(self):
        """Mock deployment manager"""
        mock_manager = AsyncMock()
        mock_manager.get_deployment.return_value = StrategyDeployment(
            deployment_id="deploy_123",
            strategy_id="test_strategy",
            version="2.0.0",
            status=DeploymentStatus.DEPLOYED,
            allocated_capital=Decimal('100000')
        )
        return mock_manager
    
    @pytest.mark.asyncio
    async def test_rollback_plan_creation(self, rollback_service, mock_deployment_manager):
        """Test creating rollback plan"""
        rollback_service.deployment_manager = mock_deployment_manager
        
        # Create rollback plan
        plan = await rollback_service.create_rollback_plan(
            strategy_id="test_strategy",
            target_version="1.5.0",
            reason="Performance degradation detected"
        )
        
        assert isinstance(plan, RollbackPlan)
        assert plan.strategy_id == "test_strategy"
        assert plan.current_version == "2.0.0"
        assert plan.target_version == "1.5.0"
        assert plan.reason == "Performance degradation detected"
        assert len(plan.rollback_steps) > 0
    
    @pytest.mark.asyncio
    async def test_rollback_execution(self, rollback_service, mock_deployment_manager):
        """Test rollback execution"""
        rollback_service.deployment_manager = mock_deployment_manager
        
        # Mock successful rollback steps
        mock_deployment_manager.undeploy_strategy.return_value = True
        mock_deployment_manager.deploy_strategy.return_value = StrategyDeployment(
            deployment_id="deploy_rollback",
            strategy_id="test_strategy",
            version="1.5.0",
            status=DeploymentStatus.DEPLOYED,
            allocated_capital=Decimal('100000')
        )
        
        # Create and execute rollback
        plan = await rollback_service.create_rollback_plan(
            strategy_id="test_strategy",
            target_version="1.5.0"
        )
        
        result = await rollback_service.execute_rollback(plan.plan_id)
        
        assert result.status == RollbackStatus.COMPLETED
        assert result.completed_steps == len(plan.rollback_steps)
        assert len(result.error_messages) == 0
    
    @pytest.mark.asyncio
    async def test_rollback_failure_recovery(self, rollback_service, mock_deployment_manager):
        """Test rollback failure and recovery"""
        rollback_service.deployment_manager = mock_deployment_manager
        
        # Mock deployment failure during rollback
        mock_deployment_manager.undeploy_strategy.return_value = True
        mock_deployment_manager.deploy_strategy.side_effect = Exception("Deployment failed")
        
        # Create and execute rollback
        plan = await rollback_service.create_rollback_plan(
            strategy_id="test_strategy",
            target_version="1.5.0"
        )
        
        result = await rollback_service.execute_rollback(plan.plan_id)
        
        assert result.status == RollbackStatus.FAILED
        assert len(result.error_messages) > 0
        assert "Deployment failed" in str(result.error_messages)
    
    def test_rollback_validation(self, rollback_service):
        """Test rollback plan validation"""
        # Test invalid rollback scenarios
        invalid_plans = [
            {"strategy_id": "", "target_version": "1.0.0"},  # Empty strategy ID
            {"strategy_id": "test", "target_version": ""},   # Empty target version
            {"strategy_id": "test", "target_version": "invalid_version"}  # Invalid version format
        ]
        
        for invalid_plan in invalid_plans:
            validation_result = rollback_service.validate_rollback_plan(invalid_plan)
            assert validation_result["is_valid"] is False
            assert len(validation_result["errors"]) > 0


class TestPipelineMonitor:
    """Test deployment pipeline monitoring functionality"""
    
    @pytest.fixture
    def pipeline_monitor(self):
        """Create pipeline monitor for testing"""
        return PipelineMonitor()
    
    def test_deployment_metrics_tracking(self, pipeline_monitor):
        """Test tracking of deployment metrics"""
        # Mock deployment events
        deployment_events = [
            {"event_type": "deployment_started", "strategy_id": "strategy_1", "timestamp": datetime.utcnow() - timedelta(minutes=10)},
            {"event_type": "deployment_completed", "strategy_id": "strategy_1", "timestamp": datetime.utcnow() - timedelta(minutes=5)},
            {"event_type": "deployment_started", "strategy_id": "strategy_2", "timestamp": datetime.utcnow() - timedelta(minutes=3)},
            {"event_type": "deployment_failed", "strategy_id": "strategy_2", "timestamp": datetime.utcnow() - timedelta(minutes=1)}
        ]
        
        # Process events
        for event in deployment_events:
            pipeline_monitor.record_deployment_event(event)
        
        # Get metrics
        metrics = pipeline_monitor.get_deployment_metrics(time_window_hours=1)
        
        assert isinstance(metrics, DeploymentMetrics)
        assert metrics.total_deployments == 2
        assert metrics.successful_deployments == 1
        assert metrics.failed_deployments == 1
        assert metrics.success_rate == 0.5
        assert metrics.average_deployment_time > timedelta(0)
    
    def test_pipeline_health_monitoring(self, pipeline_monitor):
        """Test pipeline health monitoring"""
        # Mock active deployments
        active_deployments = [
            {"strategy_id": "strategy_1", "status": "RUNNING", "last_heartbeat": datetime.utcnow() - timedelta(seconds=30)},
            {"strategy_id": "strategy_2", "status": "RUNNING", "last_heartbeat": datetime.utcnow() - timedelta(minutes=5)},  # Stale
            {"strategy_id": "strategy_3", "status": "STOPPED", "last_heartbeat": datetime.utcnow() - timedelta(hours=1)}
        ]
        
        health_status = pipeline_monitor.check_pipeline_health(active_deployments)
        
        assert health_status["overall_status"] in ["HEALTHY", "WARNING", "CRITICAL"]
        assert "active_strategies" in health_status
        assert "unhealthy_strategies" in health_status
        assert len(health_status["unhealthy_strategies"]) >= 1  # strategy_2 should be flagged as stale
    
    def test_performance_degradation_detection(self, pipeline_monitor):
        """Test detection of strategy performance degradation"""
        # Mock performance data showing degradation
        performance_data = [
            {"timestamp": datetime.utcnow() - timedelta(hours=5), "strategy_id": "strategy_1", "daily_return": 0.02},
            {"timestamp": datetime.utcnow() - timedelta(hours=4), "strategy_id": "strategy_1", "daily_return": 0.015},
            {"timestamp": datetime.utcnow() - timedelta(hours=3), "strategy_id": "strategy_1", "daily_return": -0.005},
            {"timestamp": datetime.utcnow() - timedelta(hours=2), "strategy_id": "strategy_1", "daily_return": -0.015},
            {"timestamp": datetime.utcnow() - timedelta(hours=1), "strategy_id": "strategy_1", "daily_return": -0.025}
        ]
        
        degradation_alerts = pipeline_monitor.detect_performance_degradation(performance_data)
        
        assert len(degradation_alerts) > 0
        alert = degradation_alerts[0]
        assert alert["strategy_id"] == "strategy_1"
        assert alert["alert_type"] == "PERFORMANCE_DEGRADATION"
        assert "trend_score" in alert
    
    @pytest.mark.asyncio
    async def test_automated_response_triggers(self, pipeline_monitor):
        """Test automated response to pipeline issues"""
        triggered_responses = []
        
        async def mock_response_handler(alert_type, strategy_id, details):
            triggered_responses.append({
                "alert_type": alert_type,
                "strategy_id": strategy_id,
                "details": details
            })
        
        # Register response handler
        pipeline_monitor.add_response_handler(mock_response_handler)
        
        # Simulate critical alert
        critical_alert = {
            "alert_type": "CRITICAL_ERROR",
            "strategy_id": "strategy_1",
            "details": {"error": "Memory usage exceeded 90%"}
        }
        
        await pipeline_monitor.process_alert(critical_alert)
        
        # Verify response was triggered
        assert len(triggered_responses) == 1
        assert triggered_responses[0]["alert_type"] == "CRITICAL_ERROR"
        assert triggered_responses[0]["strategy_id"] == "strategy_1"


class TestStrategyFrameworkIntegration:
    """Integration tests for strategy framework components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_deployment_pipeline(self):
        """Test complete deployment pipeline from development to production"""
        # Initialize components
        version_control = StrategyVersionControl()
        strategy_tester = StrategyTester()
        deployment_manager = DeploymentManager()
        pipeline_monitor = PipelineMonitor()
        
        # Create temporary repository
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_repo = Path(temp_dir)
            
            # 1. Create strategy version
            strategy_code = "# Test strategy implementation"
            version = version_control.create_version(
                strategy_id="integration_test_strategy",
                version_number="1.0.0",
                strategy_code=strategy_code,
                repository_path=temp_repo
            )
            
            assert version.status == VersionStatus.DRAFT
            
            # 2. Promote to testing
            version_control.promote_version(version.version_id, VersionStatus.TESTING)
            
            # 3. Mock successful testing
            with patch.object(strategy_tester, 'run_backtest') as mock_backtest:
                mock_backtest.return_value = TestResult(
                    test_id="integration_test",
                    strategy_id="integration_test_strategy",
                    total_return=0.15,
                    total_trades=100,
                    final_portfolio_value=Decimal('115000'),
                    max_drawdown=-0.05
                )
                
                test_result = await strategy_tester.run_backtest(Mock())
                assert test_result.total_return > 0.1  # Good performance
            
            # 4. Promote to production
            version_control.promote_version(version.version_id, VersionStatus.PRODUCTION)
            
            # 5. Mock deployment
            with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
                mock_validate.return_value = (True, [])
                
                deployment_config = DeploymentConfig(
                    strategy_id="integration_test_strategy",
                    strategy_name="Integration Test Strategy",
                    version="1.0.0",
                    allocated_capital=Decimal('100000')
                )
                
                deployment_manager.nautilus_engine = AsyncMock()
                deployment_manager.nautilus_engine.add_strategy.return_value = True
                deployment_manager.nautilus_engine.start_strategy.return_value = True
                
                deployment = await deployment_manager.deploy_strategy(deployment_config)
                assert deployment.status == DeploymentStatus.DEPLOYED
            
            # 6. Monitor deployment
            pipeline_monitor.record_deployment_event({
                "event_type": "deployment_completed",
                "strategy_id": "integration_test_strategy",
                "timestamp": datetime.utcnow()
            })
            
            metrics = pipeline_monitor.get_deployment_metrics(time_window_hours=1)
            assert metrics.total_deployments >= 1
    
    def test_error_handling_across_components(self):
        """Test error handling and resilience across framework components"""
        deployment_manager = DeploymentManager()
        
        # Test invalid configuration handling
        invalid_config = DeploymentConfig(
            strategy_id="",  # Invalid empty ID
            strategy_name="Test Strategy",
            version="1.0.0",
            allocated_capital=Decimal('-1000')  # Invalid negative capital
        )
        
        # Should handle validation gracefully
        with patch.object(deployment_manager, '_validate_strategy') as mock_validate:
            mock_validate.return_value = (False, ["Invalid strategy configuration"])
            
            result = asyncio.run(deployment_manager.deploy_strategy(invalid_config))
            assert result.status == DeploymentStatus.FAILED
            assert len(result.error_messages) > 0
    
    def test_performance_under_concurrent_operations(self):
        """Test framework performance under concurrent operations"""
        import time
        import threading
        
        deployment_manager = DeploymentManager()
        pipeline_monitor = PipelineMonitor()
        
        # Simulate concurrent deployment monitoring
        def monitor_deployment(strategy_id):
            for i in range(10):
                pipeline_monitor.record_deployment_event({
                    "event_type": "heartbeat",
                    "strategy_id": strategy_id,
                    "timestamp": datetime.utcnow()
                })
                time.sleep(0.001)  # Small delay
        
        # Run concurrent monitoring for multiple strategies
        threads = []
        for i in range(5):
            thread = threading.Thread(target=monitor_deployment, args=[f"strategy_{i}"])
            threads.append(thread)
        
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # Should handle concurrent operations efficiently
        assert execution_time < 1.0  # Should complete within 1 second
        
        # Verify all events were recorded
        metrics = pipeline_monitor.get_deployment_metrics(time_window_hours=1)
        assert metrics.total_events >= 50  # 5 strategies * 10 events each
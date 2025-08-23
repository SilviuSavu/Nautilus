"""
Integration Tests for Strategy Deployment Pipeline

Comprehensive tests for all pipeline components and their interactions
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from .strategy_tester import StrategyTester, TestStatus, TestType
from .deployment_manager import DeploymentManager, DeploymentEnvironment, DeploymentStatus
from .version_control import VersionControl, BranchType
from .rollback_service import RollbackService, RollbackTriggerType
from .pipeline_monitor import PipelineMonitor, AlertSeverity
from . import StrategyPipeline


class TestStrategyTester:
    """Test cases for StrategyTester component"""
    
    def setup_method(self):
        self.tester = StrategyTester()
    
    @pytest.mark.asyncio
    async def test_syntax_validation(self):
        """Test strategy code syntax validation"""
        
        # Valid Python code
        valid_code = """
class TestStrategy:
    def __init__(self):
        self.name = "test"
    
    def on_start(self):
        pass
"""
        
        result = await self.tester._run_syntax_validation(valid_code)
        assert result.valid is True
        assert len(result.errors) == 0
        assert "TestStrategy" in result.classes
        assert "__init__" in result.methods
        assert "on_start" in result.methods
    
    @pytest.mark.asyncio
    async def test_syntax_validation_with_errors(self):
        """Test syntax validation with invalid code"""
        
        invalid_code = """
class TestStrategy
    def __init__(self)
        self.name = "test"
"""
        
        result = await self.tester._run_syntax_validation(invalid_code)
        assert result.valid is False
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_code_analysis(self):
        """Test code quality analysis"""
        
        code = """
class SimpleStrategy:
    def __init__(self):
        self.counter = 0
    
    def process_data(self, data):
        if data > 0:
            self.counter += 1
        return self.counter
"""
        
        result = await self.tester._run_code_analysis(code)
        assert isinstance(result.complexity_score, float)
        assert isinstance(result.maintainability_index, float)
        assert isinstance(result.code_quality_score, float)
        assert result.metrics["classes"] > 0
        assert result.metrics["functions"] > 0
    
    @pytest.mark.asyncio
    async def test_full_test_suite(self):
        """Test complete test suite execution"""
        
        strategy_code = """
from nautilus_trader import Strategy

class TestStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.counter = 0
    
    def on_start(self):
        self.log.info("Strategy started")
    
    def on_data(self, data):
        self.counter += 1
"""
        
        strategy_config = {
            "id": "test_strategy",
            "name": "Test Strategy",
            "parameters": {
                "instrument": "EUR/USD.SIM",
                "trade_size": 100000
            }
        }
        
        test_suite = await self.tester.run_full_test_suite(
            strategy_code, strategy_config
        )
        
        assert isinstance(test_suite.suite_id, str)
        assert test_suite.strategy_id == "test_strategy"
        assert len(test_suite.tests) > 0
        assert isinstance(test_suite.overall_score, float)
        assert test_suite.overall_score >= 0
        
        # Check that all expected test types were run
        test_types = [test.test_type for test in test_suite.tests]
        assert TestType.SYNTAX_VALIDATION in test_types
        assert TestType.CODE_ANALYSIS in test_types


class TestDeploymentManager:
    """Test cases for DeploymentManager component"""
    
    def setup_method(self):
        self.deployment_manager = DeploymentManager()
        self.mock_nautilus_service = AsyncMock()
        self.deployment_manager.nautilus_engine_service = self.mock_nautilus_service
    
    @pytest.mark.asyncio
    async def test_create_deployment_request(self):
        """Test deployment request creation"""
        
        # Mock version control
        self.deployment_manager.version_control = MagicMock()
        self.deployment_manager.version_control.get_version.return_value = MagicMock()
        
        request = await self.deployment_manager.create_deployment_request(
            strategy_id="test_strategy",
            version="1.0.0",
            target_environment=DeploymentEnvironment.DEVELOPMENT,
            requested_by="test_user"
        )
        
        assert isinstance(request.request_id, str)
        assert request.strategy_id == "test_strategy"
        assert request.version == "1.0.0"
        assert request.target_environment == DeploymentEnvironment.DEVELOPMENT
        assert request.requested_by == "test_user"
    
    @pytest.mark.asyncio
    async def test_deploy_strategy_direct(self):
        """Test direct deployment strategy"""
        
        # Mock services
        self.deployment_manager.version_control = MagicMock()
        self.deployment_manager.version_control.get_version.return_value = MagicMock()
        
        self.mock_nautilus_service.start_engine.return_value = {
            "success": True,
            "deployment_id": "nautilus_deployment_123"
        }
        
        deployment = await self.deployment_manager.deploy_strategy(
            strategy_id="test_strategy",
            version="1.0.0",
            target_environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        assert isinstance(deployment.deployment_id, str)
        assert deployment.strategy_id == "test_strategy"
        assert deployment.version == "1.0.0"
        assert deployment.environment == DeploymentEnvironment.DEVELOPMENT
        assert deployment.status in [DeploymentStatus.DEPLOYED, DeploymentStatus.DEPLOYING]
    
    @pytest.mark.asyncio
    async def test_deployment_with_pre_tests(self):
        """Test deployment with pre-deployment testing"""
        
        # Mock strategy tester
        mock_tester = AsyncMock()
        test_suite_mock = MagicMock()
        test_suite_mock.status = TestStatus.PASSED
        test_suite_mock.overall_score = 85.0
        mock_tester.run_full_test_suite.return_value = test_suite_mock
        
        self.deployment_manager.strategy_tester = mock_tester
        self.deployment_manager.version_control = MagicMock()
        self.deployment_manager.version_control.get_version.return_value = MagicMock(
            strategy_code="test_code",
            strategy_config={"test": "config"}
        )
        
        self.mock_nautilus_service.start_engine.return_value = {
            "success": True,
            "deployment_id": "nautilus_deployment_123"
        }
        
        # Create deployment request for staging (requires pre-tests)
        request = await self.deployment_manager.create_deployment_request(
            strategy_id="test_strategy",
            version="1.0.0",
            target_environment=DeploymentEnvironment.STAGING
        )
        
        # Execute deployment
        deployment = await self.deployment_manager._execute_deployment_request(request)
        
        # Verify tests were run
        mock_tester.run_full_test_suite.assert_called_once()
        assert deployment.performance_metrics.get("pre_deployment_test_score") == 85.0


class TestVersionControl:
    """Test cases for VersionControl component"""
    
    def setup_method(self):
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.version_control = VersionControl(storage_path=self.temp_dir)
    
    def test_commit_version(self):
        """Test version commit"""
        
        version_info = self.version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code="print('Hello World')",
            strategy_config={"param1": "value1"},
            commit_message="Initial commit",
            author="test_user"
        )
        
        assert isinstance(version_info.version_id, str)
        assert version_info.strategy_id == "test_strategy"
        assert version_info.version_number == "1.0.0"  # First version
        assert version_info.branch_name == "main"
        assert version_info.commit_message == "Initial commit"
        assert version_info.author == "test_user"
    
    def test_create_branch(self):
        """Test branch creation"""
        
        # First commit to main
        self.version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code="print('Hello World')",
            strategy_config={}
        )
        
        # Create feature branch
        branch = self.version_control.create_branch(
            branch_name="feature/new-indicator",
            base_branch="main",
            branch_type=BranchType.FEATURE,
            created_by="test_user"
        )
        
        assert branch.branch_name == "feature/new-indicator"
        assert branch.branch_type == BranchType.FEATURE
        assert branch.base_branch == "main"
        assert branch.created_by == "test_user"
    
    def test_merge_without_conflicts(self):
        """Test merge without conflicts"""
        
        # Initial commit to main
        main_version = self.version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code="class Strategy: pass",
            strategy_config={"param1": "value1"}
        )
        
        # Create and commit to feature branch
        self.version_control.create_branch("feature/test", "main", BranchType.FEATURE)
        feature_version = self.version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code="class Strategy:\n    def new_method(self): pass",
            strategy_config={"param1": "value1", "param2": "value2"},
            branch_name="feature/test"
        )
        
        # Create merge request
        merge_request = self.version_control.create_merge_request(
            source_branch="feature/test",
            target_branch="main",
            title="Add new method",
            description="Adding new method to strategy"
        )
        
        assert merge_request.source_branch == "feature/test"
        assert merge_request.target_branch == "main"
        assert merge_request.has_conflicts is False  # No conflicts expected
    
    def test_version_diff(self):
        """Test version diff calculation"""
        
        # Create two versions
        version1 = self.version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code="print('v1')",
            strategy_config={"param1": "value1"}
        )
        
        version2 = self.version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code="print('v2')",
            strategy_config={"param1": "value2", "param2": "new_value"}
        )
        
        diff = self.version_control.get_version_diff(
            version1.version_id, version2.version_id
        )
        
        assert diff["code_changed"] is True
        assert diff["config_changed"] is True
        assert "param1" in diff["parameter_changes"]
        assert "param2" in diff["parameter_changes"]
        assert diff["parameter_changes"]["param1"]["change_type"] == "modified"
        assert diff["parameter_changes"]["param2"]["change_type"] == "added"


class TestRollbackService:
    """Test cases for RollbackService component"""
    
    def setup_method(self):
        self.rollback_service = RollbackService()
        self.mock_nautilus_service = AsyncMock()
        self.rollback_service.nautilus_engine_service = self.mock_nautilus_service
    
    def test_configure_rollback_triggers(self):
        """Test rollback trigger configuration"""
        
        from .rollback_service import RollbackTrigger
        
        triggers = [
            RollbackTrigger(
                trigger_type=RollbackTriggerType.PERFORMANCE_THRESHOLD,
                max_drawdown_percent=15.0,
                min_sharpe_ratio=0.8
            ),
            RollbackTrigger(
                trigger_type=RollbackTriggerType.ERROR_RATE,
                max_error_rate=3.0,
                error_window_minutes=20
            )
        ]
        
        result = self.rollback_service.configure_rollback_triggers(
            "test_strategy", triggers
        )
        
        assert result is True
        configured_triggers = self.rollback_service.get_rollback_triggers("test_strategy")
        assert len(configured_triggers) == 2
        assert configured_triggers[0].trigger_type == RollbackTriggerType.PERFORMANCE_THRESHOLD
        assert configured_triggers[1].trigger_type == RollbackTriggerType.ERROR_RATE
    
    @pytest.mark.asyncio
    async def test_execute_rollback(self):
        """Test rollback execution"""
        
        # Mock version control
        mock_version_control = MagicMock()
        self.rollback_service.version_control = mock_version_control
        
        # Mock deployment manager
        mock_deployment_manager = MagicMock()
        mock_deployment_manager.list_deployments.return_value = [
            MagicMock(version="2.0.0", status="deployed", nautilus_deployment_id="deploy_123")
        ]
        self.rollback_service.deployment_manager = mock_deployment_manager
        
        # Mock nautilus service
        self.mock_nautilus_service.stop_engine.return_value = {"success": True}
        self.mock_nautilus_service.start_engine.return_value = {"success": True}
        self.mock_nautilus_service.get_engine_status.return_value = {"status": "running"}
        
        result = await self.rollback_service.execute_rollback(
            strategy_id="test_strategy",
            environment="staging",
            target_version="1.0.0",
            reason="Performance degradation",
            executed_by="test_user"
        )
        
        assert result["success"] is True
        assert isinstance(result["execution_id"], str)
        
        # Verify rollback was recorded
        execution = self.rollback_service.get_rollback_execution(result["execution_id"])
        assert execution is not None
        assert execution.strategy_id == "test_strategy"
        assert execution.to_version == "1.0.0"
        assert execution.trigger_reason == "Performance degradation"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_trigger(self):
        """Test performance-based rollback trigger"""
        
        from .rollback_service import RollbackTrigger, PerformanceSnapshot
        
        # Configure performance trigger
        trigger = RollbackTrigger(
            trigger_type=RollbackTriggerType.PERFORMANCE_THRESHOLD,
            max_drawdown_percent=10.0,
            enabled=True
        )
        
        # Create snapshot that exceeds threshold
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            total_pnl=-5000.0,
            unrealized_pnl=-500.0,
            drawdown_percent=15.0,  # Exceeds 10% threshold
            sharpe_ratio=0.5,
            win_rate=0.4,
            total_trades=20,
            error_count=1,
            health_status="healthy"
        )
        
        # Test trigger evaluation
        triggered = await self.rollback_service._check_trigger(
            trigger, "test_strategy", "staging", snapshot
        )
        
        assert triggered is True


class TestPipelineMonitor:
    """Test cases for PipelineMonitor component"""
    
    def setup_method(self):
        self.monitor = PipelineMonitor()
    
    def test_initialize_default_alert_rules(self):
        """Test default alert rules initialization"""
        
        assert len(self.monitor._alert_rules) > 0
        
        # Check for key default rules
        rule_names = [rule.name for rule in self.monitor._alert_rules.values()]
        assert "High Deployment Failure Rate" in rule_names
        assert "Pipeline Duration Exceeded" in rule_names
        assert "Frequent Rollbacks" in rule_names
    
    @pytest.mark.asyncio
    async def test_create_alert(self):
        """Test alert creation and notification"""
        
        alert_id = await self.monitor._create_alert(
            strategy_id="test_strategy",
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            details={"test_key": "test_value"}
        )
        
        # Check alert was created
        assert len(self.monitor._active_alerts) > 0
        
        # Find the created alert
        created_alert = None
        for alert in self.monitor._active_alerts.values():
            if alert.title == "Test Alert":
                created_alert = alert
                break
        
        assert created_alert is not None
        assert created_alert.strategy_id == "test_strategy"
        assert created_alert.severity == AlertSeverity.WARNING
        assert created_alert.message == "This is a test alert"
        assert created_alert.details["test_key"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_pipeline_monitoring(self):
        """Test pipeline status monitoring"""
        
        # Start monitoring
        await self.monitor.start_monitoring(
            pipeline_id="test_pipeline",
            strategy_id="test_strategy", 
            version="1.0.0"
        )
        
        # Check pipeline status was created
        pipeline_status = self.monitor.get_pipeline_status("test_pipeline")
        assert pipeline_status is not None
        assert pipeline_status.strategy_id == "test_strategy"
        assert pipeline_status.version == "1.0.0"
        assert pipeline_status.status == "running"
        
        # Stop monitoring
        await self.monitor.stop_monitoring("test_pipeline")
        
        # Check monitoring task was stopped
        assert "test_pipeline" not in self.monitor._monitoring_tasks
    
    def test_alert_rule_management(self):
        """Test alert rule addition and removal"""
        
        from .pipeline_monitor import AlertRule, MetricType
        
        # Add custom rule
        custom_rule = AlertRule(
            rule_id="custom_test_rule",
            name="Custom Test Rule",
            description="Custom rule for testing",
            metric_type=MetricType.PERFORMANCE_SCORE,
            threshold_value=50.0,
            threshold_operator="<",
            severity=AlertSeverity.ERROR
        )
        
        initial_count = len(self.monitor._alert_rules)
        self.monitor.add_alert_rule(custom_rule)
        
        assert len(self.monitor._alert_rules) == initial_count + 1
        assert "custom_test_rule" in self.monitor._alert_rules
        
        # Remove rule
        removed = self.monitor.remove_alert_rule("custom_test_rule")
        assert removed is True
        assert len(self.monitor._alert_rules) == initial_count
        assert "custom_test_rule" not in self.monitor._alert_rules


class TestStrategyPipeline:
    """Test cases for integrated StrategyPipeline"""
    
    def setup_method(self):
        self.mock_nautilus_service = AsyncMock()
        self.pipeline = StrategyPipeline(self.mock_nautilus_service)
        
        # Mock nautilus service responses
        self.mock_nautilus_service.start_engine.return_value = {
            "success": True,
            "deployment_id": "nautilus_deploy_123"
        }
        self.mock_nautilus_service.get_engine_status.return_value = {
            "status": "running",
            "performance_metrics": {
                "total_pnl": 1000.0,
                "unrealized_pnl": 100.0,
                "max_drawdown": 200.0,
                "sharpe_ratio": 1.5,
                "win_rate": 0.65
            },
            "runtime_info": {"orders_placed": 10},
            "error_log": [],
            "health_status": "healthy"
        }
    
    @pytest.mark.asyncio
    async def test_full_pipeline_deployment(self):
        """Test complete pipeline deployment workflow"""
        
        strategy_code = """
from nautilus_trader import Strategy

class TestPipelineStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.trade_count = 0
    
    def on_start(self):
        self.log.info("Pipeline test strategy started")
    
    def on_data(self, data):
        # Simple trading logic
        if self.trade_count < 10:
            self.trade_count += 1
"""
        
        strategy_config = {
            "id": "pipeline_test_strategy",
            "name": "Pipeline Test Strategy",
            "parameters": {
                "instrument": "EUR/USD.SIM",
                "trade_size": 100000,
                "max_trades": 10
            }
        }
        
        # Execute full pipeline
        result = await self.pipeline.deploy_strategy_with_pipeline(
            strategy_id="pipeline_test_strategy",
            strategy_code=strategy_code,
            strategy_config=strategy_config,
            target_environment="staging",
            commit_message="Deploy pipeline test strategy",
            author="test_user"
        )
        
        # Verify pipeline success
        assert result["success"] is True
        assert result["pipeline_id"] is not None
        assert result["version_info"] is not None
        assert result["test_results"] is not None
        assert result["deployment"] is not None
        assert result["monitoring_started"] is True
        assert len(result["errors"]) == 0
        
        # Verify version was created
        version_info = result["version_info"]
        assert version_info.strategy_id == "pipeline_test_strategy"
        assert version_info.commit_message == "Deploy pipeline test strategy"
        assert version_info.author == "test_user"
        
        # Verify tests were run
        test_results = result["test_results"]
        assert len(test_results.tests) > 0
        assert test_results.overall_score >= 0
        
        # Verify deployment was created
        deployment = result["deployment"]
        assert deployment.strategy_id == "pipeline_test_strategy"
        assert deployment.environment == DeploymentEnvironment.STAGING
    
    @pytest.mark.asyncio
    async def test_pipeline_with_test_failure(self):
        """Test pipeline behavior when tests fail"""
        
        # Invalid strategy code that should fail tests
        invalid_code = """
class BadStrategy
    def broken_method(self)
        return "This has syntax errors
"""
        
        strategy_config = {
            "id": "bad_strategy",
            "name": "Bad Strategy",
            "parameters": {}
        }
        
        result = await self.pipeline.deploy_strategy_with_pipeline(
            strategy_id="bad_strategy",
            strategy_code=invalid_code,
            strategy_config=strategy_config
        )
        
        # Pipeline should fail due to test failures
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "Tests failed" in str(result["errors"]) or "failed" in str(result["errors"]).lower()
        
        # Version should still be created
        assert result["version_info"] is not None
        
        # But deployment should not occur
        assert result["deployment"] is None
        assert result["monitoring_started"] is False
    
    def test_get_strategy_overview(self):
        """Test comprehensive strategy overview"""
        
        # First create some test data
        strategy_id = "overview_test_strategy"
        
        # Create a version
        self.pipeline.version_control.commit_version(
            strategy_id=strategy_id,
            strategy_code="class TestStrategy: pass",
            strategy_config={"param1": "value1"},
            commit_message="Test commit"
        )
        
        # Get overview
        overview = self.pipeline.get_strategy_overview(strategy_id)
        
        # Verify overview structure
        assert overview["strategy_id"] == strategy_id
        assert "versions" in overview
        assert "deployments" in overview
        assert "test_history" in overview
        assert "rollback_history" in overview
        assert "active_alerts" in overview
        assert "performance_summary" in overview
        
        # Verify versions were included
        assert len(overview["versions"]) > 0
        version = overview["versions"][0]
        assert "version_id" in version
        assert "version_number" in version
        assert "branch_name" in version
        assert "status" in version


class TestPipelineIntegration:
    """Integration tests for component interactions"""
    
    def setup_method(self):
        self.mock_nautilus_service = AsyncMock()
        self.mock_nautilus_service.start_engine.return_value = {"success": True, "deployment_id": "test_deploy"}
        self.mock_nautilus_service.get_engine_status.return_value = {"status": "running"}
        
        # Create integrated pipeline
        self.pipeline = StrategyPipeline(self.mock_nautilus_service)
    
    @pytest.mark.asyncio
    async def test_version_control_to_deployment_flow(self):
        """Test flow from version control through deployment"""
        
        strategy_id = "integration_test_strategy"
        
        # Step 1: Create version
        version = self.pipeline.version_control.commit_version(
            strategy_id=strategy_id,
            strategy_code="class IntegrationStrategy: pass",
            strategy_config={"test": True},
            commit_message="Integration test"
        )
        
        # Step 2: Deploy the version
        deployment = await self.pipeline.deployment_manager.deploy_strategy(
            strategy_id=strategy_id,
            version=version.version_number,
            target_environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        # Verify integration
        assert deployment.strategy_id == strategy_id
        assert deployment.version == version.version_number
        assert version.version_id in self.pipeline.version_control._versions
    
    @pytest.mark.asyncio
    async def test_deployment_to_monitoring_flow(self):
        """Test flow from deployment through monitoring"""
        
        strategy_id = "monitoring_test_strategy"
        
        # Create version first
        version = self.pipeline.version_control.commit_version(
            strategy_id=strategy_id,
            strategy_code="class MonitoringStrategy: pass",
            strategy_config={}
        )
        
        # Deploy
        deployment = await self.pipeline.deployment_manager.deploy_strategy(
            strategy_id=strategy_id,
            version=version.version_number,
            target_environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        # Start monitoring
        await self.pipeline.pipeline_monitor.start_monitoring(
            pipeline_id=deployment.deployment_id,
            strategy_id=strategy_id,
            version=version.version_number
        )
        
        # Verify monitoring is active
        pipeline_status = self.pipeline.pipeline_monitor.get_pipeline_status(
            deployment.deployment_id
        )
        assert pipeline_status is not None
        assert pipeline_status.strategy_id == strategy_id
        assert pipeline_status.version == version.version_number
    
    @pytest.mark.asyncio
    async def test_deployment_to_rollback_integration(self):
        """Test integration between deployment and rollback services"""
        
        strategy_id = "rollback_test_strategy"
        
        # Create two versions
        version1 = self.pipeline.version_control.commit_version(
            strategy_id=strategy_id,
            strategy_code="class StrategyV1: pass",
            strategy_config={"version": 1}
        )
        
        version2 = self.pipeline.version_control.commit_version(
            strategy_id=strategy_id,
            strategy_code="class StrategyV2: pass",
            strategy_config={"version": 2}
        )
        
        # Deploy version 2
        deployment = await self.pipeline.deployment_manager.deploy_strategy(
            strategy_id=strategy_id,
            version=version2.version_number,
            target_environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        # Execute rollback to version 1
        rollback_result = await self.pipeline.rollback_service.execute_rollback(
            strategy_id=strategy_id,
            environment="development",
            target_version=version1.version_number,
            reason="Integration test rollback"
        )
        
        # Verify rollback
        assert rollback_result["success"] is True
        
        rollback_execution = self.pipeline.rollback_service.get_rollback_execution(
            rollback_result["execution_id"]
        )
        assert rollback_execution.strategy_id == strategy_id
        assert rollback_execution.to_version == version1.version_number

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_with_rollback(self):
        """Test complete end-to-end pipeline including rollback scenario"""
        
        strategy_id = "e2e_test_strategy"
        
        # Deploy initial version
        result1 = await self.pipeline.deploy_strategy_with_pipeline(
            strategy_id=strategy_id,
            strategy_code="class E2EStrategyV1: pass",
            strategy_config={"id": strategy_id, "version": 1},
            commit_message="Deploy v1"
        )
        
        assert result1["success"] is True
        version1_number = result1["version_info"].version_number
        
        # Deploy second version
        result2 = await self.pipeline.deploy_strategy_with_pipeline(
            strategy_id=strategy_id,
            strategy_code="class E2EStrategyV2: pass", 
            strategy_config={"id": strategy_id, "version": 2},
            commit_message="Deploy v2"
        )
        
        assert result2["success"] is True
        
        # Simulate performance issue requiring rollback
        rollback_result = await self.pipeline.rollback_service.execute_rollback(
            strategy_id=strategy_id,
            environment="staging",
            target_version=version1_number,
            reason="Performance degradation in v2"
        )
        
        # Verify rollback succeeded
        assert rollback_result["success"] is True
        
        # Check overall system state
        overview = self.pipeline.get_strategy_overview(strategy_id)
        
        # Should have 2 versions
        assert len(overview["versions"]) == 2
        
        # Should have deployments
        assert len(overview["deployments"]) >= 1
        
        # Should have rollback history
        assert len(overview["rollback_history"]) == 1
        assert overview["rollback_history"][0]["trigger_type"] == "manual"


# Pytest configuration and fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
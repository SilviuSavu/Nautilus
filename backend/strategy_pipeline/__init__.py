"""
Strategy Deployment Pipeline Package

A comprehensive CI/CD pipeline for trading strategies with automated testing, deployment orchestration,
version control, rollback capabilities, and monitoring.

Components:
- StrategyTester: Automated testing framework with syntax validation, code analysis, backtesting, and performance benchmarking
- DeploymentManager: Orchestrates deployments across multiple environments with various deployment strategies
- VersionControl: Git-like versioning system with branch management, merge conflict resolution, and release tagging
- RollbackService: Automated and manual rollback capabilities with performance-based triggers
- PipelineMonitor: Comprehensive monitoring with metrics collection, alerting, and failure detection

Key Features:
- Multi-environment deployments (dev, testing, staging, production)
- Blue-green, canary, and rolling deployment strategies
- Automated rollback on performance degradation
- Comprehensive test suites with performance benchmarking
- Real-time monitoring with customizable alerts
- Version control with Git-like workflow
- Audit trails and comprehensive logging
"""

from .strategy_tester import (
    StrategyTester,
    TestStatus,
    TestType,
    TestResult,
    TestSuite,
    ValidationResult,
    CodeAnalysisResult,
    BacktestResult,
    BenchmarkResult,
    strategy_tester
)

from .deployment_manager import (
    DeploymentManager,
    DeploymentEnvironment,
    DeploymentStatus,
    DeploymentStrategy,
    ApprovalStatus,
    DeploymentConfig,
    EnvironmentConfig,
    DeploymentRequest,
    Deployment,
    DeploymentPipeline,
    deployment_manager
)

from .version_control import (
    VersionControl,
    VersionStatus,
    BranchType,
    MergeStrategy,
    ConflictType,
    VersionInfo,
    BranchInfo,
    MergeConflict,
    MergeRequest,
    ReleaseTag,
    version_control
)

from .rollback_service import (
    RollbackService,
    RollbackTriggerType,
    RollbackStatus,
    RollbackStrategy,
    RollbackTrigger,
    PerformanceSnapshot,
    RollbackPlan,
    RollbackExecution,
    RollbackValidationResult,
    rollback_service
)

from .pipeline_monitor import (
    PipelineMonitor,
    AlertSeverity,
    AlertChannel,
    MetricType,
    PipelineStage,
    MonitoringAlert,
    MetricValue,
    PipelineMetrics,
    PipelineStatus,
    AlertRule,
    NotificationConfig,
    pipeline_monitor
)

# Version information
__version__ = "1.0.0"
__author__ = "Nautilus Trading Platform"
__description__ = "Strategy Deployment Pipeline for Trading Strategies"

# Global pipeline instance with integrated services
class StrategyPipeline:
    """
    Integrated strategy deployment pipeline
    
    This class provides a unified interface to all pipeline components:
    - Testing and validation
    - Version control and branching
    - Deployment orchestration
    - Monitoring and alerting
    - Rollback capabilities
    """
    
    def __init__(self, nautilus_engine_service=None):
        # Initialize core services
        self.version_control = version_control
        self.strategy_tester = strategy_tester
        self.rollback_service = rollback_service
        self.deployment_manager = deployment_manager
        self.pipeline_monitor = pipeline_monitor
        
        # Configure service dependencies
        self.strategy_tester.nautilus_engine_service = nautilus_engine_service
        self.deployment_manager.nautilus_engine_service = nautilus_engine_service
        self.deployment_manager.strategy_tester = self.strategy_tester
        self.deployment_manager.version_control = self.version_control
        self.deployment_manager.rollback_service = self.rollback_service
        
        self.rollback_service.nautilus_engine_service = nautilus_engine_service
        self.rollback_service.version_control = self.version_control
        self.rollback_service.deployment_manager = self.deployment_manager
        
        self.pipeline_monitor.strategy_tester = self.strategy_tester
        self.pipeline_monitor.deployment_manager = self.deployment_manager
        self.pipeline_monitor.version_control = self.version_control
        self.pipeline_monitor.rollback_service = self.rollback_service
    
    async def deploy_strategy_with_pipeline(self, 
                                            strategy_id: str,
                                            strategy_code: str,
                                            strategy_config: dict,
                                            target_environment: str = "staging",
                                            commit_message: str = "Deploy strategy",
                                            author: str = "system") -> dict:
        """
        Deploy strategy through complete pipeline
        
        This is the main entry point for strategy deployments that includes:
        1. Version control commit
        2. Automated testing
        3. Deployment orchestration
        4. Monitoring setup
        5. Rollback configuration
        """
        
        pipeline_results = {
            "success": False,
            "pipeline_id": None,
            "version_info": None,
            "test_results": None,
            "deployment": None,
            "monitoring_started": False,
            "errors": []
        }
        
        try:
            # Step 1: Commit new version
            version_info = self.version_control.commit_version(
                strategy_id=strategy_id,
                strategy_code=strategy_code,
                strategy_config=strategy_config,
                commit_message=commit_message,
                author=author
            )
            pipeline_results["version_info"] = version_info
            
            # Step 2: Run comprehensive tests
            test_suite = await self.strategy_tester.run_full_test_suite(
                strategy_code=strategy_code,
                strategy_config=strategy_config
            )
            pipeline_results["test_results"] = test_suite
            
            # Step 3: Deploy if tests pass
            if test_suite.overall_score >= 70:
                deployment = await self.deployment_manager.deploy_strategy(
                    strategy_id=strategy_id,
                    version=version_info.version_number,
                    target_environment=DeploymentEnvironment(target_environment),
                    requested_by=author
                )
                pipeline_results["deployment"] = deployment
                
                # Step 4: Start monitoring
                if deployment.status == DeploymentStatus.DEPLOYED:
                    await self.pipeline_monitor.start_monitoring(
                        pipeline_id=deployment.deployment_id,
                        strategy_id=strategy_id,
                        version=version_info.version_number
                    )
                    pipeline_results["monitoring_started"] = True
                    
                    # Step 5: Configure rollback monitoring
                    await self.rollback_service.start_monitoring(
                        strategy_id=strategy_id,
                        environment=target_environment,
                        deployment_id=deployment.deployment_id
                    )
                    
                    pipeline_results["success"] = True
                    pipeline_results["pipeline_id"] = deployment.deployment_id
                else:
                    pipeline_results["errors"].append("Deployment failed")
            else:
                pipeline_results["errors"].append(f"Tests failed with score {test_suite.overall_score}")
        
        except Exception as e:
            pipeline_results["errors"].append(str(e))
        
        return pipeline_results
    
    def get_pipeline_status(self, pipeline_id: str) -> dict:
        """Get comprehensive pipeline status"""
        
        status = {
            "pipeline_id": pipeline_id,
            "deployment": None,
            "monitoring": None,
            "alerts": [],
            "metrics": None
        }
        
        # Get deployment status
        deployment = self.deployment_manager.get_deployment(pipeline_id)
        if deployment:
            status["deployment"] = {
                "status": deployment.status.value,
                "environment": deployment.environment.value,
                "version": deployment.version,
                "deployed_at": deployment.deployed_at.isoformat()
            }
        
        # Get monitoring status
        pipeline_status = self.pipeline_monitor.get_pipeline_status(pipeline_id)
        if pipeline_status:
            status["monitoring"] = {
                "current_stage": pipeline_status.current_stage.value,
                "progress_percent": pipeline_status.progress_percent,
                "status": pipeline_status.status
            }
        
        # Get active alerts
        if deployment:
            alerts = self.pipeline_monitor.list_active_alerts(
                strategy_id=deployment.strategy_id
            )
            status["alerts"] = [
                {
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts[:5]  # Latest 5 alerts
            ]
            
            # Get recent metrics
            metrics = self.pipeline_monitor.get_strategy_metrics(
                strategy_id=deployment.strategy_id,
                hours=1
            )
            if metrics:
                latest_metrics = metrics[0]
                status["metrics"] = {
                    "performance_score": latest_metrics.performance_score,
                    "resource_cpu_percent": latest_metrics.resource_cpu_percent,
                    "resource_memory_percent": latest_metrics.resource_memory_percent,
                    "deployments_successful": latest_metrics.deployments_successful,
                    "deployments_failed": latest_metrics.deployments_failed
                }
        
        return status
    
    def get_strategy_overview(self, strategy_id: str) -> dict:
        """Get comprehensive strategy overview across all pipeline components"""
        
        overview = {
            "strategy_id": strategy_id,
            "versions": [],
            "deployments": [],
            "test_history": [],
            "rollback_history": [],
            "active_alerts": [],
            "performance_summary": {}
        }
        
        # Version information
        versions = self.version_control.list_versions(strategy_id)
        overview["versions"] = [
            {
                "version_id": v.version_id,
                "version_number": v.version_number,
                "branch_name": v.branch_name,
                "status": v.status.value,
                "created_at": v.created_at.isoformat(),
                "tags": v.tags
            }
            for v in versions[:10]  # Latest 10 versions
        ]
        
        # Deployment information
        deployments = self.deployment_manager.list_deployments(strategy_id=strategy_id)
        overview["deployments"] = [
            {
                "deployment_id": d.deployment_id,
                "environment": d.environment.value,
                "version": d.version,
                "status": d.status.value,
                "deployed_at": d.deployed_at.isoformat()
            }
            for d in deployments[:10]  # Latest 10 deployments
        ]
        
        # Test history
        test_suites = self.strategy_tester.list_test_suites(strategy_id=strategy_id)
        overview["test_history"] = [
            {
                "suite_id": ts.suite_id,
                "overall_score": ts.overall_score,
                "status": ts.status.value,
                "started_at": ts.started_at.isoformat(),
                "test_count": len(ts.tests)
            }
            for ts in test_suites[:10]  # Latest 10 test suites
        ]
        
        # Rollback history
        rollbacks = self.rollback_service.list_rollback_history(strategy_id=strategy_id)
        overview["rollback_history"] = [
            {
                "execution_id": rb.execution_id,
                "trigger_type": rb.trigger_type.value,
                "from_version": rb.from_version,
                "to_version": rb.to_version,
                "success": rb.success,
                "started_at": rb.started_at.isoformat()
            }
            for rb in rollbacks[:10]  # Latest 10 rollbacks
        ]
        
        # Active alerts
        alerts = self.pipeline_monitor.list_active_alerts(strategy_id=strategy_id)
        overview["active_alerts"] = [
            {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "created_at": alert.created_at.isoformat()
            }
            for alert in alerts
        ]
        
        # Performance summary
        recent_metrics = self.pipeline_monitor.get_strategy_metrics(
            strategy_id=strategy_id,
            hours=24
        )
        
        if recent_metrics:
            avg_performance = sum(m.performance_score for m in recent_metrics) / len(recent_metrics)
            avg_cpu = sum(m.resource_cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.resource_memory_percent for m in recent_metrics) / len(recent_metrics)
            total_deployments = sum(m.deployments_successful + m.deployments_failed for m in recent_metrics)
            successful_deployments = sum(m.deployments_successful for m in recent_metrics)
            
            overview["performance_summary"] = {
                "average_performance_score": avg_performance,
                "average_cpu_utilization": avg_cpu,
                "average_memory_utilization": avg_memory,
                "deployment_success_rate": (successful_deployments / total_deployments * 100) if total_deployments > 0 else 100,
                "metrics_collected": len(recent_metrics),
                "period_hours": 24
            }
        
        return overview


# Global pipeline instance
strategy_pipeline = StrategyPipeline()

# Export main classes and instances
__all__ = [
    # Core classes
    "StrategyTester", "DeploymentManager", "VersionControl", 
    "RollbackService", "PipelineMonitor", "StrategyPipeline",
    
    # Enums
    "TestStatus", "TestType", "DeploymentEnvironment", "DeploymentStatus", 
    "DeploymentStrategy", "VersionStatus", "BranchType", "RollbackTriggerType",
    "RollbackStatus", "AlertSeverity", "AlertChannel", "MetricType",
    
    # Data models
    "TestResult", "TestSuite", "ValidationResult", "DeploymentConfig",
    "Deployment", "VersionInfo", "BranchInfo", "RollbackExecution",
    "MonitoringAlert", "PipelineMetrics", "PipelineStatus",
    
    # Global instances
    "strategy_tester", "deployment_manager", "version_control",
    "rollback_service", "pipeline_monitor", "strategy_pipeline"
]
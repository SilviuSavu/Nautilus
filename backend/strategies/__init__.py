"""
Strategy Deployment Framework - Sprint 3

Comprehensive strategy deployment infrastructure including:
- Automated testing and validation
- Deployment orchestration with multiple strategies
- Version control and rollback capabilities  
- Pipeline monitoring and alerting

This framework provides production-ready deployment capabilities for NautilusTrader strategies
with comprehensive risk management, testing, and monitoring.
"""

# Core imports
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
    RollbackStrategy as RollbackStrategyType,
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
__version__ = "3.0.0"
__author__ = "Nautilus Strategy Deployment Framework"

# Framework configuration
FRAMEWORK_CONFIG = {
    "version": __version__,
    "components": [
        "strategy_tester",
        "deployment_manager", 
        "version_control",
        "rollback_service",
        "pipeline_monitor"
    ],
    "environments": [
        "development",
        "testing", 
        "staging",
        "production"
    ],
    "deployment_strategies": [
        "direct",
        "blue_green",
        "canary",
        "rolling"
    ]
}

def initialize_framework(nautilus_engine_service=None, **kwargs):
    """
    Initialize the complete strategy deployment framework with all components integrated.
    
    Args:
        nautilus_engine_service: NautilusTrader engine service instance
        **kwargs: Additional configuration options
        
    Returns:
        dict: Initialized framework components
    """
    
    # Initialize core components with dependencies
    tester = StrategyTester(nautilus_engine_service=nautilus_engine_service)
    version_ctrl = VersionControl()
    rollback_svc = RollbackService(
        nautilus_engine_service=nautilus_engine_service,
        version_control=version_ctrl
    )
    
    deployment_mgr = DeploymentManager(
        nautilus_engine_service=nautilus_engine_service,
        strategy_tester=tester,
        version_control=version_ctrl,
        rollback_service=rollback_svc
    )
    
    # Complete rollback service initialization
    rollback_svc.deployment_manager = deployment_mgr
    
    monitor = PipelineMonitor(
        strategy_tester=tester,
        deployment_manager=deployment_mgr,
        version_control=version_ctrl,
        rollback_service=rollback_svc
    )
    
    framework_components = {
        "strategy_tester": tester,
        "deployment_manager": deployment_mgr,
        "version_control": version_ctrl,
        "rollback_service": rollback_svc,
        "pipeline_monitor": monitor,
        "config": FRAMEWORK_CONFIG
    }
    
    return framework_components

def get_framework_info():
    """Get framework information and status"""
    return {
        "name": "Nautilus Strategy Deployment Framework",
        "version": __version__,
        "description": "Production-ready strategy deployment infrastructure for NautilusTrader",
        "components": FRAMEWORK_CONFIG["components"],
        "features": [
            "Automated strategy testing and validation",
            "Multi-environment deployment pipelines", 
            "Git-like version control with branching",
            "Intelligent rollback with performance monitoring",
            "Real-time pipeline monitoring and alerting",
            "Blue-green and canary deployment strategies",
            "Risk-aware deployment processes",
            "Integration with NautilusTrader engine"
        ],
        "environments_supported": FRAMEWORK_CONFIG["environments"],
        "deployment_strategies": FRAMEWORK_CONFIG["deployment_strategies"]
    }

# Export framework initialization functions
__all__ = [
    # Core components
    "StrategyTester", "DeploymentManager", "VersionControl", 
    "RollbackService", "PipelineMonitor",
    
    # Global instances
    "strategy_tester", "deployment_manager", "version_control",
    "rollback_service", "pipeline_monitor",
    
    # Enums and types
    "TestStatus", "TestType", "DeploymentEnvironment", "DeploymentStatus",
    "DeploymentStrategy", "VersionStatus", "BranchType", "MergeStrategy",
    "RollbackTriggerType", "RollbackStatus", "AlertSeverity", "AlertChannel",
    "MetricType", "PipelineStage",
    
    # Data models
    "TestResult", "TestSuite", "ValidationResult", "CodeAnalysisResult",
    "BacktestResult", "BenchmarkResult", "DeploymentConfig", "EnvironmentConfig",
    "DeploymentRequest", "Deployment", "DeploymentPipeline", "VersionInfo",
    "BranchInfo", "MergeConflict", "MergeRequest", "ReleaseTag", "RollbackTrigger",
    "PerformanceSnapshot", "RollbackPlan", "RollbackExecution", 
    "RollbackValidationResult", "MonitoringAlert", "MetricValue", 
    "PipelineMetrics", "PipelineStatus", "AlertRule", "NotificationConfig",
    
    # Framework functions
    "initialize_framework", "get_framework_info",
    
    # Framework metadata
    "__version__", "FRAMEWORK_CONFIG"
]
"""
Strategy Deployment Manager
Orchestrates deployment workflows, environment promotion, rollback capabilities, and configuration management
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from .strategy_tester import StrategyTester, TestStatus
from .version_control import VersionControl, VersionInfo
from .rollback_service import RollbackService

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    TESTING = "testing"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    DIRECT = "direct"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"


class ApprovalStatus(Enum):
    """Approval status for deployments"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class DeploymentConfig(BaseModel):
    """Deployment configuration"""
    strategy: DeploymentStrategy = DeploymentStrategy.DIRECT
    auto_rollback: bool = True
    rollback_threshold: float = 0.05  # 5% loss threshold
    canary_percentage: Optional[float] = None
    approval_required: bool = False
    notification_channels: List[str] = []
    resource_limits: Dict[str, Any] = {}
    health_checks: Dict[str, Any] = {}


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration"""
    environment: DeploymentEnvironment
    requires_approval: bool = False
    auto_promote: bool = False
    promotion_criteria: Dict[str, Any] = {}
    resource_limits: Dict[str, str] = {}
    allowed_strategies: List[DeploymentStrategy] = []


class DeploymentRequest(BaseModel):
    """Deployment request"""
    request_id: str
    strategy_id: str
    version: str
    source_environment: Optional[DeploymentEnvironment] = None
    target_environment: DeploymentEnvironment
    deployment_config: DeploymentConfig
    requested_by: str
    requested_at: datetime
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING


class Deployment(BaseModel):
    """Active deployment"""
    deployment_id: str
    request_id: str
    strategy_id: str
    version: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    deployment_config: DeploymentConfig
    nautilus_deployment_id: Optional[str] = None
    previous_version: Optional[str] = None
    canary_deployment_id: Optional[str] = None
    
    # Metadata
    deployed_by: str
    deployed_at: datetime
    completed_at: Optional[datetime] = None
    
    # Performance tracking
    performance_metrics: Dict[str, Any] = {}
    health_status: str = "unknown"
    error_log: List[Dict[str, Any]] = []


class DeploymentPipeline(BaseModel):
    """Complete deployment pipeline"""
    pipeline_id: str
    strategy_id: str
    version: str
    environments: List[DeploymentEnvironment]
    current_stage: int = 0
    deployments: List[Deployment] = []
    status: DeploymentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None


class DeploymentManager:
    """Strategy deployment orchestration service"""
    
    def __init__(self, 
                 nautilus_engine_service=None,
                 strategy_tester: StrategyTester = None,
                 version_control: VersionControl = None,
                 rollback_service: RollbackService = None):
        self.nautilus_engine_service = nautilus_engine_service
        self.strategy_tester = strategy_tester or StrategyTester()
        self.version_control = version_control or VersionControl()
        self.rollback_service = rollback_service or RollbackService()
        
        # State management
        self._deployment_requests: Dict[str, DeploymentRequest] = {}
        self._active_deployments: Dict[str, Deployment] = {}
        self._deployment_pipelines: Dict[str, DeploymentPipeline] = {}
        self._environment_configs: Dict[DeploymentEnvironment, EnvironmentConfig] = {}
        
        # Initialize default environment configurations
        self._initialize_environment_configs()
    
    def _initialize_environment_configs(self):
        """Initialize default environment configurations"""
        self._environment_configs[DeploymentEnvironment.DEVELOPMENT] = EnvironmentConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
            requires_approval=False,
            auto_promote=False,
            allowed_strategies=[DeploymentStrategy.DIRECT]
        )
        
        self._environment_configs[DeploymentEnvironment.TESTING] = EnvironmentConfig(
            environment=DeploymentEnvironment.TESTING,
            requires_approval=False,
            auto_promote=True,
            promotion_criteria={"test_score": 80.0, "max_test_duration": 300},
            allowed_strategies=[DeploymentStrategy.DIRECT, DeploymentStrategy.ROLLING]
        )
        
        self._environment_configs[DeploymentEnvironment.STAGING] = EnvironmentConfig(
            environment=DeploymentEnvironment.STAGING,
            requires_approval=True,
            auto_promote=False,
            promotion_criteria={"min_uptime_hours": 24, "performance_score": 75.0},
            allowed_strategies=[DeploymentStrategy.DIRECT, DeploymentStrategy.BLUE_GREEN]
        )
        
        self._environment_configs[DeploymentEnvironment.PRODUCTION] = EnvironmentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            requires_approval=True,
            auto_promote=False,
            allowed_strategies=[DeploymentStrategy.BLUE_GREEN, DeploymentStrategy.CANARY],
            resource_limits={"max_memory": "4g", "max_cpu": "2.0"}
        )
    
    async def create_deployment_request(self, 
                                        strategy_id: str, 
                                        version: str,
                                        target_environment: DeploymentEnvironment,
                                        deployment_config: DeploymentConfig = None,
                                        requested_by: str = "system") -> DeploymentRequest:
        """Create a new deployment request"""
        request_id = str(uuid.uuid4())
        
        if not deployment_config:
            deployment_config = DeploymentConfig()
        
        # Validate version exists
        version_info = self.version_control.get_version(strategy_id, version)
        if not version_info:
            raise ValueError(f"Version {version} not found for strategy {strategy_id}")
        
        # Check environment configuration
        env_config = self._environment_configs.get(target_environment)
        if not env_config:
            raise ValueError(f"Environment {target_environment.value} not configured")
        
        # Validate deployment strategy is allowed
        if deployment_config.strategy not in env_config.allowed_strategies:
            raise ValueError(f"Deployment strategy {deployment_config.strategy.value} not allowed in {target_environment.value}")
        
        deployment_request = DeploymentRequest(
            request_id=request_id,
            strategy_id=strategy_id,
            version=version,
            target_environment=target_environment,
            deployment_config=deployment_config,
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            approval_status=ApprovalStatus.PENDING if env_config.requires_approval else ApprovalStatus.APPROVED
        )
        
        self._deployment_requests[request_id] = deployment_request
        
        # Auto-approve if not required
        if not env_config.requires_approval:
            deployment_request.approval_status = ApprovalStatus.APPROVED
            deployment_request.approved_by = "system"
            deployment_request.approved_at = datetime.utcnow()
        
        logger.info(f"Created deployment request {request_id} for strategy {strategy_id} v{version} to {target_environment.value}")
        return deployment_request
    
    async def approve_deployment_request(self, 
                                         request_id: str, 
                                         approved_by: str,
                                         approved: bool = True) -> bool:
        """Approve or reject a deployment request"""
        request = self._deployment_requests.get(request_id)
        if not request:
            return False
        
        request.approval_status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.approved_by = approved_by
        request.approved_at = datetime.utcnow()
        
        if approved:
            # Trigger deployment
            await self._execute_deployment_request(request)
        
        logger.info(f"Deployment request {request_id} {'approved' if approved else 'rejected'} by {approved_by}")
        return True
    
    async def deploy_strategy(self, 
                              strategy_id: str, 
                              version: str,
                              target_environment: DeploymentEnvironment,
                              deployment_config: DeploymentConfig = None,
                              requested_by: str = "system") -> Deployment:
        """Deploy strategy directly (for development/testing)"""
        # Create and auto-approve request
        request = await self.create_deployment_request(
            strategy_id, version, target_environment, deployment_config, requested_by
        )
        
        if request.approval_status != ApprovalStatus.APPROVED:
            raise ValueError(f"Deployment requires approval")
        
        return await self._execute_deployment_request(request)
    
    async def _execute_deployment_request(self, request: DeploymentRequest) -> Deployment:
        """Execute approved deployment request"""
        deployment_id = str(uuid.uuid4())
        
        # Get current deployed version for rollback
        previous_version = await self._get_current_deployed_version(
            request.strategy_id, request.target_environment
        )
        
        deployment = Deployment(
            deployment_id=deployment_id,
            request_id=request.request_id,
            strategy_id=request.strategy_id,
            version=request.version,
            environment=request.target_environment,
            status=DeploymentStatus.TESTING,
            deployment_config=request.deployment_config,
            previous_version=previous_version,
            deployed_by=request.requested_by,
            deployed_at=datetime.utcnow()
        )
        
        self._active_deployments[deployment_id] = deployment
        
        try:
            # Step 1: Run pre-deployment tests
            if request.target_environment in [DeploymentEnvironment.STAGING, DeploymentEnvironment.PRODUCTION]:
                await self._run_pre_deployment_tests(deployment)
            
            # Step 2: Execute deployment based on strategy
            deployment.status = DeploymentStatus.DEPLOYING
            await self._execute_deployment_strategy(deployment)
            
            # Step 3: Post-deployment verification
            await self._verify_deployment(deployment)
            
            deployment.status = DeploymentStatus.DEPLOYED
            deployment.completed_at = datetime.utcnow()
            deployment.health_status = "healthy"
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {str(e)}")
            deployment.status = DeploymentStatus.FAILED
            deployment.error_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "phase": "deployment"
            })
            
            # Auto-rollback if enabled
            if request.deployment_config.auto_rollback and previous_version:
                await self._trigger_auto_rollback(deployment)
        
        return deployment
    
    async def _run_pre_deployment_tests(self, deployment: Deployment):
        """Run pre-deployment tests"""
        logger.info(f"Running pre-deployment tests for {deployment.deployment_id}")
        
        # Get strategy code and config from version control
        version_info = self.version_control.get_version(deployment.strategy_id, deployment.version)
        if not version_info:
            raise ValueError(f"Version {deployment.version} not found")
        
        # Run test suite
        test_config = {
            "backtest": {"duration_days": 30},
            "paper_trading": {"duration_minutes": 10}
        }
        
        test_suite = await self.strategy_tester.run_full_test_suite(
            version_info.strategy_code,
            version_info.strategy_config,
            test_config
        )
        
        if test_suite.status != TestStatus.PASSED or test_suite.overall_score < 70:
            raise ValueError(f"Pre-deployment tests failed: score {test_suite.overall_score}")
        
        deployment.performance_metrics["pre_deployment_test_score"] = test_suite.overall_score
    
    async def _execute_deployment_strategy(self, deployment: Deployment):
        """Execute deployment based on strategy"""
        strategy = deployment.deployment_config.strategy
        
        if strategy == DeploymentStrategy.DIRECT:
            await self._execute_direct_deployment(deployment)
        elif strategy == DeploymentStrategy.BLUE_GREEN:
            await self._execute_blue_green_deployment(deployment)
        elif strategy == DeploymentStrategy.CANARY:
            await self._execute_canary_deployment(deployment)
        elif strategy == DeploymentStrategy.ROLLING:
            await self._execute_rolling_deployment(deployment)
        else:
            raise ValueError(f"Unsupported deployment strategy: {strategy}")
    
    async def _execute_direct_deployment(self, deployment: Deployment):
        """Execute direct deployment"""
        logger.info(f"Executing direct deployment for {deployment.deployment_id}")
        
        if self.nautilus_engine_service:
            # Stop current strategy if running
            current_deployment = await self._get_current_deployment(
                deployment.strategy_id, deployment.environment
            )
            if current_deployment and current_deployment.nautilus_deployment_id:
                await self.nautilus_engine_service.stop_engine(force=True)
            
            # Deploy new version
            engine_config = await self._build_engine_config(deployment)
            result = await self.nautilus_engine_service.start_engine(engine_config)
            
            if not result.get("success"):
                raise ValueError(f"Failed to start NautilusTrader engine: {result.get('error')}")
            
            deployment.nautilus_deployment_id = result.get("deployment_id")
        else:
            # Mock deployment
            deployment.nautilus_deployment_id = str(uuid.uuid4())
            await asyncio.sleep(2)  # Simulate deployment time
    
    async def _execute_blue_green_deployment(self, deployment: Deployment):
        """Execute blue-green deployment"""
        logger.info(f"Executing blue-green deployment for {deployment.deployment_id}")
        
        # Deploy to green environment
        green_config = await self._build_engine_config(deployment, environment_suffix="_green")
        
        if self.nautilus_engine_service:
            result = await self.nautilus_engine_service.start_engine(green_config)
            if not result.get("success"):
                raise ValueError(f"Failed to start green environment: {result.get('error')}")
            
            green_deployment_id = result.get("deployment_id")
            
            # Validate green environment
            await asyncio.sleep(30)  # Wait for warmup
            health_check = await self._check_deployment_health(green_deployment_id)
            
            if not health_check.get("healthy"):
                raise ValueError(f"Green environment health check failed")
            
            # Switch traffic from blue to green
            await self._switch_traffic(deployment, green_deployment_id)
            
            deployment.nautilus_deployment_id = green_deployment_id
        else:
            # Mock blue-green deployment
            deployment.nautilus_deployment_id = str(uuid.uuid4())
            await asyncio.sleep(5)
    
    async def _execute_canary_deployment(self, deployment: Deployment):
        """Execute canary deployment"""
        logger.info(f"Executing canary deployment for {deployment.deployment_id}")
        
        canary_percentage = deployment.deployment_config.canary_percentage or 10.0
        
        # Deploy canary version
        canary_config = await self._build_engine_config(
            deployment, 
            resource_scaling=canary_percentage / 100.0
        )
        
        if self.nautilus_engine_service:
            result = await self.nautilus_engine_service.start_engine(canary_config)
            if not result.get("success"):
                raise ValueError(f"Failed to start canary deployment: {result.get('error')}")
            
            canary_deployment_id = result.get("deployment_id")
            deployment.canary_deployment_id = canary_deployment_id
            
            # Monitor canary for specified duration
            await self._monitor_canary_deployment(deployment, canary_deployment_id)
            
            # Promote canary to full deployment if successful
            await self._promote_canary_deployment(deployment)
        else:
            # Mock canary deployment
            deployment.canary_deployment_id = str(uuid.uuid4())
            deployment.nautilus_deployment_id = str(uuid.uuid4())
            await asyncio.sleep(10)
    
    async def _execute_rolling_deployment(self, deployment: Deployment):
        """Execute rolling deployment"""
        logger.info(f"Executing rolling deployment for {deployment.deployment_id}")
        
        # For single strategy deployments, rolling is similar to direct
        # In a multi-instance scenario, this would update instances incrementally
        await self._execute_direct_deployment(deployment)
    
    async def _verify_deployment(self, deployment: Deployment):
        """Verify deployment success"""
        if deployment.nautilus_deployment_id:
            # Check engine status
            if self.nautilus_engine_service:
                status = await self.nautilus_engine_service.get_engine_status()
                if status.get("status") != "running":
                    raise ValueError(f"Engine not running after deployment")
            
            # Run health checks
            health_checks = deployment.deployment_config.health_checks
            if health_checks:
                await self._run_health_checks(deployment, health_checks)
        
        logger.info(f"Deployment {deployment.deployment_id} verified successfully")
    
    async def _build_engine_config(self, 
                                   deployment: Deployment, 
                                   environment_suffix: str = "",
                                   resource_scaling: float = 1.0) -> Dict[str, Any]:
        """Build NautilusTrader engine configuration"""
        env_config = self._environment_configs[deployment.environment]
        
        config = {
            "engine_type": "live" if deployment.environment == DeploymentEnvironment.PRODUCTION else "paper",
            "instance_id": f"{deployment.strategy_id}-{deployment.version}{environment_suffix}",
            "log_level": "INFO",
            "trading_mode": "live" if deployment.environment == DeploymentEnvironment.PRODUCTION else "paper"
        }
        
        # Apply resource limits
        if env_config.resource_limits:
            config.update(env_config.resource_limits)
        
        # Apply scaling for canary deployments
        if resource_scaling != 1.0:
            if "max_memory" in config:
                memory_value = config["max_memory"].rstrip("g")
                config["max_memory"] = f"{float(memory_value) * resource_scaling:.1f}g"
            if "max_cpu" in config:
                config["max_cpu"] = str(float(config["max_cpu"]) * resource_scaling)
        
        return config
    
    async def _get_current_deployed_version(self, 
                                            strategy_id: str, 
                                            environment: DeploymentEnvironment) -> Optional[str]:
        """Get currently deployed version in environment"""
        for deployment in self._active_deployments.values():
            if (deployment.strategy_id == strategy_id and 
                deployment.environment == environment and 
                deployment.status == DeploymentStatus.DEPLOYED):
                return deployment.version
        return None
    
    async def _get_current_deployment(self, 
                                      strategy_id: str, 
                                      environment: DeploymentEnvironment) -> Optional[Deployment]:
        """Get current active deployment"""
        for deployment in self._active_deployments.values():
            if (deployment.strategy_id == strategy_id and 
                deployment.environment == environment and 
                deployment.status == DeploymentStatus.DEPLOYED):
                return deployment
        return None
    
    async def _check_deployment_health(self, deployment_id: str) -> Dict[str, Any]:
        """Check deployment health"""
        if self.nautilus_engine_service:
            return await self.nautilus_engine_service.health_check()
        
        # Mock health check
        return {"healthy": True, "status": "running"}
    
    async def _switch_traffic(self, deployment: Deployment, new_deployment_id: str):
        """Switch traffic to new deployment (blue-green)"""
        logger.info(f"Switching traffic to deployment {new_deployment_id}")
        # In a real implementation, this would update load balancer or proxy configuration
        await asyncio.sleep(1)
    
    async def _monitor_canary_deployment(self, deployment: Deployment, canary_deployment_id: str):
        """Monitor canary deployment performance"""
        logger.info(f"Monitoring canary deployment {canary_deployment_id}")
        
        # Monitor for 5 minutes
        monitoring_duration = 300
        check_interval = 30
        checks = monitoring_duration // check_interval
        
        for i in range(checks):
            health = await self._check_deployment_health(canary_deployment_id)
            if not health.get("healthy"):
                raise ValueError(f"Canary deployment health check failed")
            
            await asyncio.sleep(check_interval)
        
        logger.info(f"Canary deployment {canary_deployment_id} monitoring completed successfully")
    
    async def _promote_canary_deployment(self, deployment: Deployment):
        """Promote canary deployment to full deployment"""
        if deployment.canary_deployment_id:
            logger.info(f"Promoting canary deployment {deployment.canary_deployment_id}")
            
            # Scale up canary to full deployment
            full_config = await self._build_engine_config(deployment)
            
            # Stop canary and start full deployment
            if self.nautilus_engine_service:
                # This would scale the canary deployment to full capacity
                deployment.nautilus_deployment_id = deployment.canary_deployment_id
            
            deployment.canary_deployment_id = None
    
    async def _run_health_checks(self, deployment: Deployment, health_checks: Dict[str, Any]):
        """Run custom health checks"""
        logger.info(f"Running health checks for deployment {deployment.deployment_id}")
        
        for check_name, check_config in health_checks.items():
            # Mock health check implementation
            await asyncio.sleep(1)
            logger.info(f"Health check '{check_name}' passed")
    
    async def _trigger_auto_rollback(self, deployment: Deployment):
        """Trigger automatic rollback"""
        if deployment.previous_version and self.rollback_service:
            logger.warning(f"Triggering auto-rollback for deployment {deployment.deployment_id}")
            
            rollback_result = await self.rollback_service.execute_rollback(
                deployment.strategy_id,
                deployment.environment,
                deployment.previous_version,
                reason="auto_rollback_on_deployment_failure"
            )
            
            if rollback_result.get("success"):
                deployment.status = DeploymentStatus.ROLLED_BACK
            else:
                logger.error(f"Auto-rollback failed: {rollback_result.get('error')}")
    
    # Pipeline Management
    async def create_deployment_pipeline(self, 
                                         strategy_id: str, 
                                         version: str,
                                         environments: List[DeploymentEnvironment]) -> DeploymentPipeline:
        """Create multi-environment deployment pipeline"""
        pipeline_id = str(uuid.uuid4())
        
        pipeline = DeploymentPipeline(
            pipeline_id=pipeline_id,
            strategy_id=strategy_id,
            version=version,
            environments=environments,
            status=DeploymentStatus.PENDING,
            started_at=datetime.utcnow()
        )
        
        self._deployment_pipelines[pipeline_id] = pipeline
        
        # Start first environment deployment
        if environments:
            await self._advance_pipeline(pipeline)
        
        return pipeline
    
    async def _advance_pipeline(self, pipeline: DeploymentPipeline):
        """Advance pipeline to next environment"""
        if pipeline.current_stage >= len(pipeline.environments):
            pipeline.status = DeploymentStatus.DEPLOYED
            pipeline.completed_at = datetime.utcnow()
            return
        
        current_env = pipeline.environments[pipeline.current_stage]
        
        try:
            deployment = await self.deploy_strategy(
                pipeline.strategy_id,
                pipeline.version,
                current_env
            )
            
            pipeline.deployments.append(deployment)
            
            if deployment.status == DeploymentStatus.DEPLOYED:
                pipeline.current_stage += 1
                
                # Auto-advance if environment allows it
                env_config = self._environment_configs[current_env]
                if env_config.auto_promote and pipeline.current_stage < len(pipeline.environments):
                    await asyncio.sleep(5)  # Brief delay
                    await self._advance_pipeline(pipeline)
            else:
                pipeline.status = DeploymentStatus.FAILED
                
        except Exception as e:
            logger.error(f"Pipeline {pipeline.pipeline_id} failed at stage {pipeline.current_stage}: {str(e)}")
            pipeline.status = DeploymentStatus.FAILED
    
    # Query Methods
    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID"""
        return self._active_deployments.get(deployment_id)
    
    def list_deployments(self, 
                         strategy_id: str = None,
                         environment: DeploymentEnvironment = None,
                         status: DeploymentStatus = None) -> List[Deployment]:
        """List deployments with optional filtering"""
        deployments = list(self._active_deployments.values())
        
        if strategy_id:
            deployments = [d for d in deployments if d.strategy_id == strategy_id]
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        return sorted(deployments, key=lambda d: d.deployed_at, reverse=True)
    
    def get_deployment_pipeline(self, pipeline_id: str) -> Optional[DeploymentPipeline]:
        """Get deployment pipeline by ID"""
        return self._deployment_pipelines.get(pipeline_id)
    
    def list_deployment_requests(self, 
                                 strategy_id: str = None,
                                 status: ApprovalStatus = None) -> List[DeploymentRequest]:
        """List deployment requests"""
        requests = list(self._deployment_requests.values())
        
        if strategy_id:
            requests = [r for r in requests if r.strategy_id == strategy_id]
        if status:
            requests = [r for r in requests if r.approval_status == status]
        
        return sorted(requests, key=lambda r: r.requested_at, reverse=True)
    
    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        all_deployments = list(self._active_deployments.values())
        
        total_deployments = len(all_deployments)
        successful_deployments = len([d for d in all_deployments if d.status == DeploymentStatus.DEPLOYED])
        failed_deployments = len([d for d in all_deployments if d.status == DeploymentStatus.FAILED])
        
        # Environment breakdown
        env_stats = {}
        for env in DeploymentEnvironment:
            env_deployments = [d for d in all_deployments if d.environment == env]
            env_stats[env.value] = {
                "total": len(env_deployments),
                "deployed": len([d for d in env_deployments if d.status == DeploymentStatus.DEPLOYED]),
                "failed": len([d for d in env_deployments if d.status == DeploymentStatus.FAILED])
            }
        
        return {
            "total_deployments": total_deployments,
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "success_rate": (successful_deployments / total_deployments * 100) if total_deployments > 0 else 0,
            "environment_stats": env_stats,
            "pending_requests": len([r for r in self._deployment_requests.values() 
                                   if r.approval_status == ApprovalStatus.PENDING]),
            "active_pipelines": len([p for p in self._deployment_pipelines.values() 
                                   if p.status not in [DeploymentStatus.DEPLOYED, DeploymentStatus.FAILED]]),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
deployment_manager = DeploymentManager()
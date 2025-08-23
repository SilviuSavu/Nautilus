"""
Automated Rollback Service
Handles performance-based rollback triggers, manual rollback procedures, state restoration, and rollback validation
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RollbackTriggerType(Enum):
    """Types of rollback triggers"""
    MANUAL = "manual"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    ERROR_RATE = "error_rate"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    EXTERNAL_SIGNAL = "external_signal"
    TIMEOUT = "timeout"


class RollbackStatus(Enum):
    """Rollback execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RollbackStrategy(Enum):
    """Rollback strategies"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CANARY_FIRST = "canary_first"
    BLUE_GREEN_SWITCH = "blue_green_switch"


class RollbackTrigger(BaseModel):
    """Rollback trigger configuration"""
    trigger_type: RollbackTriggerType
    enabled: bool = True
    
    # Performance thresholds
    max_drawdown_percent: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None
    max_loss_amount: Optional[float] = None
    max_consecutive_losses: Optional[int] = None
    
    # Error rate thresholds
    max_error_rate: Optional[float] = None
    error_window_minutes: Optional[int] = None
    
    # Timeout settings
    max_deployment_time_minutes: Optional[int] = None
    max_downtime_minutes: Optional[int] = None
    
    # Health check settings
    health_check_failures: Optional[int] = None
    health_check_window_minutes: Optional[int] = None


class PerformanceSnapshot(BaseModel):
    """Performance snapshot for rollback decisions"""
    timestamp: datetime
    total_pnl: float
    unrealized_pnl: float
    drawdown_percent: float
    sharpe_ratio: Optional[float]
    win_rate: float
    total_trades: int
    error_count: int
    last_error_time: Optional[datetime]
    health_status: str


class RollbackPlan(BaseModel):
    """Rollback execution plan"""
    plan_id: str
    strategy_id: str
    environment: str
    current_version: str
    target_version: str
    rollback_strategy: RollbackStrategy
    
    # Steps
    execution_steps: List[Dict[str, Any]]
    estimated_duration_minutes: int
    
    # Validation
    validation_checks: List[str]
    rollback_verification: Dict[str, Any]


class RollbackExecution(BaseModel):
    """Rollback execution record"""
    execution_id: str
    strategy_id: str
    environment: str
    trigger_type: RollbackTriggerType
    trigger_reason: str
    
    # Version information
    from_version: str
    to_version: str
    
    # Execution details
    rollback_plan: RollbackPlan
    status: RollbackStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    success: bool = False
    error_message: Optional[str] = None
    performance_before: Optional[PerformanceSnapshot] = None
    performance_after: Optional[PerformanceSnapshot] = None
    
    # Metadata
    executed_by: str
    execution_log: List[Dict[str, Any]] = []


class RollbackValidationResult(BaseModel):
    """Rollback validation result"""
    validation_id: str
    execution_id: str
    timestamp: datetime
    
    # Validation checks
    deployment_healthy: bool
    performance_improved: bool
    error_rate_acceptable: bool
    functionality_verified: bool
    
    # Metrics
    validation_score: float
    issues_found: List[str]
    recommendations: List[str]


class RollbackService:
    """Automated rollback service for trading strategies"""
    
    def __init__(self, 
                 nautilus_engine_service=None,
                 version_control=None,
                 deployment_manager=None):
        self.nautilus_engine_service = nautilus_engine_service
        self.version_control = version_control
        self.deployment_manager = deployment_manager
        
        # State management
        self._rollback_triggers: Dict[str, List[RollbackTrigger]] = {}
        self._performance_history: Dict[str, List[PerformanceSnapshot]] = {}
        self._active_rollbacks: Dict[str, RollbackExecution] = {}
        self._rollback_history: Dict[str, RollbackExecution] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Default triggers
        self._initialize_default_triggers()
    
    def _initialize_default_triggers(self):
        """Initialize default rollback triggers"""
        default_triggers = [
            RollbackTrigger(
                trigger_type=RollbackTriggerType.PERFORMANCE_THRESHOLD,
                max_drawdown_percent=10.0,
                min_sharpe_ratio=0.5,
                max_loss_amount=5000.0,
                max_consecutive_losses=5
            ),
            RollbackTrigger(
                trigger_type=RollbackTriggerType.ERROR_RATE,
                max_error_rate=5.0,  # 5% error rate
                error_window_minutes=30
            ),
            RollbackTrigger(
                trigger_type=RollbackTriggerType.HEALTH_CHECK_FAILURE,
                health_check_failures=3,
                health_check_window_minutes=15
            ),
            RollbackTrigger(
                trigger_type=RollbackTriggerType.TIMEOUT,
                max_deployment_time_minutes=60,
                max_downtime_minutes=5
            )
        ]
        
        # Apply default triggers to all strategies
        self._rollback_triggers["default"] = default_triggers
    
    def configure_rollback_triggers(self, 
                                    strategy_id: str,
                                    triggers: List[RollbackTrigger]) -> bool:
        """Configure rollback triggers for a strategy"""
        self._rollback_triggers[strategy_id] = triggers
        
        logger.info(f"Configured {len(triggers)} rollback triggers for strategy {strategy_id}")
        return True
    
    def get_rollback_triggers(self, strategy_id: str) -> List[RollbackTrigger]:
        """Get rollback triggers for a strategy"""
        # Strategy-specific triggers take precedence over defaults
        return self._rollback_triggers.get(strategy_id, self._rollback_triggers.get("default", []))
    
    async def start_monitoring(self, 
                               strategy_id: str, 
                               environment: str,
                               deployment_id: str) -> bool:
        """Start monitoring a deployment for rollback triggers"""
        
        monitoring_key = f"{strategy_id}_{environment}"
        
        # Stop existing monitoring if any
        if monitoring_key in self._monitoring_tasks:
            self._monitoring_tasks[monitoring_key].cancel()
        
        # Start new monitoring task
        task = asyncio.create_task(
            self._monitor_deployment(strategy_id, environment, deployment_id)
        )
        self._monitoring_tasks[monitoring_key] = task
        
        logger.info(f"Started rollback monitoring for {strategy_id} in {environment}")
        return True
    
    async def stop_monitoring(self, strategy_id: str, environment: str) -> bool:
        """Stop monitoring a deployment"""
        
        monitoring_key = f"{strategy_id}_{environment}"
        
        if monitoring_key in self._monitoring_tasks:
            self._monitoring_tasks[monitoring_key].cancel()
            del self._monitoring_tasks[monitoring_key]
            
            logger.info(f"Stopped rollback monitoring for {strategy_id} in {environment}")
            return True
        
        return False
    
    async def _monitor_deployment(self, 
                                  strategy_id: str, 
                                  environment: str,
                                  deployment_id: str):
        """Monitor deployment for rollback triggers"""
        
        triggers = self.get_rollback_triggers(strategy_id)
        monitoring_interval = 30  # seconds
        
        try:
            while True:
                # Collect current performance snapshot
                snapshot = await self._collect_performance_snapshot(
                    strategy_id, environment, deployment_id
                )
                
                # Store snapshot
                key = f"{strategy_id}_{environment}"
                if key not in self._performance_history:
                    self._performance_history[key] = []
                
                self._performance_history[key].append(snapshot)
                
                # Keep only recent snapshots (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self._performance_history[key] = [
                    s for s in self._performance_history[key] 
                    if s.timestamp > cutoff_time
                ]
                
                # Check triggers
                for trigger in triggers:
                    if trigger.enabled:
                        triggered = await self._check_trigger(
                            trigger, strategy_id, environment, snapshot
                        )
                        
                        if triggered:
                            await self._trigger_rollback(
                                strategy_id, environment, trigger, snapshot
                            )
                            return  # Stop monitoring after triggering rollback
                
                await asyncio.sleep(monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for {strategy_id} in {environment}")
        except Exception as e:
            logger.error(f"Monitoring error for {strategy_id}: {str(e)}")
    
    async def _collect_performance_snapshot(self, 
                                            strategy_id: str, 
                                            environment: str,
                                            deployment_id: str) -> PerformanceSnapshot:
        """Collect current performance snapshot"""
        
        try:
            if self.nautilus_engine_service:
                # Get real performance data from NautilusTrader
                status = await self.nautilus_engine_service.get_engine_status()
                
                # Mock data for now - in production, extract from actual status
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.utcnow(),
                    total_pnl=status.get("performance_metrics", {}).get("total_pnl", 0.0),
                    unrealized_pnl=status.get("performance_metrics", {}).get("unrealized_pnl", 0.0),
                    drawdown_percent=abs(status.get("performance_metrics", {}).get("max_drawdown", 0.0)),
                    sharpe_ratio=status.get("performance_metrics", {}).get("sharpe_ratio"),
                    win_rate=status.get("performance_metrics", {}).get("win_rate", 0.0),
                    total_trades=status.get("runtime_info", {}).get("orders_placed", 0),
                    error_count=len(status.get("error_log", [])),
                    health_status=status.get("health_status", "unknown")
                )
            else:
                # Mock performance snapshot
                import random
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.utcnow(),
                    total_pnl=random.uniform(-1000, 2000),
                    unrealized_pnl=random.uniform(-200, 300),
                    drawdown_percent=random.uniform(0, 15),
                    sharpe_ratio=random.uniform(-0.5, 2.0),
                    win_rate=random.uniform(0.4, 0.8),
                    total_trades=random.randint(10, 100),
                    error_count=random.randint(0, 5),
                    health_status="healthy" if random.random() > 0.1 else "unhealthy"
                )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to collect performance snapshot: {str(e)}")
            # Return empty snapshot
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                total_pnl=0.0,
                unrealized_pnl=0.0,
                drawdown_percent=0.0,
                sharpe_ratio=None,
                win_rate=0.0,
                total_trades=0,
                error_count=999,  # High error count to trigger rollback
                health_status="error"
            )
    
    async def _check_trigger(self, 
                             trigger: RollbackTrigger,
                             strategy_id: str,
                             environment: str,
                             current_snapshot: PerformanceSnapshot) -> bool:
        """Check if a rollback trigger should fire"""
        
        key = f"{strategy_id}_{environment}"
        history = self._performance_history.get(key, [])
        
        if trigger.trigger_type == RollbackTriggerType.PERFORMANCE_THRESHOLD:
            # Check drawdown
            if (trigger.max_drawdown_percent and 
                current_snapshot.drawdown_percent > trigger.max_drawdown_percent):
                logger.warning(f"Drawdown trigger fired: {current_snapshot.drawdown_percent}% > {trigger.max_drawdown_percent}%")
                return True
            
            # Check Sharpe ratio
            if (trigger.min_sharpe_ratio and 
                current_snapshot.sharpe_ratio is not None and
                current_snapshot.sharpe_ratio < trigger.min_sharpe_ratio):
                logger.warning(f"Sharpe ratio trigger fired: {current_snapshot.sharpe_ratio} < {trigger.min_sharpe_ratio}")
                return True
            
            # Check absolute loss
            if (trigger.max_loss_amount and 
                current_snapshot.total_pnl < -trigger.max_loss_amount):
                logger.warning(f"Loss amount trigger fired: ${current_snapshot.total_pnl} < -${trigger.max_loss_amount}")
                return True
        
        elif trigger.trigger_type == RollbackTriggerType.ERROR_RATE:
            if trigger.error_window_minutes and history:
                # Calculate error rate in window
                window_start = datetime.utcnow() - timedelta(minutes=trigger.error_window_minutes)
                window_snapshots = [s for s in history if s.timestamp > window_start]
                
                if window_snapshots:
                    total_errors = sum(s.error_count for s in window_snapshots)
                    total_operations = sum(s.total_trades for s in window_snapshots)
                    
                    if total_operations > 0:
                        error_rate = (total_errors / total_operations) * 100
                        if error_rate > trigger.max_error_rate:
                            logger.warning(f"Error rate trigger fired: {error_rate:.2f}% > {trigger.max_error_rate}%")
                            return True
        
        elif trigger.trigger_type == RollbackTriggerType.HEALTH_CHECK_FAILURE:
            if trigger.health_check_window_minutes and history:
                # Check for consecutive health check failures
                window_start = datetime.utcnow() - timedelta(minutes=trigger.health_check_window_minutes)
                window_snapshots = [s for s in history if s.timestamp > window_start]
                
                consecutive_failures = 0
                for snapshot in reversed(window_snapshots):  # Most recent first
                    if snapshot.health_status != "healthy":
                        consecutive_failures += 1
                    else:
                        break
                
                if consecutive_failures >= trigger.health_check_failures:
                    logger.warning(f"Health check trigger fired: {consecutive_failures} consecutive failures")
                    return True
        
        return False
    
    async def _trigger_rollback(self, 
                                strategy_id: str,
                                environment: str,
                                trigger: RollbackTrigger,
                                snapshot: PerformanceSnapshot):
        """Trigger automatic rollback"""
        
        logger.warning(f"Triggering automatic rollback for {strategy_id} in {environment}")
        
        # Find previous stable version
        target_version = await self._find_stable_version(strategy_id, environment)
        if not target_version:
            logger.error(f"No stable version found for rollback of {strategy_id}")
            return
        
        # Execute rollback
        await self.execute_rollback(
            strategy_id=strategy_id,
            environment=environment,
            target_version=target_version,
            reason=f"Automatic rollback triggered by {trigger.trigger_type.value}",
            executed_by="system"
        )
    
    async def execute_rollback(self, 
                               strategy_id: str,
                               environment: str,
                               target_version: str,
                               reason: str = "Manual rollback",
                               executed_by: str = "system",
                               rollback_strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE) -> Dict[str, Any]:
        """Execute strategy rollback"""
        
        execution_id = str(uuid.uuid4())
        
        # Get current deployment info
        current_deployment = None
        if self.deployment_manager:
            deployments = self.deployment_manager.list_deployments(
                strategy_id=strategy_id, 
                environment=environment
            )
            current_deployment = next((d for d in deployments if d.status == "deployed"), None)
        
        current_version = current_deployment.version if current_deployment else "unknown"
        
        # Create rollback execution record
        rollback_execution = RollbackExecution(
            execution_id=execution_id,
            strategy_id=strategy_id,
            environment=environment,
            trigger_type=RollbackTriggerType.MANUAL,
            trigger_reason=reason,
            from_version=current_version,
            to_version=target_version,
            rollback_plan=await self._create_rollback_plan(
                strategy_id, environment, current_version, target_version, rollback_strategy
            ),
            status=RollbackStatus.EXECUTING,
            started_at=datetime.utcnow(),
            executed_by=executed_by
        )
        
        # Collect performance before rollback
        rollback_execution.performance_before = await self._collect_performance_snapshot(
            strategy_id, environment, current_deployment.nautilus_deployment_id if current_deployment else ""
        )
        
        self._active_rollbacks[execution_id] = rollback_execution
        
        try:
            # Execute rollback steps
            await self._execute_rollback_plan(rollback_execution)
            
            # Validate rollback
            validation_result = await self._validate_rollback(rollback_execution)
            
            if validation_result.validation_score >= 80:
                rollback_execution.status = RollbackStatus.COMPLETED
                rollback_execution.success = True
                rollback_execution.completed_at = datetime.utcnow()
                
                logger.info(f"Rollback {execution_id} completed successfully")
            else:
                rollback_execution.status = RollbackStatus.FAILED
                rollback_execution.error_message = f"Rollback validation failed: score {validation_result.validation_score}"
                
                logger.error(f"Rollback {execution_id} validation failed")
        
        except Exception as e:
            rollback_execution.status = RollbackStatus.FAILED
            rollback_execution.error_message = str(e)
            rollback_execution.completed_at = datetime.utcnow()
            
            logger.error(f"Rollback {execution_id} failed: {str(e)}")
        
        finally:
            # Move to history
            self._rollback_history[execution_id] = rollback_execution
            if execution_id in self._active_rollbacks:
                del self._active_rollbacks[execution_id]
        
        return {
            "execution_id": execution_id,
            "success": rollback_execution.success,
            "status": rollback_execution.status.value,
            "error_message": rollback_execution.error_message
        }
    
    async def _find_stable_version(self, strategy_id: str, environment: str) -> Optional[str]:
        """Find a stable version for rollback"""
        
        if self.version_control:
            # Get version history
            versions = self.version_control.list_versions(
                strategy_id=strategy_id, 
                branch_name="main",
                status=None
            )
            
            # Find tagged versions (more stable)
            for version in versions:
                if version.tags:
                    return version.version_id
            
            # Fall back to recent committed version
            if versions:
                return versions[0].version_id
        
        # Mock stable version
        return f"stable-{strategy_id}-v1.0.0"
    
    async def _create_rollback_plan(self, 
                                    strategy_id: str,
                                    environment: str,
                                    current_version: str,
                                    target_version: str,
                                    rollback_strategy: RollbackStrategy) -> RollbackPlan:
        """Create rollback execution plan"""
        
        plan_id = str(uuid.uuid4())
        
        if rollback_strategy == RollbackStrategy.IMMEDIATE:
            execution_steps = [
                {"step": "stop_current_deployment", "description": "Stop current strategy deployment"},
                {"step": "backup_state", "description": "Backup current strategy state"},
                {"step": "deploy_target_version", "description": f"Deploy target version {target_version}"},
                {"step": "verify_deployment", "description": "Verify successful deployment"},
                {"step": "restore_state", "description": "Restore compatible state data"}
            ]
            estimated_duration = 10
            
        elif rollback_strategy == RollbackStrategy.BLUE_GREEN_SWITCH:
            execution_steps = [
                {"step": "deploy_target_version", "description": "Deploy target version to green environment"},
                {"step": "validate_green", "description": "Validate green environment"},
                {"step": "switch_traffic", "description": "Switch traffic to green environment"},
                {"step": "stop_blue", "description": "Stop blue environment"}
            ]
            estimated_duration = 20
        
        else:  # Default to immediate
            execution_steps = [
                {"step": "immediate_stop", "description": "Immediately stop current deployment"},
                {"step": "emergency_deploy", "description": "Emergency deploy target version"}
            ]
            estimated_duration = 5
        
        return RollbackPlan(
            plan_id=plan_id,
            strategy_id=strategy_id,
            environment=environment,
            current_version=current_version,
            target_version=target_version,
            rollback_strategy=rollback_strategy,
            execution_steps=execution_steps,
            estimated_duration_minutes=estimated_duration,
            validation_checks=[
                "deployment_health",
                "basic_functionality",
                "performance_baseline"
            ],
            rollback_verification={
                "verify_version": target_version,
                "verify_health": True,
                "performance_check_duration": 300  # 5 minutes
            }
        )
    
    async def _execute_rollback_plan(self, rollback_execution: RollbackExecution):
        """Execute the rollback plan"""
        
        plan = rollback_execution.rollback_plan
        
        for i, step in enumerate(plan.execution_steps):
            step_start = datetime.utcnow()
            
            try:
                # Execute step
                await self._execute_rollback_step(step, rollback_execution)
                
                # Log success
                rollback_execution.execution_log.append({
                    "step_number": i + 1,
                    "step": step["step"],
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_seconds": (datetime.utcnow() - step_start).total_seconds()
                })
                
            except Exception as e:
                # Log failure
                rollback_execution.execution_log.append({
                    "step_number": i + 1,
                    "step": step["step"],
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_seconds": (datetime.utcnow() - step_start).total_seconds()
                })
                raise
    
    async def _execute_rollback_step(self, 
                                     step: Dict[str, Any], 
                                     rollback_execution: RollbackExecution):
        """Execute a single rollback step"""
        
        step_type = step["step"]
        
        if step_type == "stop_current_deployment":
            if self.nautilus_engine_service:
                await self.nautilus_engine_service.stop_engine(force=True)
            await asyncio.sleep(2)
        
        elif step_type == "backup_state":
            # Mock state backup
            await asyncio.sleep(1)
        
        elif step_type == "deploy_target_version":
            if self.deployment_manager:
                await self.deployment_manager.deploy_strategy(
                    rollback_execution.strategy_id,
                    rollback_execution.to_version,
                    rollback_execution.environment
                )
            await asyncio.sleep(5)
        
        elif step_type == "verify_deployment":
            if self.nautilus_engine_service:
                status = await self.nautilus_engine_service.get_engine_status()
                if status.get("status") != "running":
                    raise ValueError("Deployment verification failed")
            await asyncio.sleep(2)
        
        elif step_type == "restore_state":
            # Mock state restoration
            await asyncio.sleep(1)
        
        else:
            # Generic step execution
            await asyncio.sleep(1)
    
    async def _validate_rollback(self, rollback_execution: RollbackExecution) -> RollbackValidationResult:
        """Validate rollback success"""
        
        validation_id = str(uuid.uuid4())
        
        # Wait for deployment to stabilize
        await asyncio.sleep(30)
        
        # Collect post-rollback performance
        rollback_execution.performance_after = await self._collect_performance_snapshot(
            rollback_execution.strategy_id,
            rollback_execution.environment,
            "rollback-deployment"
        )
        
        # Run validation checks
        deployment_healthy = rollback_execution.performance_after.health_status == "healthy"
        performance_improved = (
            rollback_execution.performance_after.total_pnl > 
            rollback_execution.performance_before.total_pnl
        ) if rollback_execution.performance_before else True
        
        error_rate_acceptable = rollback_execution.performance_after.error_count < 3
        functionality_verified = True  # Mock verification
        
        # Calculate validation score
        checks = [deployment_healthy, performance_improved, error_rate_acceptable, functionality_verified]
        validation_score = (sum(checks) / len(checks)) * 100
        
        issues_found = []
        if not deployment_healthy:
            issues_found.append("Deployment health check failed")
        if not performance_improved:
            issues_found.append("Performance did not improve after rollback")
        if not error_rate_acceptable:
            issues_found.append("Error rate still too high")
        
        return RollbackValidationResult(
            validation_id=validation_id,
            execution_id=rollback_execution.execution_id,
            timestamp=datetime.utcnow(),
            deployment_healthy=deployment_healthy,
            performance_improved=performance_improved,
            error_rate_acceptable=error_rate_acceptable,
            functionality_verified=functionality_verified,
            validation_score=validation_score,
            issues_found=issues_found,
            recommendations=["Monitor performance closely", "Consider gradual re-deployment of new version"]
        )
    
    # Query Methods
    def get_rollback_execution(self, execution_id: str) -> Optional[RollbackExecution]:
        """Get rollback execution by ID"""
        return self._rollback_history.get(execution_id) or self._active_rollbacks.get(execution_id)
    
    def list_rollback_history(self, 
                              strategy_id: str = None,
                              environment: str = None,
                              status: RollbackStatus = None) -> List[RollbackExecution]:
        """List rollback execution history"""
        
        executions = list(self._rollback_history.values()) + list(self._active_rollbacks.values())
        
        if strategy_id:
            executions = [e for e in executions if e.strategy_id == strategy_id]
        
        if environment:
            executions = [e for e in executions if e.environment == environment]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return sorted(executions, key=lambda e: e.started_at, reverse=True)
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback service statistics"""
        
        all_executions = list(self._rollback_history.values()) + list(self._active_rollbacks.values())
        
        total_rollbacks = len(all_executions)
        successful_rollbacks = len([e for e in all_executions if e.success])
        failed_rollbacks = len([e for e in all_executions if e.status == RollbackStatus.FAILED])
        active_rollbacks = len([e for e in all_executions if e.status == RollbackStatus.EXECUTING])
        
        # Trigger statistics
        trigger_stats = {}
        for trigger_type in RollbackTriggerType:
            count = len([e for e in all_executions if e.trigger_type == trigger_type])
            trigger_stats[trigger_type.value] = count
        
        return {
            "total_rollbacks": total_rollbacks,
            "successful_rollbacks": successful_rollbacks,
            "failed_rollbacks": failed_rollbacks,
            "active_rollbacks": active_rollbacks,
            "success_rate": (successful_rollbacks / total_rollbacks * 100) if total_rollbacks > 0 else 0,
            "trigger_statistics": trigger_stats,
            "active_monitoring": len(self._monitoring_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
rollback_service = RollbackService()
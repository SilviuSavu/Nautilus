"""
Strategy Deployment Service
Handles deployment logic, approval workflow, and strategy lifecycle management
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DeploymentStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class StrategyState(str, Enum):
    DEPLOYING = "deploying"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    EMERGENCY_STOPPED = "emergency_stopped"

class DeploymentService:
    """Service for managing strategy deployments"""
    
    def __init__(self):
        self.deployments: Dict[str, Dict] = {}
        self.live_strategies: Dict[str, Dict] = {}
        self.approval_workflows: Dict[str, List[Dict]] = {}
        self.rollout_monitors: Dict[str, Dict] = {}
        
    async def create_deployment_request(
        self,
        strategy_id: str,
        version: str,
        proposed_config: Dict[str, Any],
        rollout_plan: Dict[str, Any],
        backtest_results: Optional[Dict[str, Any]] = None,
        risk_assessment: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new deployment request"""
        
        deployment_id = str(uuid.uuid4())
        
        # Validate deployment request
        validation_errors = await self._validate_deployment_request(
            strategy_id, version, proposed_config, rollout_plan
        )
        
        if validation_errors:
            raise ValueError(f"Deployment validation failed: {validation_errors}")
        
        # Perform automated risk assessment if not provided
        if not risk_assessment:
            risk_assessment = await self._perform_automated_risk_assessment(
                strategy_id, proposed_config, backtest_results
            )
        
        # Create approval workflow
        approval_chain = self._create_approval_workflow(
            deployment_id, risk_assessment.get("risk_level", "medium")
        )
        
        deployment = {
            "deployment_id": deployment_id,
            "strategy_id": strategy_id,
            "version": version,
            "deployment_config": proposed_config,
            "rollout_plan": rollout_plan,
            "backtest_results": backtest_results,
            "risk_assessment": risk_assessment,
            "status": DeploymentStatus.PENDING_APPROVAL,
            "created_by": "current_user",  # In production, get from auth context
            "created_at": datetime.utcnow(),
            "approval_chain": approval_chain,
            "deployment_history": []
        }
        
        self.deployments[deployment_id] = deployment
        self.approval_workflows[deployment_id] = approval_chain
        
        logger.info(f"Created deployment request {deployment_id} for strategy {strategy_id}")
        
        # Trigger approval notifications
        await self._notify_approvers(deployment_id)
        
        return deployment_id
    
    async def approve_deployment(
        self,
        deployment_id: str,
        approver_id: str,
        approved: bool,
        comments: Optional[str] = None
    ) -> bool:
        """Approve or reject a deployment request"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment["status"] != DeploymentStatus.PENDING_APPROVAL:
            raise ValueError(f"Deployment {deployment_id} is not pending approval")
        
        # Find and update the approval
        approval_chain = self.approval_workflows.get(deployment_id, [])
        approval_updated = False
        
        for approval in approval_chain:
            if approval["approver_id"] == approver_id and approval["status"] == "pending":
                approval["status"] = "approved" if approved else "rejected"
                approval["approved_at"] = datetime.utcnow()
                approval["comments"] = comments
                approval_updated = True
                break
        
        if not approval_updated:
            raise ValueError(f"No pending approval found for approver {approver_id}")
        
        # Check if all approvals are complete
        if approved:
            all_approved = all(
                approval["status"] == "approved" 
                for approval in approval_chain
            )
            
            if all_approved:
                deployment["status"] = DeploymentStatus.APPROVED
                deployment["approved_at"] = datetime.utcnow()
                logger.info(f"Deployment {deployment_id} fully approved")
                
                # Trigger auto-deployment if configured
                await self._check_auto_deployment(deployment_id)
        else:
            deployment["status"] = "rejected"
            deployment["rejected_at"] = datetime.utcnow()
            deployment["rejection_reason"] = comments
            logger.info(f"Deployment {deployment_id} rejected by {approver_id}")
        
        return True
    
    async def deploy_strategy(self, deployment_id: str, force_restart: bool = False) -> str:
        """Deploy an approved strategy"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment["status"] != DeploymentStatus.APPROVED:
            raise ValueError(f"Deployment must be approved before deployment")
        
        strategy_instance_id = str(uuid.uuid4())
        
        try:
            # Update deployment status
            deployment["status"] = DeploymentStatus.DEPLOYING
            deployment["deployed_at"] = datetime.utcnow()
            deployment["strategy_instance_id"] = strategy_instance_id
            
            # Create live strategy instance
            live_strategy = await self._create_live_strategy_instance(
                deployment, strategy_instance_id
            )
            
            self.live_strategies[strategy_instance_id] = live_strategy
            
            # Start Docker container with strategy
            await self._deploy_to_container(deployment, strategy_instance_id)
            
            # Initialize rollout monitoring
            await self._initialize_rollout_monitoring(deployment_id, strategy_instance_id)
            
            # Update statuses
            deployment["status"] = DeploymentStatus.DEPLOYED
            live_strategy["state"] = StrategyState.RUNNING
            
            logger.info(f"Successfully deployed strategy {deployment['strategy_id']} as {strategy_instance_id}")
            
            return strategy_instance_id
            
        except Exception as e:
            # Handle deployment failure
            deployment["status"] = DeploymentStatus.FAILED
            deployment["failure_reason"] = str(e)
            logger.error(f"Deployment {deployment_id} failed: {e}")
            raise
    
    async def control_strategy(
        self,
        strategy_instance_id: str,
        action: str,
        reason: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Control a live strategy (pause, resume, stop, emergency_stop)"""
        
        strategy = self.live_strategies.get(strategy_instance_id)
        if not strategy:
            raise ValueError(f"Strategy instance {strategy_instance_id} not found")
        
        previous_state = strategy["state"]
        
        if action == "pause":
            if strategy["state"] != StrategyState.RUNNING:
                raise ValueError("Strategy must be running to pause")
            strategy["state"] = StrategyState.PAUSED
            await self._send_control_command(strategy_instance_id, "pause")
            
        elif action == "resume":
            if strategy["state"] != StrategyState.PAUSED:
                raise ValueError("Strategy must be paused to resume")
            strategy["state"] = StrategyState.RUNNING
            await self._send_control_command(strategy_instance_id, "resume")
            
        elif action == "stop":
            strategy["state"] = StrategyState.STOPPED
            await self._send_control_command(strategy_instance_id, "stop")
            
        elif action == "emergency_stop":
            strategy["state"] = StrategyState.EMERGENCY_STOPPED
            await self._send_control_command(strategy_instance_id, "emergency_stop", force=True)
            
            # Trigger emergency procedures
            await self._execute_emergency_procedures(strategy_instance_id, reason)
        
        # Log the action
        strategy.setdefault("control_history", []).append({
            "action": action,
            "reason": reason,
            "timestamp": datetime.utcnow(),
            "previous_state": previous_state,
            "new_state": strategy["state"],
            "force": force
        })
        
        logger.info(f"Strategy {strategy_instance_id} {action}: {reason}")
        
        return {
            "success": True,
            "new_state": strategy["state"],
            "message": f"Strategy {action} completed",
            "executed_at": datetime.utcnow()
        }
    
    async def rollback_deployment(
        self,
        deployment_id: str,
        target_version: str,
        reason: str,
        immediate: bool = False
    ) -> str:
        """Rollback a deployment to a previous version"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        rollback_id = str(uuid.uuid4())
        
        # Find live strategy instance
        strategy_instance_id = deployment.get("strategy_instance_id")
        if strategy_instance_id:
            strategy = self.live_strategies.get(strategy_instance_id)
            if strategy:
                # Stop current strategy
                await self.control_strategy(
                    strategy_instance_id, "stop", f"Rollback to {target_version}"
                )
        
        # Create rollback plan
        rollback_plan = await self._create_rollback_plan(
            deployment_id, target_version, immediate
        )
        
        # Execute rollback
        if immediate:
            await self._execute_rollback(rollback_id, rollback_plan)
        else:
            # Schedule rollback execution
            asyncio.create_task(self._execute_rollback(rollback_id, rollback_plan))
        
        # Update deployment status
        deployment["status"] = DeploymentStatus.ROLLED_BACK
        deployment["rollback_id"] = rollback_id
        deployment["rollback_reason"] = reason
        deployment["rollback_at"] = datetime.utcnow()
        
        logger.info(f"Initiated rollback {rollback_id} for deployment {deployment_id}")
        
        return rollback_id
    
    async def get_live_strategies(self) -> List[Dict[str, Any]]:
        """Get all live strategies"""
        return list(self.live_strategies.values())
    
    async def get_live_strategy(self, strategy_instance_id: str) -> Dict[str, Any]:
        """Get specific live strategy"""
        strategy = self.live_strategies.get(strategy_instance_id)
        if not strategy:
            raise ValueError(f"Strategy instance {strategy_instance_id} not found")
        return strategy
    
    async def get_strategy_metrics(self, strategy_instance_id: str) -> Dict[str, Any]:
        """Get real-time strategy metrics"""
        strategy = self.live_strategies.get(strategy_instance_id)
        if not strategy:
            raise ValueError(f"Strategy instance {strategy_instance_id} not found")
        
        # Update metrics from live data sources
        await self._update_strategy_metrics(strategy_instance_id)
        
        return {
            "performanceMetrics": strategy["performance_metrics"],
            "riskMetrics": strategy["risk_metrics"],
            "positions": [strategy["current_position"]],
            "alerts": strategy.get("alerts", []),
            "timestamp": datetime.utcnow(),
            "healthStatus": strategy["health_status"]
        }
    
    # Private helper methods
    
    async def _validate_deployment_request(
        self, strategy_id: str, version: str, config: Dict, rollout_plan: Dict
    ) -> List[str]:
        """Validate deployment request"""
        errors = []
        
        if not strategy_id:
            errors.append("Strategy ID is required")
        
        if not version:
            errors.append("Version is required")
        
        if not rollout_plan.get("phases"):
            errors.append("Rollout plan must have at least one phase")
        
        # Validate risk limits
        risk_engine = config.get("risk_engine", {})
        if risk_engine.get("max_daily_loss", 0) <= 0:
            errors.append("Max daily loss must be greater than 0")
        
        return errors
    
    async def _perform_automated_risk_assessment(
        self, strategy_id: str, config: Dict, backtest_results: Optional[Dict]
    ) -> Dict[str, Any]:
        """Perform automated risk assessment"""
        
        risk_level = "low"
        warnings = []
        blockers = []
        recommendations = []
        
        # Analyze backtest results
        if backtest_results:
            max_dd = backtest_results.get("max_drawdown", 0)
            sharpe = backtest_results.get("sharpe_ratio", 0)
            
            if max_dd > 0.15:
                risk_level = "high"
                warnings.append("Maximum drawdown exceeds 15%")
            
            if sharpe < 1.0:
                warnings.append("Sharpe ratio below 1.0")
            
            if max_dd > 0.25:
                blockers.append("Maximum drawdown exceeds 25% limit")
        
        # Analyze configuration
        max_positions = config.get("risk_engine", {}).get("position_limits", {}).get("max_positions", 0)
        if max_positions > 10:
            warnings.append("High number of simultaneous positions")
        
        recommendations.append("Monitor correlation with existing positions")
        
        return {
            "risk_level": risk_level,
            "portfolio_impact": "medium" if risk_level == "high" else "low",
            "correlation_risk": "low",
            "max_drawdown_estimate": backtest_results.get("max_drawdown", 0.05) if backtest_results else 0.05,
            "var_estimate": 0.03,
            "liquidity_risk": "low",
            "warnings": warnings,
            "blockers": blockers,
            "recommendations": recommendations
        }
    
    def _create_approval_workflow(self, deployment_id: str, risk_level: str) -> List[Dict]:
        """Create approval workflow based on risk level"""
        
        workflow = [
            {
                "approval_id": str(uuid.uuid4()),
                "deployment_id": deployment_id,
                "approver_id": "senior_trader",
                "approver_name": "Senior Trader",
                "approval_level": 1,
                "status": "pending",
                "required_role": "senior_trader"
            }
        ]
        
        # Add risk manager approval for medium/high risk
        if risk_level in ["medium", "high"]:
            workflow.append({
                "approval_id": str(uuid.uuid4()),
                "deployment_id": deployment_id,
                "approver_id": "risk_manager",
                "approver_name": "Risk Manager",
                "approval_level": 2,
                "status": "pending",
                "required_role": "risk_manager"
            })
        
        # Add compliance approval for high risk
        if risk_level == "high":
            workflow.append({
                "approval_id": str(uuid.uuid4()),
                "deployment_id": deployment_id,
                "approver_id": "compliance",
                "approver_name": "Compliance Officer",
                "approval_level": 3,
                "status": "pending",
                "required_role": "compliance"
            })
        
        return workflow
    
    async def _notify_approvers(self, deployment_id: str):
        """Notify approvers of pending deployment"""
        # In production, this would send notifications via email, Slack, etc.
        logger.info(f"Notifying approvers for deployment {deployment_id}")
    
    async def _check_auto_deployment(self, deployment_id: str):
        """Check if deployment should be automatically deployed"""
        # Could implement auto-deployment logic here
        pass
    
    async def _create_live_strategy_instance(
        self, deployment: Dict, strategy_instance_id: str
    ) -> Dict[str, Any]:
        """Create live strategy instance"""
        
        return {
            "strategy_instance_id": strategy_instance_id,
            "deployment_id": deployment["deployment_id"],
            "strategy_id": deployment["strategy_id"],
            "version": deployment["version"],
            "state": StrategyState.DEPLOYING,
            "current_position": {
                "instrument": "EURUSD",
                "side": "FLAT",
                "quantity": 0,
                "avg_price": 0,
                "market_value": 0,
                "unrealized_pnl": 0,
                "last_updated": datetime.utcnow()
            },
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "performance_metrics": {
                "total_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "daily_pnl": 0.0,
                "weekly_pnl": 0.0,
                "monthly_pnl": 0.0,
                "total_volume": 0.0,
                "avg_trade_size": 0.0,
                "fill_rate": 1.0,
                "slippage_avg": 0.0,
                "execution_quality": 1.0,
                "vs_backtest_deviation": 0.0,
                "last_updated": datetime.utcnow()
            },
            "risk_metrics": {
                "current_drawdown": 0.0,
                "max_drawdown_today": 0.0,
                "value_at_risk": 0.0,
                "expected_shortfall": 0.0,
                "leverage_ratio": 0.0,
                "concentration_risk": 0.0,
                "correlation_to_portfolio": 0.0,
                "last_risk_check": datetime.utcnow()
            },
            "health_status": {
                "overall": "healthy",
                "heartbeat": "active",
                "data_feed": "connected",
                "order_execution": "normal",
                "risk_compliance": "compliant",
                "last_health_check": datetime.utcnow()
            },
            "alerts": [],
            "last_heartbeat": datetime.utcnow(),
            "started_at": datetime.utcnow()
        }
    
    async def _deploy_to_container(self, deployment: Dict, strategy_instance_id: str):
        """Deploy strategy to Docker container"""
        
        # Generate strategy configuration file
        config_path = f"/app/strategies/deployment_configs/strategy_config_{strategy_instance_id}.json"
        
        # Create deployment command
        container_name = deployment["deployment_config"]["environment"]["container_name"]
        
        # In production, this would execute actual Docker commands
        logger.info(f"Deploying strategy {strategy_instance_id} to container {container_name}")
        
        # Simulate deployment time
        await asyncio.sleep(2)
    
    async def _initialize_rollout_monitoring(self, deployment_id: str, strategy_instance_id: str):
        """Initialize rollout monitoring"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return
        
        rollout_plan = deployment["rollout_plan"]
        
        monitor = {
            "deployment_id": deployment_id,
            "strategy_instance_id": strategy_instance_id,
            "current_phase": rollout_plan.get("current_phase", 0),
            "phase_start_time": datetime.utcnow(),
            "auto_advance": True,
            "phase_history": []
        }
        
        self.rollout_monitors[deployment_id] = monitor
        
        # Start rollout monitoring task
        asyncio.create_task(self._monitor_rollout_progress(deployment_id))
    
    async def _monitor_rollout_progress(self, deployment_id: str):
        """Monitor rollout progress and auto-advance phases"""
        
        while deployment_id in self.rollout_monitors:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                monitor = self.rollout_monitors.get(deployment_id)
                if not monitor:
                    break
                
                deployment = self.deployments.get(deployment_id)
                if not deployment or deployment["status"] != DeploymentStatus.DEPLOYED:
                    break
                
                # Check if current phase criteria are met
                await self._check_phase_advancement(deployment_id)
                
            except Exception as e:
                logger.error(f"Error monitoring rollout {deployment_id}: {e}")
                await asyncio.sleep(60)
    
    async def _check_phase_advancement(self, deployment_id: str):
        """Check if current phase can be advanced"""
        
        monitor = self.rollout_monitors.get(deployment_id)
        if not monitor:
            return
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return
        
        rollout_plan = deployment["rollout_plan"]
        phases = rollout_plan["phases"]
        current_phase_idx = monitor["current_phase"]
        
        if current_phase_idx >= len(phases):
            logger.info(f"Rollout {deployment_id} completed")
            return
        
        phase = phases[current_phase_idx]
        strategy_instance_id = monitor["strategy_instance_id"]
        strategy = self.live_strategies.get(strategy_instance_id)
        
        if not strategy:
            return
        
        # Check success criteria
        criteria_met = await self._evaluate_phase_criteria(phase, strategy, monitor)
        
        if criteria_met and monitor["auto_advance"]:
            await self._advance_rollout_phase(deployment_id)
    
    async def _evaluate_phase_criteria(self, phase: Dict, strategy: Dict, monitor: Dict) -> bool:
        """Evaluate if phase success criteria are met"""
        
        criteria = phase.get("success_criteria", {})
        
        # Check minimum trades
        if criteria.get("min_trades"):
            if strategy["performance_metrics"]["total_trades"] < criteria["min_trades"]:
                return False
        
        # Check maximum drawdown
        if criteria.get("max_drawdown"):
            if strategy["risk_metrics"]["current_drawdown"] > (criteria["max_drawdown"] * 100):
                return False
        
        # Check P&L threshold
        if criteria.get("pnl_threshold"):
            total_pnl = strategy["realized_pnl"] + strategy["unrealized_pnl"]
            if total_pnl < (criteria["pnl_threshold"] * 1000):
                return False
        
        # Check time duration
        phase_duration = phase.get("duration", -1)
        if phase_duration > 0:
            elapsed = (datetime.utcnow() - monitor["phase_start_time"]).total_seconds()
            if elapsed < phase_duration:
                return False
        
        return True
    
    async def _advance_rollout_phase(self, deployment_id: str):
        """Advance to next rollout phase"""
        
        monitor = self.rollout_monitors.get(deployment_id)
        if not monitor:
            return
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return
        
        rollout_plan = deployment["rollout_plan"]
        current_phase_idx = monitor["current_phase"]
        
        # Record phase completion
        monitor["phase_history"].append({
            "phase_index": current_phase_idx,
            "completed_at": datetime.utcnow(),
            "duration": (datetime.utcnow() - monitor["phase_start_time"]).total_seconds()
        })
        
        # Advance to next phase
        next_phase_idx = current_phase_idx + 1
        phases = rollout_plan["phases"]
        
        if next_phase_idx >= len(phases):
            logger.info(f"Rollout {deployment_id} completed successfully")
            del self.rollout_monitors[deployment_id]
            return
        
        monitor["current_phase"] = next_phase_idx
        monitor["phase_start_time"] = datetime.utcnow()
        rollout_plan["current_phase"] = next_phase_idx
        
        next_phase = phases[next_phase_idx]
        logger.info(f"Advanced rollout {deployment_id} to phase {next_phase['name']}")
        
        # Update position sizing for new phase
        await self._update_position_sizing(deployment_id, next_phase["position_size_percent"])
    
    async def _update_position_sizing(self, deployment_id: str, position_size_percent: int):
        """Update position sizing for current rollout phase"""
        
        # In production, this would send commands to the trading engine
        logger.info(f"Updated position sizing to {position_size_percent}% for deployment {deployment_id}")
    
    async def _send_control_command(self, strategy_instance_id: str, command: str, force: bool = False):
        """Send control command to live strategy"""
        
        # In production, this would send commands to the Docker container
        logger.info(f"Sending {command} command to strategy {strategy_instance_id}")
        
        # Simulate command execution
        await asyncio.sleep(1)
    
    async def _execute_emergency_procedures(self, strategy_instance_id: str, reason: str):
        """Execute emergency procedures"""
        
        strategy = self.live_strategies.get(strategy_instance_id)
        if not strategy:
            return
        
        # Close all positions
        strategy["current_position"]["side"] = "FLAT"
        strategy["current_position"]["quantity"] = 0
        
        # Create emergency alert
        alert = {
            "alert_id": str(uuid.uuid4()),
            "strategy_instance_id": strategy_instance_id,
            "type": "emergency_stop",
            "severity": "critical",
            "message": f"Emergency stop executed: {reason}",
            "timestamp": datetime.utcnow(),
            "acknowledged": False
        }
        
        strategy.setdefault("alerts", []).append(alert)
        
        # Notify risk team
        logger.critical(f"Emergency stop executed for {strategy_instance_id}: {reason}")
    
    async def _create_rollback_plan(
        self, deployment_id: str, target_version: str, immediate: bool
    ) -> List[Dict]:
        """Create rollback execution plan"""
        
        return [
            {
                "step_id": "stop_current",
                "description": "Stop current strategy instance",
                "estimated_duration": 60,
                "critical": True,
                "status": "pending"
            },
            {
                "step_id": "backup_config",
                "description": "Backup current configuration",
                "estimated_duration": 30,
                "critical": False,
                "status": "pending"
            },
            {
                "step_id": "deploy_previous",
                "description": f"Deploy version {target_version}",
                "estimated_duration": 120,
                "critical": True,
                "status": "pending"
            },
            {
                "step_id": "validate_rollback",
                "description": "Validate rollback success",
                "estimated_duration": 90,
                "critical": True,
                "status": "pending"
            }
        ]
    
    async def _execute_rollback(self, rollback_id: str, rollback_plan: List[Dict]):
        """Execute rollback plan"""
        
        logger.info(f"Executing rollback {rollback_id}")
        
        for step in rollback_plan:
            try:
                step["status"] = "executing"
                
                # Simulate step execution
                await asyncio.sleep(step["estimated_duration"])
                
                step["status"] = "completed"
                step["completed_at"] = datetime.utcnow()
                
                logger.info(f"Completed rollback step: {step['description']}")
                
            except Exception as e:
                step["status"] = "failed"
                step["error"] = str(e)
                logger.error(f"Rollback step failed: {step['description']} - {e}")
                
                if step["critical"]:
                    logger.critical(f"Critical rollback step failed for {rollback_id}")
                    break
        
        logger.info(f"Rollback {rollback_id} execution completed")
    
    async def _update_strategy_metrics(self, strategy_instance_id: str):
        """Update strategy metrics from live data sources"""
        
        strategy = self.live_strategies.get(strategy_instance_id)
        if not strategy:
            return
        
        # Simulate metric updates
        import random
        
        # Update performance metrics
        strategy["performance_metrics"]["daily_pnl"] += random.uniform(-50, 50)
        strategy["performance_metrics"]["total_pnl"] = (
            strategy["realized_pnl"] + strategy["unrealized_pnl"]
        )
        
        # Update risk metrics
        strategy["risk_metrics"]["current_drawdown"] = max(0, random.uniform(0, 8))
        strategy["risk_metrics"]["value_at_risk"] = random.uniform(100, 1000)
        strategy["risk_metrics"]["last_risk_check"] = datetime.utcnow()
        
        # Update health status
        strategy["health_status"]["last_health_check"] = datetime.utcnow()
        strategy["last_heartbeat"] = datetime.utcnow()

# Global service instance
deployment_service = DeploymentService()
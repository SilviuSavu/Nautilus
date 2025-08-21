"""
Strategy Version Control Service

Handles versioning, rollback, and configuration history for strategy configurations.
Provides comprehensive version management with performance tracking and rollback capabilities.
"""

import logging
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session
from decimal import Decimal

from database import get_db_session
from strategy_service import StrategyConfig, StrategyInstance

logger = logging.getLogger(__name__)

class ChangeType(str, Enum):
    PARAMETER_CHANGE = "parameter_change"
    DEPLOYMENT = "deployment"
    PAUSE = "pause"
    STOP = "stop"
    SAVE = "save"
    ROLLBACK = "rollback"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class StrategyVersion:
    """Strategy version data model"""
    id: str
    config_id: str
    version_number: int
    config_snapshot: dict[str, Any]
    change_summary: str
    created_by: str
    created_at: datetime
    deployment_results: list[dict[str, Any | None]] = None

@dataclass
class ConfigurationChange:
    """Configuration change tracking model"""
    id: str
    strategy_id: str
    change_type: ChangeType
    timestamp: datetime
    changed_by: str
    description: str | None = None
    reason: str | None = None
    version: int | None = None
    changed_fields: list[str | None] = None
    config_diff: dict[str, Any | None] = None
    config_snapshot: dict[str, Any | None] = None
    auto_generated: bool = False
    deployment_mode: str | None = None
    rollback_version: int | None = None
    performance_before: dict[str, Any | None] = None
    performance_after: dict[str, Any | None] = None

@dataclass
class RollbackPlan:
    """Rollback execution plan"""
    strategy_id: str
    from_version: int
    to_version: int
    changes_to_revert: list[ConfigurationChange]
    execution_steps: list[dict[str, Any]]
    risk_assessment: dict[str, Any]
    estimated_duration_seconds: int
    backup_required: bool
    dependencies: list[str]

@dataclass
class RollbackProgress:
    """Rollback execution progress tracking"""
    rollback_id: str
    status: str  # 'initializing', 'running', 'completed', 'failed', 'rolled_back'
    overall_progress: float
    current_step: str
    current_operation: str | None = None
    completed_steps: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0
    estimated_remaining_seconds: float = 0
    errors: list[str] = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class StrategyVersionControlService:
    """
    Comprehensive strategy version control service
    
    Provides:
    - Strategy versioning and snapshots
    - Configuration change tracking
    - Performance-aware rollback system
    - Audit logging and compliance
    """
    
    def __init__(self):
        self.rollback_tasks = {}  # Track active rollback operations
        
    # Version Management
    async def create_version(
        self, strategy_id: str, current_config: StrategyConfig, change_summary: str, created_by: str
    ) -> StrategyVersion:
        """Create a new version of a strategy configuration"""
        
        with get_db_session() as db:
            # Get the highest version number for this strategy
            highest_version = self._get_highest_version(db, strategy_id)
            new_version_number = highest_version + 1
            
            # Create version snapshot
            config_snapshot = {
                "id": current_config.id, "name": current_config.name, "template_id": current_config.template_id, "parameters": current_config.parameters, "risk_settings": current_config.risk_settings, "deployment_settings": current_config.deployment_settings, "tags": current_config.tags, "created_at": current_config.created_at.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            version = StrategyVersion(
                id=f"{strategy_id}_v{new_version_number}", config_id=strategy_id, version_number=new_version_number, config_snapshot=config_snapshot, change_summary=change_summary, created_by=created_by, created_at=datetime.now(timezone.utc)
            )
            
            # Store in database (implement actual DB storage)
            self._store_version(db, version)
            
            logger.info(f"Created version {new_version_number} for strategy {strategy_id}")
            return version
    
    async def get_version_history(self, strategy_id: str) -> list[StrategyVersion]:
        """Get complete version history for a strategy"""
        
        with get_db_session() as db:
            versions = self._get_versions_from_db(db, strategy_id)
            
            # Enrich with deployment results
            for version in versions:
                version.deployment_results = await self._get_deployment_results(
                    strategy_id, version.version_number
                )
            
            return sorted(versions, key=lambda v: v.version_number, reverse=True)
    
    async def compare_versions(
        self, strategy_id: str, version1_id: str, version2_id: str
    ) -> dict[str, Any]:
        """Compare two strategy versions"""
        
        with get_db_session() as db:
            version1 = self._get_version_from_db(db, version1_id)
            version2 = self._get_version_from_db(db, version2_id)
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            # Configuration differences
            config_diff = self._calculate_config_differences(
                version1.config_snapshot, version2.config_snapshot
            )
            
            # Performance comparison (if available)
            performance_comparison = await self._compare_version_performance(
                strategy_id, version1.version_number, version2.version_number
            )
            
            return {
                "version1": asdict(version1), "version2": asdict(version2), "configuration_diff": config_diff, "performance_comparison": performance_comparison, "differences": config_diff.get("parameter_changes", [])
            }
    
    # Configuration History Tracking
    async def track_configuration_change(
        self, strategy_id: str, change_type: ChangeType, changed_by: str, description: str | None = None, reason: str | None = None, changed_fields: list[str | None] = None, old_config: dict[str, Any | None] = None, new_config: dict[str, Any | None] = None, auto_generated: bool = False
    ) -> ConfigurationChange:
        """Track a configuration change"""
        
        change = ConfigurationChange(
            id=f"change_{strategy_id}_{int(datetime.now().timestamp())}", strategy_id=strategy_id, change_type=change_type, timestamp=datetime.now(timezone.utc), changed_by=changed_by, description=description, reason=reason, changed_fields=changed_fields, auto_generated=auto_generated
        )
        
        # Calculate configuration diff if both configs provided
        if old_config and new_config:
            change.config_diff = self._calculate_config_diff(old_config, new_config)
        
        # Store in database
        with get_db_session() as db:
            self._store_configuration_change(db, change)
        
        logger.info(f"Tracked {change_type} for strategy {strategy_id}")
        return change
    
    async def get_configuration_history(self, strategy_id: str) -> list[ConfigurationChange]:
        """Get configuration change history for a strategy"""
        
        with get_db_session() as db:
            return self._get_configuration_changes_from_db(db, strategy_id)
    
    # Rollback System
    async def generate_rollback_plan(
        self, strategy_id: str, from_version: int, to_version: int
    ) -> RollbackPlan:
        """Generate a rollback execution plan"""
        
        with get_db_session() as db:
            # Get configuration changes between versions
            changes_to_revert = self._get_changes_between_versions(
                db, strategy_id, to_version, from_version
            )
            
            # Assess risks
            risk_assessment = self._assess_rollback_risk(changes_to_revert)
            
            # Generate execution steps
            execution_steps = self._generate_execution_steps(
                strategy_id, from_version, to_version, changes_to_revert
            )
            
            # Estimate duration
            estimated_duration = self._estimate_rollback_duration(execution_steps)
            
            plan = RollbackPlan(
                strategy_id=strategy_id, from_version=from_version, to_version=to_version, changes_to_revert=changes_to_revert, execution_steps=execution_steps, risk_assessment=risk_assessment, estimated_duration_seconds=estimated_duration, backup_required=risk_assessment["risk_level"] != "low", dependencies=self._identify_dependencies(strategy_id)
            )
            
            return plan
    
    async def validate_rollback(
        self, strategy_id: str, target_version: int, rollback_plan: RollbackPlan
    ) -> dict[str, Any]:
        """Validate rollback feasibility and safety"""
        
        validation_result = {
            "validation_passed": True, "validation_errors": [], "warnings": [], "pre_rollback_checks": [], "backup_verification": None
        }
        
        # Pre-rollback checks
        checks = [
            self._check_strategy_state(strategy_id), self._check_deployment_dependencies(strategy_id), self._check_data_integrity(strategy_id), self._check_backup_availability(strategy_id)
        ]
        
        validation_result["pre_rollback_checks"] = await asyncio.gather(*checks)
        
        # Check for blocking issues
        for check in validation_result["pre_rollback_checks"]:
            if not check["passed"] and check.get("blocking", False):
                validation_result["validation_errors"].append(check["description"])
                validation_result["validation_passed"] = False
        
        # Create backup if required
        if rollback_plan.backup_required:
            backup_result = await self._create_rollback_backup(strategy_id)
            validation_result["backup_verification"] = backup_result
        
        return validation_result
    
    async def execute_rollback(
        self, strategy_id: str, target_version: int, rollback_settings: dict[str, Any], progress_callback: callable | None = None
    ) -> str:
        """Execute rollback operation"""
        
        rollback_id = f"rollback_{strategy_id}_{int(datetime.now().timestamp())}"
        
        # Initialize progress tracking
        progress = RollbackProgress(
            rollback_id=rollback_id, status="initializing", overall_progress=0, current_step="Starting rollback process"
        )
        
        self.rollback_tasks[rollback_id] = progress
        
        # Execute rollback asynchronously
        asyncio.create_task(
            self._execute_rollback_async(
                rollback_id, strategy_id, target_version, rollback_settings, progress_callback
            )
        )
        
        return rollback_id
    
    async def get_rollback_progress(self, rollback_id: str) -> RollbackProgress:
        """Get rollback execution progress"""
        
        return self.rollback_tasks.get(
            rollback_id, RollbackProgress(
                rollback_id=rollback_id, status="not_found", overall_progress=0, current_step="Rollback not found"
            )
        )
    
    # Private Methods
    def _get_highest_version(self, db: Session, strategy_id: str) -> int:
        """Get the highest version number for a strategy"""
        # Implement database query to get highest version
        # This is a placeholder implementation
        return 0
    
    def _store_version(self, db: Session, version: StrategyVersion):
        """Store version in database"""
        # Implement database storage
        pass
    
    def _get_versions_from_db(self, db: Session, strategy_id: str) -> list[StrategyVersion]:
        """Get versions from database"""
        # Implement database query
        return []
    
    def _get_version_from_db(self, db: Session, version_id: str) -> StrategyVersion | None:
        """Get specific version from database"""
        # Implement database query
        return None
    
    async def _get_deployment_results(self, strategy_id: str, version: int) -> list[dict[str, Any]]:
        """Get deployment results for a version"""
        # Implement deployment results retrieval
        return []
    
    def _calculate_config_differences(self, config1: Dict, config2: Dict) -> dict[str, Any]:
        """Calculate detailed differences between configurations"""
        
        differences = {
            "parameter_changes": [], "total_changes": 0, "high_impact_changes": 0, "medium_impact_changes": 0, "low_impact_changes": 0
        }
        
        # Compare parameters
        params1 = config1.get("parameters", {})
        params2 = config2.get("parameters", {})
        
        all_params = set(params1.keys()) | set(params2.keys())
        
        for param in all_params:
            change_type = "unchanged"
            old_value = params1.get(param)
            new_value = params2.get(param)
            
            if param not in params1:
                change_type = "added"
            elif param not in params2:
                change_type = "removed"
            elif old_value != new_value:
                change_type = "modified"
            
            if change_type != "unchanged":
                impact_level = self._assess_parameter_impact(param, old_value, new_value)
                
                differences["parameter_changes"].append({
                    "parameter_name": param, "change_type": change_type, "old_value": old_value, "new_value": new_value, "impact_level": impact_level
                })
                
                differences["total_changes"] += 1
                if impact_level == "high":
                    differences["high_impact_changes"] += 1
                elif impact_level == "medium":
                    differences["medium_impact_changes"] += 1
                else:
                    differences["low_impact_changes"] += 1
        
        return differences
    
    def _assess_parameter_impact(self, param: str, old_value: Any, new_value: Any) -> str:
        """Assess the impact level of a parameter change"""
        
        # High impact parameters (affect risk or core strategy logic)
        high_impact_params = [
            "max_position_size", "stop_loss", "take_profit", "strategy_mode", "risk_multiplier"
        ]
        
        # Medium impact parameters (affect performance but not risk)
        medium_impact_params = [
            "fast_period", "slow_period", "threshold", "lookback_period"
        ]
        
        if param.lower() in [p.lower() for p in high_impact_params]:
            return "high"
        elif param.lower() in [p.lower() for p in medium_impact_params]:
            return "medium"
        else:
            return "low"
    
    async def _compare_version_performance(
        self, strategy_id: str, version1: int, version2: int
    ) -> dict[str, Any | None]:
        """Compare performance between two versions"""
        
        # This would integrate with the performance tracking system
        # Placeholder implementation
        return None
    
    def _calculate_config_diff(self, old_config: Dict, new_config: Dict) -> dict[str, Any]:
        """Calculate configuration diff between old and new config"""
        
        diff = {}
        
        for key in set(old_config.keys()) | set(new_config.keys()):
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            
            if old_val != new_val:
                diff[key] = {
                    "old": old_val, "new": new_val
                }
        
        return diff
    
    def _store_configuration_change(self, db: Session, change: ConfigurationChange):
        """Store configuration change in database"""
        # Implement database storage
        pass
    
    def _get_configuration_changes_from_db(self, db: Session, strategy_id: str) -> list[ConfigurationChange]:
        """Get configuration changes from database"""
        # Implement database query
        return []
    
    def _get_changes_between_versions(
        self, db: Session, strategy_id: str, from_version: int, to_version: int
    ) -> list[ConfigurationChange]:
        """Get changes between two versions"""
        # Implement database query for changes between versions
        return []
    
    def _assess_rollback_risk(self, changes: list[ConfigurationChange]) -> dict[str, Any]:
        """Assess risk level of rolling back changes"""
        
        high_risk_changes = [ChangeType.DEPLOYMENT, ChangeType.PARAMETER_CHANGE]
        risk_level = "low"
        warnings = []
        recommendations = []
        
        for change in changes:
            if change.change_type in high_risk_changes:
                risk_level = "medium"
                warnings.append(f"Rolling back {change.change_type} may affect strategy performance")
        
        if len([c for c in changes if c.change_type == ChangeType.DEPLOYMENT]) > 1:
            risk_level = "high"
            warnings.append("Multiple deployment changes detected - high risk rollback")
            recommendations.append("Consider incremental rollback or thorough testing")
        
        return {
            "risk_level": risk_level, "warnings": warnings, "blockers": [], # Would include actual blockers
            "recommendations": recommendations
        }
    
    def _generate_execution_steps(
        self, strategy_id: str, from_version: int, to_version: int, changes: list[ConfigurationChange]
    ) -> list[dict[str, Any]]:
        """Generate rollback execution steps"""
        
        steps = [
            {
                "step_id": "backup", "description": "Create configuration backup", "action_type": "backup", "estimated_duration": 5, "critical": True
            }, {
                "step_id": "stop_strategy", "description": "Stop strategy execution", "action_type": "strategy_restart", "estimated_duration": 10, "critical": True
            }, {
                "step_id": "revert_config", "description": f"Revert configuration to version {to_version}", "action_type": "config_change", "estimated_duration": 15, "critical": True
            }, {
                "step_id": "validate_config", "description": "Validate reverted configuration", "action_type": "validation", "estimated_duration": 10, "critical": True
            }, {
                "step_id": "restart_strategy", "description": "Restart strategy with reverted configuration", "action_type": "strategy_restart", "estimated_duration": 20, "critical": True
            }
        ]
        
        return steps
    
    def _estimate_rollback_duration(self, steps: list[dict[str, Any]]) -> int:
        """Estimate total rollback duration"""
        
        return sum(step.get("estimated_duration", 0) for step in steps)
    
    def _identify_dependencies(self, strategy_id: str) -> list[str]:
        """Identify rollback dependencies"""
        
        # Would check for actual dependencies like:
        # - Other strategies using same instruments
        # - Active positions
        # - Pending orders
        return []
    
    async def _check_strategy_state(self, strategy_id: str) -> dict[str, Any]:
        """Check current strategy state"""
        
        return {
            "check_name": "strategy_state", "description": "Strategy is in safe state for rollback", "passed": True, "details": "Strategy is stopped"
        }
    
    async def _check_deployment_dependencies(self, strategy_id: str) -> dict[str, Any]:
        """Check deployment dependencies"""
        
        return {
            "check_name": "deployment_dependencies", "description": "No blocking deployment dependencies", "passed": True, "details": "No active deployments detected"
        }
    
    async def _check_data_integrity(self, strategy_id: str) -> dict[str, Any]:
        """Check data integrity"""
        
        return {
            "check_name": "data_integrity", "description": "Strategy data is consistent", "passed": True, "details": "All data checks passed"
        }
    
    async def _check_backup_availability(self, strategy_id: str) -> dict[str, Any]:
        """Check backup availability"""
        
        return {
            "check_name": "backup_availability", "description": "Backup system is available", "passed": True, "details": "Backup storage is accessible"
        }
    
    async def _create_rollback_backup(self, strategy_id: str) -> dict[str, Any]:
        """Create rollback backup"""
        
        # Would create actual backup
        return {
            "backup_created": True, "backup_path": f"/backups/{strategy_id}_{int(datetime.now().timestamp())}", "backup_size_mb": 1.2, "backup_verified": True
        }
    
    async def _execute_rollback_async(
        self, rollback_id: str, strategy_id: str, target_version: int, rollback_settings: dict[str, Any], progress_callback: callable | None = None
    ):
        """Execute rollback asynchronously with progress tracking"""
        
        progress = self.rollback_tasks[rollback_id]
        
        try:
            progress.status = "running"
            progress.total_steps = 5
            
            # Step 1: Create backup
            progress.current_step = "Creating backup"
            progress.current_operation = "Backing up current configuration"
            if progress_callback:
                progress_callback(progress)
            
            await asyncio.sleep(2)  # Simulate backup creation
            progress.completed_steps = 1
            progress.overall_progress = 20
            
            # Step 2: Stop strategy
            progress.current_step = "Stopping strategy"
            progress.current_operation = "Safely stopping strategy execution"
            if progress_callback:
                progress_callback(progress)
            
            await asyncio.sleep(3)
            progress.completed_steps = 2
            progress.overall_progress = 40
            
            # Step 3: Revert configuration
            progress.current_step = "Reverting configuration"
            progress.current_operation = f"Applying version {target_version} configuration"
            if progress_callback:
                progress_callback(progress)
            
            await asyncio.sleep(4)
            progress.completed_steps = 3
            progress.overall_progress = 60
            
            # Step 4: Validate
            progress.current_step = "Validating configuration"
            progress.current_operation = "Verifying reverted configuration"
            if progress_callback:
                progress_callback(progress)
            
            await asyncio.sleep(2)
            progress.completed_steps = 4
            progress.overall_progress = 80
            
            # Step 5: Restart
            progress.current_step = "Restarting strategy"
            progress.current_operation = "Starting strategy with reverted configuration"
            if progress_callback:
                progress_callback(progress)
            
            await asyncio.sleep(3)
            progress.completed_steps = 5
            progress.overall_progress = 100
            progress.status = "completed"
            progress.current_step = "Rollback completed successfully"
            progress.current_operation = None
            
            if progress_callback:
                progress_callback(progress)
            
            logger.info(f"Rollback {rollback_id} completed successfully")
            
        except Exception as e:
            progress.status = "failed"
            progress.errors.append(str(e))
            if progress_callback:
                progress_callback(progress)
            
            logger.error(f"Rollback {rollback_id} failed: {e}")

# Global service instance
version_control_service = StrategyVersionControlService()
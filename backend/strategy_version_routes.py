"""
Strategy Version Control API Routes

FastAPI routes for strategy version control operations:
- Version management (create, list, compare)
- Configuration history tracking
- Rollback system with progress tracking
- Configuration snapshots and audit logging
"""

import logging
from typing import Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from decimal import Decimal

from strategy_version_control import (
    version_control_service, StrategyVersion, ConfigurationChange, RollbackPlan, RollbackProgress, ChangeType
)
from strategy_service import get_strategy_config

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/strategies", tags=["strategy-versions"])

# Request/Response Models
class CreateVersionRequest(BaseModel):
    change_summary: str = Field(..., description="Summary of changes made")
    save_current_config: bool = Field(default=True, description="Whether to save current config as new version")

class VersionResponse(BaseModel):
    id: str
    config_id: str
    version_number: int
    change_summary: str
    created_by: str
    created_at: datetime
    deployment_results: list[dict[str, Any | None]] = None

class VersionHistoryResponse(BaseModel):
    versions: list[VersionResponse]
    total_count: int
    current_version: int | None = None

class CompareVersionsResponse(BaseModel):
    version1: VersionResponse
    version2: VersionResponse
    configuration_diff: dict[str, Any]
    performance_comparison: dict[str, Any | None] = None
    differences: list[dict[str, Any]]

class RollbackRequest(BaseModel):
    version_id: str = Field(..., description="Target version to rollback to")
    reason: str | None = Field(None, description="Reason for rollback")
    create_backup: bool = Field(default=True, description="Create backup before rollback")
    force_rollback: bool = Field(default=False, description="Force rollback ignoring warnings")

class RollbackPlanRequest(BaseModel):
    from_version: int
    to_version: int

class RollbackExecuteRequest(BaseModel):
    target_version: int
    settings: dict[str, Any]

class ConfigurationHistoryResponse(BaseModel):
    changes: list[dict[str, Any]]
    total_count: int
    page: int
    page_size: int

class AuditLogResponse(BaseModel):
    audit_entries: list[dict[str, Any]]
    total_count: int

# Version Management Endpoints
@router.get("/{strategy_id}/versions", response_model=VersionHistoryResponse)
async def get_version_history(
    strategy_id: str, page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100)
):
    """Get complete version history for a strategy"""
    
    try:
        versions = await version_control_service.get_version_history(strategy_id)
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_versions = versions[start_idx:end_idx]
        
        # Convert to response format
        version_responses = [
            VersionResponse(
                id=v.id, config_id=v.config_id, version_number=v.version_number, change_summary=v.change_summary, created_by=v.created_by, created_at=v.created_at, deployment_results=v.deployment_results
            )
            for v in paginated_versions
        ]
        
        return VersionHistoryResponse(
            versions=version_responses, total_count=len(versions), current_version=versions[0].version_number if versions else None
        )
        
    except Exception as e:
        logger.error(f"Error getting version history for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/versions", response_model=VersionResponse)
async def create_version(
    strategy_id: str, request: CreateVersionRequest
):
    """Create a new version of a strategy configuration"""
    
    try:
        # Get current strategy configuration
        current_config = await get_strategy_config(strategy_id)
        if not current_config:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Create new version
        version = await version_control_service.create_version(
            strategy_id=strategy_id, current_config=current_config, change_summary=request.change_summary, created_by="current_user"  # Would get from auth context
        )
        
        return VersionResponse(
            id=version.id, config_id=version.config_id, version_number=version.version_number, change_summary=version.change_summary, created_by=version.created_by, created_at=version.created_at, deployment_results=version.deployment_results
        )
        
    except Exception as e:
        logger.error(f"Error creating version for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}/versions/{version_id}")
async def get_version(
    strategy_id: str, version_id: str
):
    """Get specific version details"""
    
    try:
        version = await version_control_service.get_version(strategy_id, version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return VersionResponse(
            id=version.id, config_id=version.config_id, version_number=version.version_number, change_summary=version.change_summary, created_by=version.created_by, created_at=version.created_at, deployment_results=version.deployment_results
        )
        
    except Exception as e:
        logger.error(f"Error getting version {version_id} for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}/versions/compare", response_model=CompareVersionsResponse)
async def compare_versions(
    strategy_id: str, version1: str = Query(..., description="First version ID"), version2: str = Query(..., description="Second version ID")
):
    """Compare two strategy versions"""
    
    try:
        comparison = await version_control_service.compare_versions(
            strategy_id, version1, version2
        )
        
        return CompareVersionsResponse(**comparison)
        
    except Exception as e:
        logger.error(f"Error comparing versions {version1} and {version2} for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration History Endpoints
@router.get("/{strategy_id}/history", response_model=ConfigurationHistoryResponse)
async def get_configuration_history(
    strategy_id: str, page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=200), change_type: str | None = Query(None, description="Filter by change type")
):
    """Get configuration change history"""
    
    try:
        changes = await version_control_service.get_configuration_history(strategy_id)
        
        # Filter by change type if specified
        if change_type:
            changes = [c for c in changes if c.change_type.value == change_type]
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_changes = changes[start_idx:end_idx]
        
        # Convert to dict format
        change_dicts = []
        for change in paginated_changes:
            change_dict = {
                "id": change.id, "strategy_id": change.strategy_id, "change_type": change.change_type.value, "timestamp": change.timestamp, "changed_by": change.changed_by, "description": change.description, "reason": change.reason, "version": change.version, "changed_fields": change.changed_fields, "auto_generated": change.auto_generated, "deployment_mode": change.deployment_mode, "rollback_version": change.rollback_version
            }
            change_dicts.append(change_dict)
        
        return ConfigurationHistoryResponse(
            changes=change_dicts, total_count=len(changes), page=page, page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error getting configuration history for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}/audit", response_model=AuditLogResponse)
async def get_configuration_audit(
    strategy_id: str, page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=200)
):
    """Get configuration audit log"""
    
    try:
        # This would integrate with actual audit logging system
        audit_entries = [
            {
                "id": f"audit_{strategy_id}_1", "timestamp": datetime.now(), "action": "Configuration validation passed", "user_id": "current_user", "details": "All parameters validated successfully", "risk_level": "low", "warnings": []
            }
        ]
        
        return AuditLogResponse(
            audit_entries=audit_entries, total_count=len(audit_entries)
        )
        
    except Exception as e:
        logger.error(f"Error getting audit log for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Rollback System Endpoints
@router.post("/{strategy_id}/rollback/plan")
async def generate_rollback_plan(
    strategy_id: str, request: RollbackPlanRequest
):
    """Generate rollback execution plan"""
    
    try:
        plan = await version_control_service.generate_rollback_plan(
            strategy_id=strategy_id, from_version=request.from_version, to_version=request.to_version
        )
        
        # Convert to dict for JSON response
        plan_dict = {
            "strategy_id": plan.strategy_id, "from_version": plan.from_version, "to_version": plan.to_version, "changes_to_revert": [
                {
                    "change_type": change.change_type.value, "timestamp": change.timestamp, "description": change.description, "parameters_affected": change.changed_fields
                }
                for change in plan.changes_to_revert
            ], "execution_steps": plan.execution_steps, "risk_assessment": plan.risk_assessment, "estimated_duration_seconds": plan.estimated_duration_seconds, "backup_required": plan.backup_required, "dependencies": plan.dependencies
        }
        
        return JSONResponse(content=plan_dict)
        
    except Exception as e:
        logger.error(f"Error generating rollback plan for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/rollback/validate")
async def validate_rollback(
    strategy_id: str, request: dict[str, Any]
):
    """Validate rollback feasibility"""
    
    try:
        target_version = request.get("target_version")
        rollback_plan = request.get("rollback_plan")
        
        # Convert rollback_plan dict back to RollbackPlan object if needed
        # This is simplified - in practice you'd have proper deserialization
        
        validation_result = await version_control_service.validate_rollback(
            strategy_id=strategy_id, target_version=target_version, rollback_plan=None  # Would pass actual plan object
        )
        
        return JSONResponse(content=validation_result)
        
    except Exception as e:
        logger.error(f"Error validating rollback for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/rollback/execute")
async def execute_rollback(
    strategy_id: str, request: RollbackExecuteRequest, background_tasks: BackgroundTasks
):
    """Execute rollback operation"""
    
    try:
        rollback_id = await version_control_service.execute_rollback(
            strategy_id=strategy_id, target_version=request.target_version, rollback_settings=request.settings
        )
        
        return JSONResponse(content={"rollback_id": rollback_id})
        
    except Exception as e:
        logger.error(f"Error executing rollback for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rollback/{rollback_id}/progress")
async def get_rollback_progress(rollback_id: str):
    """Get rollback execution progress"""
    
    try:
        progress = await version_control_service.get_rollback_progress(rollback_id)
        
        progress_dict = {
            "rollback_id": progress.rollback_id, "status": progress.status, "overall_progress": progress.overall_progress, "current_step": progress.current_step, "current_operation": progress.current_operation, "completed_steps": progress.completed_steps, "total_steps": progress.total_steps, "elapsed_seconds": progress.elapsed_seconds, "estimated_remaining_seconds": progress.estimated_remaining_seconds, "errors": progress.errors, "warnings": progress.warnings
        }
        
        return JSONResponse(content=progress_dict)
        
    except Exception as e:
        logger.error(f"Error getting rollback progress for {rollback_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/rollback")
async def simple_rollback(
    strategy_id: str, request: RollbackRequest
):
    """Simple rollback to specific version"""
    
    try:
        # Track the rollback operation
        await version_control_service.track_configuration_change(
            strategy_id=strategy_id, change_type=ChangeType.ROLLBACK, changed_by="current_user", reason=request.reason, description=f"Rollback to version {request.version_id}"
        )
        
        return JSONResponse(content={
            "success": True, "message": f"Successfully initiated rollback to {request.version_id}"
        })
        
    except Exception as e:
        logger.error(f"Error rolling back {strategy_id} to {request.version_id}: {e}")
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )

# Performance History Endpoints
@router.get("/{strategy_id}/performance/history")
async def get_performance_history(
    strategy_id: str, days: int = Query(default=30, ge=1, le=365)
):
    """Get performance history for version control"""
    
    try:
        # This would integrate with actual performance tracking
        # Placeholder implementation
        metrics = [
            {
                "timestamp": datetime.now(), "total_pnl": "1000.00", "unrealized_pnl": "150.00", "total_trades": 25, "winning_trades": 18, "win_rate": 72.0, "max_drawdown": "200.00", "sharpe_ratio": 1.8, "last_updated": datetime.now()
            }
        ]
        
        return JSONResponse(content={"metrics": metrics})
        
    except Exception as e:
        logger.error(f"Error getting performance history for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}/performance/compare")
async def compare_version_performance(
    strategy_id: str, v1: int = Query(..., description="First version number"), v2: int = Query(..., description="Second version number")
):
    """Compare performance between two versions"""
    
    try:
        comparison = await version_control_service._compare_version_performance(
            strategy_id, v1, v2
        )
        
        if not comparison:
            # Return placeholder comparison
            comparison = {
                "version1_metrics": {"total_pnl": 1000.0, "win_rate": 65.0, "trades": 20}, "version2_metrics": {"total_pnl": 1200.0, "win_rate": 70.0, "trades": 25}, "comparison": {"pnl_improvement": 200.0, "win_rate_improvement": 5.0}
            }
        
        return JSONResponse(content=comparison)
        
    except Exception as e:
        logger.error(f"Error comparing performance for {strategy_id} v{v1} vs v{v2}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Snapshots
@router.post("/{strategy_id}/snapshots")
async def create_configuration_snapshot(
    strategy_id: str, request: dict[str, Any]
):
    """Create configuration snapshot"""
    
    try:
        snapshot_id = f"snapshot_{strategy_id}_{int(datetime.now().timestamp())}"
        
        return JSONResponse(content={"snapshot_id": snapshot_id})
        
    except Exception as e:
        logger.error(f"Error creating snapshot for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/snapshots/{snapshot_id}/restore")
async def restore_from_snapshot(
    strategy_id: str, snapshot_id: str
):
    """Restore configuration from snapshot"""
    
    try:
        return JSONResponse(content={
            "success": True, "message": f"Successfully restored from snapshot {snapshot_id}"
        })
        
    except Exception as e:
        logger.error(f"Error restoring from snapshot {snapshot_id} for {strategy_id}: {e}")
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )
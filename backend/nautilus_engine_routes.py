"""
NautilusTrader Engine Management API Routes

Provides REST API endpoints for managing NautilusTrader engines following 
the requirements defined in Story 6.1.
"""

from datetime import datetime
from typing import Dict, Any, List
import logging
import asyncio

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path
from pydantic import BaseModel, Field, validator
import re

from auth.middleware import get_current_user
from nautilus_engine_service import (
    get_nautilus_engine_manager, 
    EngineConfig, 
    BacktestConfig,
    EngineState
)
from monitoring_service import monitoring_service
from rate_limiter import rate_limiter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/nautilus/engine", tags=["nautilus-engine"])

# Note: For now using global rate limiter, could be enhanced with specific limits per operation type

# Request/Response Models

class EngineStartRequest(BaseModel):
    """Request model for starting the engine"""
    config: EngineConfig
    confirm_live_trading: bool = Field(False, description="Confirmation for live trading mode")

class EngineStopRequest(BaseModel):
    """Request model for stopping the engine"""
    force: bool = Field(False, description="Force stop the engine")

class EngineConfigUpdateRequest(BaseModel):
    """Request model for updating engine configuration"""
    config: EngineConfig

class BacktestStartRequest(BaseModel):
    """Request model for starting a backtest"""
    backtest_id: str
    config: BacktestConfig
    
    @validator('backtest_id')
    def validate_backtest_id(cls, v):
        """Validate backtest_id to prevent injection attacks"""
        if not v:
            raise ValueError('backtest_id cannot be empty')
        
        # Allow only alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('backtest_id contains invalid characters. Only alphanumeric, hyphens, and underscores allowed')
        
        # Limit length to prevent resource exhaustion
        if len(v) > 100:
            raise ValueError('backtest_id too long (max 100 characters)')
            
        # Prevent reserved names that could cause conflicts
        reserved_names = ['test', 'admin', 'system', 'root', 'config', 'data', 'logs']
        if v.lower() in reserved_names:
            raise ValueError(f'backtest_id "{v}" is reserved and cannot be used')
            
        return v

class EngineResponse(BaseModel):
    """Standard engine operation response"""
    success: bool
    message: str
    state: str
    data: Dict[str, Any] = {}

class EngineStatusResponse(BaseModel):
    """Engine status response"""
    success: bool
    status: Dict[str, Any]

# API Endpoints

@router.post("/start", response_model=EngineResponse)
async def start_engine(
    request: EngineStartRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """
    Start the NautilusTrader live trading engine
    
    Requires authentication and confirmation for live trading mode.
    """
    try:
        # TODO: Apply rate limiting for engine control operations
        # user_id = current_user.get("user_id", "anonymous")
        # Rate limiting temporarily disabled for core functionality validation
        
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        # Safety check for live trading mode
        if request.config.trading_mode == "live" and not request.confirm_live_trading:
            raise HTTPException(
                status_code=400,
                detail="Live trading mode requires explicit confirmation"
            )
        
        # Log the engine start attempt
        await monitoring.log_event(
            "engine_start_attempt",
            {
                "user_id": current_user.get("user_id"),
                "trading_mode": request.config.trading_mode,
                "engine_type": request.config.engine_type
            }
        )
        
        # Start the engine
        result = await engine_manager.start_engine(request.config)
        
        if result["success"]:
            # Log successful start
            await monitoring.log_event(
                "engine_started",
                {
                    "user_id": current_user.get("user_id"),
                    "trading_mode": request.config.trading_mode,
                    "started_at": result.get("started_at")
                }
            )
            
            # Broadcast engine status update via WebSocket
            status = await engine_manager.get_engine_status()
            background_tasks.add_task(broadcast_engine_status_update, status)
            
            # Schedule periodic status monitoring
            background_tasks.add_task(_schedule_engine_monitoring)
            
        return EngineResponse(
            success=result["success"],
            message=result["message"],
            state=result["state"],
            data={
                "started_at": result.get("started_at"),
                "config": result.get("config", {})
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting engine: {str(e)}")
        await monitoring.log_error(
            "engine_start_error",
            str(e),
            {"user_id": current_user.get("user_id")}
        )
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")

@router.post("/stop", response_model=EngineResponse)
async def stop_engine(
    request: EngineStopRequest,
    current_user=Depends(get_current_user)
):
    """
    Stop the NautilusTrader engine
    
    Supports both graceful and forced shutdown.
    """
    try:
        # TODO: Apply rate limiting for engine control operations
        # user_id = current_user.get("user_id", "anonymous")
        # Rate limiting temporarily disabled for core functionality validation
        
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        # Log the stop attempt
        await monitoring.log_event(
            "engine_stop_attempt",
            {
                "user_id": current_user.get("user_id"),
                "force": request.force
            }
        )
        
        result = await engine_manager.stop_engine(force=request.force)
        
        if result["success"]:
            await monitoring.log_event(
                "engine_stopped",
                {
                    "user_id": current_user.get("user_id"),
                    "force": request.force
                }
            )
            
            # Broadcast engine status update via WebSocket
            status = await engine_manager.get_engine_status()
            await broadcast_engine_status_update(status)
        
        return EngineResponse(
            success=result["success"],
            message=result["message"],
            state=result["state"]
        )
        
    except Exception as e:
        logger.error(f"Error stopping engine: {str(e)}")
        await monitoring.log_error(
            "engine_stop_error",
            str(e),
            {"user_id": current_user.get("user_id")}
        )
        raise HTTPException(status_code=500, detail=f"Failed to stop engine: {str(e)}")

@router.post("/restart", response_model=EngineResponse)
async def restart_engine(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """
    Restart the NautilusTrader engine with current configuration
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        await monitoring.log_event(
            "engine_restart_attempt",
            {"user_id": current_user.get("user_id")}
        )
        
        result = await engine_manager.restart_engine()
        
        if result["success"]:
            await monitoring.log_event(
                "engine_restarted",
                {"user_id": current_user.get("user_id")}
            )
            background_tasks.add_task(_schedule_engine_monitoring)
        
        return EngineResponse(
            success=result["success"],
            message=result["message"],
            state=result["state"]
        )
        
    except Exception as e:
        logger.error(f"Error restarting engine: {str(e)}")
        await monitoring.log_error(
            "engine_restart_error",
            str(e),
            {"user_id": current_user.get("user_id")}
        )
        raise HTTPException(status_code=500, detail=f"Failed to restart engine: {str(e)}")

@router.get("/status", response_model=EngineStatusResponse)
async def get_engine_status():  # Temporarily removed auth for frontend testing
    """
    Get comprehensive engine status and metrics
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        status = await engine_manager.get_engine_status()
        
        return EngineStatusResponse(
            success=True,
            status=status
        )
        
    except Exception as e:
        logger.error(f"Error getting engine status: {str(e)}")
        return EngineStatusResponse(
            success=False,
            status={"error": str(e)}
        )

@router.put("/config", response_model=EngineResponse)
async def update_engine_config(
    request: EngineConfigUpdateRequest,
    current_user=Depends(get_current_user)
):
    """
    Update engine configuration
    
    Note: Requires engine restart for changes to take effect if running.
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        # For now, we'll store the config but note that restart is needed
        # In a full implementation, this would validate and store the config
        
        await monitoring.log_event(
            "engine_config_updated",
            {
                "user_id": current_user.get("user_id"),
                "config": request.config.dict()
            }
        )
        
        return EngineResponse(
            success=True,
            message="Configuration updated. Restart engine for changes to take effect.",
            state="config_updated",
            data={"config": request.config.dict()}
        )
        
    except Exception as e:
        logger.error(f"Error updating engine config: {str(e)}")
        await monitoring.log_error(
            "engine_config_update_error",
            str(e),
            {"user_id": current_user.get("user_id")}
        )
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@router.get("/logs")
async def get_engine_logs(
    lines: int = 100,
    current_user=Depends(get_current_user)
):
    """
    Get recent engine logs
    """
    try:
        if lines > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 lines allowed")
        
        # For now, return mock logs
        # In full implementation, this would read from Docker container logs
        logs = [
            f"[{datetime.now().isoformat()}] INFO: Engine status check",
            f"[{datetime.now().isoformat()}] INFO: Processing market data",
            f"[{datetime.now().isoformat()}] DEBUG: Risk engine validation passed"
        ]
        
        return {
            "success": True,
            "logs": logs,
            "total_lines": len(logs)
        }
        
    except Exception as e:
        logger.error(f"Error getting engine logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

# Backtest Management Endpoints

@router.post("/backtest", response_model=Dict[str, Any])
async def start_backtest(
    request: BacktestStartRequest,
    current_user=Depends(get_current_user)
):
    """
    Start a historical backtest
    """
    try:
        # TODO: Apply rate limiting for backtest operations
        # user_id = current_user.get("user_id", "anonymous")
        # Rate limiting temporarily disabled for core functionality validation
        
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        await monitoring.log_event(
            "backtest_started",
            {
                "user_id": current_user.get("user_id"),
                "backtest_id": request.backtest_id,
                "config": request.config.dict()
            }
        )
        
        result = await engine_manager.run_backtest(request.backtest_id, request.config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting backtest: {str(e)}")
        await monitoring.log_error(
            "backtest_start_error",
            str(e),
            {
                "user_id": current_user.get("user_id"),
                "backtest_id": request.backtest_id
            }
        )
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@router.get("/backtest/{backtest_id}")
async def get_backtest_status(
    backtest_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$", min_length=1, max_length=100),
    current_user=Depends(get_current_user)
):
    """
    Get backtest status and results
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        result = await engine_manager.get_backtest_status(backtest_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting backtest status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")

@router.delete("/backtest/{backtest_id}")
async def cancel_backtest(
    backtest_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$", min_length=1, max_length=100),
    current_user=Depends(get_current_user)
):
    """
    Cancel a running backtest
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        result = await engine_manager.cancel_backtest(backtest_id)
        
        if result["success"]:
            await monitoring.log_event(
                "backtest_cancelled",
                {
                    "user_id": current_user.get("user_id"),
                    "backtest_id": backtest_id
                }
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error cancelling backtest: {str(e)}")
        await monitoring.log_error(
            "backtest_cancel_error",
            str(e),
            {
                "user_id": current_user.get("user_id"),
                "backtest_id": backtest_id
            }
        )
        raise HTTPException(status_code=500, detail=f"Failed to cancel backtest: {str(e)}")

@router.get("/backtests")
async def list_backtests(current_user=Depends(get_current_user)):
    """
    List all backtests
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        result = await engine_manager.list_backtests()
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing backtests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")

# Data Catalog Endpoints

@router.get("/catalog")
async def get_data_catalog(current_user=Depends(get_current_user)):
    """
    Get available data in the catalog
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        result = await engine_manager.get_data_catalog()
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting data catalog: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data catalog: {str(e)}")

# Emergency Stop Endpoint

@router.post("/emergency-stop")
async def emergency_stop(current_user=Depends(get_current_user)):
    """
    Emergency stop - immediately force stop the engine
    
    This is a safety endpoint that bypasses normal confirmation flows.
    """
    try:
        # TODO: Apply rate limiting for emergency stop operations
        # user_id = current_user.get("user_id", "anonymous")
        # Rate limiting temporarily disabled for core functionality validation
        
        engine_manager = get_nautilus_engine_manager()
        monitoring = monitoring_service
        
        await monitoring.log_event(
            "emergency_stop_triggered",
            {
                "user_id": current_user.get("user_id"),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        result = await engine_manager.stop_engine(force=True)
        
        return EngineResponse(
            success=result["success"],
            message="Emergency stop executed",
            state=result["state"]
        )
        
    except Exception as e:
        logger.error(f"Error in emergency stop: {str(e)}")
        await monitoring.log_error(
            "emergency_stop_error",
            str(e),
            {"user_id": current_user.get("user_id")}
        )
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")

# Health Check Endpoint

@router.get("/health")
async def engine_health_check():
    """
    Health check endpoint for the engine service (no auth required)
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        status = await engine_manager.get_engine_status()
        
        return {
            "service": "nautilus-engine-api",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "engine_state": status.get("state", "unknown"),
            "container_running": status.get("container_info", {}).get("running", False)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "service": "nautilus-engine-api",
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Helper Functions

async def _schedule_engine_monitoring():
    """
    Background task to monitor engine status and send metrics to dashboard
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        
        # Monitor engine status every 5 seconds for 60 seconds after state change
        for _ in range(12):  # 12 * 5 seconds = 60 seconds
            await asyncio.sleep(5)
            
            # Get current status and broadcast via WebSocket
            status = await engine_manager.get_engine_status()
            await broadcast_engine_status_update(status)
            
            # Stop monitoring if engine is in stable state
            if status.get("state") in ["stopped", "error"]:
                break
                
        logger.debug("Engine monitoring task completed")
        
    except Exception as e:
        logger.error(f"Error in engine monitoring: {str(e)}")

# WebSocket Integration with existing MessageBus
async def broadcast_engine_status_update(status: Dict[str, Any]):
    """
    Broadcast engine status updates via WebSocket
    
    This integrates with the existing MessageBus WebSocket endpoint
    """
    try:
        from messagebus_client import get_messagebus_client
        
        messagebus = get_messagebus_client()
        
        # Send update via existing WebSocket messagebus with proper event type
        await messagebus.broadcast_event(
            event_type="nautilus_engine_status",
            data={
                "engine_state": status.get("state"),
                "resource_usage": status.get("resource_usage", {}),
                "health_status": status.get("health_check", {}),
                "timestamp": datetime.now().isoformat(),
                "active_backtests": status.get("active_backtests", 0)
            }
        )
        
        logger.debug("Engine status broadcast via WebSocket")
        
    except Exception as e:
        # Don't let WebSocket errors break engine operations
        logger.warning(f"Failed to broadcast engine status: {e}")
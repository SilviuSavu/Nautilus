"""
FastAPI routes for Strategy Management System
Handles REST API endpoints for strategy templates, configuration, and deployment
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from fastapi.responses import JSONResponse
from typing import Any
from pydantic import BaseModel, Field
from decimal import Decimal
from datetime import datetime
import logging

from strategy_service import strategy_service, StrategyService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/strategies", tags=["strategies"])

# Pydantic models for request/response validation
class ConfigureStrategyRequest(BaseModel):
    template_id: str = Field(..., description="ID of the strategy template to use")
    name: str = Field(..., min_length=1, max_length=100, description="Name for the strategy configuration")
    parameters: dict[str, Any] = Field(..., description="Strategy parameters as key-value pairs")
    risk_settings: dict[str, Any | None] = Field(None, description="Risk management settings")

    class Config:
        schema_extra = {
            "example": {
                "template_id": "ma_cross_001", "name": "EUR/USD MA Cross", "parameters": {
                    "instrument_id": "EUR/USD.SIM", "fast_period": 10, "slow_period": 20, "trade_size": "100000"
                }, "risk_settings": {
                    "max_position_size": "1000000", "stop_loss_atr": 2.0
                }
            }
        }

class DeployStrategyRequest(BaseModel):
    strategy_id: str = Field(..., description="ID of the strategy configuration to deploy")
    deployment_mode: str = Field(..., pattern="^(live|paper|backtest)$", description="Deployment mode")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "config-uuid-here", "deployment_mode": "paper"
            }
        }

class ControlStrategyRequest(BaseModel):
    action: str = Field(..., pattern="^(start|stop|pause|resume)$", description="Control action to perform")
    force: bool | None = Field(False, description="Force action even if strategy is in transition state")

    class Config:
        schema_extra = {
            "example": {
                "action": "start", "force": False
            }
        }

class ParameterValidationRequest(BaseModel):
    template_id: str = Field(..., description="ID of the strategy template")
    parameters: dict[str, Any] = Field(..., description="Parameters to validate")

    class Config:
        schema_extra = {
            "example": {
                "template_id": "ma_cross_001", "parameters": {
                    "fast_period": 10, "slow_period": 20
                }
            }
        }

# Dependency to get strategy service instance
def get_strategy_service() -> StrategyService:
    return strategy_service

# Template Management Endpoints
@router.get("/templates")
async def get_strategy_templates(
    category: str | None = Query(None, description="Filter by strategy category"), search: str | None = Query(None, description="Search templates by name or description"), service: StrategyService = Depends(get_strategy_service)
):
    """
    Get available strategy templates with optional filtering
    """
    try:
        result = service.get_templates(category=category, search=search)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/templates/{template_id}")
async def get_strategy_template(
    template_id: str, service: StrategyService = Depends(get_strategy_service)
):
    """
    Get specific strategy template by ID
    """
    try:
        template = service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        return JSONResponse(content=template)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")

# Configuration Management Endpoints
@router.post("/configure")
async def create_strategy_configuration(
    request: ConfigureStrategyRequest, service: StrategyService = Depends(get_strategy_service)
):
    """
    Create a new strategy configuration from a template
    """
    try:
        result = service.create_configuration(
            template_id=request.template_id, name=request.name, parameters=request.parameters, risk_settings=request.risk_settings
        )
        return JSONResponse(content=result, status_code=201)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create configuration: {str(e)}")

@router.get("/configure/{config_id}")
async def get_strategy_configuration(
    config_id: str, service: StrategyService = Depends(get_strategy_service)
):
    """
    Get strategy configuration by ID
    """
    try:
        config = service.get_configuration(config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Configuration {config_id} not found")
        return JSONResponse(content=config)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting configuration {config_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.get("/configurations")
async def list_strategy_configurations(
    user_id: str | None = Query(None, description="Filter by user ID"), service: StrategyService = Depends(get_strategy_service)
):
    """
    List all strategy configurations, optionally filtered by user
    """
    try:
        configs = service.list_configurations(user_id=user_id)
        return JSONResponse(content=configs)
    except Exception as e:
        logger.error(f"Error listing configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list configurations: {str(e)}")

@router.delete("/configure/{config_id}")
async def delete_strategy_configuration(
    config_id: str, service: StrategyService = Depends(get_strategy_service)
):
    """
    Delete a strategy configuration
    """
    try:
        success = service.delete_configuration(config_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Configuration {config_id} not found")
        return JSONResponse(content={"message": "Configuration deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration {config_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete configuration: {str(e)}")

# Deployment Endpoints
@router.post("/deploy")
async def deploy_strategy(
    request: DeployStrategyRequest, service: StrategyService = Depends(get_strategy_service)
):
    """
    Deploy a strategy configuration to start trading
    """
    try:
        result = service.deploy_strategy(
            config_id=request.strategy_id, deployment_mode=request.deployment_mode
        )
        return JSONResponse(content=result, status_code=201)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deploying strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy strategy: {str(e)}")

# Control Endpoints
@router.post("/control/{strategy_id}")
async def control_strategy(
    strategy_id: str, request: ControlStrategyRequest, service: StrategyService = Depends(get_strategy_service)
):
    """
    Control strategy execution (start, stop, pause, resume)
    """
    try:
        result = service.control_strategy(
            strategy_id=strategy_id, action=request.action, force=request.force
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error controlling strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to control strategy: {str(e)}")

@router.get("/status/{strategy_id}")
async def get_strategy_status(
    strategy_id: str, service: StrategyService = Depends(get_strategy_service)
):
    """
    Get current status and performance metrics for a deployed strategy
    """
    try:
        status = service.get_strategy_status(strategy_id)
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting strategy status {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategy status: {str(e)}")

# Validation Endpoints
@router.post("/validate")
async def validate_strategy_parameters(
    request: ParameterValidationRequest, service: StrategyService = Depends(get_strategy_service)
):
    """
    Validate strategy parameters against template definition
    """
    try:
        result = service.validate_parameters(
            template_id=request.template_id, parameters=request.parameters
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error validating parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate parameters: {str(e)}")

# Utility Endpoints
@router.get("/instruments/available")
async def get_available_instruments(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Get list of available trading instruments
    """
    try:
        instruments = service.get_available_instruments()
        return JSONResponse(content=instruments)
    except Exception as e:
        logger.error(f"Error getting instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get instruments: {str(e)}")

@router.get("/timeframes/available")
async def get_available_timeframes(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Get list of available timeframes
    """
    try:
        timeframes = service.get_available_timeframes()
        return JSONResponse(content=timeframes)
    except Exception as e:
        logger.error(f"Error getting timeframes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeframes: {str(e)}")

@router.get("/venues/available")
async def get_available_venues(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Get list of available trading venues
    """
    try:
        venues = service.get_available_venues()
        return JSONResponse(content=venues)
    except Exception as e:
        logger.error(f"Error getting venues: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get venues: {str(e)}")

@router.get("/health")
async def strategy_service_health(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Strategy service health check endpoint
    """
    try:
        health_status = service.health_check()
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Additional endpoints for future extension
@router.get("/deployments/{deployment_id}/status")
async def get_deployment_status(
    deployment_id: str, service: StrategyService = Depends(get_strategy_service)
):
    """
    Get deployment status by deployment ID
    """
    try:
        # This would be implemented to track specific deployments
        return JSONResponse(content={
            "deployment_id": deployment_id, "status": "running", "message": "Deployment status tracking not yet implemented"
        })
    except Exception as e:
        logger.error(f"Error getting deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get deployment status: {str(e)}")

# Error handling is done through FastAPI's built-in exception handling

# Additional endpoints for Performance Dashboard integration
@router.get("/active")
async def get_active_strategies(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Get all active strategy instances for performance monitoring
    """
    try:
        # Mock data for now - replace with actual database queries
        import random
        from datetime import datetime, timedelta
        
        instances = []
        strategy_ids = ["momentum_1", "mean_revert_2", "arbitrage_3", "market_maker_4"]
        
        for strategy_id in strategy_ids:
            instances.append({
                "id": strategy_id, "config_id": f"config_{strategy_id}", "nautilus_strategy_id": f"nautilus_{strategy_id}", "deployment_id": f"deploy_{strategy_id}", "state": random.choice(["running", "paused", "stopped", "error"]), "performance_metrics": {
                    "total_pnl": str(random.uniform(-2000, 10000)), "unrealized_pnl": str(random.uniform(-500, 2000)), "total_trades": random.randint(50, 300), "winning_trades": random.randint(25, 200), "win_rate": random.uniform(0.45, 0.7), "max_drawdown": str(random.uniform(2, 15)), "sharpe_ratio": random.uniform(0.5, 2.5), "last_updated": datetime.now().isoformat()
                }, "runtime_info": {
                    "orders_placed": random.randint(50, 300), "positions_opened": random.randint(20, 100), "last_signal_time": (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(), "cpu_usage": random.uniform(10, 80), "memory_usage": random.uniform(20, 70), "uptime_seconds": random.randint(3600, 604800)
                }, "started_at": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat()
            })
        
        return JSONResponse(content={"instances": instances})
    except Exception as e:
        logger.error(f"Error getting active strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active strategies: {str(e)}")

@router.get("/monitoring")
async def get_strategy_monitoring_data(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Get real-time monitoring data for all strategies
    """
    try:
        # This endpoint is handled by performance_routes.py
        # Redirecting or providing basic strategy list
        active_strategies = await get_active_strategies(service)
        strategies_data = active_strategies.body.decode()
        
        import json
        strategies = json.loads(strategies_data)
        
        # Enhance with monitoring-specific data
        monitoring_data = []
        for strategy in strategies.get("instances", []):
            monitoring_data.append({
                **strategy, "health_score": random.randint(70, 100) if strategy["state"] == "running" else random.randint(30, 70), "connection_status": random.choice(["connected", "disconnected", "reconnecting"]), "last_signal": random.choice(["buy", "sell", "hold"]) if strategy["state"] == "running" else None, "active_positions": random.randint(0, 10), "pending_orders": random.randint(0, 5), "recent_trades": random.randint(0, 25), "latency_ms": random.uniform(20, 200), "error_rate": random.uniform(0, 5)
            })
        
        return JSONResponse(content={"strategies": monitoring_data})
    except Exception as e:
        logger.error(f"Error getting monitoring data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")

@router.get("/{strategy_id}/performance")
async def get_strategy_performance(
    strategy_id: str, start_date: str | None = Query(None), end_date: str | None = Query(None), service: StrategyService = Depends(get_strategy_service)
):
    """
    Get performance metrics for a specific strategy
    """
    try:
        import random
        from datetime import datetime
        
        # Mock performance data
        total_trades = random.randint(50, 300)
        winning_trades = int(total_trades * random.uniform(0.45, 0.7))
        
        performance = {
            "strategy_id": strategy_id, "total_pnl": random.uniform(-2000, 10000), "unrealized_pnl": random.uniform(-500, 2000), "total_trades": total_trades, "winning_trades": winning_trades, "win_rate": winning_trades / total_trades, "max_drawdown": random.uniform(2, 15), "sharpe_ratio": random.uniform(0.5, 2.5), "daily_pnl_change": random.uniform(-5, 8), "weekly_pnl_change": random.uniform(-12, 20), "monthly_pnl_change": random.uniform(-25, 45), "volatility": random.uniform(10, 35), "calmar_ratio": random.uniform(0.5, 2.8), "sortino_ratio": random.uniform(0.8, 3.2), "max_consecutive_wins": random.randint(3, 12), "max_consecutive_losses": random.randint(2, 8), "profit_factor": random.uniform(1.1, 2.5), "recovery_factor": random.uniform(1.5, 4.0), "daily_returns": [random.uniform(-3, 3) for _ in range(30)], "last_updated": datetime.now().isoformat()
        }
        
        return JSONResponse(content=performance)
    except Exception as e:
        logger.error(f"Error getting strategy performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategy performance: {str(e)}")

@router.get("/signals/recent")
async def get_recent_strategy_signals(
    limit: int = Query(50, ge=1, le=100), service: StrategyService = Depends(get_strategy_service)
):
    """
    Get recent signals from all strategies
    """
    try:
        import random
        from datetime import datetime, timedelta
        
        signals = []
        for i in range(min(limit, 20)):
            signals.append({
                "id": f"signal_{i}", "strategy_id": f"strategy_{i % 4 + 1}", "signal_type": random.choice(["buy", "sell", "hold"]), "instrument": random.choice(["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]), "confidence": random.uniform(0.6, 0.95), "generated_at": (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(), "executed": random.choice([True, False]), "reasoning": f"Technical indicator triggered for position sizing"
            })
        
        return JSONResponse(content={"signals": signals})
    except Exception as e:
        logger.error(f"Error getting recent signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent signals: {str(e)}")

@router.get("/positions/active") 
async def get_active_strategy_positions(
    service: StrategyService = Depends(get_strategy_service)
):
    """
    Get active positions across all strategies
    """
    try:
        import random
        from datetime import datetime
        
        positions = []
        for i in range(8):
            entry_price = random.uniform(100, 300)
            current_price = random.uniform(90, 320)
            quantity = random.randint(100, 1000)
            side = random.choice(["long", "short"])
            
            unrealized_pnl = (current_price - entry_price) * quantity
            if side == "short":
                unrealized_pnl = -unrealized_pnl
                
            positions.append({
                "id": f"position_{i}", "strategy_id": f"strategy_{i % 4 + 1}", "instrument": random.choice(["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]), "side": side, "quantity": quantity, "entry_price": entry_price, "current_price": current_price, "unrealized_pnl": unrealized_pnl, "duration_hours": random.uniform(0.5, 48), "risk_percentage": random.uniform(1, 5)
            })
        
        return JSONResponse(content={"positions": positions})
    except Exception as e:
        logger.error(f"Error getting active positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active positions: {str(e)}")

import random
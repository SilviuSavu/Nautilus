"""
Enhanced Strategy Management API Routes with NautilusTrader Integration
Complete strategy lifecycle management through NautilusTrader execution engine.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Any, Union
from pydantic import BaseModel, Field
from decimal import Decimal
from datetime import datetime
import logging
import asyncio

from strategy_execution_engine import (
    get_strategy_execution_engine, StrategyExecutionEngine, StrategyDeploymentConfig, StrategyState
)
from strategy_serialization import get_strategy_serializer, StrategyConfigSerializer
from strategy_runtime_manager import get_strategy_runtime_manager, StrategyRuntimeManager
from strategy_error_handler import (
    get_strategy_error_handler, log_configuration_error, log_execution_error, ErrorSeverity, ErrorCategory
)

logger = logging.getLogger(__name__)

# Create router for NautilusTrader integration
router = APIRouter(prefix="/api/v1/nautilus/strategies", tags=["nautilus-strategies"])

# Enhanced Pydantic models for NautilusTrader integration

class NautilusDeployRequest(BaseModel):
    """Enhanced deployment request for NautilusTrader"""
    strategy_id: str = Field(..., description="Strategy configuration ID")
    name: str = Field(..., description="Strategy instance name")
    strategy_class: str = Field(..., description="NautilusTrader strategy class name")
    parameters: dict[str, Any] = Field(..., description="Strategy parameters")
    risk_settings: dict[str, Any] = Field(default_factory=dict, description="Risk management settings")
    deployment_mode: str = Field("paper", pattern="^(live|paper|backtest)$", description="Deployment mode")
    auto_start: bool = Field(True, description="Automatically start strategy after deployment")
    risk_check: bool = Field(True, description="Perform risk validation before deployment")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy-uuid-here", "name": "EUR/USD MA Cross - Live", "strategy_class": "MovingAverageCross", "parameters": {
                    "instrument_id": "EUR/USD.SIM", "fast_period": 10, "slow_period": 20, "trade_size": "100000"
                }, "risk_settings": {
                    "max_position_size": "1000000", "stop_loss_atr": 2.0
                }, "deployment_mode": "paper", "auto_start": True, "risk_check": True
            }
        }

class NautilusControlRequest(BaseModel):
    """Enhanced control request for NautilusTrader strategies"""
    action: str = Field(..., pattern="^(start|stop|pause|resume|restart)$", description="Control action")
    force: bool = Field(False, description="Force action even if strategy is in transition")
    reason: str | None = Field(None, description="Reason for the control action")

    class Config:
        schema_extra = {
            "example": {
                "action": "start", "force": False, "reason": "Manual start by user"
            }
        }

class AlertRuleRequest(BaseModel):
    """Alert rule configuration request"""
    name: str = Field(..., description="Alert rule name")
    strategy_id: str | None = Field(None, description="Strategy ID (None for global rules)")
    metric: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., pattern="^(gt|lt|eq|change_pct)$", description="Condition type")
    threshold: float = Field(..., description="Threshold value")
    enabled: bool = Field(True, description="Enable alert rule")

    class Config:
        schema_extra = {
            "example": {
                "name": "High Drawdown Alert", "strategy_id": None, "metric": "current_drawdown", "condition": "gt", "threshold": 0.05, "enabled": True
            }
        }

# Dependency functions
async def get_execution_engine() -> StrategyExecutionEngine:
    """Get strategy execution engine"""
    engine = get_strategy_execution_engine()
    if not engine.trading_node:
        if not await engine.initialize_trading_node():
            raise HTTPException(status_code=503, detail="Strategy execution engine not available")
    return engine

async def get_runtime_manager() -> StrategyRuntimeManager:
    """Get strategy runtime manager"""
    manager = get_strategy_runtime_manager()
    if manager.status.value != "running":
        await manager.start()
    return manager

def get_serializer() -> StrategyConfigSerializer:
    """Get strategy serializer"""
    return get_strategy_serializer()

# Template Management Endpoints (Enhanced)

@router.get("/templates")
async def get_nautilus_strategy_templates(
    category: str | None = Query(None, description="Filter by category"), serializer: StrategyConfigSerializer = Depends(get_serializer)
):
    """Get NautilusTrader-compatible strategy templates"""
    try:
        templates = serializer.get_strategy_templates()
        
        # Filter by category if provided
        if category:
            templates = {
                k: v for k, v in templates.items() 
                if v.get("category") == category
            }
        
        return JSONResponse(content={
            "templates": templates, "categories": list(set(t.get("category", "unknown") for t in templates.values())), "total_count": len(templates)
        })
    except Exception as e:
        log_configuration_error("templates", f"Failed to get strategy templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/templates/{template_id}")
async def get_nautilus_strategy_template(
    template_id: str, serializer: StrategyConfigSerializer = Depends(get_serializer)
):
    """Get specific NautilusTrader strategy template"""
    try:
        template = serializer.get_template_by_id(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        return JSONResponse(content=template)
    except HTTPException:
        raise
    except Exception as e:
        log_configuration_error("templates", f"Failed to get template {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")

# Configuration and Validation Endpoints

@router.post("/validate")
async def validate_nautilus_strategy_config(
    template_id: str = Body(...), parameters: dict[str, Any] = Body(...), serializer: StrategyConfigSerializer = Depends(get_serializer)
):
    """Validate strategy configuration for NautilusTrader"""
    try:
        validation_result = serializer.validate_strategy_config(template_id, parameters)
        
        return JSONResponse(content={
            "is_valid": validation_result.is_valid, "errors": validation_result.errors, "warnings": validation_result.warnings, "normalized_values": validation_result.normalized_values
        })
    except Exception as e:
        log_configuration_error("validation", f"Failed to validate configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/serialize")
async def serialize_to_nautilus_config(
    template_id: str = Body(...), parameters: dict[str, Any] = Body(...), strategy_name: str = Body(...), serializer: StrategyConfigSerializer = Depends(get_serializer)
):
    """Serialize frontend config to NautilusTrader format"""
    try:
        nautilus_config = serializer.serialize_to_nautilus_config(template_id, parameters, strategy_name)
        json_config = serializer.serialize_to_json(nautilus_config)
        
        return JSONResponse(content={
            "nautilus_config": nautilus_config, "json_config": json_config, "template_id": template_id, "strategy_name": strategy_name
        })
    except Exception as e:
        log_configuration_error("serialization", f"Failed to serialize configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Serialization failed: {str(e)}")

# Strategy Deployment and Management

@router.post("/deploy")
async def deploy_nautilus_strategy(
    request: NautilusDeployRequest, background_tasks: BackgroundTasks, engine: StrategyExecutionEngine = Depends(get_execution_engine), serializer: StrategyConfigSerializer = Depends(get_serializer)
):
    """Deploy strategy to NautilusTrader execution engine"""
    try:
        # Create deployment configuration
        deployment_config = StrategyDeploymentConfig(
            strategy_id=request.strategy_id, name=request.name, strategy_class=request.strategy_class, parameters=request.parameters, risk_settings=request.risk_settings, deployment_mode=request.deployment_mode, auto_start=request.auto_start, risk_check=request.risk_check
        )
        
        # Deploy strategy
        result = await engine.deploy_strategy(deployment_config)
        
        if result.get("status") == "deployed":
            return JSONResponse(content=result, status_code=201)
        else:
            await log_execution_error(
                f"Strategy deployment failed: {result.get('error_message', 'Unknown error')}", strategy_id=request.strategy_id, details={"deployment_config": request.dict(), "result": result}
            )
            raise HTTPException(status_code=400, detail=result.get("error_message", "Deployment failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        await log_execution_error(f"Failed to deploy strategy: {str(e)}", strategy_id=request.strategy_id, exception=e)
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@router.post("/control/{deployment_id}")
async def control_nautilus_strategy(
    deployment_id: str, request: NautilusControlRequest, engine: StrategyExecutionEngine = Depends(get_execution_engine)
):
    """Control NautilusTrader strategy execution"""
    try:
        result = await engine.control_strategy(deployment_id, request.action, request.force)
        
        if result.get("status") == "success":
            return JSONResponse(content=result)
        else:
            await log_execution_error(
                f"Strategy control action '{request.action}' failed: {result.get('message')}", strategy_id=deployment_id, details={"action": request.action, "force": request.force, "result": result}
            )
            raise HTTPException(status_code=400, detail=result.get("message", "Control action failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        await log_execution_error(f"Failed to control strategy: {str(e)}", strategy_id=deployment_id, exception=e)
        raise HTTPException(status_code=500, detail=f"Control action failed: {str(e)}")

@router.get("/status/{deployment_id}")
async def get_nautilus_strategy_status(
    deployment_id: str, include_metrics: bool = Query(True, description="Include performance metrics"), engine: StrategyExecutionEngine = Depends(get_execution_engine)
):
    """Get NautilusTrader strategy status and metrics"""
    try:
        status = await engine.get_strategy_status(deployment_id)
        
        if include_metrics:
            # Get additional metrics from runtime manager
            try:
                manager = get_strategy_runtime_manager(engine)
                metrics = manager.get_strategy_metrics(deployment_id, limit=10)
                status["metrics_history"] = metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics history: {e}")
        
        return JSONResponse(content=status)
    except Exception as e:
        await log_execution_error(f"Failed to get strategy status: {str(e)}", strategy_id=deployment_id, exception=e)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/deployed")
async def list_deployed_strategies(
    state: str | None = Query(None, description="Filter by strategy state"), engine: StrategyExecutionEngine = Depends(get_execution_engine)
):
    """List all deployed strategies"""
    try:
        deployed_strategies = engine.get_deployed_strategies()
        
        # Filter by state if provided
        if state:
            try:
                state_enum = StrategyState(state.lower())
                deployed_strategies = {
                    k: v for k, v in deployed_strategies.items()
                    if v.state == state_enum
                }
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid state: {state}")
        
        # Convert to serializable format
        result = {}
        for deployment_id, instance in deployed_strategies.items():
            result[deployment_id] = {
                "id": instance.id, "config_id": instance.config_id, "nautilus_strategy_id": instance.nautilus_strategy_id, "state": instance.state.value, "strategy_class": instance.strategy_class, "parameters": instance.parameters, "performance_metrics": instance.performance_metrics, "runtime_info": instance.runtime_info, "started_at": instance.started_at.isoformat(), "stopped_at": instance.stopped_at.isoformat() if instance.stopped_at else None, "error_count": len(instance.error_log)
            }
        
        return JSONResponse(content={
            "strategies": result, "total_count": len(result), "filtered_by_state": state
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list deployed strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list strategies: {str(e)}")

@router.delete("/deployed/{deployment_id}")
async def remove_deployed_strategy(
    deployment_id: str, force: bool = Query(False, description="Force removal even if strategy is running"), engine: StrategyExecutionEngine = Depends(get_execution_engine)
):
    """Remove a deployed strategy"""
    try:
        success = await engine.remove_strategy(deployment_id)
        
        if success:
            return JSONResponse(content={"message": "Strategy removed successfully"})
        else:
            raise HTTPException(status_code=404, detail="Strategy not found or could not be removed")
            
    except HTTPException:
        raise
    except Exception as e:
        await log_execution_error(f"Failed to remove strategy: {str(e)}", strategy_id=deployment_id, exception=e)
        raise HTTPException(status_code=500, detail=f"Failed to remove strategy: {str(e)}")

# Runtime Management and Monitoring

@router.get("/runtime/status")
async def get_runtime_manager_status(
    manager: StrategyRuntimeManager = Depends(get_runtime_manager)
):
    """Get strategy runtime manager status"""
    try:
        status = manager.get_runtime_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Failed to get runtime status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get runtime status: {str(e)}")

@router.get("/metrics/{deployment_id}")
async def get_strategy_metrics(
    deployment_id: str, limit: int = Query(100, ge=1, le=1000, description="Number of metrics to return"), manager: StrategyRuntimeManager = Depends(get_runtime_manager)
):
    """Get strategy performance metrics history"""
    try:
        metrics = manager.get_strategy_metrics(deployment_id, limit)
        return JSONResponse(content={
            "deployment_id": deployment_id, "metrics": metrics, "count": len(metrics)
        })
    except Exception as e:
        logger.error(f"Failed to get strategy metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/metrics/system")
async def get_system_metrics(
    limit: int = Query(100, ge=1, le=1000, description="Number of metrics to return"), manager: StrategyRuntimeManager = Depends(get_runtime_manager)
):
    """Get system resource metrics"""
    try:
        metrics = manager.get_system_metrics(limit)
        return JSONResponse(content={
            "metrics": metrics, "count": len(metrics)
        })
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

# Alert Management

@router.get("/alerts/rules")
async def get_alert_rules(
    manager: StrategyRuntimeManager = Depends(get_runtime_manager)
):
    """Get all alert rules"""
    try:
        rules = manager.get_alert_rules()
        return JSONResponse(content={"rules": rules, "count": len(rules)})
    except Exception as e:
        logger.error(f"Failed to get alert rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")

@router.post("/alerts/rules")
async def create_alert_rule(
    request: AlertRuleRequest, manager: StrategyRuntimeManager = Depends(get_runtime_manager)
):
    """Create a new alert rule"""
    try:
        from strategy_runtime_manager import AlertRule
        import uuid
        
        rule = AlertRule(
            id=str(uuid.uuid4()), name=request.name, strategy_id=request.strategy_id, metric=request.metric, condition=request.condition, threshold=request.threshold, enabled=request.enabled
        )
        
        success = manager.add_alert_rule(rule)
        
        if success:
            return JSONResponse(content={"message": "Alert rule created", "rule_id": rule.id}, status_code=201)
        else:
            raise HTTPException(status_code=400, detail="Alert rule already exists")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert rule: {str(e)}")

# Error Management

@router.get("/errors")
async def get_strategy_errors(
    severity: str | None = Query(None, description="Filter by severity"), category: str | None = Query(None, description="Filter by category"), strategy_id: str | None = Query(None, description="Filter by strategy ID"), limit: int = Query(100, ge=1, le=1000, description="Number of errors to return")
):
    """Get strategy errors with filtering"""
    try:
        error_handler = get_strategy_error_handler()
        
        # Convert string parameters to enums
        severity_enum = None
        if severity:
            try:
                severity_enum = ErrorSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        category_enum = None
        if category:
            try:
                category_enum = ErrorCategory(category.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        errors = error_handler.get_errors(
            severity=severity_enum, category=category_enum, strategy_id=strategy_id, limit=limit
        )
        
        return JSONResponse(content={
            "errors": errors, "count": len(errors), "filters": {
                "severity": severity, "category": category, "strategy_id": strategy_id
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get errors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get errors: {str(e)}")

@router.get("/errors/summary")
async def get_error_summary():
    """Get error summary statistics"""
    try:
        error_handler = get_strategy_error_handler()
        summary = error_handler.get_error_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Failed to get error summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get error summary: {str(e)}")

# Health and Status

@router.get("/health")
async def nautilus_strategy_health():
    """Health check for NautilusTrader strategy system"""
    try:
        # Check execution engine
        engine = get_strategy_execution_engine()
        engine_status = "available" if engine.trading_node else "not_initialized"
        
        # Check runtime manager
        manager = get_strategy_runtime_manager()
        manager_status = manager.status.value
        
        # Check error handler
        error_handler = get_strategy_error_handler()
        error_summary = error_handler.get_error_summary()
        
        return JSONResponse(content={
            "status": "healthy", "timestamp": datetime.now().isoformat(), "components": {
                "execution_engine": engine_status, "runtime_manager": manager_status, "error_handler": "available"
            }, "error_summary": {
                "total_errors": error_summary["total_errors"], "errors_last_hour": error_summary["errors_last_hour"]
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@router.get("/strategy-classes")
async def get_available_strategy_classes(
    engine: StrategyExecutionEngine = Depends(get_execution_engine)
):
    """Get available NautilusTrader strategy classes"""
    try:
        classes = engine.get_strategy_classes()
        return JSONResponse(content={
            "strategy_classes": classes, "count": len(classes)
        })
    except Exception as e:
        logger.error(f"Failed to get strategy classes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategy classes: {str(e)}")

# Initialize the execution engine on startup
@router.on_event("startup")
async def startup_nautilus_strategies():
    """Initialize NautilusTrader strategy system on startup"""
    try:
        logger.info("Initializing NautilusTrader strategy system...")
        
        # Initialize execution engine
        engine = get_strategy_execution_engine()
        await engine.initialize_trading_node()
        
        # Start runtime manager
        manager = get_strategy_runtime_manager(engine)
        await manager.start()
        
        logger.info("NautilusTrader strategy system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize NautilusTrader strategy system: {e}")

@router.on_event("shutdown")
async def shutdown_nautilus_strategies():
    """Shutdown NautilusTrader strategy system"""
    try:
        logger.info("Shutting down NautilusTrader strategy system...")
        
        # Shutdown runtime manager
        manager = get_strategy_runtime_manager()
        await manager.stop()
        
        # Shutdown execution engine
        engine = get_strategy_execution_engine()
        await engine.shutdown()
        
        logger.info("NautilusTrader strategy system shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
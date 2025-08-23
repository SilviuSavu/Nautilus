"""
Sprint 3 Enhanced API Endpoints
Comprehensive API routes for WebSocket streaming, advanced analytics, and risk management
"""

from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import json
import uuid
from decimal import Decimal

from websocket.websocket_manager import get_websocket_manager
from analytics.performance_calculator import get_performance_calculator
from analytics.risk_analytics import get_risk_analytics
from analytics.strategy_analytics import get_strategy_analytics
from analytics.execution_analytics import get_execution_analytics
from analytics.analytics_aggregator import get_analytics_aggregator, AggregationInterval, DataType
from risk.risk_monitor import get_risk_monitor
from risk_management.limit_engine import limit_engine
from risk_management.breach_detector import BreachDetector
from risk_management.risk_reporter import RiskReporter
from strategies.deployment_manager import deployment_manager, DeploymentEnvironment, DeploymentStrategy, DeploymentConfig
from strategies.version_control import VersionControl
from strategies.rollback_service import RollbackService
from strategies.strategy_tester import StrategyTester
from strategies.pipeline_monitor import PipelineMonitor
from websocket.subscription_manager import get_subscription_manager

logger = logging.getLogger(__name__)

# Request/Response Models

class WebSocketSubscriptionRequest(BaseModel):
    event_types: List[str] = Field(..., description="List of event types to subscribe to")
    filters: Dict[str, Any] = Field(default={}, description="Optional filters")
    
class MarketDataRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to subscribe to")
    data_types: List[str] = Field(default=["trades", "quotes"], description="Data types")
    
class PerformanceAnalysisRequest(BaseModel):
    portfolio_ids: List[str] = Field(..., description="Portfolio IDs to analyze")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    benchmark: Optional[str] = "SPY"
    
class RiskAnalysisRequest(BaseModel):
    portfolio_ids: List[str] = Field(..., description="Portfolio IDs for risk analysis")
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="VaR confidence levels")
    time_horizons: List[int] = Field(default=[1, 5, 10], description="Time horizons in days")
    
class StrategyAnalysisRequest(BaseModel):
    strategy_ids: List[str] = Field(..., description="Strategy IDs to analyze")
    period: str = Field(default="quarterly", description="Analysis period")
    benchmark: Optional[str] = "SPY"
    
class ExecutionAnalysisRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    venue: Optional[str] = None
    strategy_id: Optional[str] = None
    
class AggregationRequest(BaseModel):
    data_type: str = Field(..., description="Data type to aggregate")
    interval: str = Field(..., description="Aggregation interval")
    start_date: datetime
    end_date: datetime
    filters: Dict[str, Any] = Field(default={})
    
class RiskMonitoringRequest(BaseModel):
    portfolio_ids: List[str] = Field(..., description="Portfolios to monitor")
    update_interval: int = Field(default=30, description="Update interval in seconds")
    
class AlertAcknowledgmentRequest(BaseModel):
    alert_ids: List[str] = Field(..., description="Alert IDs to acknowledge")
    acknowledged_by: str = Field(..., description="User acknowledging alerts")

# Risk Management Models
class RiskLimitRequest(BaseModel):
    name: str = Field(..., description="Limit name")
    portfolio_id: Optional[str] = None
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    limit_type: str = Field(..., description="Type of limit (var, concentration, etc.)")
    threshold_value: float = Field(..., description="Critical threshold value")
    warning_threshold: float = Field(..., description="Warning threshold value")
    action: str = Field(..., description="Action to take on breach")
    parameters: Dict[str, Any] = Field(default={}, description="Additional parameters")

# Strategy Management Models
class StrategyDeploymentRequest(BaseModel):
    strategy_id: str = Field(..., description="Strategy ID to deploy")
    version: str = Field(..., description="Version to deploy")
    target_environment: str = Field(..., description="Target environment")
    deployment_strategy: str = Field(default="direct", description="Deployment strategy")
    auto_rollback: bool = Field(default=True, description="Enable auto rollback")
    rollback_threshold: float = Field(default=0.05, description="Loss threshold for rollback")
    canary_percentage: Optional[float] = Field(default=None, description="Canary deployment percentage")
    approval_required: bool = Field(default=False, description="Require manual approval")

class VersionCreateRequest(BaseModel):
    strategy_id: str = Field(..., description="Strategy ID")
    version: str = Field(..., description="Version identifier")
    strategy_code: str = Field(..., description="Strategy source code")
    strategy_config: Dict[str, Any] = Field(..., description="Strategy configuration")
    description: Optional[str] = Field(default=None, description="Version description")
    tags: List[str] = Field(default=[], description="Version tags")

# WebSocket Management Models
class WebSocketConnectionRequest(BaseModel):
    connection_id: str = Field(..., description="Connection ID")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
class SubscriptionRequest(BaseModel):
    connection_id: str = Field(..., description="Connection ID")
    topic: str = Field(..., description="Topic to subscribe to")
    filters: Dict[str, Any] = Field(default={}, description="Subscription filters")
    
class BroadcastRequest(BaseModel):
    topic: str = Field(..., description="Topic to broadcast to")
    message: Dict[str, Any] = Field(..., description="Message to broadcast")
    target_connections: Optional[List[str]] = Field(default=None, description="Specific connections to target")

# System Monitoring Models
class AlertRequest(BaseModel):
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

# API Router
router = APIRouter(prefix="/api/v1/sprint3", tags=["Sprint 3 - Advanced Trading Infrastructure"])

# WebSocket Management Endpoints

@router.post("/websocket/subscribe")
async def subscribe_websocket_events(
    request: WebSocketSubscriptionRequest,
    background_tasks: BackgroundTasks
):
    """
    Subscribe to WebSocket events for real-time updates
    """
    try:
        ws_manager = get_websocket_manager()
        
        # This would typically be called from a WebSocket connection
        # For REST API, we return subscription configuration
        
        return JSONResponse({
            "success": True,
            "subscription_id": f"sub_{int(datetime.utcnow().timestamp())}",
            "event_types": request.event_types,
            "filters": request.filters,
            "message": "Subscription configured. Connect to WebSocket endpoint to receive events."
        })
        
    except Exception as e:
        logger.error(f"Error configuring WebSocket subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/websocket/connections")
async def get_websocket_connections():
    """
    Get current WebSocket connection statistics
    """
    try:
        ws_manager = get_websocket_manager()
        stats = await ws_manager.get_connection_stats()
        
        return JSONResponse({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting WebSocket statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Analytics Endpoints

@router.post("/analytics/performance/analyze")
async def analyze_performance(request: PerformanceAnalysisRequest):
    """
    Comprehensive performance analysis with real-time calculations
    """
    try:
        perf_calc = get_performance_calculator()
        
        results = {}
        
        for portfolio_id in request.portfolio_ids:
            analysis = await perf_calc.calculate_portfolio_performance(
                portfolio_id=portfolio_id,
                start_date=request.start_date,
                end_date=request.end_date or datetime.utcnow(),
                benchmark=request.benchmark
            )
            
            results[portfolio_id] = {
                "total_return": float(analysis.total_return),
                "annualized_return": float(analysis.annualized_return),
                "volatility": float(analysis.volatility),
                "sharpe_ratio": float(analysis.sharpe_ratio),
                "max_drawdown": float(analysis.max_drawdown),
                "var_95": float(analysis.var_95) if analysis.var_95 else None,
                "alpha": float(analysis.alpha) if analysis.alpha else None,
                "beta": float(analysis.beta) if analysis.beta else None,
                "last_updated": analysis.timestamp.isoformat()
            }
        
        return JSONResponse({
            "success": True,
            "analysis_results": results,
            "benchmark": request.benchmark,
            "period": {
                "start": request.start_date.isoformat() if request.start_date else None,
                "end": (request.end_date or datetime.utcnow()).isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/risk/analyze")
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Advanced risk analytics with VaR, stress testing, and exposure analysis
    """
    try:
        risk_analytics = get_risk_analytics()
        
        results = {}
        
        for portfolio_id in request.portfolio_ids:
            # VaR calculations
            var_results = {}
            for confidence_level in request.confidence_levels:
                for time_horizon in request.time_horizons:
                    var_calc = await risk_analytics.calculate_var(
                        portfolio_id=portfolio_id,
                        confidence_level=confidence_level,
                        time_horizon=time_horizon
                    )
                    
                    var_results[f"var_{int(confidence_level*100)}_{time_horizon}d"] = {
                        "var_amount": float(var_calc.var_amount),
                        "expected_shortfall": float(var_calc.expected_shortfall),
                        "method": var_calc.calculation_method
                    }
            
            # Exposure analysis
            exposure_analysis = await risk_analytics.analyze_portfolio_exposure(portfolio_id)
            
            # Stress testing
            stress_results = await risk_analytics.run_stress_test(
                portfolio_id=portfolio_id,
                stress_scenarios=["market_crash", "interest_rate_shock", "sector_rotation"]
            )
            
            results[portfolio_id] = {
                "var_calculations": var_results,
                "exposure_analysis": {
                    "total_exposure": float(exposure_analysis.total_exposure),
                    "net_exposure": float(exposure_analysis.net_exposure),
                    "gross_exposure": float(exposure_analysis.gross_exposure),
                    "sector_exposures": exposure_analysis.sector_exposures,
                    "concentration_risk": float(exposure_analysis.concentration_risk)
                },
                "stress_test_results": [
                    {
                        "scenario": result.scenario_name,
                        "expected_loss": float(result.expected_loss),
                        "probability": result.probability,
                        "impact_breakdown": result.impact_breakdown
                    }
                    for result in stress_results
                ]
            }
        
        return JSONResponse({
            "success": True,
            "risk_analysis": results,
            "calculation_timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/strategy/analyze")
async def analyze_strategies(request: StrategyAnalysisRequest):
    """
    Strategy performance analysis and comparison
    """
    try:
        strategy_analytics = get_strategy_analytics()
        
        # Individual strategy analysis
        strategy_performances = {}
        for strategy_id in request.strategy_ids:
            performance = await strategy_analytics.calculate_strategy_performance(
                strategy_id=strategy_id,
                benchmark=request.benchmark
            )
            
            strategy_performances[strategy_id] = {
                "total_return": performance.total_return,
                "annualized_return": performance.annualized_return,
                "volatility": performance.volatility,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "win_rate": performance.win_rate,
                "profit_factor": performance.profit_factor,
                "alpha": performance.alpha,
                "beta": performance.beta,
                "total_trades": performance.total_trades
            }
        
        # Strategy comparison (if multiple strategies)
        comparison_result = None
        if len(request.strategy_ids) > 1:
            comparison = await strategy_analytics.compare_strategies(
                strategy_ids=request.strategy_ids,
                benchmark=request.benchmark
            )
            
            comparison_result = {
                "correlation_matrix": comparison.correlation_matrix.tolist(),
                "ranking": comparison.ranking,
                "best_performer": comparison.best_performer,
                "worst_performer": comparison.worst_performer
            }
        
        return JSONResponse({
            "success": True,
            "strategy_performances": strategy_performances,
            "strategy_comparison": comparison_result,
            "analysis_period": request.period,
            "benchmark": request.benchmark
        })
        
    except Exception as e:
        logger.error(f"Error in strategy analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/execution/analyze")
async def analyze_execution(request: ExecutionAnalysisRequest):
    """
    Trade execution quality analysis
    """
    try:
        execution_analytics = get_execution_analytics()
        
        # Calculate execution metrics
        metrics = await execution_analytics.calculate_execution_metrics(
            start_date=request.start_date,
            end_date=request.end_date or datetime.utcnow(),
            strategy_id=request.strategy_id,
            venue=request.venue
        )
        
        # Venue analysis
        venue_analyses = await execution_analytics.analyze_venue_performance(
            start_date=request.start_date,
            end_date=request.end_date or datetime.utcnow()
        )
        
        return JSONResponse({
            "success": True,
            "execution_metrics": {
                "total_orders": metrics.total_orders,
                "filled_orders": metrics.filled_orders,
                "fill_rate": metrics.fill_rate,
                "avg_execution_time_ms": metrics.avg_execution_time_ms,
                "avg_slippage_bps": metrics.avg_slippage_bps,
                "total_slippage_cost": float(metrics.total_slippage_cost),
                "market_impact_bps": metrics.market_impact_bps
            },
            "venue_analysis": [
                {
                    "venue": venue.venue,
                    "order_count": venue.order_count,
                    "fill_rate": venue.fill_rate,
                    "avg_slippage_bps": venue.avg_slippage_bps,
                    "market_share_pct": venue.market_share_pct,
                    "quality_score": venue.execution_quality_score
                }
                for venue in venue_analyses
            ]
        })
        
    except Exception as e:
        logger.error(f"Error in execution analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Aggregation Endpoints

@router.post("/aggregation/create-job")
async def create_aggregation_job(request: AggregationRequest):
    """
    Create data aggregation job for historical analysis
    """
    try:
        aggregator = get_analytics_aggregator()
        
        from analytics.analytics_aggregator import AggregationJob, CompressionLevel
        
        job_config = AggregationJob(
            job_id=f"agg_{int(datetime.utcnow().timestamp())}",
            data_type=DataType(request.data_type),
            interval=AggregationInterval(request.interval),
            start_date=request.start_date,
            end_date=request.end_date,
            filters=request.filters,
            compression_level=CompressionLevel.MEDIUM,
            auto_cleanup=True,
            retention_days=365
        )
        
        job_id = await aggregator.create_aggregation_job(job_config)
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "status": "created",
            "config": {
                "data_type": request.data_type,
                "interval": request.interval,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating aggregation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/aggregation/run/{job_id}")
async def run_aggregation_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Execute aggregation job
    """
    try:
        aggregator = get_analytics_aggregator()
        
        # Run job in background
        background_tasks.add_task(aggregator.run_aggregation_job, job_id)
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "status": "running",
            "message": "Aggregation job started in background"
        })
        
    except Exception as e:
        logger.error(f"Error running aggregation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregation/data/{data_type}/{interval}")
async def get_aggregated_data(
    data_type: str,
    interval: str,
    start_date: datetime = Query(..., description="Start date for data retrieval"),
    end_date: datetime = Query(..., description="End date for data retrieval")
):
    """
    Retrieve aggregated analytical data
    """
    try:
        aggregator = get_analytics_aggregator()
        
        data = await aggregator.get_aggregated_data(
            data_type=DataType(data_type),
            interval=AggregationInterval(interval),
            start_date=start_date,
            end_date=end_date
        )
        
        return JSONResponse({
            "success": True,
            "data": data,
            "record_count": len(data)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving aggregated data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Monitoring Endpoints

@router.post("/risk/monitoring/start")
async def start_risk_monitoring(request: RiskMonitoringRequest):
    """
    Start real-time risk monitoring for portfolios
    """
    try:
        risk_monitor = get_risk_monitor()
        
        await risk_monitor.start_monitoring(
            portfolio_ids=request.portfolio_ids,
            load_thresholds=True
        )
        
        return JSONResponse({
            "success": True,
            "message": "Risk monitoring started",
            "portfolios": request.portfolio_ids,
            "update_interval": request.update_interval,
            "status": "active"
        })
        
    except Exception as e:
        logger.error(f"Error starting risk monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/monitoring/stop")
async def stop_risk_monitoring():
    """
    Stop real-time risk monitoring
    """
    try:
        risk_monitor = get_risk_monitor()
        await risk_monitor.stop_monitoring()
        
        return JSONResponse({
            "success": True,
            "message": "Risk monitoring stopped",
            "status": "stopped"
        })
        
    except Exception as e:
        logger.error(f"Error stopping risk monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/realtime/{portfolio_id}")
async def get_realtime_risk(portfolio_id: str):
    """
    Get real-time risk metrics for portfolio
    """
    try:
        risk_monitor = get_risk_monitor()
        
        portfolio_risk = await risk_monitor.get_real_time_risk(portfolio_id)
        
        if not portfolio_risk:
            raise HTTPException(status_code=404, detail="Portfolio risk data not found")
        
        return JSONResponse({
            "success": True,
            "portfolio_risk": {
                "portfolio_id": portfolio_risk.portfolio_id,
                "total_value": float(portfolio_risk.total_value),
                "var_95": float(portfolio_risk.var_95),
                "var_99": float(portfolio_risk.var_99),
                "expected_shortfall": float(portfolio_risk.expected_shortfall),
                "leverage_ratio": portfolio_risk.leverage_ratio,
                "concentration_risk": portfolio_risk.concentration_risk,
                "risk_level": portfolio_risk.risk_level.value,
                "positions_count": portfolio_risk.positions_count,
                "last_updated": portfolio_risk.last_updated.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting real-time risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/alerts/{portfolio_id}")
async def check_risk_alerts(portfolio_id: str):
    """
    Check for active risk alerts for portfolio
    """
    try:
        risk_monitor = get_risk_monitor()
        
        alerts = await risk_monitor.check_risk_breaches(portfolio_id)
        
        return JSONResponse({
            "success": True,
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "priority": alert.priority.value,
                    "description": alert.description,
                    "current_value": float(alert.current_value),
                    "threshold_value": float(alert.threshold_value),
                    "breach_percentage": alert.breach_percentage,
                    "recommended_action": alert.recommended_action,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts
            ],
            "alert_count": len(alerts)
        })
        
    except Exception as e:
        logger.error(f"Error checking risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Status and Health Endpoints

@router.get("/system/health")
async def get_system_health():
    """
    Get comprehensive system health status for Sprint 3 components
    """
    try:
        health_status = {
            "websocket_manager": "unknown",
            "performance_calculator": "unknown",
            "risk_analytics": "unknown",
            "strategy_analytics": "unknown", 
            "execution_analytics": "unknown",
            "analytics_aggregator": "unknown",
            "risk_monitor": "unknown"
        }
        
        # Check each component
        try:
            ws_manager = get_websocket_manager()
            health_status["websocket_manager"] = "healthy" if ws_manager else "unhealthy"
        except:
            health_status["websocket_manager"] = "unhealthy"
        
        try:
            perf_calc = get_performance_calculator()
            health_status["performance_calculator"] = "healthy" if perf_calc else "unhealthy"
        except:
            health_status["performance_calculator"] = "unhealthy"
        
        try:
            risk_analytics = get_risk_analytics()
            health_status["risk_analytics"] = "healthy" if risk_analytics else "unhealthy"
        except:
            health_status["risk_analytics"] = "unhealthy"
        
        try:
            risk_monitor = get_risk_monitor()
            status = risk_monitor.monitoring_status.value if risk_monitor else "stopped"
            health_status["risk_monitor"] = status
        except:
            health_status["risk_monitor"] = "unhealthy"
        
        overall_status = "healthy" if all(
            status in ["healthy", "active", "stopped"] 
            for status in health_status.values()
        ) else "degraded"
        
        return JSONResponse({
            "success": True,
            "overall_status": overall_status,
            "components": health_status,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/metrics")
async def get_system_metrics():
    """
    Get system performance metrics for monitoring
    """
    try:
        # This would integrate with Prometheus metrics
        metrics = {
            "websocket_connections": 0,
            "active_portfolios": 0,
            "calculations_per_minute": 0,
            "alerts_generated": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0
        }
        
        # Get WebSocket stats
        try:
            ws_manager = get_websocket_manager()
            ws_stats = await ws_manager.get_connection_stats()
            metrics["websocket_connections"] = ws_stats.get("active_connections", 0)
        except:
            pass
        
        # Get risk monitor stats
        try:
            risk_monitor = get_risk_monitor()
            metrics["active_portfolios"] = len(risk_monitor.active_portfolios)
        except:
            pass
        
        return JSONResponse({
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Management Endpoints

@router.post("/risk/limits")
async def create_risk_limit(request: RiskLimitRequest):
    """
    Create a new risk limit
    """
    try:
        from risk_management.limit_engine import RiskLimit, LimitType, LimitAction
        
        limit = RiskLimit(
            id=f"limit_{int(datetime.utcnow().timestamp())}",
            name=request.name,
            portfolio_id=request.portfolio_id,
            user_id=request.user_id,
            strategy_id=request.strategy_id,
            limit_type=LimitType(request.limit_type),
            threshold_value=Decimal(str(request.threshold_value)),
            warning_threshold=Decimal(str(request.warning_threshold)),
            action=LimitAction(request.action),
            active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="api",
            parameters=request.parameters
        )
        
        limit_id = await limit_engine.add_limit(limit)
        
        return JSONResponse({
            "success": True,
            "limit_id": limit_id,
            "limit": limit.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error creating risk limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/limits")
async def list_risk_limits(portfolio_id: Optional[str] = Query(None)):
    """
    List all risk limits with optional portfolio filtering
    """
    try:
        status = await limit_engine.get_limit_status(portfolio_id)
        
        return JSONResponse({
            "success": True,
            "limits": status.get("limits", []),
            "active_breaches": status.get("active_breaches", []),
            "statistics": {
                "total_limits": status.get("total_limits", 0),
                "monitoring_active": status.get("monitoring_active", False),
                "check_count": status.get("check_count", 0),
                "breach_prevention_count": status.get("breach_prevention_count", 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/risk/limits/{limit_id}")
async def update_risk_limit(limit_id: str, updates: Dict[str, Any]):
    """
    Update an existing risk limit
    """
    try:
        success = await limit_engine.update_limit(limit_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Limit not found")
        
        return JSONResponse({
            "success": True,
            "message": "Limit updated successfully",
            "limit_id": limit_id
        })
        
    except Exception as e:
        logger.error(f"Error updating risk limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/risk/limits/{limit_id}")
async def delete_risk_limit(limit_id: str):
    """
    Delete a risk limit
    """
    try:
        success = await limit_engine.remove_limit(limit_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Limit not found")
        
        return JSONResponse({
            "success": True,
            "message": "Limit deleted successfully",
            "limit_id": limit_id
        })
        
    except Exception as e:
        logger.error(f"Error deleting risk limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/limits/monitoring/start")
async def start_limit_monitoring():
    """
    Start real-time limit monitoring
    """
    try:
        await limit_engine.start_monitoring()
        
        return JSONResponse({
            "success": True,
            "message": "Limit monitoring started",
            "status": "active"
        })
        
    except Exception as e:
        logger.error(f"Error starting limit monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/limits/monitoring/stop")
async def stop_limit_monitoring():
    """
    Stop real-time limit monitoring
    """
    try:
        await limit_engine.stop_monitoring()
        
        return JSONResponse({
            "success": True,
            "message": "Limit monitoring stopped",
            "status": "stopped"
        })
        
    except Exception as e:
        logger.error(f"Error stopping limit monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/pre-trade-check")
async def check_pre_trade_limits(portfolio_id: str, trade_request: Dict[str, Any]):
    """
    Check if a proposed trade would breach any limits
    """
    try:
        check_result = await limit_engine.check_pre_trade_limits(portfolio_id, trade_request)
        
        return JSONResponse({
            "success": True,
            "check_result": check_result,
            "trade_approved": check_result.get("trade_approved", False)
        })
        
    except Exception as e:
        logger.error(f"Error checking pre-trade limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/breaches/{portfolio_id}")
async def get_risk_breaches(portfolio_id: str):
    """
    Get risk breaches for a portfolio
    """
    try:
        status = await limit_engine.get_limit_status(portfolio_id)
        breaches = status.get("active_breaches", [])
        
        return JSONResponse({
            "success": True,
            "portfolio_id": portfolio_id,
            "active_breaches": breaches,
            "breach_count": len(breaches)
        })
        
    except Exception as e:
        logger.error(f"Error getting risk breaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Strategy Management Endpoints

@router.post("/strategy/deploy")
async def deploy_strategy(request: StrategyDeploymentRequest):
    """
    Deploy a strategy to specified environment
    """
    try:
        # Convert string enums
        target_env = DeploymentEnvironment(request.target_environment)
        deployment_strategy = DeploymentStrategy(request.deployment_strategy)
        
        # Create deployment config
        config = DeploymentConfig(
            strategy=deployment_strategy,
            auto_rollback=request.auto_rollback,
            rollback_threshold=request.rollback_threshold,
            canary_percentage=request.canary_percentage,
            approval_required=request.approval_required
        )
        
        # Execute deployment
        deployment = await deployment_manager.deploy_strategy(
            strategy_id=request.strategy_id,
            version=request.version,
            target_environment=target_env,
            deployment_config=config
        )
        
        return JSONResponse({
            "success": True,
            "deployment": {
                "deployment_id": deployment.deployment_id,
                "strategy_id": deployment.strategy_id,
                "version": deployment.version,
                "environment": deployment.environment.value,
                "status": deployment.status.value,
                "deployed_at": deployment.deployed_at.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error deploying strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/deployments")
async def list_strategy_deployments(
    strategy_id: Optional[str] = Query(None),
    environment: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    List strategy deployments with optional filtering
    """
    try:
        # Convert string filters to enums if provided
        env_filter = DeploymentEnvironment(environment) if environment else None
        from strategies.deployment_manager import DeploymentStatus
        status_filter = DeploymentStatus(status) if status else None
        
        deployments = deployment_manager.list_deployments(
            strategy_id=strategy_id,
            environment=env_filter,
            status=status_filter
        )
        
        deployment_list = [
            {
                "deployment_id": d.deployment_id,
                "strategy_id": d.strategy_id,
                "version": d.version,
                "environment": d.environment.value,
                "status": d.status.value,
                "deployed_by": d.deployed_by,
                "deployed_at": d.deployed_at.isoformat(),
                "completed_at": d.completed_at.isoformat() if d.completed_at else None,
                "health_status": d.health_status
            }
            for d in deployments
        ]
        
        return JSONResponse({
            "success": True,
            "deployments": deployment_list,
            "count": len(deployment_list)
        })
        
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/deployments/{deployment_id}")
async def get_strategy_deployment(deployment_id: str):
    """
    Get detailed deployment information
    """
    try:
        deployment = deployment_manager.get_deployment(deployment_id)
        
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return JSONResponse({
            "success": True,
            "deployment": {
                "deployment_id": deployment.deployment_id,
                "request_id": deployment.request_id,
                "strategy_id": deployment.strategy_id,
                "version": deployment.version,
                "environment": deployment.environment.value,
                "status": deployment.status.value,
                "deployment_config": deployment.deployment_config.dict(),
                "nautilus_deployment_id": deployment.nautilus_deployment_id,
                "previous_version": deployment.previous_version,
                "deployed_by": deployment.deployed_by,
                "deployed_at": deployment.deployed_at.isoformat(),
                "completed_at": deployment.completed_at.isoformat() if deployment.completed_at else None,
                "performance_metrics": deployment.performance_metrics,
                "health_status": deployment.health_status,
                "error_log": deployment.error_log
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/versions")
async def create_strategy_version(request: VersionCreateRequest):
    """
    Create a new strategy version
    """
    try:
        version_control = VersionControl()
        
        version_info = await version_control.create_version(
            strategy_id=request.strategy_id,
            version=request.version,
            strategy_code=request.strategy_code,
            strategy_config=request.strategy_config,
            description=request.description,
            tags=request.tags
        )
        
        return JSONResponse({
            "success": True,
            "version_info": {
                "strategy_id": version_info.strategy_id,
                "version": version_info.version,
                "created_at": version_info.created_at.isoformat(),
                "created_by": version_info.created_by,
                "description": version_info.description,
                "tags": version_info.tags,
                "checksum": version_info.checksum
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating strategy version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/versions/{strategy_id}")
async def list_strategy_versions(strategy_id: str):
    """
    List all versions of a strategy
    """
    try:
        version_control = VersionControl()
        versions = version_control.list_versions(strategy_id)
        
        version_list = [
            {
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "created_by": v.created_by,
                "description": v.description,
                "tags": v.tags,
                "checksum": v.checksum
            }
            for v in versions
        ]
        
        return JSONResponse({
            "success": True,
            "strategy_id": strategy_id,
            "versions": version_list,
            "count": len(version_list)
        })
        
    except Exception as e:
        logger.error(f"Error listing strategy versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/rollback")
async def rollback_strategy(
    strategy_id: str,
    environment: str,
    target_version: str,
    reason: Optional[str] = "manual_rollback"
):
    """
    Rollback strategy to previous version
    """
    try:
        rollback_service = RollbackService()
        target_env = DeploymentEnvironment(environment)
        
        result = await rollback_service.execute_rollback(
            strategy_id=strategy_id,
            environment=target_env,
            target_version=target_version,
            reason=reason
        )
        
        return JSONResponse({
            "success": result.get("success", False),
            "rollback_result": result,
            "message": result.get("message", "Rollback completed")
        })
        
    except Exception as e:
        logger.error(f"Error rolling back strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/statistics")
async def get_deployment_statistics():
    """
    Get deployment statistics and metrics
    """
    try:
        stats = deployment_manager.get_deployment_statistics()
        
        return JSONResponse({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting deployment statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket Management Endpoints

@router.post("/websocket/connections")
async def register_websocket_connection(request: WebSocketConnectionRequest):
    """
    Register a new WebSocket connection
    """
    try:
        ws_manager = get_websocket_manager()
        
        connection_info = {
            "connection_id": request.connection_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "connected_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        # Register connection with manager
        await ws_manager.register_connection(request.connection_id, connection_info)
        
        return JSONResponse({
            "success": True,
            "connection_info": connection_info
        })
        
    except Exception as e:
        logger.error(f"Error registering WebSocket connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/websocket/connections/{connection_id}")
async def unregister_websocket_connection(connection_id: str):
    """
    Unregister a WebSocket connection
    """
    try:
        ws_manager = get_websocket_manager()
        await ws_manager.unregister_connection(connection_id)
        
        return JSONResponse({
            "success": True,
            "message": f"Connection {connection_id} unregistered"
        })
        
    except Exception as e:
        logger.error(f"Error unregistering WebSocket connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/websocket/subscriptions")
async def create_subscription(request: SubscriptionRequest):
    """
    Create a WebSocket topic subscription
    """
    try:
        subscription_manager = get_subscription_manager()
        
        subscription_id = await subscription_manager.subscribe(
            connection_id=request.connection_id,
            topic=request.topic,
            filters=request.filters
        )
        
        return JSONResponse({
            "success": True,
            "subscription_id": subscription_id,
            "connection_id": request.connection_id,
            "topic": request.topic,
            "filters": request.filters
        })
        
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/websocket/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: str):
    """
    Delete a WebSocket subscription
    """
    try:
        subscription_manager = get_subscription_manager()
        await subscription_manager.unsubscribe(subscription_id)
        
        return JSONResponse({
            "success": True,
            "message": f"Subscription {subscription_id} deleted"
        })
        
    except Exception as e:
        logger.error(f"Error deleting subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/websocket/broadcast")
async def broadcast_message(request: BroadcastRequest):
    """
    Broadcast message to WebSocket subscribers
    """
    try:
        ws_manager = get_websocket_manager()
        
        broadcast_count = await ws_manager.broadcast_to_topic(
            topic=request.topic,
            message=request.message,
            target_connections=request.target_connections
        )
        
        return JSONResponse({
            "success": True,
            "topic": request.topic,
            "broadcast_count": broadcast_count,
            "message": "Message broadcast successfully"
        })
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/websocket/stats")
async def get_websocket_statistics():
    """
    Get WebSocket connection and subscription statistics
    """
    try:
        ws_manager = get_websocket_manager()
        subscription_manager = get_subscription_manager()
        
        ws_stats = await ws_manager.get_connection_stats()
        subscription_stats = await subscription_manager.get_statistics()
        
        return JSONResponse({
            "success": True,
            "websocket_stats": ws_stats,
            "subscription_stats": subscription_stats,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting WebSocket statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Monitoring Endpoints

@router.post("/system/alerts")
async def create_alert(request: AlertRequest):
    """
    Create a system alert
    """
    try:
        alert_id = f"alert_{int(datetime.utcnow().timestamp())}"
        
        alert = {
            "id": alert_id,
            "type": request.alert_type,
            "severity": request.severity,
            "message": request.message,
            "metadata": request.metadata,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "acknowledged": False
        }
        
        # Store alert (in production, would use proper storage)
        # For now, we'll use the WebSocket manager to broadcast it
        ws_manager = get_websocket_manager()
        await ws_manager.broadcast_to_topic("system.alerts", alert)
        
        return JSONResponse({
            "success": True,
            "alert": alert
        })
        
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/performance")
async def get_system_performance():
    """
    Get system performance metrics
    """
    try:
        try:
            import psutil
            import os
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            performance_metrics = {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_used_gb": round(memory.used / (1024**3), 2),
                    "memory_percent": memory.percent,
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_percent": round((disk.used / disk.total) * 100, 2)
                },
                "process": {
                    "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                    "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                    "cpu_percent": process.cpu_percent()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except ImportError:
            # Fallback if psutil is not available
            performance_metrics = {
                "system": {
                    "cpu_percent": 0,
                    "memory_percent": 0,
                    "disk_percent": 0
                },
                "process": {
                    "memory_rss_mb": 0,
                    "cpu_percent": 0
                },
                "timestamp": datetime.utcnow().isoformat(),
                "note": "Detailed metrics unavailable - psutil not installed"
            }
        
        return JSONResponse({
            "success": True,
            "performance_metrics": performance_metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting system performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/components")
async def get_component_status():
    """
    Get status of all Sprint 3 components
    """
    try:
        components = {
            "websocket_manager": {"status": "unknown", "last_check": datetime.utcnow().isoformat()},
            "analytics_aggregator": {"status": "unknown", "last_check": datetime.utcnow().isoformat()},
            "risk_monitor": {"status": "unknown", "last_check": datetime.utcnow().isoformat()},
            "limit_engine": {"status": "unknown", "last_check": datetime.utcnow().isoformat()},
            "deployment_manager": {"status": "unknown", "last_check": datetime.utcnow().isoformat()}
        }
        
        # Check each component
        try:
            ws_manager = get_websocket_manager()
            components["websocket_manager"]["status"] = "healthy" if ws_manager else "unhealthy"
        except:
            components["websocket_manager"]["status"] = "unhealthy"
        
        try:
            aggregator = get_analytics_aggregator()
            components["analytics_aggregator"]["status"] = "healthy" if aggregator else "unhealthy"
        except:
            components["analytics_aggregator"]["status"] = "unhealthy"
        
        try:
            risk_monitor = get_risk_monitor()
            components["risk_monitor"]["status"] = "healthy" if risk_monitor else "unhealthy"
            if risk_monitor:
                components["risk_monitor"]["monitoring_active"] = risk_monitor.monitoring_status.value
        except:
            components["risk_monitor"]["status"] = "unhealthy"
        
        try:
            components["limit_engine"]["status"] = "healthy" if limit_engine else "unhealthy"
            components["limit_engine"]["monitoring_active"] = limit_engine.monitoring_active
        except:
            components["limit_engine"]["status"] = "unhealthy"
        
        try:
            components["deployment_manager"]["status"] = "healthy" if deployment_manager else "unhealthy"
            stats = deployment_manager.get_deployment_statistics()
            components["deployment_manager"]["active_deployments"] = stats.get("total_deployments", 0)
        except:
            components["deployment_manager"]["status"] = "unhealthy"
        
        overall_status = "healthy" if all(c["status"] == "healthy" for c in components.values()) else "degraded"
        
        return JSONResponse({
            "success": True,
            "overall_status": overall_status,
            "components": components,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting component status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional Portfolio Analytics Endpoints

@router.get("/analytics/portfolio/{portfolio_id}/summary")
async def get_portfolio_summary(portfolio_id: str):
    """
    Get comprehensive portfolio summary with real-time metrics
    """
    try:
        # Get performance metrics
        perf_calc = get_performance_calculator()
        performance = await perf_calc.calculate_portfolio_performance(
            portfolio_id=portfolio_id,
            end_date=datetime.utcnow()
        )
        
        # Get risk metrics
        risk_monitor = get_risk_monitor()
        risk_metrics = await risk_monitor.get_real_time_risk(portfolio_id)
        
        # Get limit status
        limit_status = await limit_engine.get_limit_status(portfolio_id)
        
        summary = {
            "portfolio_id": portfolio_id,
            "performance": {
                "total_return": float(performance.total_return) if performance else 0,
                "annualized_return": float(performance.annualized_return) if performance else 0,
                "volatility": float(performance.volatility) if performance else 0,
                "sharpe_ratio": float(performance.sharpe_ratio) if performance else 0,
                "max_drawdown": float(performance.max_drawdown) if performance else 0
            },
            "risk": {
                "var_95": float(risk_metrics.var_95) if risk_metrics else 0,
                "var_99": float(risk_metrics.var_99) if risk_metrics else 0,
                "leverage_ratio": risk_metrics.leverage_ratio if risk_metrics else 0,
                "concentration_risk": risk_metrics.concentration_risk if risk_metrics else 0,
                "risk_level": risk_metrics.risk_level.value if risk_metrics else "unknown"
            },
            "limits": {
                "active_limits": len(limit_status.get("limits", [])),
                "active_breaches": len(limit_status.get("active_breaches", [])),
                "monitoring_active": limit_status.get("monitoring_active", False)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse({
            "success": True,
            "portfolio_summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export router
sprint3_router = router
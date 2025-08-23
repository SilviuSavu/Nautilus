"""
FastAPI routes for Risk Management System
Handles REST API endpoints for risk calculations, alerts, and limits
Enhanced with Sprint 3 risk management capabilities
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from fastapi.responses import JSONResponse
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
from decimal import Decimal
from datetime import datetime
import logging
import numpy as np

from risk_service import risk_service, RiskService, PortfolioRisk, RiskMetrics, ExposureAnalysis, RiskAlert, RiskLimit
from risk_calculator import risk_calculator

# Enhanced risk management integration
try:
    from risk_management.risk_monitor import risk_monitor
    from risk_management.limit_engine import limit_engine, RiskLimit as EnhancedRiskLimit, LimitType, LimitAction
    from risk_management.breach_detector import breach_detector
    from risk_management.risk_reporter import risk_reporter, ReportType, ReportFormat
    from risk_management.enhanced_risk_calculator import enhanced_risk_calculator
    ENHANCED_RISK_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced risk management not available")
    ENHANCED_RISK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/risk", tags=["risk"])

@router.get("/status")
async def risk_status():
    """Risk service status for frontend health checks"""
    return {
        "status": "healthy",
        "service": "risk-management",
        "available_endpoints": [
            "/portfolio/{portfolio_id}",
            "/calculate",
            "/exposure/{portfolio_id}",
            "/alerts/{portfolio_id}"
        ]
    }

# Pydantic models for request/response validation
class RiskCalculationRequest(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio identifier")
    calculation_type: str = Field(..., pattern="^(var|correlation|exposure|stress_test|all)$", description="Type of risk calculation")
    confidence_levels: Optional[List[float]] = Field([0.95, 0.99], description="Confidence levels for VaR")
    time_horizons: Optional[List[int]] = Field([1, 7, 30], description="Time horizons in days")
    include_stress_tests: Optional[bool] = Field(False, description="Include stress test scenarios")

class RiskLimitRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Risk limit name")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    limit_type: str = Field(..., pattern="^(var|concentration|position_size|leverage|correlation)$", description="Type of risk limit")
    threshold_value: Decimal = Field(..., description="Threshold value for the limit")
    warning_threshold: Decimal = Field(..., description="Warning threshold value")
    action: str = Field(..., pattern="^(warn|block|reduce|notify)$", description="Action to take when limit is breached")
    active: bool = Field(True, description="Whether the limit is active")

class PreTradeRiskRequest(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio identifier")
    instrument: str = Field(..., description="Instrument symbol")
    quantity: Decimal = Field(..., description="Trade quantity")
    side: str = Field(..., pattern="^(buy|sell)$", description="Trade side")
    order_type: str = Field(..., description="Order type")

class StressTestRequest(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio identifier")
    scenario_name: str = Field(..., description="Stress test scenario name")
    market_shocks: List[Dict[str, Any]] = Field(..., description="Market shock parameters")

# Dependencies
def get_risk_service() -> RiskService:
    return risk_service

# Portfolio Risk Overview Routes
@router.get("/portfolio/{portfolio_id}")
async def get_portfolio_risk(
    portfolio_id: str,
    service: RiskService = Depends(get_risk_service)
):
    """Get comprehensive portfolio risk overview"""
    try:
        # In a real implementation, we would fetch actual position and price data
        # For now, we'll return a mock response to establish the API structure
        mock_positions = [
            {
                'symbol': 'AAPL',
                'quantity': '100',
                'market_value': '15000.00',
                'unrealized_pnl': '500.00'
            },
            {
                'symbol': 'GOOGL',
                'quantity': '50',
                'market_value': '12000.00',
                'unrealized_pnl': '-200.00'
            }
        ]
        
        # Mock price history (in real implementation, fetch from market data service)
        mock_price_history = {
            'AAPL': [150, 152, 148, 151, 149, 153, 150],
            'GOOGL': [2400, 2420, 2380, 2410, 2390, 2430, 2400]
        }
        
        portfolio_risk = await service.calculate_portfolio_risk(
            portfolio_id, mock_positions, mock_price_history
        )
        
        return {
            "portfolio_id": portfolio_risk.portfolio_id,
            "var_1d": str(portfolio_risk.var_1d_95),
            "var_1w": str(portfolio_risk.var_1w_95),
            "var_1m": str(portfolio_risk.var_1m_95),
            "expected_shortfall": str(portfolio_risk.expected_shortfall_95),
            "beta": portfolio_risk.beta_vs_market,
            "correlation_matrix": [],  # Will be empty for mock data
            "concentration_risk": [],  # Will be empty for mock data
            "total_exposure": str(sum(float(pos['market_value']) for pos in mock_positions)),
            "last_calculated": portfolio_risk.calculated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate")
async def calculate_risk_metrics(
    request: RiskCalculationRequest,
    service: RiskService = Depends(get_risk_service)
):
    """Calculate specific risk metrics based on request parameters"""
    try:
        # Mock implementation - in reality would fetch data and calculate
        calculation_result = {
            "portfolio_id": request.portfolio_id,
            "status": "success",
            "calculations": {
                "risk_metrics": {
                    "var_1d_95": "2500.50",
                    "var_1d_99": "3200.75",
                    "var_1w_95": "6500.25",
                    "expected_shortfall_95": "3800.00",
                    "beta_vs_market": 1.2,
                    "portfolio_volatility": 0.18,
                    "calculated_at": datetime.utcnow().isoformat()
                }
            },
            "calculation_time_ms": 150,
            "warnings": []
        }
        
        return calculation_result
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{portfolio_id}")
async def get_risk_metrics(
    portfolio_id: str,
    service: RiskService = Depends(get_risk_service)
):
    """Get risk metrics for a specific portfolio"""
    try:
        # Mock positions for demonstration
        mock_positions = [
            {
                'symbol': 'AAPL',
                'quantity': '100',
                'market_value': '15000.00',
                'unrealized_pnl': '500.00'
            },
            {
                'symbol': 'GOOGL',
                'quantity': '50',
                'market_value': '12000.00',
                'unrealized_pnl': '-200.00'
            }
        ]
        
        # Mock price history
        mock_price_history = {
            'AAPL': [150, 152, 148, 151, 149, 153, 150],
            'GOOGL': [2400, 2420, 2380, 2410, 2390, 2430, 2400]
        }
        
        risk_metrics = await service.calculate_portfolio_risk(
            portfolio_id, mock_positions, mock_price_history
        )
        
        return {
            "portfolio_id": risk_metrics.portfolio_id,
            "var_1d_95": str(risk_metrics.var_1d_95),
            "var_1d_99": str(risk_metrics.var_1d_99),
            "var_1w_95": str(risk_metrics.var_1w_95),
            "var_1w_99": str(risk_metrics.var_1w_99),
            "var_1m_95": str(risk_metrics.var_1m_95),
            "var_1m_99": str(risk_metrics.var_1m_99),
            "expected_shortfall_95": str(risk_metrics.expected_shortfall_95),
            "expected_shortfall_99": str(risk_metrics.expected_shortfall_99),
            "beta_vs_market": risk_metrics.beta_vs_market,
            "portfolio_volatility": risk_metrics.portfolio_volatility,
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "max_drawdown": str(risk_metrics.max_drawdown),
            "correlation_with_market": risk_metrics.correlation_with_market,
            "tracking_error": risk_metrics.tracking_error,
            "information_ratio": risk_metrics.information_ratio,
            "calculated_at": risk_metrics.calculated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Exposure Analysis Routes
@router.get("/exposure/{portfolio_id}")
async def get_exposure_analysis(
    portfolio_id: str,
    service: RiskService = Depends(get_risk_service)
):
    """Get detailed portfolio exposure analysis"""
    try:
        # Mock positions for demonstration
        mock_positions = [
            {
                'symbol': 'AAPL',
                'quantity': '100',
                'market_value': '15000.00',
                'unrealized_pnl': '500.00'
            },
            {
                'symbol': 'GOOGL',
                'quantity': '50',
                'market_value': '12000.00',
                'unrealized_pnl': '-200.00'
            }
        ]
        
        exposure_analysis = await service.get_exposure_analysis(portfolio_id, mock_positions)
        
        return {
            "total_exposure": str(exposure_analysis.total_exposure),
            "long_exposure": str(exposure_analysis.long_exposure),
            "short_exposure": str(exposure_analysis.short_exposure),
            "net_exposure": str(exposure_analysis.net_exposure),
            "by_instrument": exposure_analysis.by_instrument,
            "by_sector": exposure_analysis.by_sector,
            "by_currency": exposure_analysis.by_currency,
            "by_geography": exposure_analysis.by_geography
        }
        
    except Exception as e:
        logger.error(f"Error getting exposure analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exposure/{portfolio_id}/breakdown")
async def get_exposure_breakdown(
    portfolio_id: str,
    breakdown_type: str = Query(..., pattern="^(instrument|sector|currency|geography)$"),
    service: RiskService = Depends(get_risk_service)
):
    """Get exposure breakdown by specific category"""
    try:
        # Mock breakdown data
        breakdown_data = {
            "breakdown_type": breakdown_type,
            "data": [
                {"category": "Technology", "exposure": "15000.00", "percentage": 55.6},
                {"category": "Healthcare", "exposure": "8000.00", "percentage": 29.6},
                {"category": "Financial", "exposure": "4000.00", "percentage": 14.8}
            ],
            "total_exposure": "27000.00"
        }
        
        return breakdown_data
        
    except Exception as e:
        logger.error(f"Error getting exposure breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Alerts Routes
@router.get("/alerts/{portfolio_id}")
async def get_risk_alerts(
    portfolio_id: str,
    active_only: bool = Query(True, description="Return only active alerts"),
    service: RiskService = Depends(get_risk_service)
):
    """Get risk alerts for a portfolio"""
    try:
        # Mock alerts
        alerts = [
            {
                "id": "alert-001",
                "portfolio_id": portfolio_id,
                "alert_type": "limit_breach",
                "severity": "warning",
                "message": "VaR 95% exceeds warning threshold",
                "triggered_at": datetime.utcnow().isoformat(),
                "acknowledged": False
            }
        ]
        
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Body(..., embed=True),
    service: RiskService = Depends(get_risk_service)
):
    """Acknowledge a risk alert"""
    try:
        # Mock acknowledgment
        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Limits Routes
@router.get("/limits/{portfolio_id}")
async def get_risk_limits(
    portfolio_id: str,
    active_only: bool = Query(True, description="Return only active limits"),
    service: RiskService = Depends(get_risk_service)
):
    """Get risk limits for a portfolio"""
    try:
        # Mock limits
        limits = [
            {
                "id": "limit-001",
                "name": "Daily VaR Limit",
                "portfolio_id": portfolio_id,
                "limit_type": "var",
                "threshold_value": "5000.00",
                "warning_threshold": "4000.00",
                "action": "warn",
                "active": True,
                "breach_count": 0,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return {"limits": limits}
        
    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits")
async def create_risk_limit(
    request: RiskLimitRequest,
    service: RiskService = Depends(get_risk_service)
):
    """Create a new risk limit"""
    try:
        # Create new limit
        new_limit = {
            "id": f"limit-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "name": request.name,
            "portfolio_id": request.portfolio_id,
            "limit_type": request.limit_type,
            "threshold_value": str(request.threshold_value),
            "warning_threshold": str(request.warning_threshold),
            "action": request.action,
            "active": request.active,
            "breach_count": 0,
            "last_breach": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return new_limit
        
    except Exception as e:
        logger.error(f"Error creating risk limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/limits/{limit_id}")
async def update_risk_limit(
    limit_id: str,
    updates: Dict[str, Any],
    service: RiskService = Depends(get_risk_service)
):
    """Update an existing risk limit"""
    try:
        # Mock update
        updated_limit = {
            "id": limit_id,
            "updated_at": datetime.utcnow().isoformat(),
            **updates
        }
        
        return updated_limit
        
    except Exception as e:
        logger.error(f"Error updating risk limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/limits/{limit_id}")
async def delete_risk_limit(
    limit_id: str,
    service: RiskService = Depends(get_risk_service)
):
    """Delete a risk limit"""
    try:
        return {"deleted": True, "limit_id": limit_id}
        
    except Exception as e:
        logger.error(f"Error deleting risk limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Pre-Trade Risk Assessment
@router.post("/assess")
async def assess_pre_trade_risk(
    request: PreTradeRiskRequest,
    service: RiskService = Depends(get_risk_service)
):
    """Assess risk impact of a proposed trade"""
    try:
        # Mock assessment
        assessment = {
            "portfolio_id": request.portfolio_id,
            "proposed_trade": {
                "instrument": request.instrument,
                "quantity": str(request.quantity),
                "side": request.side,
                "order_type": request.order_type
            },
            "risk_impact": {
                "var_impact": "150.25",
                "concentration_impact": 2.5,
                "correlation_impact": 0.1,
                "leverage_impact": 0.05
            },
            "limit_violations": [],
            "recommendation": "approve",
            "risk_score": 25.5
        }
        
        return assessment
        
    except Exception as e:
        logger.error(f"Error assessing pre-trade risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Position Sizing Routes
@router.get("/position-sizing/{portfolio_id}/{instrument}")
async def get_position_sizing_recommendation(
    portfolio_id: str,
    instrument: str,
    target_risk_pct: Optional[float] = Query(None, description="Target risk percentage"),
    service: RiskService = Depends(get_risk_service)
):
    """Get position sizing recommendation"""
    try:
        # Mock recommendation
        recommendation = {
            "portfolio_id": portfolio_id,
            "instrument": instrument,
            "current_position": "100.00",
            "recommended_position": "150.00",
            "max_position_size": "200.00",
            "reasoning": "Based on VaR and correlation analysis",
            "risk_adjusted_size": "145.00",
            "var_based_size": "140.00",
            "volatility_adjusted_size": "155.00",
            "confidence_score": 85.5
        }
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Error getting position sizing recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chart Data Routes
@router.get("/charts/{portfolio_id}")
async def get_risk_chart_data(
    portfolio_id: str,
    chart_type: str = Query(..., pattern="^(var_history|correlation_matrix|concentration|exposure_timeline)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    service: RiskService = Depends(get_risk_service)
):
    """Get risk chart data for visualization"""
    try:
        # Mock chart data
        chart_data = {
            "chart_type": chart_type,
            "data": [
                {"date": "2024-01-01", "var_95": 2500.0, "var_99": 3200.0},
                {"date": "2024-01-02", "var_95": 2600.0, "var_99": 3300.0}
            ],
            "metadata": {
                "portfolio_id": portfolio_id,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Error getting risk chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check
@router.get("/health")
async def risk_health_check(service: RiskService = Depends(get_risk_service)):
    """Health check for risk management service"""
    try:
        health_status = await service.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"Error in risk health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Stress Testing Routes
@router.post("/stress-tests/run")
async def run_stress_test(
    request: StressTestRequest,
    service: RiskService = Depends(get_risk_service)
):
    """Run a stress test scenario"""
    try:
        # Mock stress test result
        result = {
            "scenario_name": request.scenario_name,
            "portfolio_id": request.portfolio_id,
            "portfolio_pnl_impact": "-5000.00",
            "var_impact": "1500.00",
            "position_impacts": [
                {
                    "instrument": "AAPL",
                    "current_value": "15000.00",
                    "shocked_value": "13500.00",
                    "pnl_impact": "-1500.00"
                }
            ],
            "risk_metrics_impact": {
                "var_change": "500.00",
                "beta_change": 0.1,
                "correlation_change": 0.05
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stress-tests/scenarios")
async def get_stress_test_scenarios(service: RiskService = Depends(get_risk_service)):
    """Get available stress test scenarios"""
    try:
        scenarios = [
            {
                "id": "scenario-001",
                "name": "Market Crash 2008",
                "description": "Simulates 2008 financial crisis conditions",
                "scenario_type": "historical"
            },
            {
                "id": "scenario-002",
                "name": "Interest Rate Shock",
                "description": "Sudden 2% increase in interest rates",
                "scenario_type": "hypothetical"
            }
        ]
        
        return {"scenarios": scenarios}
        
    except Exception as e:
        logger.error(f"Error getting stress test scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Position Integration Routes (Story 4.3 Task 6)
@router.get("/position-risk/{portfolio_id}")
async def get_position_based_risk(
    portfolio_id: str = "main",
    service: RiskService = Depends(get_risk_service)
):
    """Calculate risk metrics using real position data from Story 3.4"""
    try:
        risk_analysis = await service.calculate_position_risk(portfolio_id)
        return risk_analysis
        
    except Exception as e:
        logger.error(f"Error calculating position-based risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pre-trade-assessment")
async def assess_pre_trade_risk(
    trade_request: Dict[str, Any],
    portfolio_id: str = Query("main", description="Portfolio ID"),
    service: RiskService = Depends(get_risk_service)
):
    """Assess risk impact of a proposed trade before execution"""
    try:
        assessment = await service.assess_pre_trade_risk(portfolio_id, trade_request)
        return assessment
        
    except Exception as e:
        logger.error(f"Error in pre-trade risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/position-breakdown/{portfolio_id}")
async def get_position_risk_breakdown(
    portfolio_id: str = "main",
    service: RiskService = Depends(get_risk_service)
):
    """Get risk breakdown by individual position"""
    try:
        breakdown = await service.get_position_risk_breakdown(portfolio_id)
        return {"portfolio_id": portfolio_id, "positions": breakdown}
        
    except Exception as e:
        logger.error(f"Error getting position risk breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ENHANCED RISK MANAGEMENT ENDPOINTS (Sprint 3)
# =============================================================================

# Pydantic models for enhanced endpoints
class EnhancedRiskLimitRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    portfolio_id: Optional[str] = None
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    limit_type: str = Field(..., description="var, concentration, position_size, leverage, correlation, exposure, drawdown, volatility, notional")
    threshold_value: Decimal = Field(..., gt=0)
    warning_threshold: Decimal = Field(..., gt=0)
    action: str = Field(..., description="warn, block, reduce, notify, liquidate, freeze")
    active: bool = True
    parameters: Optional[Dict[str, Any]] = None

class ScenarioAnalysisRequest(BaseModel):
    portfolio_id: str
    scenario_name: str
    custom_shocks: Optional[Dict[str, float]] = None
    monte_carlo_runs: int = Field(10000, ge=1000, le=100000)

class ReportGenerationRequest(BaseModel):
    report_type: str = Field(..., description="daily_risk, weekly_summary, monthly_summary, regulatory, stress_test, etc.")
    portfolio_ids: List[str]
    format: str = Field("json", description="json, pdf, excel, csv, html")
    parameters: Optional[Dict[str, Any]] = None

# Real-time Risk Monitoring Endpoints
@router.get("/monitor/status")
async def get_risk_monitoring_status():
    """Get real-time risk monitoring status"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        status = await risk_monitor.get_monitoring_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor/start")
async def start_risk_monitoring(portfolio_ids: List[str] = Body(...)):
    """Start real-time risk monitoring for specified portfolios"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        await risk_monitor.start_monitoring(portfolio_ids)
        return {"status": "started", "portfolios": portfolio_ids}
        
    except Exception as e:
        logger.error(f"Error starting risk monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor/stop")
async def stop_risk_monitoring():
    """Stop real-time risk monitoring"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        await risk_monitor.stop_monitoring()
        return {"status": "stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping risk monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitor/realtime/{portfolio_id}")
async def get_realtime_risk_metrics(portfolio_id: str):
    """Get real-time risk metrics for a portfolio"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            # Fallback to basic calculator
            snapshot = await risk_calculator.get_real_time_risk_snapshot(portfolio_id)
            if snapshot:
                return snapshot
            raise HTTPException(status_code=404, detail="No real-time data available")
        
        snapshot = await risk_monitor.get_current_risk_metrics(portfolio_id)
        if snapshot:
            return snapshot.to_dict()
        
        raise HTTPException(status_code=404, detail="Portfolio not found or not monitored")
        
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitor/history/{portfolio_id}")
async def get_risk_history(
    portfolio_id: str,
    hours: int = Query(1, ge=1, le=168, description="Hours of history (max 1 week)")
):
    """Get risk metrics history for a portfolio"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        history = await risk_monitor.get_risk_history(portfolio_id, hours)
        return {
            "portfolio_id": portfolio_id,
            "hours": hours,
            "history": [snapshot.to_dict() for snapshot in history]
        }
        
    except Exception as e:
        logger.error(f"Error getting risk history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Risk Limit Management
@router.get("/limits/enhanced")
async def get_enhanced_risk_limits(portfolio_id: Optional[str] = Query(None)):
    """Get enhanced risk limits with utilization"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        status = await limit_engine.get_limit_status(portfolio_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting enhanced limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits/enhanced")
async def create_enhanced_risk_limit(request: EnhancedRiskLimitRequest):
    """Create enhanced risk limit with real-time monitoring"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        # Create enhanced risk limit
        from risk_management.limit_engine import RiskLimit
        
        limit = RiskLimit(
            id=f"limit_{int(datetime.utcnow().timestamp())}",
            name=request.name,
            portfolio_id=request.portfolio_id,
            user_id=request.user_id,
            strategy_id=request.strategy_id,
            limit_type=LimitType(request.limit_type),
            threshold_value=request.threshold_value,
            warning_threshold=request.warning_threshold,
            action=LimitAction(request.action),
            active=request.active,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="api",
            parameters=request.parameters
        )
        
        limit_id = await limit_engine.add_limit(limit)
        return {"limit_id": limit_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating enhanced limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits/enhanced/start-monitoring")
async def start_limit_monitoring():
    """Start real-time limit monitoring"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        await limit_engine.start_monitoring()
        return {"status": "started", "message": "Limit monitoring started"}
        
    except Exception as e:
        logger.error(f"Error starting limit monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits/pre-trade-check")
async def check_pre_trade_limits(
    portfolio_id: str,
    trade_request: Dict[str, Any] = Body(...)
):
    """Check if proposed trade would breach limits"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        check_result = await limit_engine.check_pre_trade_limits(portfolio_id, trade_request)
        return check_result
        
    except Exception as e:
        logger.error(f"Error checking pre-trade limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Breach Detection and Alerts
@router.get("/alerts/active")
async def get_active_alerts(portfolio_id: Optional[str] = Query(None)):
    """Get active risk alerts"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        alerts = await breach_detector.get_active_alerts(portfolio_id)
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_risk_alert(
    alert_id: str,
    acknowledged_by: str = Body(..., embed=True)
):
    """Acknowledge a risk alert"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        success = await breach_detector.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return {"alert_id": alert_id, "acknowledged": True, "acknowledged_by": acknowledged_by}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/statistics")
async def get_breach_statistics(
    portfolio_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168)
):
    """Get breach and alert statistics"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        stats = await breach_detector.get_breach_statistics(portfolio_id, hours)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting breach statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Risk Analytics
@router.post("/analytics/comprehensive")
async def get_comprehensive_risk_analysis(
    portfolio_id: str = Body(...),
    include_scenarios: bool = Body(False),
    include_attribution: bool = Body(True)
):
    """Get comprehensive risk analysis with advanced metrics"""
    try:
        # Mock data for demonstration
        mock_returns_data = {
            'AAPL': [0.01, -0.02, 0.015, -0.01, 0.02],
            'GOOGL': [0.015, -0.015, 0.02, -0.012, 0.018]
        }
        
        mock_positions = {
            'AAPL': {'market_value': 15000, 'quantity': 100},
            'GOOGL': {'market_value': 12000, 'quantity': 50}
        }
        
        mock_weights = {'AAPL': 0.6, 'GOOGL': 0.4}
        
        # Use enhanced calculator if available
        if ENHANCED_RISK_AVAILABLE:
            analysis = await enhanced_risk_calculator.comprehensive_portfolio_analysis(
                portfolio_id=portfolio_id,
                returns_data={k: np.array(v) for k, v in mock_returns_data.items()},
                positions=mock_positions,
                portfolio_weights=mock_weights,
                analysis_config={'include_scenarios': include_scenarios}
            )
        else:
            # Fallback to basic analysis
            analysis = await risk_calculator.calculate_enhanced_risk_metrics(
                portfolio_id=portfolio_id,
                returns_data={k: np.array(v) for k, v in mock_returns_data.items()},
                positions=mock_positions,
                portfolio_weights=mock_weights,
                include_scenarios=include_scenarios
            )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/scenario")
async def run_scenario_analysis(request: ScenarioAnalysisRequest):
    """Run scenario analysis or stress testing"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        # Mock portfolio returns for demonstration
        mock_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.025, -0.015])
        
        scenario_result = await enhanced_risk_calculator.run_custom_scenario(
            portfolio_returns=mock_returns,
            scenario_name=request.scenario_name,
            custom_shocks=request.custom_shocks,
            monte_carlo_runs=request.monte_carlo_runs
        )
        
        return scenario_result
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/attribution/{portfolio_id}")
async def get_risk_attribution(portfolio_id: str):
    """Get risk attribution analysis"""
    try:
        # Mock data
        mock_returns_data = {
            'AAPL': np.array([0.01, -0.02, 0.015, -0.01, 0.02]),
            'GOOGL': np.array([0.015, -0.015, 0.02, -0.012, 0.018])
        }
        mock_weights = {'AAPL': 0.6, 'GOOGL': 0.4}
        
        attribution = await risk_calculator.calculate_position_risk_attribution(
            mock_returns_data, mock_weights
        )
        
        return {
            "portfolio_id": portfolio_id,
            "attribution": attribution
        }
        
    except Exception as e:
        logger.error(f"Error getting risk attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Reporting and Dashboards
@router.get("/dashboard/{portfolio_id}")
async def get_risk_dashboard(portfolio_id: str):
    """Get real-time risk dashboard data"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        dashboard_data = await risk_reporter.get_dashboard_data(portfolio_id)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reports/generate")
async def generate_risk_report(request: ReportGenerationRequest):
    """Generate risk report"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        report = await risk_reporter.generate_report(
            report_type=ReportType(request.report_type),
            portfolio_ids=request.portfolio_ids,
            parameters=request.parameters,
            format=ReportFormat(request.format)
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/status")
async def get_reporting_status():
    """Get risk reporting system status"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        status = await risk_reporter.get_report_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting reporting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Integration and Status
@router.get("/enhanced/status")
async def get_enhanced_risk_status():
    """Get status of enhanced risk management system"""
    try:
        status = {
            "enhanced_available": ENHANCED_RISK_AVAILABLE,
            "components": {}
        }
        
        if ENHANCED_RISK_AVAILABLE:
            status["components"] = {
                "risk_monitor": await risk_monitor.get_monitoring_status(),
                "limit_engine": await limit_engine.get_limit_status(),
                "breach_detector": await breach_detector.get_monitoring_status(),
                "risk_reporter": await risk_reporter.get_report_status(),
                "enhanced_calculator": await enhanced_risk_calculator.get_calculation_status()
            }
        
        # Add integration status from basic calculator
        status["integration"] = risk_calculator.get_integration_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting enhanced risk status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enhanced/initialize")
async def initialize_enhanced_risk_system():
    """Initialize the enhanced risk management system"""
    try:
        if not ENHANCED_RISK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced risk management not available")
        
        # Start all services
        initialization_results = {}
        
        try:
            await risk_monitor.start_monitoring(['main'])  # Default portfolio
            initialization_results['risk_monitor'] = 'started'
        except Exception as e:
            initialization_results['risk_monitor'] = f'error: {e}'
        
        try:
            await limit_engine.start_monitoring()
            initialization_results['limit_engine'] = 'started'
        except Exception as e:
            initialization_results['limit_engine'] = f'error: {e}'
        
        try:
            await breach_detector.start_monitoring()
            initialization_results['breach_detector'] = 'started'
        except Exception as e:
            initialization_results['breach_detector'] = f'error: {e}'
        
        try:
            await risk_reporter.start_services()
            initialization_results['risk_reporter'] = 'started'
        except Exception as e:
            initialization_results['risk_reporter'] = f'error: {e}'
        
        return {
            "status": "initialized",
            "results": initialization_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initializing enhanced risk system: {e}")
        raise HTTPException(status_code=500, detail=str(e))
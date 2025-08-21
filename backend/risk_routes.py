"""
FastAPI routes for Risk Management System
Handles REST API endpoints for risk calculations, alerts, and limits
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from fastapi.responses import JSONResponse
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
from decimal import Decimal
from datetime import datetime
import logging

from risk_service import risk_service, RiskService, PortfolioRisk, RiskMetrics, ExposureAnalysis, RiskAlert, RiskLimit

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
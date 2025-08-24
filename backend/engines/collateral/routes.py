"""
Collateral Management API Routes
===============================

FastAPI routes for the collateral management engine providing:
- Real-time margin calculations
- Cross-margining optimization
- Regulatory compliance monitoring
- Live margin alerts and monitoring
- Stress testing and scenario analysis
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from fastapi.responses import StreamingResponse
import json

from collateral_engine import CollateralManagementEngine
from models import (
    Portfolio, Position, AssetClass, MarginAlert, AlertSeverity
)
from margin_monitor import MonitoringConfig

logger = logging.getLogger(__name__)

# Global engine instance
_collateral_engine: Optional[CollateralManagementEngine] = None


def get_collateral_engine() -> CollateralManagementEngine:
    """Get or create the global collateral engine instance"""
    global _collateral_engine
    if _collateral_engine is None:
        _collateral_engine = CollateralManagementEngine()
        # Initialize in background
        asyncio.create_task(_collateral_engine.initialize())
    return _collateral_engine


# Pydantic models for API requests/responses
class PositionRequest(BaseModel):
    """Request model for position data"""
    id: str
    symbol: str
    quantity: float
    market_value: float
    asset_class: str
    currency: str = "USD"
    sector: Optional[str] = None
    country: Optional[str] = None
    duration: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    @validator('asset_class')
    def validate_asset_class(cls, v):
        if v.upper() not in [ac.value.upper() for ac in AssetClass]:
            raise ValueError(f"Invalid asset class: {v}")
        return v.lower()


class PortfolioRequest(BaseModel):
    """Request model for portfolio data"""
    id: str
    name: str
    positions: List[PositionRequest]
    available_cash: float
    currency: str = "USD"
    leverage_ratio: Optional[float] = None


class MarginCalculationRequest(BaseModel):
    """Request for margin calculation"""
    portfolio: PortfolioRequest
    optimize: bool = Field(True, description="Apply cross-margining optimization")
    include_stress_test: bool = Field(False, description="Include stress test analysis")


class MonitoringRequest(BaseModel):
    """Request to start/stop monitoring"""
    portfolio: PortfolioRequest
    monitoring_config: Optional[Dict[str, Any]] = None


class CollateralAllocationRequest(BaseModel):
    """Request for collateral allocation optimization"""
    portfolio: PortfolioRequest
    available_collateral: Dict[str, float]


class StressTestRequest(BaseModel):
    """Request for stress testing"""
    portfolio: PortfolioRequest
    scenarios: Optional[List[str]] = None


def _convert_portfolio_request(portfolio_req: PortfolioRequest) -> Portfolio:
    """Convert API request to internal Portfolio model"""
    positions = []
    for pos_req in portfolio_req.positions:
        # Map asset class string to enum
        asset_class_map = {ac.value: ac for ac in AssetClass}
        asset_class = asset_class_map[pos_req.asset_class.lower()]
        
        position = Position(
            id=pos_req.id,
            symbol=pos_req.symbol,
            quantity=Decimal(str(pos_req.quantity)),
            market_value=Decimal(str(pos_req.market_value)),
            asset_class=asset_class,
            currency=pos_req.currency,
            sector=pos_req.sector,
            country=pos_req.country,
            duration=Decimal(str(pos_req.duration)) if pos_req.duration else None,
            implied_volatility=Decimal(str(pos_req.implied_volatility)) if pos_req.implied_volatility else None,
            delta=Decimal(str(pos_req.delta)) if pos_req.delta else None,
            gamma=Decimal(str(pos_req.gamma)) if pos_req.gamma else None,
            theta=Decimal(str(pos_req.theta)) if pos_req.theta else None,
            vega=Decimal(str(pos_req.vega)) if pos_req.vega else None
        )
        positions.append(position)
    
    return Portfolio(
        id=portfolio_req.id,
        name=portfolio_req.name,
        positions=positions,
        available_cash=Decimal(str(portfolio_req.available_cash)),
        currency=portfolio_req.currency,
        leverage_ratio=Decimal(str(portfolio_req.leverage_ratio)) if portfolio_req.leverage_ratio else None
    )


# Initialize router
router = APIRouter(prefix="/api/v1/collateral", tags=["Collateral Management"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        engine = get_collateral_engine()
        status = await engine.get_engine_status()
        return {
            "status": "healthy",
            "engine_status": status['engine_status'],
            "active_portfolios": status['active_portfolios'],
            "hardware_acceleration": status['hardware_acceleration'],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@router.get("/status")
async def get_engine_status():
    """Get detailed engine status and performance metrics"""
    try:
        engine = get_collateral_engine()
        return await engine.get_engine_status()
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/margin/calculate")
async def calculate_margin(request: MarginCalculationRequest):
    """
    Calculate comprehensive margin requirements for a portfolio
    
    This endpoint provides:
    - Base margin calculations by position and asset class
    - Cross-margining optimization for capital efficiency
    - Regulatory capital requirements
    - Integration with Risk Engine (if available)
    - Optional stress testing
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(request.portfolio)
        
        # Calculate margin with optimization
        result = await engine.calculate_portfolio_margin(portfolio, request.optimize)
        
        # Add stress testing if requested
        if request.include_stress_test:
            stress_results = await engine.run_margin_stress_test(portfolio)
            result['stress_test'] = stress_results
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error calculating margin: {e}")
        raise HTTPException(status_code=500, detail=f"Margin calculation failed: {str(e)}")


@router.post("/margin/optimize")
async def optimize_margin(request: PortfolioRequest):
    """
    Optimize margin requirements through advanced cross-margining strategies
    
    Returns detailed analysis of potential margin savings and capital efficiency improvements
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(request)
        
        # Run optimization
        optimization_result = await engine.collateral_optimizer.optimize_portfolio_margin(portfolio)
        
        return {
            "success": True,
            "data": {
                "original_margin": float(optimization_result.original_margin),
                "optimized_margin": float(optimization_result.optimized_margin),
                "margin_savings": float(optimization_result.margin_savings),
                "capital_efficiency_improvement": float(optimization_result.capital_efficiency_improvement),
                "cross_margin_benefits": [
                    {
                        "asset_class": benefit.asset_class.value,
                        "position_count": len(benefit.position_ids),
                        "gross_margin": float(benefit.gross_margin),
                        "cross_margin_offset": float(benefit.cross_margin_offset),
                        "offset_percentage": float(benefit.offset_percentage),
                        "calculation_method": benefit.calculation_method
                    }
                    for benefit in optimization_result.cross_margin_benefits
                ],
                "computation_time_ms": optimization_result.computation_time_ms,
                "optimization_method": optimization_result.optimization_method
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing margin: {e}")
        raise HTTPException(status_code=500, detail=f"Margin optimization failed: {str(e)}")


@router.post("/monitoring/start")
async def start_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks):
    """
    Start real-time margin monitoring for a portfolio
    
    Provides continuous monitoring with:
    - Real-time margin calculations (every 5 seconds)
    - Predictive margin call alerts
    - Multiple alert severity levels
    - Integration with MessageBus for notifications
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(request.portfolio)
        
        # Convert monitoring config if provided
        monitoring_config = None
        if request.monitoring_config:
            monitoring_config = MonitoringConfig(**request.monitoring_config)
        
        # Start monitoring in background
        success = await engine.start_real_time_monitoring(portfolio)
        
        if success:
            return {
                "success": True,
                "message": f"Real-time monitoring started for portfolio {portfolio.id}",
                "portfolio_id": portfolio.id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start monitoring")
            
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/monitoring/stop/{portfolio_id}")
async def stop_monitoring(portfolio_id: str):
    """Stop real-time margin monitoring for a portfolio"""
    try:
        engine = get_collateral_engine()
        success = await engine.stop_real_time_monitoring(portfolio_id)
        
        if success:
            return {
                "success": True,
                "message": f"Monitoring stopped for portfolio {portfolio_id}",
                "portfolio_id": portfolio_id
            }
        else:
            raise HTTPException(status_code=404, detail="Portfolio not found or monitoring not active")
            
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/monitoring/status/{portfolio_id}")
async def get_monitoring_status(portfolio_id: str):
    """Get current monitoring status for a portfolio"""
    try:
        engine = get_collateral_engine()
        status = await engine.margin_monitor.get_monitoring_status(portfolio_id)
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "monitoring_status": status
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {str(e)}")


@router.post("/collateral/optimize-allocation")
async def optimize_collateral_allocation(request: CollateralAllocationRequest):
    """
    Optimize allocation of different types of collateral to minimize margin requirements
    
    Takes available collateral (cash, bonds, equities) and suggests optimal allocation
    considering haircuts and preferences
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(request.portfolio)
        
        # Convert collateral amounts to Decimal
        available_collateral = {
            k: Decimal(str(v)) for k, v in request.available_collateral.items()
        }
        
        result = await engine.optimize_collateral_allocation(portfolio, available_collateral)
        
        # Convert Decimal values back to float for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            return obj
        
        return {
            "success": True,
            "data": convert_decimals(result)
        }
        
    except Exception as e:
        logger.error(f"Error optimizing collateral allocation: {e}")
        raise HTTPException(status_code=500, detail=f"Collateral optimization failed: {str(e)}")


@router.post("/stress-test")
async def run_stress_test(request: StressTestRequest):
    """
    Run comprehensive margin stress tests under various market scenarios
    
    Tests portfolio margin requirements under:
    - Market crash scenarios
    - Volatility spikes
    - Liquidity crises
    - Custom scenarios
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(request.portfolio)
        
        result = await engine.run_margin_stress_test(portfolio, request.scenarios)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")


@router.get("/regulatory/report/{portfolio_id}")
async def generate_regulatory_report(
    portfolio_request: PortfolioRequest,
    jurisdiction: str = Query("US", description="Regulatory jurisdiction"),
    entity_type: str = Query("hedge_fund", description="Entity type for regulatory purposes")
):
    """
    Generate comprehensive regulatory compliance report
    
    Covers:
    - Basel III requirements
    - Dodd-Frank compliance
    - EMIR requirements
    - Local regulatory requirements
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(portfolio_request)
        
        # Create regulatory calculator for specific jurisdiction
        from regulatory_calculator import RegulatoryCapitalCalculator
        reg_calculator = RegulatoryCapitalCalculator(jurisdiction, entity_type)
        
        report = await reg_calculator.generate_regulatory_report(portfolio)
        
        return {
            "success": True,
            "data": report
        }
        
    except Exception as e:
        logger.error(f"Error generating regulatory report: {e}")
        raise HTTPException(status_code=500, detail=f"Regulatory report generation failed: {str(e)}")


@router.post("/reports/comprehensive")
async def generate_comprehensive_report(request: PortfolioRequest):
    """
    Generate comprehensive collateral management report including:
    - Complete margin analysis with optimization
    - Stress test results
    - Regulatory compliance status
    - Real-time monitoring status
    - Actionable recommendations
    """
    try:
        engine = get_collateral_engine()
        portfolio = _convert_portfolio_request(request)
        
        report = await engine.generate_comprehensive_report(portfolio)
        
        return {
            "success": True,
            "data": report
        }
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/alerts/stream/{portfolio_id}")
async def stream_margin_alerts(portfolio_id: str):
    """
    Stream real-time margin alerts for a portfolio
    
    Returns Server-Sent Events (SSE) stream of margin alerts and updates
    """
    async def alert_stream():
        try:
            engine = get_collateral_engine()
            
            # Check if portfolio is being monitored
            status = await engine.margin_monitor.get_monitoring_status(portfolio_id)
            if not status.get('is_monitoring', False):
                yield f"data: {json.dumps({'error': 'Portfolio not being monitored'})}\n\n"
                return
            
            # Stream alerts (simplified implementation - would need proper SSE setup in production)
            alert_queue = asyncio.Queue()
            
            # Register alert callback
            async def alert_callback(alert: MarginAlert):
                if alert.portfolio_id == portfolio_id:
                    await alert_queue.put({
                        'type': 'alert',
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'margin_utilization': float(alert.margin_utilization),
                        'time_to_margin_call_minutes': alert.time_to_margin_call_minutes,
                        'recommended_action': alert.recommended_action.value if alert.recommended_action else None,
                        'timestamp': alert.created_at.isoformat()
                    })
            
            engine.alert_callbacks.append(alert_callback)
            
            # Stream alerts for 5 minutes (timeout)
            timeout_count = 0
            while timeout_count < 60:  # 60 * 5 seconds = 5 minutes
                try:
                    alert_data = await asyncio.wait_for(alert_queue.get(), timeout=5.0)
                    yield f"data: {json.dumps(alert_data)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                    timeout_count += 1
            
            # Clean up callback
            if alert_callback in engine.alert_callbacks:
                engine.alert_callbacks.remove(alert_callback)
                
        except Exception as e:
            logger.error(f"Error in alert stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(alert_stream(), media_type="text/plain")


@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get engine performance metrics and statistics"""
    try:
        engine = get_collateral_engine()
        status = await engine.get_engine_status()
        
        return {
            "success": True,
            "data": {
                "performance_metrics": status['performance_metrics'],
                "active_portfolios": status['active_portfolios'],
                "hardware_acceleration": status['hardware_acceleration'],
                "components_status": status['components'],
                "uptime": status['uptime']
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/emergency/stop-all-monitoring")
async def emergency_stop_all_monitoring():
    """
    Emergency endpoint to stop all margin monitoring
    
    Use only in case of system issues or maintenance
    """
    try:
        engine = get_collateral_engine()
        await engine.margin_monitor.emergency_stop_monitoring()
        
        return {
            "success": True,
            "message": "All margin monitoring stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")


# Add shutdown handler for graceful cleanup
@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _collateral_engine
    if _collateral_engine:
        try:
            await _collateral_engine.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            _collateral_engine = None
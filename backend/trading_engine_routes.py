"""
Trading Engine API Routes
=========================

RESTful API endpoints for the professional trading engine with real-time
order management, execution, risk controls, and position tracking.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from trading_engine import (
    OrderManagementSystem, ExecutionEngine, RealTimeRiskEngine, PositionKeeper,
    Order, OrderType, OrderSide, TimeInForce, RiskLimits, IBKRVenue
)
from production_auth import get_current_user, require_permission, PermissionLevel

logger = logging.getLogger(__name__)

# Initialize trading engine components
oms = OrderManagementSystem()
execution_engine = ExecutionEngine()
risk_engine = RealTimeRiskEngine()
position_keeper = PositionKeeper()

# Connect components
oms.set_risk_engine(risk_engine)
oms.set_execution_engine(execution_engine)
risk_engine.set_position_keeper(position_keeper)
execution_engine.add_execution_callback(oms.handle_fill)
execution_engine.add_execution_callback(position_keeper.process_fill)

router = APIRouter(prefix="/api/v1/trading", tags=["Trading Engine"])


# Request/Response Models
class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side: buy or sell")
    order_type: str = Field(..., description="Order type: market, limit, stop, etc.")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    stop_price: Optional[float] = Field(None, description="Stop price (for stop orders)")
    time_in_force: str = Field("day", description="Time in force")
    portfolio_id: str = Field("default", description="Portfolio ID")
    strategy_id: Optional[str] = Field(None, description="Strategy ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Order tags")


class OrderResponse(BaseModel):
    order_id: str
    status: str
    message: str


class CancelOrderRequest(BaseModel):
    reason: str = Field("User requested", description="Cancellation reason")


class RiskLimitsRequest(BaseModel):
    max_position_size: float = Field(..., gt=0)
    max_portfolio_concentration: float = Field(..., gt=0, le=1)
    max_sector_concentration: float = Field(..., gt=0, le=1)
    max_daily_loss: float = Field(..., gt=0)
    max_drawdown_percent: float = Field(..., gt=0, le=1)
    max_gross_leverage: float = Field(..., gt=0)
    max_net_leverage: float = Field(..., gt=0)
    max_var_99: float = Field(..., gt=0)
    max_expected_shortfall: float = Field(..., gt=0)
    max_intraday_turnover: float = Field(..., gt=0)
    max_order_size: float = Field(..., gt=0)
    enforce_position_limits: bool = True
    enforce_loss_limits: bool = True
    enforce_leverage_limits: bool = True
    enforce_concentration_limits: bool = True
    enforce_var_limits: bool = True


class MarketDataUpdate(BaseModel):
    symbol: str
    price: float


# Order Management Endpoints
@router.post("/orders", response_model=OrderResponse)
async def submit_order(
    order_request: OrderRequest,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("trade"))
):
    """Submit a new trading order."""
    try:
        # Validate enums
        try:
            side = OrderSide(order_request.side.lower())
            order_type = OrderType(order_request.order_type.lower())
            time_in_force = TimeInForce(order_request.time_in_force.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")
        
        # Create order
        order = Order(
            symbol=order_request.symbol,
            side=side,
            order_type=order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=time_in_force,
            portfolio_id=order_request.portfolio_id,
            strategy_id=order_request.strategy_id,
            client_order_id=order_request.client_order_id,
            tags=order_request.tags
        )
        
        # Submit order
        order_id = await oms.submit_order(order)
        
        return OrderResponse(
            order_id=order_id,
            status="submitted",
            message=f"Order {order_id} submitted successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Order submission failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    cancel_request: CancelOrderRequest,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("trade"))
):
    """Cancel an existing order."""
    try:
        success = await oms.cancel_order(order_id, cancel_request.reason)
        
        if success:
            return {"message": f"Order {order_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Order not found or cannot be cancelled")
            
    except Exception as e:
        logger.error(f"Order cancellation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orders/{order_id}")
async def get_order(
    order_id: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get order details by ID."""
    order = oms.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return order.to_dict()


@router.get("/orders")
async def get_orders(
    portfolio_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    active_only: bool = False,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get orders with optional filtering."""
    try:
        if active_only:
            orders = oms.get_active_orders(portfolio_id)
        elif portfolio_id:
            orders = oms.get_orders_by_portfolio(portfolio_id)
        elif strategy_id:
            orders = oms.get_orders_by_strategy(strategy_id)
        else:
            orders = list(oms.orders.values())
        
        return [order.to_dict() for order in orders]
        
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Position Management Endpoints
@router.get("/positions/{portfolio_id}")
async def get_positions(
    portfolio_id: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get all positions for a portfolio."""
    try:
        summary = await position_keeper.get_portfolio_summary(portfolio_id)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/positions/{portfolio_id}/{symbol}")
async def get_position(
    portfolio_id: str,
    symbol: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get specific position details."""
    try:
        position = await position_keeper.get_position(portfolio_id, symbol)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        return position.to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get position: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/positions/{portfolio_id}/{symbol}/close")
async def close_position(
    portfolio_id: str,
    symbol: str,
    price: Optional[float] = None,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("trade"))
):
    """Close a position."""
    try:
        success = await position_keeper.close_position(portfolio_id, symbol, price)
        
        if success:
            return {"message": f"Position {symbol} in {portfolio_id} closed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Position not found or already flat")
            
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/positions/{portfolio_id}/{symbol}/history")
async def get_position_history(
    portfolio_id: str,
    symbol: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get position fill history."""
    try:
        history = await position_keeper.get_position_history(portfolio_id, symbol)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get position history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Risk Management Endpoints
@router.post("/risk/limits/{portfolio_id}")
async def set_risk_limits(
    portfolio_id: str,
    limits_request: RiskLimitsRequest,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("admin"))
):
    """Set risk limits for a portfolio."""
    try:
        limits = RiskLimits(
            portfolio_id=portfolio_id,
            **limits_request.dict()
        )
        
        risk_engine.set_portfolio_limits(portfolio_id, limits)
        
        return {"message": f"Risk limits updated for portfolio {portfolio_id}"}
        
    except Exception as e:
        logger.error(f"Failed to set risk limits: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/risk/limits/{portfolio_id}")
async def get_risk_limits(
    portfolio_id: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get risk limits for a portfolio."""
    limits = risk_engine.get_portfolio_limits(portfolio_id)
    if not limits:
        raise HTTPException(status_code=404, detail="Risk limits not found")
    
    return limits.__dict__


@router.get("/risk/metrics/{portfolio_id}")
async def get_risk_metrics(
    portfolio_id: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get real-time risk metrics for a portfolio."""
    try:
        positions = await position_keeper.get_positions(portfolio_id)
        metrics = await risk_engine.calculate_portfolio_metrics(portfolio_id, positions)
        
        return metrics.__dict__
        
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/risk/summary/{portfolio_id}")
async def get_risk_summary(
    portfolio_id: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get comprehensive risk summary for a portfolio."""
    try:
        summary = risk_engine.get_risk_summary(portfolio_id)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get risk summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/risk/emergency-stop/{portfolio_id}")
async def emergency_stop(
    portfolio_id: str,
    reason: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("admin"))
):
    """Emergency stop all trading for a portfolio."""
    try:
        await risk_engine.emergency_stop_trading(portfolio_id, reason)
        return {"message": f"Emergency stop activated for portfolio {portfolio_id}"}
        
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Execution Engine Endpoints
@router.get("/execution/venue-status")
async def get_venue_status(
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get status of all execution venues."""
    try:
        status = execution_engine.get_venue_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get venue status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/execution/active-orders")
async def get_active_orders_count(
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get count of active orders in execution engine."""
    try:
        count = execution_engine.get_active_orders_count()
        return {"active_orders": count}
        
    except Exception as e:
        logger.error(f"Failed to get active orders count: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Market Data Endpoints
@router.post("/market-data/update")
async def update_market_data(
    updates: List[MarketDataUpdate],
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("trade"))
):
    """Update market data for symbols."""
    try:
        for update in updates:
            await position_keeper.update_market_price(update.symbol, update.price)
        
        return {"message": f"Updated market data for {len(updates)} symbols"}
        
    except Exception as e:
        logger.error(f"Failed to update market data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/market-data")
async def get_market_data(
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get current market data."""
    try:
        market_data = position_keeper.get_market_data()
        return market_data
        
    except Exception as e:
        logger.error(f"Failed to get market data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Analytics Endpoints
@router.get("/analytics/oms-statistics")
async def get_oms_statistics(
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get order management system statistics."""
    try:
        stats = oms.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get OMS statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/performance/{portfolio_id}")
async def get_performance_analytics(
    portfolio_id: str,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("view"))
):
    """Get performance analytics for a portfolio."""
    try:
        analytics = position_keeper.get_performance_analytics(portfolio_id)
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# System Control Endpoints
@router.post("/system/start-monitoring")
async def start_monitoring(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    permissions: None = Depends(require_permission("admin"))
):
    """Start system monitoring for all engines."""
    try:
        background_tasks.add_task(execution_engine.start_monitoring)
        background_tasks.add_task(risk_engine.start_monitoring)
        background_tasks.add_task(position_keeper.start_monitoring)
        
        return {"message": "System monitoring started"}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Add callback for order events (optional - for logging/monitoring)
async def log_order_event(event):
    """Log order events for monitoring."""
    logger.info(f"Order Event: {event.event_type.value} - Order {event.order.id}")

async def log_risk_violation(violation):
    """Log risk violations for monitoring."""
    logger.warning(f"Risk Violation: {violation.description}")

async def log_position_update(update):
    """Log position updates for monitoring."""
    logger.info(f"Position Update: {update.symbol} - {update.old_quantity} -> {update.new_quantity}")

# Register callbacks
oms.add_callback(log_order_event)
risk_engine.add_violation_callback(log_risk_violation)
position_keeper.add_position_callback(log_position_update)
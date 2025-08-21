"""
Trade History API Routes
RESTful endpoints for trade history management and analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field
from enum import Enum

from trade_history_service import (
    get_trade_history_service, TradeFilter, TradeType, TradeStatus, 
    Trade, TradeSummary
)
from ib_order_manager import get_ib_order_manager
from ib_gateway_client import get_ib_gateway_client


# Pydantic models for API
class TradeResponse(BaseModel):
    """Trade API response model"""
    trade_id: str
    account_id: str
    venue: str
    symbol: str
    side: str
    quantity: str  # Decimal as string
    price: str     # Decimal as string
    commission: str # Decimal as string
    execution_time: str
    order_id: Optional[str] = None
    execution_id: Optional[str] = None
    strategy: Optional[str] = None
    notes: Optional[str] = None


class TradeFilterRequest(BaseModel):
    """Trade filter request model"""
    account_id: Optional[str] = None
    venue: Optional[str] = None
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    trade_type: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None  # ISO format
    end_date: Optional[str] = None    # ISO format
    min_pnl: Optional[str] = None     # Decimal as string
    max_pnl: Optional[str] = None     # Decimal as string
    limit: Optional[int] = Field(default=100, le=1000)
    offset: Optional[int] = Field(default=0, ge=0)


class TradeSummaryResponse(BaseModel):
    """Trade summary API response model"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: str        # Decimal as string
    total_commission: str # Decimal as string
    net_pnl: str         # Decimal as string
    average_win: str     # Decimal as string
    average_loss: str    # Decimal as string
    profit_factor: float
    max_drawdown: str    # Decimal as string
    sharpe_ratio: Optional[float]
    start_date: str
    end_date: str


class ExportFormat(str, Enum):
    """Export format options"""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"


# Create router
router = APIRouter(prefix="/api/v1/trades", tags=["Trade History"])


@router.get("/", response_model=List[TradeResponse])
async def get_trades(
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    venue: Optional[str] = Query(None, description="Filter by venue"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, le=1000, description="Maximum number of trades to return"),
    offset: int = Query(0, ge=0, description="Number of trades to skip")
):
    """Get trade history with filtering and pagination"""
    try:
        service = await get_trade_history_service()
        
        # Parse date filters
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
        
        filter_criteria = TradeFilter(
            account_id=account_id,
            venue=venue,
            symbol=symbol,
            strategy=strategy,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit,
            offset=offset
        )
        
        trades = await service.get_trades(filter_criteria)
        
        # Convert to response format
        response_trades = []
        for trade in trades:
            response_trades.append(TradeResponse(
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                venue=trade.venue,
                symbol=trade.symbol,
                side=trade.side,
                quantity=str(trade.quantity),
                price=str(trade.price),
                commission=str(trade.commission),
                execution_time=trade.execution_time.isoformat(),
                order_id=trade.order_id,
                execution_id=trade.execution_id,
                strategy=trade.strategy,
                notes=trade.notes
            ))
        
        return response_trades
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logging.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving trades: {str(e)}")


@router.get("/summary", response_model=TradeSummaryResponse)
async def get_trade_summary(
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    venue: Optional[str] = Query(None, description="Filter by venue"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)")
):
    """Get trade performance summary with P&L calculations"""
    try:
        service = await get_trade_history_service()
        
        # Parse date filters
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
        
        filter_criteria = TradeFilter(
            account_id=account_id,
            venue=venue,
            symbol=symbol,
            strategy=strategy,
            start_date=start_dt,
            end_date=end_dt
        )
        
        summary = await service.get_trade_summary(filter_criteria)
        
        return TradeSummaryResponse(
            total_trades=summary.total_trades,
            winning_trades=summary.winning_trades,
            losing_trades=summary.losing_trades,
            win_rate=summary.win_rate,
            total_pnl=str(summary.total_pnl),
            total_commission=str(summary.total_commission),
            net_pnl=str(summary.net_pnl),
            average_win=str(summary.average_win),
            average_loss=str(summary.average_loss),
            profit_factor=summary.profit_factor,
            max_drawdown=str(summary.max_drawdown),
            sharpe_ratio=summary.sharpe_ratio,
            start_date=summary.start_date.isoformat(),
            end_date=summary.end_date.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logging.error(f"Error getting trade summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving trade summary: {str(e)}")


@router.get("/export")
async def export_trades(
    format: ExportFormat = Query(ExportFormat.CSV, description="Export format"),
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    venue: Optional[str] = Query(None, description="Filter by venue"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, le=5000, description="Maximum number of trades to export")
):
    """Export trade history in various formats"""
    try:
        service = await get_trade_history_service()
        
        # Parse date filters
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
        
        filter_criteria = TradeFilter(
            account_id=account_id,
            venue=venue,
            symbol=symbol,
            strategy=strategy,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        trades = await service.get_trades(filter_criteria)
        
        if format == ExportFormat.CSV:
            content = await service.export_trades_csv(trades)
            media_type = "text/csv"
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        elif format == ExportFormat.JSON:
            content = await service.export_trades_json(trades)
            media_type = "application/json"
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        elif format == ExportFormat.EXCEL:
            # For Excel export, we'll use CSV for now (can be enhanced with openpyxl later)
            content = await service.export_trades_csv(trades)
            media_type = "application/vnd.ms-excel"
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logging.error(f"Error exporting trades: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting trades: {str(e)}")


@router.get("/debug/ib-data")
async def debug_ib_data():
    """Debug endpoint to inspect IB order manager data"""
    try:
        # Get IB order manager
        ib_client = get_ib_gateway_client()
        if not ib_client.is_connected():
            raise HTTPException(status_code=503, detail="IB Gateway not connected")
        
        ib_order_manager = get_ib_order_manager(ib_client)
        
        # Get raw data from order manager
        orders = ib_order_manager.get_all_orders()
        executions = ib_order_manager.get_executions()
        
        # Convert to serializable format
        orders_data = {}
        for order_id, order_data in orders.items():
            orders_data[str(order_id)] = {
                "order_id": order_data.order_id,
                "symbol": order_data.symbol,
                "status": order_data.status,
                "filled_quantity": str(order_data.filled_quantity),
                "avg_fill_price": str(order_data.avg_fill_price) if order_data.avg_fill_price else None,
                "executions_count": len(order_data.executions),
                "executions": [
                    {
                        "execution_id": exec.execution_id,
                        "shares": str(exec.shares),
                        "price": str(exec.price),
                        "time": exec.time
                    } for exec in order_data.executions
                ]
            }
        
        executions_data = {}
        for exec_id, execution in executions.items():
            executions_data[exec_id] = {
                "execution_id": execution.execution_id,
                "order_id": execution.order_id,
                "symbol": execution.symbol,
                "side": execution.side,
                "shares": str(execution.shares),
                "price": str(execution.price),
                "time": execution.time,
                "commission": str(execution.commission) if execution.commission else None
            }
        
        return {
            "orders_count": len(orders),
            "executions_count": len(executions),
            "orders": orders_data,
            "executions": executions_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error debugging IB data: {e}")
        raise HTTPException(status_code=500, detail=f"Error debugging IB data: {str(e)}")


@router.post("/sync/ib")
async def sync_ib_executions():
    """Sync Interactive Brokers executions to trade history"""
    try:
        service = await get_trade_history_service()
        
        # Get IB order manager
        ib_client = get_ib_gateway_client()
        if not ib_client.is_connected():
            raise HTTPException(status_code=503, detail="IB Gateway not connected")
        
        ib_order_manager = get_ib_order_manager(ib_client)
        
        # Request latest executions from IB (no filter for simplicity)
        await ib_order_manager.request_executions()
        
        # Wait for executions to be received from IB
        await asyncio.sleep(3)
        
        # Sync executions to trade history
        synced_count = await service.sync_ib_executions(ib_order_manager)
        
        return {
            "message": f"Successfully synced {synced_count} IB executions",
            "synced_count": synced_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error syncing IB executions: {e}")
        raise HTTPException(status_code=500, detail=f"Error syncing IB executions: {str(e)}")


@router.post("/manual-entry")
async def manual_trade_entry(
    symbol: str,
    side: str,
    quantity: str,
    price: str,
    commission: str = "1.0",
    execution_time: Optional[str] = None,
    execution_id: Optional[str] = None,
    account_id: str = "DU7925702",
    venue: str = "IB",
    strategy: Optional[str] = None,
    notes: Optional[str] = None
):
    """Manually record a trade execution"""
    try:
        service = await get_trade_history_service()
        
        # Parse execution time
        if execution_time:
            exec_time = datetime.fromisoformat(execution_time.replace('Z', '+00:00'))
        else:
            exec_time = datetime.now()
        
        # Generate trade ID if not provided
        if not execution_id:
            execution_id = f"MANUAL_{symbol}_{int(exec_time.timestamp())}"
        
        # Create trade object
        trade = Trade(
            trade_id=execution_id,
            account_id=account_id,
            venue=venue,
            symbol=symbol,
            side=side.upper(),
            quantity=Decimal(quantity),
            price=Decimal(price),
            commission=Decimal(commission),
            execution_time=exec_time,
            execution_id=execution_id,
            strategy=strategy,
            notes=notes or "Manually entered trade"
        )
        
        # Record the trade
        success = await service.record_trade(trade)
        
        if success:
            return {
                "message": f"Successfully recorded manual trade: {execution_id}",
                "trade_id": execution_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record trade")
        
    except Exception as e:
        logging.error(f"Error recording manual trade: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording manual trade: {str(e)}")


@router.get("/symbols")
async def get_traded_symbols(
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    venue: Optional[str] = Query(None, description="Filter by venue"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)")
):
    """Get list of symbols that have been traded"""
    try:
        service = await get_trade_history_service()
        
        # Parse date filters
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
        
        filter_criteria = TradeFilter(
            account_id=account_id,
            venue=venue,
            start_date=start_dt,
            end_date=end_dt
        )
        
        trades = await service.get_trades(filter_criteria)
        
        # Get unique symbols with trade counts
        symbol_stats = {}
        for trade in trades:
            symbol = trade.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    "symbol": symbol,
                    "venue": trade.venue,
                    "trade_count": 0,
                    "total_volume": Decimal("0"),
                    "first_trade": trade.execution_time,
                    "last_trade": trade.execution_time
                }
            
            stats = symbol_stats[symbol]
            stats["trade_count"] += 1
            stats["total_volume"] += trade.quantity
            if trade.execution_time < stats["first_trade"]:
                stats["first_trade"] = trade.execution_time
            if trade.execution_time > stats["last_trade"]:
                stats["last_trade"] = trade.execution_time
        
        # Convert to response format
        symbols = []
        for stats in symbol_stats.values():
            symbols.append({
                "symbol": stats["symbol"],
                "venue": stats["venue"],
                "trade_count": stats["trade_count"],
                "total_volume": str(stats["total_volume"]),
                "first_trade": stats["first_trade"].isoformat(),
                "last_trade": stats["last_trade"].isoformat()
            })
        
        # Sort by trade count descending
        symbols.sort(key=lambda x: x["trade_count"], reverse=True)
        
        return {"symbols": symbols}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logging.error(f"Error getting traded symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving traded symbols: {str(e)}")


@router.get("/strategies")
async def get_strategies(
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    venue: Optional[str] = Query(None, description="Filter by venue")
):
    """Get list of trading strategies that have been used"""
    try:
        service = await get_trade_history_service()
        
        filter_criteria = TradeFilter(
            account_id=account_id,
            venue=venue
        )
        
        trades = await service.get_trades(filter_criteria)
        
        # Get unique strategies with statistics
        strategy_stats = {}
        for trade in trades:
            strategy = trade.strategy or "Unspecified"
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "strategy": strategy,
                    "trade_count": 0,
                    "symbols": set(),
                    "venues": set()
                }
            
            stats = strategy_stats[strategy]
            stats["trade_count"] += 1
            stats["symbols"].add(trade.symbol)
            stats["venues"].add(trade.venue)
        
        # Convert to response format
        strategies = []
        for stats in strategy_stats.values():
            strategies.append({
                "strategy": stats["strategy"],
                "trade_count": stats["trade_count"],
                "symbol_count": len(stats["symbols"]),
                "venue_count": len(stats["venues"]),
                "symbols": sorted(list(stats["symbols"])),
                "venues": sorted(list(stats["venues"]))
            })
        
        # Sort by trade count descending
        strategies.sort(key=lambda x: x["trade_count"], reverse=True)
        
        return {"strategies": strategies}
        
    except Exception as e:
        logging.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving strategies: {str(e)}")


@router.get("/health")
async def health_check():
    """Trade history service health check"""
    try:
        service = await get_trade_history_service()
        
        return {
            "status": "healthy" if service.is_connected else "disconnected",
            "service": "trade_history",
            "timestamp": datetime.now().isoformat(),
            "database_connected": service.is_connected
        }
        
    except Exception as e:
        logging.error(f"Trade history health check failed: {e}")
        return {
            "status": "error",
            "service": "trade_history",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
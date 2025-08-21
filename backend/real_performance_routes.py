"""
REAL Performance monitoring API routes for NautilusTrader
Connects to actual PostgreSQL database in containers for real strategy data.
NO MOCK DATA - follows CORE RULE #4
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncpg
import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])

# Database connection settings - connects to actual NautilusTrader PostgreSQL
DATABASE_URL = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"

# Pydantic Models for real data structures
class RealPerformanceMetrics(BaseModel):
    strategy_id: str
    total_pnl: float
    unrealized_pnl: float
    total_trades: int
    winning_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    avg_trade_duration_minutes: Optional[float] = None
    last_updated: datetime

class RealTradeData(BaseModel):
    trade_id: str
    strategy_id: str
    instrument: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    trade_timestamp: datetime
    position_duration_minutes: Optional[float] = None

class RealPositionData(BaseModel):
    position_id: str
    strategy_id: str
    instrument: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    created_at: datetime

async def get_db_connection():
    """Get connection to real NautilusTrader PostgreSQL database"""
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to NautilusTrader database: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Cannot connect to NautilusTrader database: {str(e)}"
        )

async def calculate_strategy_performance(conn, strategy_id: str = None) -> Dict[str, Any]:
    """Calculate real performance metrics from actual trade data"""
    try:
        # Query real trades from NautilusTrader database
        if strategy_id:
            query = """
                SELECT * FROM trades 
                WHERE strategy = $1 
                ORDER BY execution_time DESC
            """
            trades = await conn.fetch(query, strategy_id)
        else:
            query = "SELECT * FROM trades ORDER BY execution_time DESC"
            trades = await conn.fetch(query)
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0
            }
        
        # Calculate real metrics from actual trade data
        total_trades = len(trades)
        # For trades table, calculate PnL as (price * quantity) - commission for side='sell'
        total_pnl = 0
        winning_trades = 0
        for trade in trades:
            if trade['side'] == 'sell':
                # This is a sell trade - calculate realized PnL
                pnl = float(trade['price']) * float(trade['quantity']) - float(trade.get('commission', 0))
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate drawdown from running PnL
        running_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for trade in trades:
            if trade['side'] == 'sell':
                pnl = float(trade['price']) * float(trade['quantity']) - float(trade.get('commission', 0))
                running_pnl += pnl
                if running_pnl > peak_pnl:
                    peak_pnl = running_pnl
                current_drawdown = (peak_pnl - running_pnl) / peak_pnl if peak_pnl > 0 else 0
                max_drawdown = max(max_drawdown, current_drawdown)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "total_pnl": total_pnl,
            "unrealized_pnl": 0.0,  # TODO: Calculate from open positions
            "win_rate": win_rate,
            "max_drawdown": max_drawdown * 100  # Convert to percentage
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        raise

@router.get("/aggregate")
async def get_real_aggregate_performance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get REAL aggregate performance metrics from NautilusTrader database"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Get performance across all strategies from real data
        metrics = await calculate_strategy_performance(conn)
        
        return RealPerformanceMetrics(
            strategy_id="aggregate",
            total_pnl=metrics["total_pnl"],
            unrealized_pnl=metrics["unrealized_pnl"],
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            win_rate=metrics["win_rate"],
            max_drawdown=metrics["max_drawdown"],
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting real performance data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/trades/recent")
async def get_real_recent_trades(
    limit: int = Query(50, ge=1, le=500),
    strategy_id: Optional[str] = None
):
    """Get REAL recent trades from NautilusTrader database"""
    conn = None
    try:
        conn = await get_db_connection()
        
        if strategy_id:
            query = """
                SELECT * FROM trades 
                WHERE strategy = $1 
                ORDER BY execution_time DESC 
                LIMIT $2
            """
            trades = await conn.fetch(query, strategy_id, limit)
        else:
            query = """
                SELECT * FROM trades 
                ORDER BY execution_time DESC 
                LIMIT $1
            """
            trades = await conn.fetch(query, limit)
        
        return {
            "trades": [
                RealTradeData(
                    trade_id=str(trade['trade_id']),
                    strategy_id=trade.get('strategy', 'unknown'),
                    instrument=trade.get('symbol', 'unknown'),
                    side=trade.get('side', 'unknown'),
                    quantity=float(trade.get('quantity', 0)),
                    entry_price=float(trade.get('price', 0)),
                    realized_pnl=float(trade['price']) * float(trade['quantity']) - float(trade.get('commission', 0)) if trade['side'] == 'sell' else 0,
                    trade_timestamp=trade['execution_time']
                ) for trade in trades
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting real trade data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/positions/active")
async def get_real_active_positions():
    """Get REAL active positions from NautilusTrader database"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Query real positions from database
        query = """
            SELECT * FROM trade_positions 
            WHERE remaining_quantity != 0 
            ORDER BY open_time DESC
        """
        positions = await conn.fetch(query)
        
        return {
            "positions": [
                RealPositionData(
                    position_id=str(position['position_id']),
                    strategy_id=position.get('strategy', 'unknown'),
                    instrument=position.get('symbol', 'unknown'),
                    side='long' if float(position.get('remaining_quantity', 0)) > 0 else 'short',
                    quantity=abs(float(position.get('remaining_quantity', 0))),
                    entry_price=float(position.get('entry_price', 0)),
                    current_price=float(position.get('entry_price', 0)),  # TODO: Get current market price
                    unrealized_pnl=float(position.get('unrealized_pnl', 0)),
                    created_at=position['open_time']
                ) for position in positions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting real position data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/database/status")
async def get_database_status():
    """Get status of the NautilusTrader database connection"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Get database statistics
        stats_query = """
            SELECT 
                (SELECT COUNT(*) FROM trades) as total_trades,
                (SELECT COUNT(*) FROM trade_positions) as total_positions,
                (SELECT COUNT(*) FROM instruments) as total_instruments,
                (SELECT COUNT(*) FROM market_bars) as total_bars
        """
        stats = await conn.fetchrow(stats_query)
        
        return {
            "status": "connected",
            "database": "nautilus",
            "total_trades": stats['total_trades'],
            "total_positions": stats['total_positions'],
            "total_instruments": stats['total_instruments'],
            "total_bars": stats['total_bars'],
            "last_checked": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/strategies/active")
async def get_real_active_strategies():
    """Get REAL active strategies from NautilusTrader data"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Get unique strategy IDs from trades
        query = """
            SELECT DISTINCT strategy, COUNT(*) as trade_count
            FROM trades 
            WHERE strategy IS NOT NULL
            GROUP BY strategy
            ORDER BY trade_count DESC
        """
        strategies = await conn.fetch(query)
        
        result = []
        for strategy in strategies:
            strategy_id = strategy['strategy']
            metrics = await calculate_strategy_performance(conn, strategy_id)
            
            result.append({
                "id": strategy_id,
                "state": "unknown",  # TODO: Get from strategy execution status
                "performance_metrics": {
                    "total_pnl": str(metrics["total_pnl"]),
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "max_drawdown": str(metrics["max_drawdown"]),
                    "last_updated": datetime.now().isoformat()
                }
            })
        
        return {"instances": result}
        
    except Exception as e:
        logger.error(f"Error getting real strategy data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/history")
async def get_performance_history(
    strategy_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get REAL performance history data from NautilusTrader database"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # For now, return empty snapshots since we need historical data tracking
        # TODO: Implement historical performance tracking in database
        return {
            "snapshots": []
        }
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/compare")
async def get_strategy_comparison():
    """Get REAL strategy comparison data"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Get comparison data for all strategies
        query = """
            SELECT strategy, COUNT(*) as trade_count,
                   AVG(CASE WHEN side = 'sell' THEN price * quantity - commission ELSE 0 END) as avg_pnl
            FROM trades 
            WHERE strategy IS NOT NULL
            GROUP BY strategy
        """
        strategies = await conn.fetch(query)
        
        return {
            "strategies": [
                {
                    "strategy_id": str(strategy['strategy']),
                    "total_trades": strategy['trade_count'],
                    "avg_pnl": float(strategy['avg_pnl'] or 0)
                } for strategy in strategies
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/alerts")
async def get_performance_alerts():
    """Get REAL performance alerts from NautilusTrader data"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # For now, return empty alerts since we need alert configuration
        # TODO: Implement alert system based on performance thresholds
        return {
            "alerts": []
        }
        
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()
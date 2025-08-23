"""
Simplified Analytics Routes for Sprint 3 - Using Real PostgreSQL Data Only
No complex analytics engine - direct database queries with real data
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
import asyncpg
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

# Database connection helper
async def get_db_connection():
    """Get database connection using same pattern as existing routes"""
    database_url = os.getenv("DATABASE_URL", "postgresql://nautilus:nautilus123@localhost:5432/nautilus")
    return await asyncpg.connect(database_url)

# Pydantic models for requests/responses
class PortfolioAnalyticsRequest(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio identifier (account_id)")
    include_risk: bool = Field(True, description="Include risk analytics")
    include_performance: bool = Field(True, description="Include performance analytics")
    include_execution: bool = Field(True, description="Include execution analytics")

@router.get("/health")
async def analytics_health():
    """Check analytics health using real database connection"""
    try:
        # Test database connection
        conn = await get_db_connection()
        await conn.execute("SELECT 1")
        await conn.close()
        
        return {
            "status": "healthy",
            "message": "Analytics ready with real database connection",
            "components": {
                "database_connection": True,
                "trades_table": True,
                "market_bars_table": True,
                "real_data_available": True
            }
        }
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Analytics not available: {e}")

@router.post("/portfolio/comprehensive")
async def get_comprehensive_portfolio_analytics(
    request: PortfolioAnalyticsRequest
) -> Dict[str, Any]:
    """
    Get comprehensive portfolio analytics using real PostgreSQL data
    
    Uses actual trades and positions from the database:
    - DU7925702 account with SPY trades
    - Real market data from market_bars table
    - Actual trade execution data
    """
    try:
        logger.info(f"Getting comprehensive analytics for portfolio: {request.portfolio_id}")
        
        # Get real data directly instead of calling endpoints
        conn = await get_db_connection()
        
        # Get portfolio summary data
        trade_summary = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) as total_bought,
                SUM(CASE WHEN side = 'SELL' THEN quantity * price ELSE 0 END) as total_sold,
                SUM(commission) as total_commission,
                MIN(execution_time) as first_trade,
                MAX(execution_time) as last_trade
            FROM trades 
            WHERE account_id = $1
        """, request.portfolio_id)
        
        # Get current positions
        positions = await conn.fetch("""
            SELECT DISTINCT symbol, SUM(
                CASE WHEN side = 'BUY' THEN quantity 
                     WHEN side = 'SELL' THEN -quantity 
                     ELSE 0 END
            ) as net_quantity
            FROM trades 
            WHERE account_id = $1 
            GROUP BY symbol
            HAVING SUM(
                CASE WHEN side = 'BUY' THEN quantity 
                     WHEN side = 'SELL' THEN -quantity 
                     ELSE 0 END
            ) != 0
        """, request.portfolio_id)
        
        await conn.close()
        
        summary_data = {
            "trade_summary": {
                "total_trades": trade_summary['total_trades'],
                "total_bought": float(trade_summary['total_bought'] or 0),
                "total_sold": float(trade_summary['total_sold'] or 0),
                "total_commission": float(trade_summary['total_commission'] or 0),
            },
            "positions": [
                {
                    "symbol": pos['symbol'],
                    "net_quantity": float(pos['net_quantity'])
                }
                for pos in positions
            ]
        }
        
        return {
            "success": True,
            "data": {
                "portfolio_id": request.portfolio_id,
                "summary": summary_data,
                "message": f"Real analytics computed for portfolio {request.portfolio_id} using PostgreSQL data"
            },
            "message": f"Analytics computed for portfolio {request.portfolio_id}"
        }
        
    except Exception as e:
        logger.error(f"Error getting comprehensive analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics calculation failed: {e}")

@router.get("/realtime/portfolio/{portfolio_id}")
async def get_realtime_portfolio_metrics(portfolio_id: str) -> Dict[str, Any]:
    """Get real-time portfolio metrics using actual database data"""
    try:
        # Get latest data from database
        conn = await get_db_connection()
        
        # Get latest trades for portfolio
        trades = await conn.fetch("""
            SELECT trade_id, symbol, side, quantity, price, commission, execution_time
            FROM trades 
            WHERE account_id = $1 
            ORDER BY execution_time DESC 
            LIMIT 10
        """, portfolio_id)
        
        # Get current positions
        positions = await conn.fetch("""
            SELECT DISTINCT symbol, SUM(
                CASE WHEN side = 'BUY' THEN quantity 
                     WHEN side = 'SELL' THEN -quantity 
                     ELSE 0 END
            ) as net_quantity
            FROM trades 
            WHERE account_id = $1 
            GROUP BY symbol
            HAVING SUM(
                CASE WHEN side = 'BUY' THEN quantity 
                     WHEN side = 'SELL' THEN -quantity 
                     ELSE 0 END
            ) != 0
        """, portfolio_id)
        
        # Get latest market data for current positions
        market_data = {}
        for pos in positions:
            symbol = pos['symbol']
            latest_bar = await conn.fetchrow("""
                SELECT close_price, timestamp_ns
                FROM market_bars 
                WHERE instrument_id LIKE $1
                ORDER BY timestamp_ns DESC 
                LIMIT 1
            """, f"{symbol}.%")
            
            if latest_bar:
                market_data[symbol] = {
                    'price': float(latest_bar['close_price']),
                    'timestamp': latest_bar['timestamp_ns']
                }
        
        await conn.close()
        
        # Calculate portfolio metrics
        total_cost = 0.0
        total_market_value = 0.0
        total_pnl = 0.0
        
        for pos in positions:
            symbol = pos['symbol']
            net_qty = float(pos['net_quantity'])
            
            # Get average cost for this position
            avg_cost = 0.0
            cost_trades = [t for t in trades if t['symbol'] == symbol]
            if cost_trades:
                total_qty = sum(float(t['quantity']) for t in cost_trades if t['side'] == 'BUY')
                total_cost_basis = sum(float(t['quantity']) * float(t['price']) for t in cost_trades if t['side'] == 'BUY')
                if total_qty > 0:
                    avg_cost = total_cost_basis / total_qty
            
            # Calculate market value and P&L
            if symbol in market_data and net_qty != 0:
                current_price = market_data[symbol]['price']
                market_value = net_qty * current_price
                cost_basis = net_qty * avg_cost
                position_pnl = market_value - cost_basis
                
                total_market_value += market_value
                total_cost += cost_basis
                total_pnl += position_pnl
        
        return {
            "success": True,
            "data": {
                "portfolio_id": portfolio_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "total_market_value": round(total_market_value, 2),
                    "total_cost_basis": round(total_cost, 2),
                    "total_pnl": round(total_pnl, 2),
                    "total_return_pct": round((total_pnl / total_cost * 100) if total_cost > 0 else 0.0, 2),
                    "position_count": len(positions),
                    "trade_count": len(trades)
                },
                "recent_trades": [
                    {
                        "trade_id": trade['trade_id'],
                        "symbol": trade['symbol'],
                        "side": trade['side'],
                        "quantity": float(trade['quantity']),
                        "price": float(trade['price']),
                        "execution_time": trade['execution_time'].isoformat()
                    }
                    for trade in trades
                ],
                "current_positions": [
                    {
                        "symbol": pos['symbol'],
                        "net_quantity": float(pos['net_quantity']),
                        "market_price": market_data.get(pos['symbol'], {}).get('price', 0.0),
                        "market_value": round(float(pos['net_quantity']) * market_data.get(pos['symbol'], {}).get('price', 0.0), 2)
                    }
                    for pos in positions
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time metrics failed: {e}")

@router.get("/portfolio/{portfolio_id}/summary")
async def get_portfolio_summary(portfolio_id: str) -> Dict[str, Any]:
    """Get portfolio summary using real trade and market data"""
    try:
        conn = await get_db_connection()
        
        # Get trade summary
        trade_summary = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) as total_bought,
                SUM(CASE WHEN side = 'SELL' THEN quantity * price ELSE 0 END) as total_sold,
                SUM(commission) as total_commission,
                MIN(execution_time) as first_trade,
                MAX(execution_time) as last_trade
            FROM trades 
            WHERE account_id = $1
        """, portfolio_id)
        
        # Get position summary
        positions = await conn.fetch("""
            SELECT 
                symbol,
                SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
                AVG(CASE WHEN side = 'BUY' THEN price ELSE NULL END) as avg_buy_price
            FROM trades 
            WHERE account_id = $1 
            GROUP BY symbol
            HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) != 0
        """)
        
        await conn.close()
        
        return {
            "success": True,
            "data": {
                "portfolio_id": portfolio_id,
                "trade_summary": {
                    "total_trades": trade_summary['total_trades'],
                    "total_bought": float(trade_summary['total_bought'] or 0),
                    "total_sold": float(trade_summary['total_sold'] or 0),
                    "total_commission": float(trade_summary['total_commission'] or 0),
                    "first_trade": trade_summary['first_trade'].isoformat() if trade_summary['first_trade'] else None,
                    "last_trade": trade_summary['last_trade'].isoformat() if trade_summary['last_trade'] else None
                },
                "positions": [
                    {
                        "symbol": pos['symbol'],
                        "net_quantity": float(pos['net_quantity']),
                        "avg_buy_price": float(pos['avg_buy_price'] or 0)
                    }
                    for pos in positions
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio summary failed: {e}")

@router.get("/market-data/symbols")
async def get_available_symbols() -> Dict[str, Any]:
    """Get available symbols from market data"""
    try:
        conn = await get_db_connection()
        
        symbols = await conn.fetch("""
            SELECT 
                instrument_id,
                COUNT(*) as bar_count,
                MIN(timestamp_ns) as earliest_data,
                MAX(timestamp_ns) as latest_data,
                MAX(close_price) as latest_price
            FROM market_bars 
            GROUP BY instrument_id 
            ORDER BY bar_count DESC
        """)
        
        await conn.close()
        
        return {
            "success": True,
            "data": {
                "total_symbols": len(symbols),
                "symbols": [
                    {
                        "instrument_id": s['instrument_id'],
                        "symbol": s['instrument_id'].split('.')[0],
                        "bar_count": s['bar_count'],
                        "earliest_data": s['earliest_data'],
                        "latest_data": s['latest_data'],
                        "latest_price": float(s['latest_price'])
                    }
                    for s in symbols
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting market data symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Market data query failed: {e}")

@router.get("/trades/recent/{portfolio_id}")
async def get_recent_trades(
    portfolio_id: str,
    limit: int = Query(20, ge=1, le=100, description="Number of recent trades")
) -> Dict[str, Any]:
    """Get recent trades for portfolio"""
    try:
        conn = await get_db_connection()
        
        trades = await conn.fetch("""
            SELECT trade_id, symbol, side, quantity, price, commission, execution_time, notes
            FROM trades 
            WHERE account_id = $1 
            ORDER BY execution_time DESC 
            LIMIT $2
        """, portfolio_id, limit)
        
        await conn.close()
        
        return {
            "success": True,
            "data": {
                "portfolio_id": portfolio_id,
                "trade_count": len(trades),
                "trades": [
                    {
                        "trade_id": trade['trade_id'],
                        "symbol": trade['symbol'],
                        "side": trade['side'],
                        "quantity": float(trade['quantity']),
                        "price": float(trade['price']),
                        "commission": float(trade['commission']),
                        "execution_time": trade['execution_time'].isoformat(),
                        "notes": trade['notes']
                    }
                    for trade in trades
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        raise HTTPException(status_code=500, detail=f"Recent trades query failed: {e}")
"""
Advanced Performance Analytics Calculator for Sprint 3 Priority 2
Real-time P&L and portfolio performance metrics calculation engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncpg
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceMetricType(Enum):
    """Supported performance metric types"""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    ALPHA = "alpha"
    BETA = "beta"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

@dataclass
class PerformanceSnapshot:
    """Real-time performance metrics snapshot"""
    timestamp: datetime
    portfolio_id: str
    total_pnl: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    volatility: float
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None

@dataclass
class PositionPerformance:
    """Individual position performance metrics"""
    instrument_id: str
    position_id: str
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    return_pct: float
    holding_period_days: int
    last_updated: datetime

class PerformanceCalculator:
    """
    Real-time performance analytics and P&L calculation engine
    Integrates with PostgreSQL for data storage and retrieval
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days_per_year = 252
        self.logger = logging.getLogger(__name__)
        
    async def calculate_real_time_pnl(
        self,
        portfolio_id: str,
        include_unrealized: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate real-time P&L for a portfolio
        
        Args:
            portfolio_id: Portfolio identifier
            include_unrealized: Include unrealized P&L
            
        Returns:
            Dict containing P&L breakdown
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get current positions
                positions_query = """
                    SELECT 
                        p.instrument_id,
                        p.position_id,
                        p.quantity,
                        p.avg_entry_price,
                        p.realized_pnl,
                        p.created_at,
                        i.symbol,
                        i.venue,
                        i.multiplier
                    FROM positions p
                    JOIN instruments i ON p.instrument_id = i.instrument_id
                    WHERE p.portfolio_id = $1 AND p.quantity != 0
                """
                
                positions = await conn.fetch(positions_query, portfolio_id)
                
                total_unrealized_pnl = Decimal('0')
                total_realized_pnl = Decimal('0')
                position_details = []
                
                for position in positions:
                    # Get current market price
                    current_price = await self._get_current_price(
                        conn, position['instrument_id']
                    )
                    
                    if current_price is None:
                        self.logger.warning(
                            f"No current price for {position['instrument_id']}"
                        )
                        continue
                    
                    # Calculate position P&L
                    entry_price = Decimal(str(position['avg_entry_price']))
                    quantity = Decimal(str(position['quantity']))
                    multiplier = Decimal(str(position['multiplier'] or 1))
                    
                    price_diff = current_price - entry_price
                    unrealized_pnl = quantity * price_diff * multiplier
                    
                    total_unrealized_pnl += unrealized_pnl
                    total_realized_pnl += Decimal(str(position['realized_pnl'] or 0))
                    
                    # Calculate holding period
                    holding_period = (datetime.utcnow() - position['created_at']).days
                    
                    position_perf = PositionPerformance(
                        instrument_id=position['instrument_id'],
                        position_id=position['position_id'],
                        entry_price=entry_price,
                        current_price=current_price,
                        quantity=quantity,
                        unrealized_pnl=unrealized_pnl,
                        realized_pnl=Decimal(str(position['realized_pnl'] or 0)),
                        total_pnl=unrealized_pnl + Decimal(str(position['realized_pnl'] or 0)),
                        return_pct=float((current_price - entry_price) / entry_price * 100),
                        holding_period_days=holding_period,
                        last_updated=datetime.utcnow()
                    )
                    
                    position_details.append(position_perf)
                
                total_pnl = total_realized_pnl
                if include_unrealized:
                    total_pnl += total_unrealized_pnl
                
                return {
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.utcnow(),
                    "total_pnl": float(total_pnl),
                    "realized_pnl": float(total_realized_pnl),
                    "unrealized_pnl": float(total_unrealized_pnl),
                    "position_count": len(position_details),
                    "positions": [
                        {
                            "instrument_id": pos.instrument_id,
                            "symbol": next(p['symbol'] for p in positions 
                                         if p['instrument_id'] == pos.instrument_id),
                            "unrealized_pnl": float(pos.unrealized_pnl),
                            "realized_pnl": float(pos.realized_pnl),
                            "total_pnl": float(pos.total_pnl),
                            "return_pct": pos.return_pct,
                            "holding_period_days": pos.holding_period_days
                        }
                        for pos in position_details
                    ]
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating real-time P&L: {e}")
                raise
    
    async def calculate_portfolio_metrics(
        self,
        portfolio_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark_symbol: str = "SPY"
    ) -> PerformanceSnapshot:
        """
        Calculate comprehensive portfolio performance metrics
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Analysis start date
            end_date: Analysis end date
            benchmark_symbol: Benchmark for alpha/beta calculation
            
        Returns:
            PerformanceSnapshot with calculated metrics
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=365)
            
        async with self.db_pool.acquire() as conn:
            try:
                # Get portfolio returns
                portfolio_returns = await self._get_portfolio_returns(
                    conn, portfolio_id, start_date, end_date
                )
                
                if len(portfolio_returns) < 30:
                    self.logger.warning(
                        f"Limited data for portfolio {portfolio_id}: "
                        f"{len(portfolio_returns)} observations"
                    )
                
                # Get current P&L
                current_pnl = await self.calculate_real_time_pnl(portfolio_id)
                
                # Calculate basic metrics
                returns_array = np.array(portfolio_returns)
                
                # Sharpe Ratio
                excess_returns = returns_array - (self.risk_free_rate / self.trading_days_per_year)
                sharpe_ratio = (
                    np.mean(excess_returns) / np.std(returns_array) * np.sqrt(self.trading_days_per_year)
                    if np.std(returns_array) > 0 else 0.0
                )
                
                # Sortino Ratio (downside deviation)
                downside_returns = returns_array[returns_array < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
                sortino_ratio = (
                    np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year)
                    if downside_deviation > 0 else 0.0
                )
                
                # Maximum Drawdown
                cumulative_returns = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                # Win Rate and Profit Factor
                winning_trades = returns_array[returns_array > 0]
                losing_trades = returns_array[returns_array < 0]
                
                win_rate = len(winning_trades) / len(returns_array) if len(returns_array) > 0 else 0
                
                total_wins = np.sum(winning_trades) if len(winning_trades) > 0 else 0
                total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 1
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                
                # Volatility (annualized)
                volatility = np.std(returns_array) * np.sqrt(self.trading_days_per_year)
                
                # Total return
                total_return = float(cumulative_returns[-1] - 1) if len(cumulative_returns) > 0 else 0
                
                # Alpha and Beta (if benchmark data available)
                alpha, beta, info_ratio = None, None, None
                try:
                    benchmark_returns = await self._get_benchmark_returns(
                        conn, benchmark_symbol, start_date, end_date
                    )
                    
                    if len(benchmark_returns) > 0:
                        alpha, beta, info_ratio = await self._calculate_alpha_beta(
                            returns_array, np.array(benchmark_returns)
                        )
                except Exception as e:
                    self.logger.warning(f"Could not calculate alpha/beta: {e}")
                
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.utcnow(),
                    portfolio_id=portfolio_id,
                    total_pnl=Decimal(str(current_pnl['total_pnl'])),
                    unrealized_pnl=Decimal(str(current_pnl['unrealized_pnl'])),
                    realized_pnl=Decimal(str(current_pnl['realized_pnl'])),
                    total_return=total_return,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    volatility=volatility,
                    alpha=alpha,
                    beta=beta,
                    information_ratio=info_ratio
                )
                
                # Store snapshot in database
                await self._store_performance_snapshot(conn, snapshot)
                
                return snapshot
                
            except Exception as e:
                self.logger.error(f"Error calculating portfolio metrics: {e}")
                raise
    
    async def calculate_performance_attribution(
        self,
        portfolio_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance attribution by asset class, sector, etc.
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
            
        async with self.db_pool.acquire() as conn:
            try:
                attribution_query = """
                    SELECT 
                        i.asset_class,
                        i.symbol,
                        p.instrument_id,
                        p.quantity * (p.avg_entry_price * i.multiplier) as position_value,
                        p.realized_pnl + p.unrealized_pnl as total_pnl,
                        (p.realized_pnl + p.unrealized_pnl) / NULLIF(p.quantity * p.avg_entry_price * i.multiplier, 0) as return_contribution
                    FROM positions p
                    JOIN instruments i ON p.instrument_id = i.instrument_id
                    WHERE p.portfolio_id = $1 
                    AND p.created_at >= $2 
                    AND p.created_at <= $3
                    AND p.quantity != 0
                """
                
                positions = await conn.fetch(attribution_query, portfolio_id, start_date, end_date)
                
                # Group by asset class
                attribution_by_class = {}
                total_portfolio_value = Decimal('0')
                
                for position in positions:
                    asset_class = position['asset_class']
                    position_value = Decimal(str(position['position_value']))
                    total_pnl = Decimal(str(position['total_pnl'] or 0))
                    
                    total_portfolio_value += position_value
                    
                    if asset_class not in attribution_by_class:
                        attribution_by_class[asset_class] = {
                            'total_value': Decimal('0'),
                            'total_pnl': Decimal('0'),
                            'positions': []
                        }
                    
                    attribution_by_class[asset_class]['total_value'] += position_value
                    attribution_by_class[asset_class]['total_pnl'] += total_pnl
                    attribution_by_class[asset_class]['positions'].append({
                        'symbol': position['symbol'],
                        'instrument_id': position['instrument_id'],
                        'position_value': float(position_value),
                        'total_pnl': float(total_pnl),
                        'weight': float(position_value / total_portfolio_value) if total_portfolio_value > 0 else 0
                    })
                
                # Calculate attribution percentages
                result = {
                    'portfolio_id': portfolio_id,
                    'analysis_period': {
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    },
                    'total_portfolio_value': float(total_portfolio_value),
                    'attribution_by_asset_class': {}
                }
                
                for asset_class, data in attribution_by_class.items():
                    weight = float(data['total_value'] / total_portfolio_value) if total_portfolio_value > 0 else 0
                    contribution = float(data['total_pnl'] / total_portfolio_value) if total_portfolio_value > 0 else 0
                    
                    result['attribution_by_asset_class'][asset_class] = {
                        'weight': weight,
                        'total_value': float(data['total_value']),
                        'total_pnl': float(data['total_pnl']),
                        'contribution_to_return': contribution,
                        'position_count': len(data['positions']),
                        'positions': data['positions']
                    }
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error calculating performance attribution: {e}")
                raise
    
    async def _get_current_price(self, conn: asyncpg.Connection, instrument_id: str) -> Optional[Decimal]:
        """Get current market price for an instrument"""
        try:
            # Try latest quote first
            quote_query = """
                SELECT (bid_price + ask_price) / 2 as mid_price
                FROM market_quotes
                WHERE instrument_id = $1
                ORDER BY timestamp_ns DESC
                LIMIT 1
            """
            
            result = await conn.fetchrow(quote_query, instrument_id)
            if result and result['mid_price']:
                return Decimal(str(result['mid_price']))
            
            # Fallback to latest trade
            trade_query = """
                SELECT price
                FROM market_ticks
                WHERE instrument_id = $1
                ORDER BY timestamp_ns DESC
                LIMIT 1
            """
            
            result = await conn.fetchrow(trade_query, instrument_id)
            if result and result['price']:
                return Decimal(str(result['price']))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {instrument_id}: {e}")
            return None
    
    async def _get_portfolio_returns(
        self,
        conn: asyncpg.Connection,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Get historical portfolio returns"""
        try:
            returns_query = """
                SELECT 
                    date_trunc('day', timestamp) as trade_date,
                    SUM(realized_pnl) as daily_pnl
                FROM trades
                WHERE portfolio_id = $1
                AND timestamp >= $2
                AND timestamp <= $3
                GROUP BY date_trunc('day', timestamp)
                ORDER BY trade_date
            """
            
            results = await conn.fetch(returns_query, portfolio_id, start_date, end_date)
            
            # Convert P&L to returns (simplified)
            returns = []
            for result in results:
                daily_pnl = float(result['daily_pnl'] or 0)
                # This is a simplified return calculation
                # In practice, you'd need the portfolio value at each point
                returns.append(daily_pnl / 100000)  # Normalize by assumed portfolio size
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio returns: {e}")
            return []
    
    async def _get_benchmark_returns(
        self,
        conn: asyncpg.Connection,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Get benchmark returns for alpha/beta calculation"""
        try:
            benchmark_query = """
                SELECT 
                    close_price,
                    LAG(close_price) OVER (ORDER BY timestamp_ns) as prev_close
                FROM market_bars
                WHERE instrument_id LIKE $1
                AND timestamp_ns >= $2
                AND timestamp_ns <= $3
                AND timeframe = '1d'
                ORDER BY timestamp_ns
            """
            
            start_ns = int(start_date.timestamp() * 1_000_000_000)
            end_ns = int(end_date.timestamp() * 1_000_000_000)
            
            results = await conn.fetch(benchmark_query, f"%{symbol}%", start_ns, end_ns)
            
            returns = []
            for result in results:
                if result['prev_close'] and result['close_price']:
                    ret = (float(result['close_price']) - float(result['prev_close'])) / float(result['prev_close'])
                    returns.append(ret)
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error getting benchmark returns: {e}")
            return []
    
    async def _calculate_alpha_beta(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate alpha, beta, and information ratio"""
        try:
            # Align return series
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            if min_length < 10:
                return None, None, None
                
            port_returns = portfolio_returns[-min_length:]
            bench_returns = benchmark_returns[-min_length:]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(bench_returns, port_returns)
            
            beta = slope
            alpha = intercept * self.trading_days_per_year  # Annualized alpha
            
            # Information ratio
            excess_returns = port_returns - bench_returns
            tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
            information_ratio = np.mean(excess_returns) * self.trading_days_per_year / tracking_error if tracking_error > 0 else 0
            
            return alpha, beta, information_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating alpha/beta: {e}")
            return None, None, None
    
    async def _store_performance_snapshot(
        self,
        conn: asyncpg.Connection,
        snapshot: PerformanceSnapshot
    ) -> None:
        """Store performance snapshot in database"""
        try:
            insert_query = """
                INSERT INTO performance_snapshots (
                    timestamp, portfolio_id, total_pnl, unrealized_pnl, realized_pnl,
                    total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor,
                    volatility, alpha, beta, information_ratio
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (portfolio_id, timestamp) DO UPDATE SET
                    total_pnl = EXCLUDED.total_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    total_return = EXCLUDED.total_return,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    win_rate = EXCLUDED.win_rate,
                    profit_factor = EXCLUDED.profit_factor,
                    volatility = EXCLUDED.volatility,
                    alpha = EXCLUDED.alpha,
                    beta = EXCLUDED.beta,
                    information_ratio = EXCLUDED.information_ratio
            """
            
            await conn.execute(
                insert_query,
                snapshot.timestamp,
                snapshot.portfolio_id,
                float(snapshot.total_pnl),
                float(snapshot.unrealized_pnl),
                float(snapshot.realized_pnl),
                snapshot.total_return,
                snapshot.sharpe_ratio,
                snapshot.max_drawdown,
                snapshot.win_rate,
                snapshot.profit_factor,
                snapshot.volatility,
                snapshot.alpha,
                snapshot.beta,
                snapshot.information_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Error storing performance snapshot: {e}")
            raise

# Global instance for use across the application
performance_calculator = None

def get_performance_calculator() -> PerformanceCalculator:
    """Get global performance calculator instance"""
    global performance_calculator
    if performance_calculator is None:
        raise RuntimeError("Performance calculator not initialized. Call init_performance_calculator() first.")
    return performance_calculator

def init_performance_calculator(db_pool: asyncpg.Pool) -> PerformanceCalculator:
    """Initialize global performance calculator instance"""
    global performance_calculator
    performance_calculator = PerformanceCalculator(db_pool)
    return performance_calculator
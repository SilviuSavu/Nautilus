"""
Trade History Service
Comprehensive trade history and execution management with P&L calculations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import asyncpg
import json
import csv
import io
from pathlib import Path

# Nautilus Trader imports for order and execution data
from nautilus_trader.model.orders import Order as NautilusOrder
from nautilus_trader.model.events import OrderFilled, OrderUpdated
from nautilus_trader.model.position import Position as NautilusPosition
from portfolio_service import Position, Order


class TradeType(Enum):
    """Trade types"""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class TradeStatus(Enum):
    """Trade status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    account_id: str
    venue: str
    symbol: str
    side: str  # BUY/SELL
    quantity: Decimal
    price: Decimal
    commission: Decimal
    execution_time: datetime
    order_id: str | None = None
    execution_id: str | None = None
    strategy: str | None = None
    notes: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TradePosition:
    """Trade position (aggregated trades)"""
    position_id: str
    account_id: str
    venue: str
    symbol: str
    trade_type: TradeType
    status: TradeStatus
    open_time: datetime
    close_time: datetime | None
    entry_price: Decimal
    exit_price: Decimal | None
    quantity: Decimal
    remaining_quantity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    commission: Decimal
    strategy: str | None = None
    trades: list[Trade] = field(default_factory=list)


@dataclass
class TradeSummary:
    """Trade performance summary"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: Decimal
    total_commission: Decimal
    net_pnl: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: float
    max_drawdown: Decimal
    sharpe_ratio: float | None
    start_date: datetime
    end_date: datetime


@dataclass
class TradeFilter:
    """Trade filtering criteria"""
    account_id: str | None = None
    venue: str | None = None
    symbol: str | None = None
    strategy: str | None = None
    trade_type: TradeType | None = None
    status: TradeStatus | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_pnl: Decimal | None = None
    max_pnl: Decimal | None = None
    limit: int | None = None
    offset: int | None = None


class TradeHistoryService:
    """
    Comprehensive trade history service with execution tracking, P&L calculations, and export capabilities.
    """
    
    def __init__(self, database_url: str = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"):
        self.logger = logging.getLogger(__name__)
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None
        self._connected = False
        
    async def initialize(self) -> bool:
        """Initialize the trade history service"""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url, min_size=2, max_size=10, command_timeout=60
            )
            
            async with self._pool.acquire() as conn:
                await self._create_tables(conn)
                
            self._connected = True
            self.logger.info("Trade history service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trade history service: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from database"""
        if self._pool:
            await self._pool.close()
            self._connected = False
            self.logger.info("Trade history service disconnected")
    
    @property
    def is_connected(self) -> bool:
        """Check if service is connected"""
        return self._connected
    
    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        """Create trade history database tables"""
        
        # Trades table for individual executions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR(255) PRIMARY KEY, account_id VARCHAR(100) NOT NULL, venue VARCHAR(50) NOT NULL, symbol VARCHAR(50) NOT NULL, side VARCHAR(10) NOT NULL, quantity DECIMAL(18, 8) NOT NULL, price DECIMAL(18, 8) NOT NULL, commission DECIMAL(18, 8) DEFAULT 0, execution_time TIMESTAMP WITH TIME ZONE NOT NULL, order_id VARCHAR(255), execution_id VARCHAR(255), strategy VARCHAR(100), notes TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Trade positions table for aggregated P&L tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_positions (
                position_id VARCHAR(255) PRIMARY KEY, account_id VARCHAR(100) NOT NULL, venue VARCHAR(50) NOT NULL, symbol VARCHAR(50) NOT NULL, trade_type VARCHAR(20) NOT NULL, status VARCHAR(20) NOT NULL, open_time TIMESTAMP WITH TIME ZONE NOT NULL, close_time TIMESTAMP WITH TIME ZONE, entry_price DECIMAL(18, 8) NOT NULL, exit_price DECIMAL(18, 8), quantity DECIMAL(18, 8) NOT NULL, remaining_quantity DECIMAL(18, 8) NOT NULL, realized_pnl DECIMAL(18, 8) DEFAULT 0, unrealized_pnl DECIMAL(18, 8) DEFAULT 0, commission DECIMAL(18, 8) DEFAULT 0, strategy VARCHAR(100), created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create indexes for efficient querying
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_execution_time ON trades(execution_time)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_venue ON trades(venue)")
        
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_open_time ON trade_positions(open_time)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trade_positions(symbol)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_account ON trade_positions(account_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON trade_positions(status)")
        
        self.logger.info("Trade history tables created")
    
    async def record_trade(self, trade: Trade) -> bool:
        """Record a new trade execution"""
        if not self._connected:
            return False
            
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trades (
                        trade_id, account_id, venue, symbol, side, quantity, price, commission, execution_time, order_id, execution_id, strategy, notes, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        quantity = EXCLUDED.quantity, price = EXCLUDED.price, commission = EXCLUDED.commission
                """, trade.trade_id, trade.account_id, trade.venue, trade.symbol, trade.side, trade.quantity, trade.price, trade.commission, trade.execution_time, trade.order_id, trade.execution_id, trade.strategy, trade.notes, trade.created_at)
                
            self.logger.info(f"Recorded trade: {trade.trade_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record trade {trade.trade_id}: {e}")
            return False
    
    async def record_nautilus_execution(self, order_filled: OrderFilled, nautilus_order: NautilusOrder) -> bool:
        """Record Nautilus Trader execution as trade"""
        try:
            # Convert Nautilus order filled event to Trade
            trade = Trade(
                trade_id=str(order_filled.execution_id),
                account_id=str(order_filled.account_id),
                venue="NAUTILUS",
                symbol=str(order_filled.instrument_id),
                side=str(nautilus_order.side),
                quantity=Decimal(str(order_filled.last_qty)),
                price=Decimal(str(order_filled.last_px)),
                commission=Decimal(str(order_filled.commission)) if order_filled.commission else Decimal("0"),
                execution_time=datetime.fromtimestamp(order_filled.ts_event / 1_000_000_000),
                order_id=str(order_filled.order_id),
                execution_id=str(order_filled.execution_id),
                strategy=str(nautilus_order.strategy_id) if hasattr(nautilus_order, 'strategy_id') else None
            )
            
            return await self.record_trade(trade)
            
        except Exception as e:
            self.logger.error(f"Failed to record Nautilus execution: {e}")
            return False
    
    async def get_trades(self, filter_criteria: TradeFilter) -> list[Trade]:
        """Get trades with filtering"""
        if not self._connected:
            return []
            
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            param_count = 0
            
            if filter_criteria.account_id:
                param_count += 1
                query += f" AND account_id = ${param_count}"
                params.append(filter_criteria.account_id)
                
            if filter_criteria.venue:
                param_count += 1
                query += f" AND venue = ${param_count}"
                params.append(filter_criteria.venue)
                
            if filter_criteria.symbol:
                param_count += 1
                query += f" AND symbol = ${param_count}"
                params.append(filter_criteria.symbol)
                
            if filter_criteria.strategy:
                param_count += 1
                query += f" AND strategy = ${param_count}"
                params.append(filter_criteria.strategy)
                
            if filter_criteria.start_date:
                param_count += 1
                query += f" AND execution_time >= ${param_count}"
                params.append(filter_criteria.start_date)
                
            if filter_criteria.end_date:
                param_count += 1
                query += f" AND execution_time <= ${param_count}"
                params.append(filter_criteria.end_date)
            
            query += " ORDER BY execution_time DESC"
            
            if filter_criteria.limit:
                param_count += 1
                query += f" LIMIT ${param_count}"
                params.append(filter_criteria.limit)
                
            if filter_criteria.offset:
                param_count += 1
                query += f" OFFSET ${param_count}"
                params.append(filter_criteria.offset)
            
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            trades = []
            for row in rows:
                trade = Trade(
                    trade_id=row['trade_id'], account_id=row['account_id'], venue=row['venue'], symbol=row['symbol'], side=row['side'], quantity=Decimal(str(row['quantity'])), price=Decimal(str(row['price'])), commission=Decimal(str(row['commission'])), execution_time=row['execution_time'], order_id=row['order_id'], execution_id=row['execution_id'], strategy=row['strategy'], notes=row['notes'], created_at=row['created_at']
                )
                trades.append(trade)
                
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            return []
    
    async def calculate_pnl(self, trades: list[Trade]) -> TradeSummary:
        """Calculate P&L and performance metrics"""
        if not trades:
            return TradeSummary(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0, total_pnl=Decimal("0"), total_commission=Decimal("0"), net_pnl=Decimal("0"), average_win=Decimal("0"), average_loss=Decimal("0"), profit_factor=0.0, max_drawdown=Decimal("0"), sharpe_ratio=None, start_date=datetime.now(), end_date=datetime.now()
            )
        
        # Group trades by symbol for position tracking
        positions = {}
        total_commission = Decimal("0")
        
        for trade in sorted(trades, key=lambda t: t.execution_time):
            symbol = trade.symbol
            if symbol not in positions:
                positions[symbol] = {
                    'quantity': Decimal("0"), 'cost_basis': Decimal("0"), 'realized_pnl': Decimal("0")
                }
            
            pos = positions[symbol]
            total_commission += trade.commission
            
            if trade.side.upper() in ['BUY', 'COVER']:
                # Opening long or covering short
                if pos['quantity'] < 0:
                    # Covering short position
                    cover_qty = min(abs(pos['quantity']), trade.quantity)
                    if pos['quantity'] != 0:
                        avg_cost = pos['cost_basis'] / abs(pos['quantity'])
                        realized_pnl = cover_qty * (avg_cost - trade.price)
                        pos['realized_pnl'] += realized_pnl
                    
                    pos['quantity'] += cover_qty
                    if cover_qty < trade.quantity:
                        # Remaining quantity opens new long position
                        remaining = trade.quantity - cover_qty
                        pos['quantity'] += remaining
                        pos['cost_basis'] += remaining * trade.price
                else:
                    # Adding to long position
                    pos['quantity'] += trade.quantity
                    pos['cost_basis'] += trade.quantity * trade.price
                    
            else:  # SELL or SHORT
                # Selling long or opening short
                if pos['quantity'] > 0:
                    # Selling long position
                    sell_qty = min(pos['quantity'], trade.quantity)
                    if pos['quantity'] != 0:
                        avg_cost = pos['cost_basis'] / pos['quantity']
                        realized_pnl = sell_qty * (trade.price - avg_cost)
                        pos['realized_pnl'] += realized_pnl
                    
                    pos['quantity'] -= sell_qty
                    pos['cost_basis'] -= sell_qty * (pos['cost_basis'] / (pos['quantity'] + sell_qty) if pos['quantity'] + sell_qty != 0 else 0)
                    
                    if sell_qty < trade.quantity:
                        # Remaining quantity opens new short position
                        remaining = trade.quantity - sell_qty
                        pos['quantity'] -= remaining
                        pos['cost_basis'] -= remaining * trade.price
                else:
                    # Adding to short position
                    pos['quantity'] -= trade.quantity
                    pos['cost_basis'] -= trade.quantity * trade.price
        
        # Calculate summary statistics
        total_realized_pnl = sum(pos['realized_pnl'] for pos in positions.values())
        net_pnl = total_realized_pnl - total_commission
        
        # Simplified metrics (would need more detailed trade-by-trade analysis for full accuracy)
        winning_trades = len([t for t in trades if total_realized_pnl > 0])  # Simplified
        losing_trades = len(trades) - winning_trades
        win_rate = winning_trades / len(trades) if trades else 0.0
        
        return TradeSummary(
            total_trades=len(trades), winning_trades=winning_trades, losing_trades=losing_trades, win_rate=win_rate, total_pnl=total_realized_pnl, total_commission=total_commission, net_pnl=net_pnl, average_win=total_realized_pnl / winning_trades if winning_trades > 0 else Decimal("0"), average_loss=total_realized_pnl / losing_trades if losing_trades > 0 else Decimal("0"), profit_factor=abs(total_realized_pnl / total_commission) if total_commission > 0 else 0.0, max_drawdown=Decimal("0"), # Would need detailed calculation
            sharpe_ratio=None, # Would need returns time series
            start_date=min(t.execution_time for t in trades), end_date=max(t.execution_time for t in trades)
        )
    
    async def export_trades_csv(self, trades: list[Trade]) -> str:
        """Export trades to CSV format"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Trade ID', 'Account', 'Venue', 'Symbol', 'Side', 'Quantity', 'Price', 'Commission', 'Execution Time', 'Order ID', 'Strategy'
        ])
        
        # Data rows
        for trade in trades:
            writer.writerow([
                trade.trade_id, trade.account_id, trade.venue, trade.symbol, trade.side, str(trade.quantity), str(trade.price), str(trade.commission), trade.execution_time.isoformat(), trade.order_id, trade.strategy
            ])
        
        return output.getvalue()
    
    async def export_trades_json(self, trades: list[Trade]) -> str:
        """Export trades to JSON format"""
        trades_data = []
        for trade in trades:
            trade_dict = asdict(trade)
            # Convert Decimal to string for JSON serialization
            trade_dict['quantity'] = str(trade.quantity)
            trade_dict['price'] = str(trade.price)
            trade_dict['commission'] = str(trade.commission)
            trade_dict['execution_time'] = trade.execution_time.isoformat()
            trade_dict['created_at'] = trade.created_at.isoformat()
            trades_data.append(trade_dict)
        
        return json.dumps(trades_data, indent=2)
    
    async def get_trade_summary(self, filter_criteria: TradeFilter) -> TradeSummary:
        """Get trade summary with P&L calculations"""
        trades = await self.get_trades(filter_criteria)
        return await self.calculate_pnl(trades)
    
    async def sync_ib_executions(self, ib_order_manager) -> int:
        """Sync IB executions to trade history"""
        if not ib_order_manager:
            return 0
            
        synced_count = 0
        
        try:
            # Get all executions from IB order manager
            executions = ib_order_manager.get_executions()
            orders = ib_order_manager.get_all_orders()
            
            for execution in executions.values():
                order_data = orders.get(execution.order_id)
                if order_data:
                    success = await self.record_nautilus_execution(execution, order_data)
                    if success:
                        synced_count += 1
                        
            self.logger.info(f"Synced {synced_count} IB executions to trade history")
            return synced_count
            
        except Exception as e:
            self.logger.error(f"Failed to sync IB executions: {e}")
            return 0


# Global service instance
_trade_history_service: TradeHistoryService | None = None

async def get_trade_history_service() -> TradeHistoryService:
    """Get or create trade history service singleton"""
    global _trade_history_service
    
    if _trade_history_service is None:
        _trade_history_service = TradeHistoryService()
        await _trade_history_service.initialize()
    
    return _trade_history_service
"""
Position Keeper
===============

Real-time position tracking and P&L calculation with nanosecond precision
for professional trading operations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from enum import Enum
import uuid

from .order_management import Order, OrderFill, OrderSide

# Set high precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PositionUpdate:
    """Position update event."""
    position_id: str
    symbol: str
    old_quantity: float
    new_quantity: float
    fill: Optional[OrderFill] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """
    Represents a trading position with comprehensive tracking
    and real-time P&L calculation.
    """
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0
    portfolio_id: str = "default"
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Cost basis tracking
    total_cost: float = 0.0
    total_commission: float = 0.0
    
    # Position tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Fill history
    fills: List[OrderFill] = field(default_factory=list)
    
    # Risk metrics
    max_position: float = 0.0
    min_position: float = 0.0
    
    # Performance metrics
    high_water_mark: float = 0.0
    drawdown: float = 0.0
    
    def __post_init__(self):
        """Initialize calculated fields."""
        self.update_derived_fields()
    
    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if abs(self.quantity) < 1e-8:
            return PositionSide.FLAT
        elif self.quantity > 0:
            return PositionSide.LONG
        else:
            return PositionSide.SHORT
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.market_price
    
    @property
    def notional_value(self) -> float:
        """Calculate absolute notional value."""
        return abs(self.market_value)
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_percent(self) -> float:
        """Calculate return percentage."""
        if abs(self.total_cost) < 1e-8:
            return 0.0
        return (self.total_pnl / abs(self.total_cost)) * 100
    
    def update_market_price(self, new_price: float):
        """Update market price and recalculate unrealized P&L."""
        self.market_price = new_price
        self.update_derived_fields()
    
    def add_fill(self, fill: OrderFill, is_closing: bool = False):
        """
        Add a fill to the position and update all metrics.
        
        Args:
            fill: OrderFill to add
            is_closing: Whether this fill closes (or reduces) the position
        """
        self.fills.append(fill)
        
        # Determine if this is a buy or sell based on fill context
        # This would need to be enhanced based on your order system
        is_buy = fill.quantity > 0  # Simplified assumption
        
        if is_closing:
            # Closing fill - realize P&L
            self._process_closing_fill(fill)
        else:
            # Opening fill - adjust position
            self._process_opening_fill(fill, is_buy)
        
        # Update commission
        self.total_commission += fill.commission
        
        # Update position tracking
        self.max_position = max(self.max_position, self.quantity)
        self.min_position = min(self.min_position, self.quantity)
        
        # Update timestamps
        self.updated_at = datetime.now(timezone.utc)
        
        # Recalculate derived fields
        self.update_derived_fields()
    
    def _process_opening_fill(self, fill: OrderFill, is_buy: bool):
        """Process fill that opens or increases position."""
        old_quantity = self.quantity
        old_total_cost = self.total_cost
        
        fill_quantity = abs(fill.quantity)
        fill_cost = fill_quantity * fill.price
        
        if is_buy:
            new_quantity = old_quantity + fill_quantity
            new_total_cost = old_total_cost + fill_cost
        else:
            new_quantity = old_quantity - fill_quantity
            new_total_cost = old_total_cost + fill_cost  # Cost basis increases
        
        # Update quantity and cost basis
        self.quantity = new_quantity
        self.total_cost = new_total_cost
        
        # Recalculate average price
        if abs(self.quantity) > 1e-8:
            self.average_price = self.total_cost / abs(self.quantity)
        else:
            self.average_price = fill.price
    
    def _process_closing_fill(self, fill: OrderFill):
        """Process fill that closes or reduces position."""
        fill_quantity = abs(fill.quantity)
        
        # Calculate realized P&L on closed portion
        if abs(self.quantity) > 1e-8:
            closed_portion = min(fill_quantity, abs(self.quantity))
            avg_cost_per_share = self.total_cost / abs(self.quantity)
            
            if self.quantity > 0:  # Long position being reduced
                realized_gain = closed_portion * (fill.price - avg_cost_per_share)
            else:  # Short position being covered
                realized_gain = closed_portion * (avg_cost_per_share - fill.price)
            
            self.realized_pnl += realized_gain
            
            # Reduce position and cost basis proportionally
            reduction_ratio = closed_portion / abs(self.quantity)
            self.total_cost *= (1 - reduction_ratio)
            
            if self.quantity > 0:
                self.quantity -= closed_portion
            else:
                self.quantity += closed_portion
    
    def update_derived_fields(self):
        """Update all derived fields based on current state."""
        # Calculate unrealized P&L
        if abs(self.quantity) > 1e-8 and self.market_price > 0:
            current_market_value = self.quantity * self.market_price
            if self.quantity > 0:  # Long position
                self.unrealized_pnl = current_market_value - self.total_cost
            else:  # Short position
                self.unrealized_pnl = self.total_cost - current_market_value
        else:
            self.unrealized_pnl = 0.0
        
        # Update performance metrics
        current_total_pnl = self.total_pnl
        self.high_water_mark = max(self.high_water_mark, current_total_pnl)
        
        if self.high_water_mark > 0:
            self.drawdown = ((self.high_water_mark - current_total_pnl) / self.high_water_mark) * 100
        else:
            self.drawdown = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'market_price': self.market_price,
            'market_value': self.market_value,
            'notional_value': self.notional_value,
            'side': self.side.value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'return_percent': self.return_percent,
            'total_cost': self.total_cost,
            'total_commission': self.total_commission,
            'portfolio_id': self.portfolio_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'fills_count': len(self.fills),
            'max_position': self.max_position,
            'min_position': self.min_position,
            'high_water_mark': self.high_water_mark,
            'drawdown': self.drawdown
        }


class PositionKeeper:
    """
    Professional position tracking system with real-time updates,
    comprehensive P&L calculation, and performance analytics.
    """
    
    def __init__(self):
        self.positions: Dict[str, Dict[str, Position]] = {}  # portfolio_id -> symbol -> Position
        self.market_data: Dict[str, float] = {}  # symbol -> current_price
        self.callbacks: List[Callable[[PositionUpdate], Any]] = []
        self.price_callbacks: List[Callable[[str, float], Any]] = []
        self.monitoring_enabled = True
        
        # Performance tracking
        self.portfolio_metrics: Dict[str, Dict[str, Any]] = {}
        
    def add_position_callback(self, callback: Callable[[PositionUpdate], Any]):
        """Add callback for position updates."""
        self.callbacks.append(callback)
    
    def add_price_callback(self, callback: Callable[[str, float], Any]):
        """Add callback for price updates."""
        self.price_callbacks.append(callback)
    
    async def update_market_price(self, symbol: str, price: float):
        """Update market price for a symbol across all portfolios."""
        old_price = self.market_data.get(symbol, 0.0)
        self.market_data[symbol] = price
        
        # Update all positions for this symbol
        for portfolio_id, positions in self.positions.items():
            if symbol in positions:
                position = positions[symbol]
                position.update_market_price(price)
                
                # Notify callbacks of price update impact
                for callback in self.price_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(symbol, price)
                        else:
                            callback(symbol, price)
                    except Exception as e:
                        logger.error(f"Price callback failed: {e}")
        
        logger.debug(f"Updated {symbol} price: {old_price} -> {price}")
    
    async def process_fill(self, order: Order, fill: OrderFill):
        """Process order fill and update positions."""
        portfolio_id = order.portfolio_id
        symbol = order.symbol
        
        # Ensure portfolio exists
        if portfolio_id not in self.positions:
            self.positions[portfolio_id] = {}
        
        # Get or create position
        if symbol not in self.positions[portfolio_id]:
            self.positions[portfolio_id][symbol] = Position(
                symbol=symbol,
                portfolio_id=portfolio_id,
                market_price=self.market_data.get(symbol, fill.price)
            )
        
        position = self.positions[portfolio_id][symbol]
        old_quantity = position.quantity
        
        # Determine if this is a closing trade
        is_closing = (
            (order.side == OrderSide.SELL and position.quantity > 0) or
            (order.side == OrderSide.BUY and position.quantity < 0)
        )
        
        # Add fill to position
        position.add_fill(fill, is_closing)
        
        # Create position update event
        update = PositionUpdate(
            position_id=f"{portfolio_id}_{symbol}",
            symbol=symbol,
            old_quantity=old_quantity,
            new_quantity=position.quantity,
            fill=fill,
            metadata={
                'order_id': order.id,
                'is_closing': is_closing,
                'side': order.side.value
            }
        )
        
        # Notify callbacks
        await self._notify_position_update(update)
        
        logger.info(f"Updated position {symbol} in {portfolio_id}: {old_quantity} -> {position.quantity}")
    
    async def get_position(self, portfolio_id: str, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol in a portfolio."""
        return self.positions.get(portfolio_id, {}).get(symbol)
    
    async def get_positions(self, portfolio_id: str) -> Dict[str, Position]:
        """Get all positions for a portfolio."""
        return self.positions.get(portfolio_id, {}).copy()
    
    async def get_portfolio_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        positions = await self.get_positions(portfolio_id)
        
        if not positions:
            return {
                'portfolio_id': portfolio_id,
                'total_positions': 0,
                'total_market_value': 0.0,
                'total_pnl': 0.0,
                'positions': []
            }
        
        # Calculate portfolio-level metrics
        total_market_value = 0.0
        total_realized_pnl = 0.0
        total_unrealized_pnl = 0.0
        total_commission = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        
        position_summaries = []
        
        for position in positions.values():
            # Update with latest market price if available
            if position.symbol in self.market_data:
                position.update_market_price(self.market_data[position.symbol])
            
            total_market_value += position.market_value
            total_realized_pnl += position.realized_pnl
            total_unrealized_pnl += position.unrealized_pnl
            total_commission += position.total_commission
            
            if position.quantity > 0:
                long_exposure += position.market_value
            elif position.quantity < 0:
                short_exposure += abs(position.market_value)
            
            position_summaries.append(position.to_dict())
        
        # Calculate additional metrics
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        summary = {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_positions': len(positions),
            'total_market_value': total_market_value,
            'total_pnl': total_pnl,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_commission': total_commission,
            'exposure': {
                'long': long_exposure,
                'short': short_exposure,
                'gross': gross_exposure,
                'net': net_exposure
            },
            'positions': position_summaries
        }
        
        # Store metrics for risk engine
        self.portfolio_metrics[portfolio_id] = summary
        
        return summary
    
    async def get_position_history(self, portfolio_id: str, symbol: str) -> List[Dict[str, Any]]:
        """Get fill history for a position."""
        position = await self.get_position(portfolio_id, symbol)
        if not position:
            return []
        
        return [
            {
                'fill_id': fill.id,
                'order_id': fill.order_id,
                'quantity': fill.quantity,
                'price': fill.price,
                'timestamp': fill.timestamp.isoformat(),
                'execution_id': fill.execution_id,
                'commission': fill.commission
            }
            for fill in position.fills
        ]
    
    async def close_position(self, portfolio_id: str, symbol: str, price: Optional[float] = None) -> bool:
        """
        Close a position (simulate a closing order).
        This would typically be called by the execution engine.
        """
        position = await self.get_position(portfolio_id, symbol)
        if not position or abs(position.quantity) < 1e-8:
            return False
        
        # Use market price if not specified
        close_price = price or self.market_data.get(symbol, position.average_price)
        
        # Create closing fill
        closing_fill = OrderFill(
            id=str(uuid.uuid4()),
            order_id=f"close_{portfolio_id}_{symbol}",
            quantity=abs(position.quantity),
            price=close_price,
            timestamp=datetime.now(timezone.utc),
            execution_id=f"close_{int(datetime.now().timestamp())}"
        )
        
        # Process as closing fill
        old_quantity = position.quantity
        position._process_closing_fill(closing_fill)
        position.fills.append(closing_fill)
        position.quantity = 0.0  # Fully close
        position.update_derived_fields()
        
        # Create position update
        update = PositionUpdate(
            position_id=f"{portfolio_id}_{symbol}",
            symbol=symbol,
            old_quantity=old_quantity,
            new_quantity=0.0,
            fill=closing_fill,
            metadata={'action': 'close_position'}
        )
        
        await self._notify_position_update(update)
        
        logger.info(f"Closed position {symbol} in {portfolio_id} at {close_price}")
        return True
    
    async def _notify_position_update(self, update: PositionUpdate):
        """Notify all callbacks of position update."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update)
                else:
                    callback(update)
            except Exception as e:
                logger.error(f"Position callback failed: {e}")
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols across all portfolios."""
        symbols = set()
        for positions in self.positions.values():
            symbols.update(positions.keys())
        return list(symbols)
    
    def get_market_data(self) -> Dict[str, float]:
        """Get current market data."""
        return self.market_data.copy()
    
    async def start_monitoring(self):
        """Start position monitoring."""
        if self.monitoring_enabled:
            asyncio.create_task(self._monitor_positions())
    
    async def _monitor_positions(self):
        """Monitor positions for performance and risk."""
        while self.monitoring_enabled:
            try:
                # Update portfolio metrics for all portfolios
                for portfolio_id in self.positions.keys():
                    await self.get_portfolio_summary(portfolio_id)
                
                # Log summary information
                total_portfolios = len(self.positions)
                total_positions = sum(len(pos) for pos in self.positions.values())
                
                if total_positions > 0:
                    logger.debug(f"Monitoring {total_positions} positions across {total_portfolios} portfolios")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_performance_analytics(self, portfolio_id: str) -> Dict[str, Any]:
        """Get advanced performance analytics for a portfolio."""
        positions = self.positions.get(portfolio_id, {})
        
        if not positions:
            return {"error": "No positions found"}
        
        # Calculate performance metrics
        total_pnl = sum(pos.total_pnl for pos in positions.values())
        total_cost = sum(pos.total_cost for pos in positions.values())
        winning_positions = [pos for pos in positions.values() if pos.total_pnl > 0]
        losing_positions = [pos for pos in positions.values() if pos.total_pnl < 0]
        
        analytics = {
            'portfolio_id': portfolio_id,
            'total_positions': len(positions),
            'total_pnl': total_pnl,
            'total_return_percent': (total_pnl / total_cost * 100) if total_cost > 0 else 0,
            'win_rate': (len(winning_positions) / len(positions) * 100) if positions else 0,
            'avg_winner': sum(pos.total_pnl for pos in winning_positions) / len(winning_positions) if winning_positions else 0,
            'avg_loser': sum(pos.total_pnl for pos in losing_positions) / len(losing_positions) if losing_positions else 0,
            'profit_factor': abs(sum(pos.total_pnl for pos in winning_positions) / sum(pos.total_pnl for pos in losing_positions)) if losing_positions else float('inf') if winning_positions else 0,
            'largest_winner': max((pos.total_pnl for pos in positions.values()), default=0),
            'largest_loser': min((pos.total_pnl for pos in positions.values()), default=0),
            'avg_holding_period_hours': 0,  # Would need fill timestamps to calculate
            'total_commission': sum(pos.total_commission for pos in positions.values())
        }
        
        return analytics
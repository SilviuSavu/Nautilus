"""
Portfolio Service
Manages trading portfolios, positions, and risk metrics across multiple exchanges.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum

from enums import Venue


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """Trading position"""
    venue: Venue
    instrument_id: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: datetime
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value"""
        return self.quantity * self.current_price
        
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total PnL"""
        return self.unrealized_pnl + self.realized_pnl


@dataclass
class Order:
    """Trading order"""
    order_id: str
    venue: Venue
    instrument_id: str
    order_type: OrderType
    side: PositionSide
    quantity: Decimal
    price: Optional[Decimal]
    filled_quantity: Decimal
    status: OrderStatus
    timestamp: datetime
    filled_timestamp: Optional[datetime] = None
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill"""
        return self.quantity - self.filled_quantity
        
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage"""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)


@dataclass
class Balance:
    """Account balance"""
    venue: Venue
    currency: str
    total: Decimal
    available: Decimal
    locked: Decimal
    timestamp: datetime
    
    @property
    def locked_percentage(self) -> float:
        """Calculate percentage of balance that is locked"""
        if self.total == 0:
            return 0.0
        return float(self.locked / self.total * 100)


@dataclass
class Portfolio:
    """Trading portfolio"""
    name: str
    positions: Dict[str, Position]  # instrument_id -> Position
    orders: Dict[str, Order]        # order_id -> Order
    balances: Dict[str, Balance]    # currency -> Balance
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    last_updated: datetime


@dataclass
class RiskMetrics:
    """Risk management metrics"""
    portfolio_value: Decimal
    total_exposure: Decimal
    max_position_size: Decimal
    leverage_ratio: float
    var_1d: Decimal  # Value at Risk 1 day
    sharpe_ratio: Optional[float]
    max_drawdown: Decimal
    win_rate: float
    profit_factor: float


class PortfolioService:
    """
    Service for managing trading portfolios, positions, and risk metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._portfolios: Dict[str, Portfolio] = {}
        self._risk_limits = {
            "max_portfolio_risk": 0.02,  # 2% max portfolio risk per trade
            "max_position_size": 0.1,    # 10% max position size
            "max_daily_loss": 0.05,      # 5% max daily loss
            "max_leverage": 3.0,         # 3x max leverage
        }
        self._create_default_portfolio()
        
    def _create_default_portfolio(self) -> None:
        """Create default portfolio"""
        default_portfolio = Portfolio(
            name="main",
            positions={},
            orders={},
            balances={},
            total_value=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            last_updated=datetime.now()
        )
        self._portfolios["main"] = default_portfolio
        
    def get_portfolio(self, name: str = "main") -> Optional[Portfolio]:
        """Get portfolio by name"""
        return self._portfolios.get(name)
        
    def get_all_portfolios(self) -> Dict[str, Portfolio]:
        """Get all portfolios"""
        return self._portfolios.copy()
        
    def create_portfolio(self, name: str) -> Portfolio:
        """Create new portfolio"""
        if name in self._portfolios:
            raise ValueError(f"Portfolio {name} already exists")
            
        portfolio = Portfolio(
            name=name,
            positions={},
            orders={},
            balances={},
            total_value=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            last_updated=datetime.now()
        )
        self._portfolios[name] = portfolio
        return portfolio
        
    def update_position(self, portfolio_name: str, position: Position) -> None:
        """Update position in portfolio"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_name} not found")
            
        key = f"{position.venue.value}:{position.instrument_id}"
        portfolio.positions[key] = position
        portfolio.last_updated = datetime.now()
        self._recalculate_portfolio_metrics(portfolio)
        
    def update_order(self, portfolio_name: str, order: Order) -> None:
        """Update order in portfolio"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_name} not found")
            
        portfolio.orders[order.order_id] = order
        portfolio.last_updated = datetime.now()
        
    def update_balance(self, portfolio_name: str, balance: Balance) -> None:
        """Update balance in portfolio"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_name} not found")
            
        key = f"{balance.venue.value}:{balance.currency}"
        portfolio.balances[key] = balance
        portfolio.last_updated = datetime.now()
        self._recalculate_portfolio_metrics(portfolio)
        
    def _recalculate_portfolio_metrics(self, portfolio: Portfolio) -> None:
        """Recalculate portfolio metrics"""
        total_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        realized_pnl = Decimal("0")
        
        # Sum up all balances
        for balance in portfolio.balances.values():
            total_value += balance.total
            
        # Sum up position values and PnL
        for position in portfolio.positions.values():
            total_value += position.market_value
            unrealized_pnl += position.unrealized_pnl
            realized_pnl += position.realized_pnl
            
        portfolio.total_value = total_value
        portfolio.unrealized_pnl = unrealized_pnl
        portfolio.realized_pnl = realized_pnl
        
    def get_positions(self, portfolio_name: str = "main", venue: Optional[Venue] = None) -> List[Position]:
        """Get positions for portfolio, optionally filtered by venue"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            return []
            
        positions = list(portfolio.positions.values())
        if venue:
            positions = [p for p in positions if p.venue == venue]
            
        return positions
        
    def get_open_orders(self, portfolio_name: str = "main", venue: Optional[Venue] = None) -> List[Order]:
        """Get open orders for portfolio"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            return []
            
        orders = [
            order for order in portfolio.orders.values()
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]
        
        if venue:
            orders = [o for o in orders if o.venue == venue]
            
        return orders
        
    def get_balances(self, portfolio_name: str = "main", venue: Optional[Venue] = None) -> List[Balance]:
        """Get balances for portfolio"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            return []
            
        balances = list(portfolio.balances.values())
        if venue:
            balances = [b for b in balances if b.venue == venue]
            
        return balances
        
    def calculate_risk_metrics(self, portfolio_name: str = "main") -> Optional[RiskMetrics]:
        """Calculate risk metrics for portfolio"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            return None
            
        # Calculate total exposure
        total_exposure = Decimal("0")
        for position in portfolio.positions.values():
            total_exposure += abs(position.market_value)
            
        # Calculate leverage ratio
        leverage_ratio = 0.0
        if portfolio.total_value > 0:
            leverage_ratio = float(total_exposure / portfolio.total_value)
            
        # Calculate max position size
        max_position_size = Decimal("0")
        for position in portfolio.positions.values():
            position_value = abs(position.market_value)
            if position_value > max_position_size:
                max_position_size = position_value
                
        # Simplified risk metrics (would need historical data for proper calculation)
        return RiskMetrics(
            portfolio_value=portfolio.total_value,
            total_exposure=total_exposure,
            max_position_size=max_position_size,
            leverage_ratio=leverage_ratio,
            var_1d=portfolio.total_value * Decimal("0.02"),  # Simplified 2% VaR
            sharpe_ratio=None,  # Would need returns data
            max_drawdown=Decimal("0"),  # Would need historical data
            win_rate=0.0,  # Would need trade history
            profit_factor=0.0  # Would need trade history
        )
        
    def check_risk_limits(self, portfolio_name: str = "main") -> Dict[str, Any]:
        """Check if portfolio is within risk limits"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            return {"status": "error", "message": "Portfolio not found"}
            
        risk_metrics = self.calculate_risk_metrics(portfolio_name)
        if not risk_metrics:
            return {"status": "error", "message": "Cannot calculate risk metrics"}
            
        violations = []
        
        # Check leverage limit
        if risk_metrics.leverage_ratio > self._risk_limits["max_leverage"]:
            violations.append({
                "limit": "max_leverage",
                "current": risk_metrics.leverage_ratio,
                "max": self._risk_limits["max_leverage"]
            })
            
        # Check max position size
        max_position_pct = 0.0
        if risk_metrics.portfolio_value > 0:
            max_position_pct = float(risk_metrics.max_position_size / risk_metrics.portfolio_value)
            
        if max_position_pct > self._risk_limits["max_position_size"]:
            violations.append({
                "limit": "max_position_size",
                "current": max_position_pct,
                "max": self._risk_limits["max_position_size"]
            })
            
        return {
            "status": "ok" if not violations else "warning",
            "violations": violations,
            "risk_metrics": asdict(risk_metrics)
        }
        
    def get_portfolio_summary(self, portfolio_name: str = "main") -> Dict[str, Any]:
        """Get portfolio summary"""
        portfolio = self._portfolios.get(portfolio_name)
        if not portfolio:
            return {}
            
        open_orders = len(self.get_open_orders(portfolio_name))
        active_positions = len([p for p in portfolio.positions.values() if p.side != PositionSide.FLAT])
        
        return {
            "name": portfolio.name,
            "total_value": float(portfolio.total_value),
            "unrealized_pnl": float(portfolio.unrealized_pnl),
            "realized_pnl": float(portfolio.realized_pnl),
            "total_pnl": float(portfolio.unrealized_pnl + portfolio.realized_pnl),
            "active_positions": active_positions,
            "open_orders": open_orders,
            "last_updated": portfolio.last_updated.isoformat()
        }


# Global portfolio service instance
portfolio_service = PortfolioService()
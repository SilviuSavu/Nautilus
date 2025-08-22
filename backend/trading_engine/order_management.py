"""
Order Management System
======================

Core order lifecycle management with real-time tracking and risk integration.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime, timezone
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted" 
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


@dataclass
class OrderFill:
    """Represents a single order fill."""
    id: str
    order_id: str
    quantity: float
    price: float
    timestamp: datetime
    execution_id: str
    commission: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Order:
    """Core order representation with comprehensive tracking."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fills: List[OrderFill] = field(default_factory=list)
    portfolio_id: str = "default"
    strategy_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return abs(self.filled_quantity - self.quantity) < 1e-8
    
    @property
    def total_commission(self) -> float:
        """Calculate total commission from all fills."""
        return sum(fill.commission for fill in self.fills)
    
    def add_fill(self, fill: OrderFill):
        """Add a fill to the order and update quantities."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        
        # Update average fill price
        total_notional = sum(f.quantity * f.price for f in self.fills)
        self.average_fill_price = total_notional / self.filled_quantity if self.filled_quantity > 0 else 0
        
        # Update status
        if self.is_complete:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.timestamp
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            'id': self.id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'fill_percentage': self.fill_percentage,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'portfolio_id': self.portfolio_id,
            'strategy_id': self.strategy_id,
            'total_commission': self.total_commission,
            'fills_count': len(self.fills),
            'tags': self.tags
        }


class OrderEventType(Enum):
    """Order event types for callbacks."""
    CREATED = "created"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderEvent:
    """Order event for callback notifications."""
    event_type: OrderEventType
    order: Order
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagementSystem:
    """
    Professional order management system with comprehensive tracking,
    risk integration, and real-time callbacks.
    """
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.orders_by_client_id: Dict[str, str] = {}
        self.callbacks: List[Callable[[OrderEvent], Any]] = []
        self.risk_engine = None
        self.execution_engine = None
        self._order_sequence = 0
        
    def add_callback(self, callback: Callable[[OrderEvent], Any]):
        """Add order event callback."""
        self.callbacks.append(callback)
    
    def set_risk_engine(self, risk_engine):
        """Set risk engine for pre-trade checks."""
        self.risk_engine = risk_engine
    
    def set_execution_engine(self, execution_engine):
        """Set execution engine for order routing."""
        self.execution_engine = execution_engine
    
    async def submit_order(self, order: Order) -> str:
        """
        Submit order for execution with comprehensive validation.
        
        Args:
            order: Order to submit
            
        Returns:
            str: Order ID
            
        Raises:
            ValueError: If order validation fails
            RuntimeError: If risk checks fail
        """
        # Validate order
        self._validate_order(order)
        
        # Store client order ID mapping
        if order.client_order_id:
            if order.client_order_id in self.orders_by_client_id:
                raise ValueError(f"Duplicate client order ID: {order.client_order_id}")
            self.orders_by_client_id[order.client_order_id] = order.id
        
        # Pre-trade risk check
        if self.risk_engine:
            risk_approved = await self.risk_engine.check_pre_trade_risk(order, order.portfolio_id)
            if not risk_approved:
                order.status = OrderStatus.REJECTED
                await self._notify_order_event(OrderEventType.REJECTED, order, 
                                             {"reason": "Risk check failed"})
                raise RuntimeError("Order rejected by risk engine")
        
        # Store order
        self.orders[order.id] = order
        await self._notify_order_event(OrderEventType.CREATED, order)
        
        # Submit to execution engine
        if self.execution_engine:
            try:
                await self.execution_engine.submit_order(order)
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now(timezone.utc)
                order.updated_at = order.submitted_at
                await self._notify_order_event(OrderEventType.SUBMITTED, order)
            except Exception as e:
                order.status = OrderStatus.REJECTED
                await self._notify_order_event(OrderEventType.REJECTED, order,
                                             {"reason": str(e)})
                raise
        
        return order.id
    
    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            bool: True if cancellation successful
        """
        order = self.orders.get(order_id)
        if not order:
            return False
        
        # Check if order can be cancelled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        # Cancel with execution engine
        if self.execution_engine:
            success = await self.execution_engine.cancel_order(order_id)
            if not success:
                return False
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc)
        
        await self._notify_order_event(OrderEventType.CANCELLED, order, {"reason": reason})
        return True
    
    async def handle_fill(self, order_id: str, fill: OrderFill):
        """
        Handle order fill notification from execution engine.
        
        Args:
            order_id: Order ID that was filled
            fill: Fill details
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"Received fill for unknown order: {order_id}")
            return
        
        # Add fill to order
        order.add_fill(fill)
        
        # Notify callbacks
        if order.status == OrderStatus.FILLED:
            await self._notify_order_event(OrderEventType.FILLED, order, {"fill": fill})
        else:
            await self._notify_order_event(OrderEventType.PARTIALLY_FILLED, order, {"fill": fill})
        
        logger.info(f"Order {order_id} filled: {fill.quantity}@{fill.price}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        order_id = self.orders_by_client_id.get(client_order_id)
        return self.orders.get(order_id) if order_id else None
    
    def get_orders_by_portfolio(self, portfolio_id: str) -> List[Order]:
        """Get all orders for a portfolio."""
        return [order for order in self.orders.values() if order.portfolio_id == portfolio_id]
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a strategy."""
        return [order for order in self.orders.values() if order.strategy_id == strategy_id]
    
    def get_active_orders(self, portfolio_id: Optional[str] = None) -> List[Order]:
        """Get all active (non-terminal) orders."""
        active_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        orders = self.orders.values()
        
        if portfolio_id:
            orders = [o for o in orders if o.portfolio_id == portfolio_id]
            
        return [order for order in orders if order.status in active_statuses]
    
    def _validate_order(self, order: Order):
        """Validate order parameters."""
        if not order.symbol:
            raise ValueError("Order symbol is required")
        
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            raise ValueError(f"Price is required for {order.order_type.value} orders")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and order.stop_price is None:
            raise ValueError(f"Stop price is required for {order.order_type.value} orders")
    
    async def _notify_order_event(self, event_type: OrderEventType, order: Order, metadata: Dict[str, Any] = None):
        """Notify all callbacks of order event."""
        event = OrderEvent(
            event_type=event_type,
            order=order,
            metadata=metadata or {}
        )
        
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Order callback failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OMS statistics."""
        orders = list(self.orders.values())
        
        if not orders:
            return {"total_orders": 0}
        
        status_counts = {}
        for status in OrderStatus:
            status_counts[status.value] = sum(1 for o in orders if o.status == status)
        
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        
        return {
            "total_orders": len(orders),
            "status_distribution": status_counts,
            "average_fill_time_seconds": self._calculate_average_fill_time(filled_orders),
            "total_commission": sum(o.total_commission for o in orders),
            "fill_rate_percentage": (len(filled_orders) / len(orders)) * 100 if orders else 0,
            "active_orders": len(self.get_active_orders())
        }
    
    def _calculate_average_fill_time(self, filled_orders: List[Order]) -> float:
        """Calculate average time from submission to fill."""
        if not filled_orders:
            return 0.0
        
        fill_times = []
        for order in filled_orders:
            if order.submitted_at and order.filled_at:
                fill_time = (order.filled_at - order.submitted_at).total_seconds()
                fill_times.append(fill_time)
        
        return sum(fill_times) / len(fill_times) if fill_times else 0.0
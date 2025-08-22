"""
Execution Engine
===============

Smart order routing and execution with multi-venue support and intelligent algorithms.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum

from .order_management import Order, OrderFill, OrderStatus

logger = logging.getLogger(__name__)


class VenueStatus(Enum):
    """Venue connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class VenueQuote:
    """Market quote from a venue."""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    venue: str
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2


@dataclass
class VenueMetrics:
    """Performance metrics for a venue."""
    venue_name: str
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    average_fill_time_ms: float = 0.0
    fill_rate_percentage: float = 0.0
    average_spread_bps: float = 0.0
    uptime_percentage: float = 100.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionVenue(ABC):
    """Abstract base class for execution venues."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = VenueStatus.DISCONNECTED
        self.metrics = VenueMetrics(venue_name=name)
        self.callbacks: List[Callable] = []
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the venue."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the venue."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        """Submit order to venue."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order at venue."""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[VenueQuote]:
        """Get current quote for symbol."""
        pass
    
    def add_callback(self, callback: Callable):
        """Add callback for venue events."""
        self.callbacks.append(callback)
    
    async def _notify_fill(self, order_id: str, fill: OrderFill):
        """Notify callbacks of order fill."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order_id, fill)
                else:
                    callback(order_id, fill)
            except Exception as e:
                logger.error(f"Venue callback failed: {e}")


class IBKRVenue(ExecutionVenue):
    """Interactive Brokers execution venue."""
    
    def __init__(self, ib_client):
        super().__init__("IBKR")
        self.ib_client = ib_client
        self.order_mapping: Dict[str, str] = {}  # internal_id -> venue_id
        
    async def connect(self) -> bool:
        """Connect to IBKR."""
        try:
            if not self.ib_client.is_connected():
                await self.ib_client.connect()
            
            self.status = VenueStatus.CONNECTED
            logger.info("Connected to IBKR venue")
            return True
            
        except Exception as e:
            self.status = VenueStatus.ERROR
            logger.error(f"Failed to connect to IBKR: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from IBKR."""
        try:
            await self.ib_client.disconnect()
            self.status = VenueStatus.DISCONNECTED
            logger.info("Disconnected from IBKR venue")
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order to IBKR."""
        try:
            if self.status != VenueStatus.CONNECTED:
                raise RuntimeError("IBKR venue not connected")
            
            # Convert to IBKR order format
            ib_order = self._convert_to_ib_order(order)
            
            # Submit to IBKR
            venue_order_id = await self.ib_client.place_order(order.symbol, ib_order)
            
            # Store mapping
            self.order_mapping[order.id] = venue_order_id
            
            # Update metrics
            self.metrics.total_orders += 1
            
            logger.info(f"Submitted order {order.id} to IBKR as {venue_order_id}")
            return True
            
        except Exception as e:
            self.metrics.rejected_orders += 1
            logger.error(f"Failed to submit order to IBKR: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order at IBKR."""
        try:
            venue_order_id = self.order_mapping.get(order_id)
            if not venue_order_id:
                logger.warning(f"No venue order ID found for {order_id}")
                return False
            
            success = await self.ib_client.cancel_order(venue_order_id)
            
            if success:
                self.metrics.cancelled_orders += 1
                logger.info(f"Cancelled order {order_id} at IBKR")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel order at IBKR: {e}")
            return False
    
    async def get_quote(self, symbol: str) -> Optional[VenueQuote]:
        """Get current quote from IBKR."""
        try:
            quote_data = await self.ib_client.get_market_data(symbol)
            
            if quote_data:
                return VenueQuote(
                    symbol=symbol,
                    bid_price=quote_data.get('bid', 0.0),
                    ask_price=quote_data.get('ask', 0.0),
                    bid_size=quote_data.get('bid_size', 0.0),
                    ask_size=quote_data.get('ask_size', 0.0),
                    timestamp=datetime.now(timezone.utc),
                    venue=self.name
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get quote from IBKR: {e}")
            return None
    
    def _convert_to_ib_order(self, order: Order) -> Dict[str, Any]:
        """Convert internal order to IBKR format."""
        ib_order = {
            'symbol': order.symbol,
            'action': 'BUY' if order.side.value == 'buy' else 'SELL',
            'quantity': order.quantity,
            'orderType': order.order_type.value.upper(),
            'timeInForce': order.time_in_force.value.upper()
        }
        
        if order.price is not None:
            ib_order['limitPrice'] = order.price
        
        if order.stop_price is not None:
            ib_order['stopPrice'] = order.stop_price
        
        return ib_order


class SmartOrderRouter:
    """
    Intelligent order routing system that selects optimal execution venues
    based on market conditions, liquidity, and historical performance.
    """
    
    def __init__(self):
        self.venues: List[ExecutionVenue] = []
        self.routing_rules: Dict[str, Any] = {}
        self.quote_cache: Dict[str, List[VenueQuote]] = {}
        self.cache_ttl_seconds = 1.0  # Quote cache TTL
        
    def add_venue(self, venue: ExecutionVenue):
        """Add execution venue."""
        self.venues.append(venue)
        venue.add_callback(self._on_venue_fill)
        logger.info(f"Added venue: {venue.name}")
    
    def set_routing_rule(self, symbol: str, rule: Dict[str, Any]):
        """Set routing rule for symbol."""
        self.routing_rules[symbol] = rule
    
    async def route_order(self, order: Order) -> bool:
        """Route order to optimal venue."""
        try:
            # Get venue selection
            venue = await self._select_optimal_venue(order)
            
            if not venue:
                logger.error(f"No suitable venue found for order {order.id}")
                return False
            
            # Submit to selected venue
            success = await venue.submit_order(order)
            
            if success:
                logger.info(f"Routed order {order.id} to {venue.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Order routing failed: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order across all venues."""
        success = False
        
        for venue in self.venues:
            try:
                if await venue.cancel_order(order_id):
                    success = True
                    break
            except Exception as e:
                logger.error(f"Failed to cancel order at {venue.name}: {e}")
        
        return success
    
    async def _select_optimal_venue(self, order: Order) -> Optional[ExecutionVenue]:
        """Select optimal venue based on multiple factors."""
        
        # Filter connected venues
        available_venues = [v for v in self.venues if v.status == VenueStatus.CONNECTED]
        
        if not available_venues:
            return None
        
        # For now, use simple selection logic
        # In production, this would consider:
        # - Liquidity and spreads
        # - Historical fill rates
        # - Latency
        # - Fees
        # - Order size vs available liquidity
        
        # Check symbol-specific routing rules
        if order.symbol in self.routing_rules:
            rule = self.routing_rules[order.symbol]
            preferred_venue = rule.get('preferred_venue')
            
            for venue in available_venues:
                if venue.name == preferred_venue:
                    return venue
        
        # Get quotes and select best venue
        venue_scores = []
        
        for venue in available_venues:
            score = await self._calculate_venue_score(venue, order)
            venue_scores.append((venue, score))
        
        # Sort by score (highest first) and return best venue
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        return venue_scores[0][0] if venue_scores else available_venues[0]
    
    async def _calculate_venue_score(self, venue: ExecutionVenue, order: Order) -> float:
        """Calculate venue score for order routing."""
        score = 0.0
        
        # Base score from venue metrics
        metrics = venue.metrics
        score += metrics.fill_rate_percentage * 0.4  # 40% weight on fill rate
        score += metrics.uptime_percentage * 0.2     # 20% weight on uptime
        
        # Penalty for high latency
        if metrics.average_fill_time_ms > 1000:  # > 1 second
            score -= 10.0
        
        # Get current quote for spread analysis
        try:
            quote = await venue.get_quote(order.symbol)
            if quote:
                # Prefer venues with tighter spreads
                spread_bps = (quote.spread / quote.mid_price) * 10000
                score += max(0, 50 - spread_bps) * 0.3  # 30% weight on spread
                
                # Consider available liquidity
                if order.side.value == 'buy':
                    liquidity_score = min(quote.ask_size / order.quantity, 1.0) * 10
                else:
                    liquidity_score = min(quote.bid_size / order.quantity, 1.0) * 10
                
                score += liquidity_score * 0.1  # 10% weight on liquidity
                
        except Exception as e:
            logger.warning(f"Failed to get quote for scoring: {e}")
        
        return max(0.0, score)
    
    async def _on_venue_fill(self, order_id: str, fill: OrderFill):
        """Handle fill notification from venue."""
        # This would be connected to the OMS
        logger.info(f"Received fill for order {order_id}: {fill.quantity}@{fill.price}")


class ExecutionEngine:
    """
    Main execution engine that coordinates order routing, venue management,
    and execution monitoring.
    """
    
    def __init__(self):
        self.router = SmartOrderRouter()
        self.active_orders: Dict[str, Order] = {}
        self.execution_callbacks: List[Callable] = []
        self.monitoring_enabled = True
        
    def add_venue(self, venue: ExecutionVenue):
        """Add execution venue."""
        self.router.add_venue(venue)
    
    def add_execution_callback(self, callback: Callable):
        """Add callback for execution events."""
        self.execution_callbacks.append(callback)
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order for execution."""
        try:
            # Store active order
            self.active_orders[order.id] = order
            
            # Route to venue
            success = await self.router.route_order(order)
            
            if not success:
                # Remove from active orders if submission failed
                self.active_orders.pop(order.id, None)
            
            return success
            
        except Exception as e:
            logger.error(f"Execution engine submission failed: {e}")
            self.active_orders.pop(order.id, None)
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            success = await self.router.cancel_order(order_id)
            
            if success:
                self.active_orders.pop(order_id, None)
            
            return success
            
        except Exception as e:
            logger.error(f"Execution engine cancellation failed: {e}")
            return False
    
    async def handle_fill(self, order_id: str, fill: OrderFill):
        """Handle fill notification."""
        order = self.active_orders.get(order_id)
        
        if order:
            # Update order
            order.add_fill(fill)
            
            # Remove if fully filled
            if order.is_complete:
                self.active_orders.pop(order_id, None)
            
            # Notify callbacks
            for callback in self.execution_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order_id, fill)
                    else:
                        callback(order_id, fill)
                except Exception as e:
                    logger.error(f"Execution callback failed: {e}")
    
    def get_venue_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all venues."""
        return {
            venue.name: {
                'status': venue.status.value,
                'metrics': {
                    'total_orders': venue.metrics.total_orders,
                    'fill_rate': venue.metrics.fill_rate_percentage,
                    'average_fill_time_ms': venue.metrics.average_fill_time_ms,
                    'uptime': venue.metrics.uptime_percentage
                }
            }
            for venue in self.router.venues
        }
    
    def get_active_orders_count(self) -> int:
        """Get count of active orders."""
        return len(self.active_orders)
    
    async def start_monitoring(self):
        """Start execution monitoring."""
        if self.monitoring_enabled:
            asyncio.create_task(self._monitor_execution())
    
    async def _monitor_execution(self):
        """Monitor execution performance and venue health."""
        while self.monitoring_enabled:
            try:
                # Update venue metrics
                for venue in self.router.venues:
                    # Check venue connectivity
                    if venue.status == VenueStatus.CONNECTED:
                        # Ping venue or check connection
                        pass
                
                # Log execution statistics
                if len(self.active_orders) > 0:
                    logger.info(f"Active orders: {len(self.active_orders)}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Execution monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
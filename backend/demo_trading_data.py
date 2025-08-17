"""
Demo Trading Data
Populates the portfolio with sample trading data for demonstration purposes.
This simulates what the system would look like with active trading.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from enums import Venue
from portfolio_service import (
    portfolio_service, Position, Order, Balance, 
    PositionSide, OrderType, OrderStatus
)
from exchange_service import exchange_service, ExchangeStatus

logger = logging.getLogger(__name__)


async def populate_demo_data():
    """Populate the system with demo trading data"""
    
    # Simulate some exchange connections (in paper trading mode)
    for venue in [Venue.BINANCE, Venue.COINBASE]:
        connection = exchange_service.get_exchange_status(venue)
        if connection:
            connection.status = ExchangeStatus.CONNECTED
            connection.last_heartbeat = datetime.now()
            logger.info(f"Demo: Set {venue.value} to connected status")
    
    # Add sample balances
    demo_balances = [
        Balance(
            venue=Venue.BINANCE,
            currency="USDT",
            total=Decimal("10000.00"),
            available=Decimal("8500.00"),
            locked=Decimal("1500.00"),
            timestamp=datetime.now()
        ),
        Balance(
            venue=Venue.BINANCE,
            currency="BTC",
            total=Decimal("0.25"),
            available=Decimal("0.15"),
            locked=Decimal("0.10"),
            timestamp=datetime.now()
        ),
        Balance(
            venue=Venue.COINBASE,
            currency="USD",
            total=Decimal("5000.00"),
            available=Decimal("4200.00"),
            locked=Decimal("800.00"),
            timestamp=datetime.now()
        ),
        Balance(
            venue=Venue.COINBASE,
            currency="ETH",
            total=Decimal("2.5"),
            available=Decimal("2.0"),
            locked=Decimal("0.5"),
            timestamp=datetime.now()
        ),
    ]
    
    for balance in demo_balances:
        portfolio_service.update_balance("main", balance)
        logger.info(f"Demo: Added {balance.currency} balance for {balance.venue.value}")
    
    # Add sample positions
    demo_positions = [
        Position(
            venue=Venue.BINANCE,
            instrument_id="BTCUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("46500.00"),
            unrealized_pnl=Decimal("150.00"),
            realized_pnl=Decimal("0.00"),
            timestamp=datetime.now() - timedelta(hours=2)
        ),
        Position(
            venue=Venue.COINBASE,
            instrument_id="ETH-USD",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("2800.00"),
            current_price=Decimal("2850.00"),
            unrealized_pnl=Decimal("25.00"),
            realized_pnl=Decimal("0.00"),
            timestamp=datetime.now() - timedelta(hours=1)
        ),
        Position(
            venue=Venue.BINANCE,
            instrument_id="ADAUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1000.00"),
            entry_price=Decimal("0.45"),
            current_price=Decimal("0.48"),
            unrealized_pnl=Decimal("30.00"),
            realized_pnl=Decimal("0.00"),
            timestamp=datetime.now() - timedelta(minutes=30)
        ),
    ]
    
    for position in demo_positions:
        portfolio_service.update_position("main", position)
        logger.info(f"Demo: Added {position.instrument_id} position for {position.venue.value}")
    
    # Add sample orders
    demo_orders = [
        Order(
            order_id="binance_order_001",
            venue=Venue.BINANCE,
            instrument_id="ETHUSDT",
            order_type=OrderType.LIMIT,
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            price=Decimal("2900.00"),
            filled_quantity=Decimal("0.0"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now() - timedelta(minutes=15)
        ),
        Order(
            order_id="coinbase_order_001",
            venue=Venue.COINBASE,
            instrument_id="BTC-USD",
            order_type=OrderType.LIMIT,
            side=PositionSide.LONG,
            quantity=Decimal("0.05"),
            price=Decimal("44800.00"),
            filled_quantity=Decimal("0.02"),
            status=OrderStatus.PARTIALLY_FILLED,
            timestamp=datetime.now() - timedelta(minutes=45)
        ),
        Order(
            order_id="binance_order_002",
            venue=Venue.BINANCE,
            instrument_id="SOLUSDT",
            order_type=OrderType.MARKET,
            side=PositionSide.LONG,
            quantity=Decimal("10.0"),
            price=None,
            filled_quantity=Decimal("10.0"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now() - timedelta(hours=3),
            filled_timestamp=datetime.now() - timedelta(hours=3)
        ),
    ]
    
    for order in demo_orders:
        portfolio_service.update_order("main", order)
        logger.info(f"Demo: Added order {order.order_id} for {order.venue.value}")
    
    logger.info("Demo data population completed!")


async def clear_demo_data():
    """Clear all demo data"""
    portfolio = portfolio_service.get_portfolio("main")
    if portfolio:
        portfolio.positions.clear()
        portfolio.orders.clear()
        portfolio.balances.clear()
        portfolio.total_value = Decimal("0")
        portfolio.unrealized_pnl = Decimal("0")
        portfolio.realized_pnl = Decimal("0")
        portfolio.last_updated = datetime.now()
        
    # Reset exchange statuses
    for venue in [Venue.BINANCE, Venue.COINBASE]:
        connection = exchange_service.get_exchange_status(venue)
        if connection:
            connection.status = ExchangeStatus.DISCONNECTED
            connection.last_heartbeat = None
            
    logger.info("Demo data cleared!")


if __name__ == "__main__":
    # Can be run standalone for testing
    asyncio.run(populate_demo_data())
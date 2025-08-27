"""
Mock Level 2 Data Generator for VPIN Testing
Generates realistic Level 2 order book data and trade flows for testing VPIN functionality.
"""

import asyncio
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

# NautilusTrader imports
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import TradeTick, QuoteTick, OrderBookDelta
from nautilus_trader.model.enums import PriceType, AggressorSide, BookAction
from nautilus_trader.model.objects import Price, Quantity

# VPIN imports
from ..models import TradeSide, MarketRegime


class MarketMicrostructure(Enum):
    """Different market microstructure scenarios"""
    NORMAL_FLOW = "normal"
    HIGH_FREQUENCY = "high_freq"
    INFORMED_TRADING = "informed"
    LIQUIDITY_CRISIS = "crisis"
    TOXIC_FLOW = "toxic"


@dataclass
class Level2Quote:
    """Level 2 order book quote"""
    price: float
    size: int
    exchange: str
    market_maker: str
    timestamp: int


@dataclass
class MarketDataConfig:
    """Configuration for market data generation"""
    symbol: str = "AAPL"
    base_price: float = 150.0
    volatility: float = 0.02
    bid_ask_spread: float = 0.01
    depth_levels: int = 10
    tick_frequency_hz: float = 10.0
    trade_frequency_hz: float = 5.0
    market_microstructure: MarketMicrostructure = MarketMicrostructure.NORMAL_FLOW


class MockLevel2Generator:
    """
    Mock Level 2 data generator for VPIN testing
    
    Generates realistic:
    - Order book depth data (10 levels)
    - Trade tick data with proper side classification
    - Market regime scenarios (normal, stressed, toxic)
    - Exchange attribution and market maker information
    """
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.current_price = config.base_price
        self.current_timestamp = int(datetime.now().timestamp() * 1_000_000_000)
        self.order_book = self._initialize_order_book()
        
        # Market makers and exchanges for realism
        self.exchanges = ["NYSE", "NASDAQ", "BATS", "EDGX", "IEX", "ARCA", "BYX"]
        self.market_makers = ["CITADEL", "VIRTU", "CDRG", "ARCA", "IEX", "JANE", "OPTIVER"]
        
        # Running statistics
        self.trades_generated = 0
        self.quotes_generated = 0
        
    def _initialize_order_book(self) -> Dict[str, List[Level2Quote]]:
        """Initialize realistic order book structure"""
        book = {"bids": [], "asks": []}
        
        # Generate initial bid/ask levels
        for level in range(self.config.depth_levels):
            # Bids (decreasing prices)
            bid_price = self.current_price - (self.config.bid_ask_spread / 2) - (level * 0.01)
            bid_size = random.randint(100, 2000) * (10 - level)  # Larger sizes at better prices
            
            bid = Level2Quote(
                price=round(bid_price, 2),
                size=bid_size,
                exchange=random.choice(self.exchanges),
                market_maker=random.choice(self.market_makers),
                timestamp=self.current_timestamp
            )
            book["bids"].append(bid)
            
            # Asks (increasing prices)
            ask_price = self.current_price + (self.config.bid_ask_spread / 2) + (level * 0.01)
            ask_size = random.randint(100, 2000) * (10 - level)
            
            ask = Level2Quote(
                price=round(ask_price, 2),
                size=ask_size,
                exchange=random.choice(self.exchanges),
                market_maker=random.choice(self.market_makers),
                timestamp=self.current_timestamp
            )
            book["asks"].append(ask)
            
        return book
        
    def _update_price_with_microstructure(self) -> float:
        """Update price based on market microstructure scenario"""
        if self.config.market_microstructure == MarketMicrostructure.NORMAL_FLOW:
            # Standard random walk
            change = np.random.normal(0, self.config.volatility * self.current_price)
            
        elif self.config.market_microstructure == MarketMicrostructure.HIGH_FREQUENCY:
            # More frequent, smaller movements
            change = np.random.normal(0, self.config.volatility * self.current_price * 0.5)
            
        elif self.config.market_microstructure == MarketMicrostructure.INFORMED_TRADING:
            # Directional bias with occasional large moves
            trend = 0.0001 * self.current_price  # Slight upward bias
            shock = np.random.normal(0, self.config.volatility * self.current_price * 2) if random.random() < 0.05 else 0
            change = trend + shock
            
        elif self.config.market_microstructure == MarketMicrostructure.LIQUIDITY_CRISIS:
            # Wider spreads, more volatile
            change = np.random.normal(0, self.config.volatility * self.current_price * 3)
            
        elif self.config.market_microstructure == MarketMicrostructure.TOXIC_FLOW:
            # High VPIN scenario - persistent directional pressure
            direction = 1 if random.random() > 0.3 else -1  # 70% probability same direction
            magnitude = np.random.exponential(self.config.volatility * self.current_price)
            change = direction * magnitude
            
        else:
            change = 0
            
        self.current_price = max(1.0, self.current_price + change)
        return self.current_price
        
    def generate_trade_tick(self) -> TradeTick:
        """Generate realistic trade tick with proper side classification"""
        self.current_timestamp += random.randint(50_000_000, 200_000_000)  # 50-200ms between trades
        
        # Update price based on microstructure
        price = self._update_price_with_microstructure()
        
        # Determine trade side and size based on market microstructure
        if self.config.market_microstructure == MarketMicrostructure.INFORMED_TRADING:
            # Larger trades, more buy-side pressure
            size = random.randint(500, 5000)
            aggressor_side = AggressorSide.BUYER if random.random() > 0.4 else AggressorSide.SELLER
            
        elif self.config.market_microstructure == MarketMicrostructure.TOXIC_FLOW:
            # Persistent order flow imbalance
            size = random.randint(1000, 10000)
            aggressor_side = AggressorSide.BUYER if random.random() > 0.25 else AggressorSide.SELLER  # 75% buy pressure
            
        else:
            # Normal trading patterns
            size = random.randint(100, 2000)
            aggressor_side = AggressorSide.BUYER if random.random() > 0.5 else AggressorSide.SELLER
            
        tick = TradeTick(
            instrument_id=InstrumentId.from_str(f"{self.config.symbol}.NASDAQ"),
            price=Price.from_str(f"{price:.2f}"),
            size=Quantity.from_int(size),
            aggressor_side=aggressor_side,
            trade_id=f"trade_{self.trades_generated}",
            ts_event=self.current_timestamp,
            ts_init=self.current_timestamp
        )
        
        self.trades_generated += 1
        return tick
        
    def generate_quote_tick(self) -> QuoteTick:
        """Generate Level 1 quote tick"""
        self.current_timestamp += random.randint(10_000_000, 100_000_000)  # 10-100ms between quotes
        
        bid_price = self.current_price - (self.config.bid_ask_spread / 2)
        ask_price = self.current_price + (self.config.bid_ask_spread / 2)
        
        # Adjust spread based on market microstructure
        if self.config.market_microstructure == MarketMicrostructure.LIQUIDITY_CRISIS:
            spread_multiplier = random.uniform(2.0, 5.0)
            bid_price = self.current_price - (self.config.bid_ask_spread * spread_multiplier / 2)
            ask_price = self.current_price + (self.config.bid_ask_spread * spread_multiplier / 2)
            
        tick = QuoteTick(
            instrument_id=InstrumentId.from_str(f"{self.config.symbol}.NASDAQ"),
            bid_price=Price.from_str(f"{bid_price:.2f}"),
            ask_price=Price.from_str(f"{ask_price:.2f}"),
            bid_size=Quantity.from_int(random.randint(100, 1000)),
            ask_size=Quantity.from_int(random.randint(100, 1000)),
            ts_event=self.current_timestamp,
            ts_init=self.current_timestamp
        )
        
        self.quotes_generated += 1
        return tick
        
    def generate_order_book_delta(self) -> OrderBookDelta:
        """Generate order book delta (Level 2 change)"""
        self.current_timestamp += random.randint(5_000_000, 50_000_000)  # 5-50ms between updates
        
        # Choose random level and side to update
        level = random.randint(0, min(4, self.config.depth_levels - 1))  # Focus on top 5 levels
        side = "bids" if random.random() > 0.5 else "asks"
        action = random.choice([BookAction.ADD, BookAction.UPDATE, BookAction.DELETE])
        
        if action == BookAction.DELETE and len(self.order_book[side]) > level:
            price = self.order_book[side][level].price
            size = 0
        else:
            if side == "bids":
                price = self.current_price - (self.config.bid_ask_spread / 2) - (level * 0.01)
            else:
                price = self.current_price + (self.config.bid_ask_spread / 2) + (level * 0.01)
            size = random.randint(100, 2000) * (10 - level)
            
        delta = OrderBookDelta(
            instrument_id=InstrumentId.from_str(f"{self.config.symbol}.NASDAQ"),
            action=action,
            order=None,  # Simplified for testing
            ts_event=self.current_timestamp,
            ts_init=self.current_timestamp
        )
        
        return delta
        
    def get_current_level2_snapshot(self) -> Dict[str, List[Dict[str, any]]]:
        """Get current Level 2 order book snapshot"""
        snapshot = {
            "symbol": self.config.symbol,
            "timestamp": self.current_timestamp,
            "bids": [],
            "asks": []
        }
        
        # Update order book with current price
        self._update_order_book_prices()
        
        for bid in self.order_book["bids"][:self.config.depth_levels]:
            snapshot["bids"].append({
                "level": len(snapshot["bids"]) + 1,
                "price": bid.price,
                "size": bid.size,
                "exchange": bid.exchange,
                "market_maker": bid.market_maker
            })
            
        for ask in self.order_book["asks"][:self.config.depth_levels]:
            snapshot["asks"].append({
                "level": len(snapshot["asks"]) + 1,
                "price": ask.price,
                "size": ask.size,
                "exchange": ask.exchange,
                "market_maker": ask.market_maker
            })
            
        return snapshot
        
    def _update_order_book_prices(self):
        """Update order book prices to reflect current market price"""
        for i, bid in enumerate(self.order_book["bids"]):
            new_price = self.current_price - (self.config.bid_ask_spread / 2) - (i * 0.01)
            bid.price = round(max(1.0, new_price), 2)
            bid.timestamp = self.current_timestamp
            
        for i, ask in enumerate(self.order_book["asks"]):
            new_price = self.current_price + (self.config.bid_ask_spread / 2) + (i * 0.01)
            ask.price = round(new_price, 2)
            ask.timestamp = self.current_timestamp
            
    async def generate_trade_stream(self, duration_seconds: int = 60) -> AsyncGenerator[TradeTick, None]:
        """Generate continuous stream of trade ticks"""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            trade = self.generate_trade_tick()
            yield trade
            
            # Wait based on trade frequency
            wait_time = 1.0 / self.config.trade_frequency_hz
            await asyncio.sleep(wait_time + random.uniform(-0.1, 0.1))
            
    async def generate_quote_stream(self, duration_seconds: int = 60) -> AsyncGenerator[QuoteTick, None]:
        """Generate continuous stream of quote ticks"""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            quote = self.generate_quote_tick()
            yield quote
            
            # Wait based on tick frequency
            wait_time = 1.0 / self.config.tick_frequency_hz
            await asyncio.sleep(wait_time + random.uniform(-0.05, 0.05))
            
    def generate_vpin_scenario(self, scenario: MarketRegime, duration_minutes: int = 5) -> List[TradeTick]:
        """Generate trade sequence for specific VPIN scenario"""
        # Configure market microstructure for scenario
        if scenario == MarketRegime.NORMAL:
            self.config.market_microstructure = MarketMicrostructure.NORMAL_FLOW
        elif scenario == MarketRegime.STRESSED:
            self.config.market_microstructure = MarketMicrostructure.HIGH_FREQUENCY
        elif scenario == MarketRegime.TOXIC:
            self.config.market_microstructure = MarketMicrostructure.INFORMED_TRADING
        elif scenario == MarketRegime.EXTREME:
            self.config.market_microstructure = MarketMicrostructure.TOXIC_FLOW
            
        # Generate trades for duration
        trades = []
        target_trades = duration_minutes * 60 * int(self.config.trade_frequency_hz)
        
        for _ in range(target_trades):
            trade = self.generate_trade_tick()
            trades.append(trade)
            
        return trades
        
    def get_statistics(self) -> Dict[str, any]:
        """Get generator statistics"""
        return {
            "trades_generated": self.trades_generated,
            "quotes_generated": self.quotes_generated,
            "current_price": self.current_price,
            "current_timestamp": self.current_timestamp,
            "market_microstructure": self.config.market_microstructure.value,
            "order_book_depth": len(self.order_book["bids"]) + len(self.order_book["asks"])
        }
        
    def reset(self):
        """Reset generator state"""
        self.current_price = self.config.base_price
        self.current_timestamp = int(datetime.now().timestamp() * 1_000_000_000)
        self.order_book = self._initialize_order_book()
        self.trades_generated = 0
        self.quotes_generated = 0


# Predefined scenarios for easy testing
VPIN_TEST_SCENARIOS = {
    "normal_market": MarketDataConfig(
        symbol="AAPL",
        base_price=150.0,
        volatility=0.015,
        market_microstructure=MarketMicrostructure.NORMAL_FLOW
    ),
    "high_frequency": MarketDataConfig(
        symbol="TSLA",
        base_price=250.0,
        volatility=0.025,
        tick_frequency_hz=50.0,
        trade_frequency_hz=25.0,
        market_microstructure=MarketMicrostructure.HIGH_FREQUENCY
    ),
    "informed_trading": MarketDataConfig(
        symbol="MSFT",
        base_price=300.0,
        volatility=0.02,
        market_microstructure=MarketMicrostructure.INFORMED_TRADING
    ),
    "liquidity_crisis": MarketDataConfig(
        symbol="GOOGL",
        base_price=130.0,
        volatility=0.04,
        bid_ask_spread=0.05,
        market_microstructure=MarketMicrostructure.LIQUIDITY_CRISIS
    ),
    "toxic_flow": MarketDataConfig(
        symbol="AMZN",
        base_price=140.0,
        volatility=0.03,
        market_microstructure=MarketMicrostructure.TOXIC_FLOW
    )
}


def create_mock_generator(scenario_name: str = "normal_market") -> MockLevel2Generator:
    """Factory function to create mock generator with predefined scenario"""
    if scenario_name not in VPIN_TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(VPIN_TEST_SCENARIOS.keys())}")
        
    config = VPIN_TEST_SCENARIOS[scenario_name]
    return MockLevel2Generator(config)
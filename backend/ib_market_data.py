"""
Interactive Brokers Market Data Management
Comprehensive market data subscription and management system.
"""

import asyncio
import logging
from typing import Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from ibapi.contract import Contract
from ibapi.common import TickerId, BarData


class IBTickType(Enum):
    """IB Tick Types"""
    BID_SIZE = 0
    BID_PRICE = 1
    ASK_PRICE = 2
    ASK_SIZE = 3
    LAST_PRICE = 4
    LAST_SIZE = 5
    HIGH = 6
    LOW = 7
    VOLUME = 8
    CLOSE_PRICE = 9
    BID_OPTION_COMPUTATION = 10
    ASK_OPTION_COMPUTATION = 11
    LAST_OPTION_COMPUTATION = 12
    MODEL_OPTION = 13
    OPEN_TICK = 14
    LOW_13_WEEK = 15
    HIGH_13_WEEK = 16
    LOW_26_WEEK = 17
    HIGH_26_WEEK = 18
    LOW_52_WEEK = 19
    HIGH_52_WEEK = 20
    AVG_VOLUME = 21


class IBBarSize(Enum):
    """IB Bar Sizes"""
    SEC_1 = "1 sec"
    SEC_5 = "5 secs"
    SEC_10 = "10 secs"
    SEC_15 = "15 secs"
    SEC_30 = "30 secs"
    MIN_1 = "1 min"
    MIN_2 = "2 mins"
    MIN_3 = "3 mins"
    MIN_5 = "5 mins"
    MIN_10 = "10 mins"
    MIN_15 = "15 mins"
    MIN_20 = "20 mins"
    MIN_30 = "30 mins"
    HOUR_1 = "1 hour"
    HOUR_2 = "2 hours"
    HOUR_3 = "3 hours"
    HOUR_4 = "4 hours"
    HOUR_8 = "8 hours"
    DAY_1 = "1 day"
    WEEK_1 = "1 week"
    MONTH_1 = "1 month"


class IBDataType(Enum):
    """IB Data Types"""
    TRADES = "TRADES"
    MIDPOINT = "MIDPOINT"
    BID = "BID"
    ASK = "ASK"
    BID_ASK = "BID_ASK"
    ADJUSTED_LAST = "ADJUSTED_LAST"
    HISTORICAL_VOLATILITY = "HISTORICAL_VOLATILITY"
    OPTION_IMPLIED_VOLATILITY = "OPTION_IMPLIED_VOLATILITY"


@dataclass
class IBTick:
    """Individual tick data"""
    tick_type: int
    value: float
    timestamp: datetime
    attrib: Any = None
    size: int | None = None


@dataclass
class IBQuote:
    """Bid/Ask quote data"""
    bid_price: float | None = None
    bid_size: int | None = None
    ask_price: float | None = None
    ask_size: int | None = None
    timestamp: datetime | None = None


@dataclass
class IBTrade:
    """Trade tick data"""
    price: float | None = None
    size: int | None = None
    timestamp: datetime | None = None


@dataclass
class IBMarketDataSnapshot:
    """Complete market data snapshot"""
    symbol: str
    contract_id: int
    timestamp: datetime
    quote: IBQuote = field(default_factory=IBQuote)
    trade: IBTrade = field(default_factory=IBTrade)
    volume: int | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    previous_close: float | None = None
    ticks: list[IBTick] = field(default_factory=list)


@dataclass
class IBBarData:
    """Historical/Real-time bar data"""
    symbol: str
    contract_id: int
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    wap: float  # Weighted Average Price
    count: int  # Number of trades


@dataclass
class IBMarketDataSubscription:
    """Market data subscription details"""
    req_id: int
    symbol: str
    contract_id: int
    contract: Contract
    tick_types: Set[int] = field(default_factory=set)
    snapshot_only: bool = False
    regulatory_snapshot: bool = False
    mkt_data_options: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_update: datetime | None = None


class IBMarketDataManager:
    """
    Interactive Brokers Market Data Manager
    
    Handles comprehensive market data subscriptions, real-time feeds, historical data, and data normalization.
    """
    
    def __init__(self, ib_client):
        self.logger = logging.getLogger(__name__)
        self.ib_client = ib_client
        
        # Subscriptions
        self.subscriptions: dict[int, IBMarketDataSubscription] = {}  # req_id -> subscription
        self.symbol_to_req_id: dict[str, int] = {}  # symbol -> req_id
        
        # Market data storage
        self.market_data: dict[str, IBMarketDataSnapshot] = {}  # symbol -> snapshot
        self.historical_bars: dict[str, list[IBBarData]] = {}  # symbol -> bars
        
        # Request tracking
        self.next_req_id = 2000
        self.pending_requests: Set[int] = set()
        
        # Callbacks
        self.tick_callbacks: list[Callable] = []
        self.quote_callbacks: list[Callable] = []
        self.trade_callbacks: list[Callable] = []
        self.bar_callbacks: list[Callable] = []
        
        # Setup IB API callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup IB API callbacks for market data"""
        if hasattr(self.ib_client, 'wrapper'):
            # Store original methods
            original_tick_price = getattr(self.ib_client.wrapper, 'tickPrice', None)
            original_tick_size = getattr(self.ib_client.wrapper, 'tickSize', None)
            original_tick_string = getattr(self.ib_client.wrapper, 'tickString', None)
            original_tick_generic = getattr(self.ib_client.wrapper, 'tickGeneric', None)
            original_real_time_bar = getattr(self.ib_client.wrapper, 'realtimeBar', None)
            original_historical_data = getattr(self.ib_client.wrapper, 'historicalData', None)
            original_historical_data_end = getattr(self.ib_client.wrapper, 'historicalDataEnd', None)
            
            # Override with enhanced handlers
            def tick_price_handler(reqId: int, tickType: int, price: float, attrib):
                self._handle_tick_price(reqId, tickType, price, attrib)
                if original_tick_price:
                    original_tick_price(reqId, tickType, price, attrib)
            
            def tick_size_handler(reqId: int, tickType: int, size: int):
                self._handle_tick_size(reqId, tickType, size)
                if original_tick_size:
                    original_tick_size(reqId, tickType, size)
            
            def tick_string_handler(reqId: int, tickType: int, value: str):
                self._handle_tick_string(reqId, tickType, value)
                if original_tick_string:
                    original_tick_string(reqId, tickType, value)
            
            def tick_generic_handler(reqId: int, tickType: int, value: float):
                self._handle_tick_generic(reqId, tickType, value)
                if original_tick_generic:
                    original_tick_generic(reqId, tickType, value)
            
            def real_time_bar_handler(reqId: int, time: int, open_: float, high: float, low: float, close: float, volume: int, wap: float, count: int):
                self._handle_real_time_bar(reqId, time, open_, high, low, close, volume, wap, count)
                if original_real_time_bar:
                    original_real_time_bar(reqId, time, open_, high, low, close, volume, wap, count)
            
            def historical_data_handler(reqId: int, bar: BarData):
                self._handle_historical_data(reqId, bar)
                if original_historical_data:
                    original_historical_data(reqId, bar)
            
            def historical_data_end_handler(reqId: int, start: str, end: str):
                self._handle_historical_data_end(reqId, start, end)
                if original_historical_data_end:
                    original_historical_data_end(reqId, start, end)
            
            # Set enhanced handlers
            self.ib_client.wrapper.tickPrice = tick_price_handler
            self.ib_client.wrapper.tickSize = tick_size_handler
            self.ib_client.wrapper.tickString = tick_string_handler
            self.ib_client.wrapper.tickGeneric = tick_generic_handler
            self.ib_client.wrapper.realtimeBar = real_time_bar_handler
            self.ib_client.wrapper.historicalData = historical_data_handler
            self.ib_client.wrapper.historicalDataEnd = historical_data_end_handler
    
    async def subscribe_market_data(self, symbol: str, contract: Contract, snapshot_only: bool = False, regulatory_snapshot: bool = False, mkt_data_options: list[str] = None) -> bool:
        """Subscribe to real-time market data"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            # Check if already subscribed
            if symbol in self.symbol_to_req_id:
                self.logger.warning(f"Already subscribed to {symbol}")
                return True
            
            req_id = self._get_next_req_id()
            
            # Create subscription
            subscription = IBMarketDataSubscription(
                req_id=req_id, symbol=symbol, contract_id=contract.conId, contract=contract, snapshot_only=snapshot_only, regulatory_snapshot=regulatory_snapshot, mkt_data_options=mkt_data_options or []
            )
            
            # Store subscription
            self.subscriptions[req_id] = subscription
            self.symbol_to_req_id[symbol] = req_id
            
            # Initialize market data snapshot
            if symbol not in self.market_data:
                self.market_data[symbol] = IBMarketDataSnapshot(
                    symbol=symbol, contract_id=contract.conId, timestamp=datetime.now()
                )
            
            # Request market data from IB
            generic_tick_list = ", ".join(mkt_data_options) if mkt_data_options else ""
            self.ib_client.reqMktData(req_id, contract, generic_tick_list, snapshot_only, regulatory_snapshot, [])
            
            self.logger.info(f"Subscribed to market data for {symbol} (req_id: {req_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to market data for {symbol}: {e}")
            return False
    
    async def unsubscribe_market_data(self, symbol: str) -> bool:
        """Unsubscribe from market data"""
        try:
            if symbol not in self.symbol_to_req_id:
                self.logger.warning(f"No subscription found for {symbol}")
                return False
            
            req_id = self.symbol_to_req_id[symbol]
            
            # Cancel subscription
            self.ib_client.cancelMktData(req_id)
            
            # Clean up
            del self.subscriptions[req_id]
            del self.symbol_to_req_id[symbol]
            
            # Optionally keep market data for reference
            # del self.market_data[symbol]
            
            self.logger.info(f"Unsubscribed from market data for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from market data for {symbol}: {e}")
            return False
    
    async def request_historical_data(self, symbol: str, contract: Contract, end_date_time: str = "", duration: str = "1 D", bar_size: str = "1 min", what_to_show: str = "TRADES", use_rth: bool = True, format_date: int = 1) -> bool:
        """Request historical data"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            req_id = self._get_next_req_id()
            self.pending_requests.add(req_id)
            
            # Initialize historical data storage
            if symbol not in self.historical_bars:
                self.historical_bars[symbol] = []
            
            # Request historical data
            self.ib_client.reqHistoricalData(
                req_id, contract, end_date_time, duration, bar_size, what_to_show, use_rth, format_date, False, []
            )
            
            self.logger.info(f"Requested historical data for {symbol} (req_id: {req_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error requesting historical data for {symbol}: {e}")
            return False
    
    def _handle_tick_price(self, req_id: int, tick_type: int, price: float, attrib):
        """Handle tick price updates"""
        if req_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[req_id]
        symbol = subscription.symbol
        timestamp = datetime.now()
        
        # Update market data
        if symbol not in self.market_data:
            self.market_data[symbol] = IBMarketDataSnapshot(
                symbol=symbol, contract_id=subscription.contract_id, timestamp=timestamp
            )
        
        snapshot = self.market_data[symbol]
        snapshot.timestamp = timestamp
        subscription.last_update = timestamp
        
        # Create tick
        tick = IBTick(tick_type=tick_type, value=price, timestamp=timestamp, attrib=attrib)
        snapshot.ticks.append(tick)
        
        # Update specific fields
        if tick_type == IBTickType.BID_PRICE.value:
            snapshot.quote.bid_price = price
            snapshot.quote.timestamp = timestamp
        elif tick_type == IBTickType.ASK_PRICE.value:
            snapshot.quote.ask_price = price
            snapshot.quote.timestamp = timestamp
        elif tick_type == IBTickType.LAST_PRICE.value:
            snapshot.trade.price = price
            snapshot.trade.timestamp = timestamp
        elif tick_type == IBTickType.HIGH.value:
            snapshot.high_price = price
        elif tick_type == IBTickType.LOW.value:
            snapshot.low_price = price
        elif tick_type == IBTickType.CLOSE_PRICE.value:
            snapshot.close_price = price
        elif tick_type == IBTickType.OPEN_TICK.value:
            snapshot.open_price = price
        
        # Notify callbacks
        asyncio.create_task(self._notify_tick_callbacks(symbol, tick))
        if tick_type in [IBTickType.BID_PRICE.value, IBTickType.ASK_PRICE.value]:
            asyncio.create_task(self._notify_quote_callbacks(symbol, snapshot.quote))
        elif tick_type == IBTickType.LAST_PRICE.value:
            asyncio.create_task(self._notify_trade_callbacks(symbol, snapshot.trade))
    
    def _handle_tick_size(self, req_id: int, tick_type: int, size: int):
        """Handle tick size updates"""
        if req_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[req_id]
        symbol = subscription.symbol
        timestamp = datetime.now()
        
        if symbol not in self.market_data:
            self.market_data[symbol] = IBMarketDataSnapshot(
                symbol=symbol, contract_id=subscription.contract_id, timestamp=timestamp
            )
        
        snapshot = self.market_data[symbol]
        snapshot.timestamp = timestamp
        subscription.last_update = timestamp
        
        # Create tick
        tick = IBTick(tick_type=tick_type, value=float(size), timestamp=timestamp, size=size)
        snapshot.ticks.append(tick)
        
        # Update specific fields
        if tick_type == IBTickType.BID_SIZE.value:
            snapshot.quote.bid_size = size
            snapshot.quote.timestamp = timestamp
        elif tick_type == IBTickType.ASK_SIZE.value:
            snapshot.quote.ask_size = size
            snapshot.quote.timestamp = timestamp
        elif tick_type == IBTickType.LAST_SIZE.value:
            snapshot.trade.size = size
            snapshot.trade.timestamp = timestamp
        elif tick_type == IBTickType.VOLUME.value:
            snapshot.volume = size
        
        # Notify callbacks
        asyncio.create_task(self._notify_tick_callbacks(symbol, tick))
        if tick_type in [IBTickType.BID_SIZE.value, IBTickType.ASK_SIZE.value]:
            asyncio.create_task(self._notify_quote_callbacks(symbol, snapshot.quote))
        elif tick_type == IBTickType.LAST_SIZE.value:
            asyncio.create_task(self._notify_trade_callbacks(symbol, snapshot.trade))
    
    def _handle_tick_string(self, req_id: int, tick_type: int, value: str):
        """Handle tick string updates"""
        if req_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[req_id]
        symbol = subscription.symbol
        timestamp = datetime.now()
        
        # Create tick
        tick = IBTick(tick_type=tick_type, value=0.0, timestamp=timestamp)
        
        if symbol in self.market_data:
            self.market_data[symbol].ticks.append(tick)
            self.market_data[symbol].timestamp = timestamp
            subscription.last_update = timestamp
        
        # Notify callbacks
        asyncio.create_task(self._notify_tick_callbacks(symbol, tick))
    
    def _handle_tick_generic(self, req_id: int, tick_type: int, value: float):
        """Handle generic tick updates"""
        if req_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[req_id]
        symbol = subscription.symbol
        timestamp = datetime.now()
        
        # Create tick
        tick = IBTick(tick_type=tick_type, value=value, timestamp=timestamp)
        
        if symbol in self.market_data:
            self.market_data[symbol].ticks.append(tick)
            self.market_data[symbol].timestamp = timestamp
            subscription.last_update = timestamp
        
        # Notify callbacks
        asyncio.create_task(self._notify_tick_callbacks(symbol, tick))
    
    def _handle_real_time_bar(self, req_id: int, time: int, open_: float, high: float, low: float, close: float, volume: int, wap: float, count: int):
        """Handle real-time bar updates"""
        # Implementation for real-time bars
        pass
    
    def _handle_historical_data(self, req_id: int, bar: BarData):
        """Handle historical data"""
        # Extract symbol from pending requests or subscriptions
        # For now, we'll need to track this separately
        pass
    
    def _handle_historical_data_end(self, req_id: int, start: str, end: str):
        """Handle end of historical data"""
        if req_id in self.pending_requests:
            self.pending_requests.remove(req_id)
    
    async def _notify_tick_callbacks(self, symbol: str, tick: IBTick):
        """Notify tick callbacks"""
        for callback in self.tick_callbacks:
            try:
                await callback(symbol, tick)
            except Exception as e:
                self.logger.error(f"Error in tick callback: {e}")
    
    async def _notify_quote_callbacks(self, symbol: str, quote: IBQuote):
        """Notify quote callbacks"""
        for callback in self.quote_callbacks:
            try:
                await callback(symbol, quote)
            except Exception as e:
                self.logger.error(f"Error in quote callback: {e}")
    
    async def _notify_trade_callbacks(self, symbol: str, trade: IBTrade):
        """Notify trade callbacks"""
        for callback in self.trade_callbacks:
            try:
                await callback(symbol, trade)
            except Exception as e:
                self.logger.error(f"Error in trade callback: {e}")
    
    def add_tick_callback(self, callback: Callable):
        """Add tick callback"""
        self.tick_callbacks.append(callback)
    
    def add_quote_callback(self, callback: Callable):
        """Add quote callback"""
        self.quote_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Add trade callback"""
        self.trade_callbacks.append(callback)
    
    def add_bar_callback(self, callback: Callable):
        """Add bar callback"""
        self.bar_callbacks.append(callback)
    
    def get_market_data(self, symbol: str) -> IBMarketDataSnapshot | None:
        """Get current market data snapshot"""
        return self.market_data.get(symbol)
    
    def get_all_market_data(self) -> dict[str, IBMarketDataSnapshot]:
        """Get all market data snapshots"""
        return self.market_data.copy()
    
    def get_subscriptions(self) -> dict[int, IBMarketDataSubscription]:
        """Get all active subscriptions"""
        return self.subscriptions.copy()
    
    def _get_next_req_id(self) -> int:
        """Get next request ID"""
        req_id = self.next_req_id
        self.next_req_id += 1
        return req_id
    
    def cleanup(self):
        """Cleanup subscriptions and data"""
        # Cancel all subscriptions
        for req_id in list(self.subscriptions.keys()):
            try:
                self.ib_client.cancelMktData(req_id)
            except:
                pass
        
        self.subscriptions.clear()
        self.symbol_to_req_id.clear()
        self.pending_requests.clear()
        self.logger.info("Market data manager cleaned up")


# Global market data manager instance
_ib_market_data_manager: IBMarketDataManager | None = None

def get_ib_market_data_manager(ib_client) -> IBMarketDataManager:
    """Get or create the IB market data manager singleton"""
    global _ib_market_data_manager
    
    if _ib_market_data_manager is None:
        _ib_market_data_manager = IBMarketDataManager(ib_client)
    
    return _ib_market_data_manager

def reset_ib_market_data_manager():
    """Reset the market data manager singleton (for testing)"""
    global _ib_market_data_manager
    if _ib_market_data_manager:
        _ib_market_data_manager.cleanup()
    _ib_market_data_manager = None
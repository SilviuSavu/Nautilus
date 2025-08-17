"""
Market Data Event Handlers
Provides specific handlers for different types of market data events.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from enums import DataType, Venue
from market_data_service import NormalizedMarketData
from data_normalizer import data_normalizer, NormalizedTick, NormalizedQuote, NormalizedBar
from redis_cache import redis_cache
from historical_data_service import historical_data_service


@dataclass
class TickData:
    """Normalized tick data structure"""
    venue: str
    instrument_id: str
    price: float
    quantity: float
    timestamp: int
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None


@dataclass
class QuoteData:
    """Normalized quote data structure"""
    venue: str
    instrument_id: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: int
    spread: Optional[float] = None


@dataclass
class BarData:
    """Normalized OHLCV bar data structure"""
    venue: str
    instrument_id: str
    timeframe: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    timestamp: int
    is_closed: bool = True


@dataclass
class OrderBookData:
    """Normalized order book data structure"""
    venue: str
    instrument_id: str
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]  # [[price, size], ...]
    timestamp: int
    sequence: Optional[int] = None


class MarketDataHandlers:
    """
    Market Data Event Handlers for processing different types of market data
    with real-time event handling and data broadcasting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._tick_handlers: List[callable] = []
        self._quote_handlers: List[callable] = []
        self._bar_handlers: List[callable] = []
        self._orderbook_handlers: List[callable] = []
        self._broadcast_callback: Optional[callable] = None
        
    def set_broadcast_callback(self, callback: callable) -> None:
        """Set callback for broadcasting data to WebSocket clients"""
        self._broadcast_callback = callback
        
    def add_tick_handler(self, handler: callable) -> None:
        """Add handler for tick data"""
        self._tick_handlers.append(handler)
        
    def add_quote_handler(self, handler: callable) -> None:
        """Add handler for quote data"""
        self._quote_handlers.append(handler)
        
    def add_bar_handler(self, handler: callable) -> None:
        """Add handler for bar data"""
        self._bar_handlers.append(handler)
        
    def add_orderbook_handler(self, handler: callable) -> None:
        """Add handler for order book data"""
        self._orderbook_handlers.append(handler)
        
    async def handle_market_data(self, data: NormalizedMarketData) -> None:
        """Main handler that routes market data to specific handlers"""
        try:
            # First normalize the data using the data normalizer
            normalized_data = data_normalizer.normalize_market_data(data)
            
            # Cache the normalized data in Redis and store in PostgreSQL
            if isinstance(normalized_data, NormalizedTick):
                await redis_cache.cache_tick(normalized_data)
                await historical_data_service.store_tick(normalized_data)
                await self._handle_normalized_tick(normalized_data)
            elif isinstance(normalized_data, NormalizedQuote):
                await redis_cache.cache_quote(normalized_data)
                await historical_data_service.store_quote(normalized_data)
                await self._handle_normalized_quote(normalized_data)
            elif isinstance(normalized_data, NormalizedBar):
                await redis_cache.cache_bar(normalized_data)
                await historical_data_service.store_bar(normalized_data)
                await self._handle_normalized_bar(normalized_data)
            else:
                self.logger.warning(f"Unknown normalized data type: {type(normalized_data)}")
                
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
            
    async def _handle_tick_data(self, data: NormalizedMarketData) -> None:
        """Handle tick data events"""
        try:
            # Extract tick-specific fields from normalized data
            tick_data = TickData(
                venue=data.venue,
                instrument_id=data.instrument_id,
                price=float(data.data.get("price", 0)),
                quantity=float(data.data.get("quantity", 0)),
                timestamp=data.timestamp,
                side=data.data.get("side", "unknown"),
                trade_id=data.data.get("trade_id")
            )
            
            self.logger.debug(f"Processing tick: {tick_data.venue} {tick_data.instrument_id} @ {tick_data.price}")
            
            # Call registered tick handlers
            for handler in self._tick_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(tick_data)
                    else:
                        handler(tick_data)
                except Exception as e:
                    self.logger.error(f"Error in tick handler: {e}")
                    
            # Broadcast to WebSocket clients
            if self._broadcast_callback:
                await self._broadcast_callback({
                    "type": "tick",
                    "data": asdict(tick_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling tick data: {e}")
            
    async def _handle_quote_data(self, data: NormalizedMarketData) -> None:
        """Handle quote data events"""
        try:
            # Extract quote-specific fields
            bid_price = float(data.data.get("bid", 0))
            ask_price = float(data.data.get("ask", 0))
            
            quote_data = QuoteData(
                venue=data.venue,
                instrument_id=data.instrument_id,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=float(data.data.get("bid_size", 0)),
                ask_size=float(data.data.get("ask_size", 0)),
                timestamp=data.timestamp,
                spread=ask_price - bid_price if bid_price > 0 and ask_price > 0 else None
            )
            
            self.logger.debug(f"Processing quote: {quote_data.venue} {quote_data.instrument_id} {quote_data.bid_price}/{quote_data.ask_price}")
            
            # Call registered quote handlers
            for handler in self._quote_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(quote_data)
                    else:
                        handler(quote_data)
                except Exception as e:
                    self.logger.error(f"Error in quote handler: {e}")
                    
            # Broadcast to WebSocket clients
            if self._broadcast_callback:
                await self._broadcast_callback({
                    "type": "quote",
                    "data": asdict(quote_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling quote data: {e}")
            
    async def _handle_bar_data(self, data: NormalizedMarketData) -> None:
        """Handle OHLCV bar data events"""
        try:
            bar_data = BarData(
                venue=data.venue,
                instrument_id=data.instrument_id,
                timeframe=data.data.get("timeframe", "1m"),
                open_price=float(data.data.get("open", 0)),
                high_price=float(data.data.get("high", 0)),
                low_price=float(data.data.get("low", 0)),
                close_price=float(data.data.get("close", 0)),
                volume=float(data.data.get("volume", 0)),
                timestamp=data.timestamp,
                is_closed=data.data.get("is_closed", True)
            )
            
            self.logger.debug(f"Processing bar: {bar_data.venue} {bar_data.instrument_id} {bar_data.timeframe} OHLCV: {bar_data.open_price}/{bar_data.high_price}/{bar_data.low_price}/{bar_data.close_price}/{bar_data.volume}")
            
            # Call registered bar handlers
            for handler in self._bar_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(bar_data)
                    else:
                        handler(bar_data)
                except Exception as e:
                    self.logger.error(f"Error in bar handler: {e}")
                    
            # Broadcast to WebSocket clients
            if self._broadcast_callback:
                await self._broadcast_callback({
                    "type": "bar",
                    "data": asdict(bar_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling bar data: {e}")
            
    async def _handle_orderbook_data(self, data: NormalizedMarketData) -> None:
        """Handle order book data events"""
        try:
            orderbook_data = OrderBookData(
                venue=data.venue,
                instrument_id=data.instrument_id,
                bids=data.data.get("bids", []),
                asks=data.data.get("asks", []),
                timestamp=data.timestamp,
                sequence=data.data.get("sequence")
            )
            
            self.logger.debug(f"Processing order book: {orderbook_data.venue} {orderbook_data.instrument_id} {len(orderbook_data.bids)} bids, {len(orderbook_data.asks)} asks")
            
            # Call registered order book handlers
            for handler in self._orderbook_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(orderbook_data)
                    else:
                        handler(orderbook_data)
                except Exception as e:
                    self.logger.error(f"Error in order book handler: {e}")
                    
            # Broadcast to WebSocket clients (limited depth for performance)
            if self._broadcast_callback:
                # Limit to top 10 levels for WebSocket broadcast
                limited_data = OrderBookData(
                    venue=orderbook_data.venue,
                    instrument_id=orderbook_data.instrument_id,
                    bids=orderbook_data.bids[:10],
                    asks=orderbook_data.asks[:10],
                    timestamp=orderbook_data.timestamp,
                    sequence=orderbook_data.sequence
                )
                
                await self._broadcast_callback({
                    "type": "orderbook",
                    "data": asdict(limited_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling order book data: {e}")
            
    async def _handle_trade_data(self, data: NormalizedMarketData) -> None:
        """Handle trade data events (similar to tick data but from trade feeds)"""
        # Convert trade data to tick format for consistent handling
        trade_as_tick = NormalizedMarketData(
            venue=data.venue,
            instrument_id=data.instrument_id,
            data_type=DataType.TICK.value,
            timestamp=data.timestamp,
            data={
                "price": data.data.get("price"),
                "quantity": data.data.get("size", data.data.get("quantity")),
                "side": data.data.get("side", "unknown"),
                "trade_id": data.data.get("trade_id", data.data.get("id"))
            },
            raw_data=data.raw_data
        )
        
        await self._handle_tick_data(trade_as_tick)
        
    async def _handle_normalized_tick(self, tick: NormalizedTick) -> None:
        """Handle normalized tick data"""
        try:
            # Convert to legacy format for existing handlers
            tick_data = TickData(
                venue=tick.venue,
                instrument_id=tick.instrument_id,
                price=float(tick.price),
                quantity=float(tick.size),
                timestamp=tick.timestamp_ns // 1000000,  # Convert to milliseconds
                side=tick.side or "unknown",
                trade_id=tick.trade_id
            )
            
            self.logger.debug(f"Processing normalized tick: {tick.venue} {tick.instrument_id} @ {tick.price}")
            
            # Call registered tick handlers
            for handler in self._tick_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(tick_data)
                    else:
                        handler(tick_data)
                except Exception as e:
                    self.logger.error(f"Error in tick handler: {e}")
                    
            # Broadcast to WebSocket clients
            if self._broadcast_callback:
                await self._broadcast_callback({
                    "type": "tick",
                    "data": asdict(tick_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling normalized tick: {e}")
            
    async def _handle_normalized_quote(self, quote: NormalizedQuote) -> None:
        """Handle normalized quote data"""
        try:
            # Convert to legacy format
            quote_data = QuoteData(
                venue=quote.venue,
                instrument_id=quote.instrument_id,
                bid_price=float(quote.bid_price),
                ask_price=float(quote.ask_price),
                bid_size=float(quote.bid_size),
                ask_size=float(quote.ask_size),
                timestamp=quote.timestamp_ns // 1000000,  # Convert to milliseconds
                spread=float(quote.ask_price - quote.bid_price)
            )
            
            self.logger.debug(f"Processing normalized quote: {quote.venue} {quote.instrument_id} {quote.bid_price}/{quote.ask_price}")
            
            # Call registered quote handlers
            for handler in self._quote_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(quote_data)
                    else:
                        handler(quote_data)
                except Exception as e:
                    self.logger.error(f"Error in quote handler: {e}")
                    
            # Broadcast to WebSocket clients
            if self._broadcast_callback:
                await self._broadcast_callback({
                    "type": "quote",
                    "data": asdict(quote_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling normalized quote: {e}")
            
    async def _handle_normalized_bar(self, bar: NormalizedBar) -> None:
        """Handle normalized bar data"""
        try:
            # Convert to legacy format
            bar_data = BarData(
                venue=bar.venue,
                instrument_id=bar.instrument_id,
                timeframe=bar.timeframe,
                open_price=float(bar.open_price),
                high_price=float(bar.high_price),
                low_price=float(bar.low_price),
                close_price=float(bar.close_price),
                volume=float(bar.volume),
                timestamp=bar.timestamp_ns // 1000000,  # Convert to milliseconds
                is_closed=bar.is_final
            )
            
            self.logger.debug(f"Processing normalized bar: {bar.venue} {bar.instrument_id} {bar.timeframe} OHLCV: {bar.open_price}/{bar.high_price}/{bar.low_price}/{bar.close_price}/{bar.volume}")
            
            # Call registered bar handlers
            for handler in self._bar_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(bar_data)
                    else:
                        handler(bar_data)
                except Exception as e:
                    self.logger.error(f"Error in bar handler: {e}")
                    
            # Broadcast to WebSocket clients
            if self._broadcast_callback:
                await self._broadcast_callback({
                    "type": "bar",
                    "data": asdict(bar_data)
                })
                
        except Exception as e:
            self.logger.error(f"Error handling normalized bar: {e}")


# Global handlers instance
market_data_handlers = MarketDataHandlers()
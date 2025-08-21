"""
Market Data Streaming Service
Provides market data subscription and processing capabilities for Nautilus Trader integration.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Tuple
from dataclasses import dataclass

from messagebus_client import MessageBusMessage, messagebus_client
from rate_limiter import rate_limiter
from enums import Venue, DataType, MessageBusTopics


@dataclass
class MarketDataSubscription:
    """Market data subscription configuration"""
    venue: Venue
    instrument_id: str
    data_type: DataType
    active: bool = True
    subscription_id: str | None = None


@dataclass
class NormalizedMarketData:
    """Normalized market data structure"""
    venue: str
    instrument_id: str
    data_type: str
    timestamp: int
    data: dict[str, Any]
    raw_data: dict[str, Any]


class MarketDataService:
    """
    Market Data Streaming Service that subscribes to NautilusTrader MessageBus
    and processes market data events with normalization and multi-venue support.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._subscriptions: dict[str, MarketDataSubscription] = {}
        self._data_handlers: list[Callable[[NormalizedMarketData], None]] = []
        self._venue_parsers: dict[Venue, Callable[[dict[str, Any]], NormalizedMarketData]] = {}
        self._running = False
        self._setup_venue_parsers()
        
    def add_data_handler(self, handler: Callable[[NormalizedMarketData], None]) -> None:
        """Add a data handler callback"""
        self._data_handlers.append(handler)
        
    def remove_data_handler(self, handler: Callable[[NormalizedMarketData], None]) -> None:
        """Remove a data handler callback"""
        if handler in self._data_handlers:
            self._data_handlers.remove(handler)
    
    async def start(self) -> None:
        """Start the market data service"""
        if self._running:
            return
            
        self.logger.info("Starting Market Data Service...")
        self._running = True
        
        # Add message handler to MessageBus client
        messagebus_client.add_message_handler(self._handle_messagebus_message)
        
        # Subscribe to market data topics
        await self._setup_subscriptions()
        
        self.logger.info("Market Data Service started")
        
    async def stop(self) -> None:
        """Stop the market data service"""
        if not self._running:
            return
            
        self.logger.info("Stopping Market Data Service...")
        self._running = False
        
        # Remove message handler
        messagebus_client.remove_message_handler(self._handle_messagebus_message)
        
        # Clear subscriptions
        self._subscriptions.clear()
        
        self.logger.info("Market Data Service stopped")
        
    async def subscribe(self, venue: Venue, instrument_id: str, data_type: DataType) -> str:
        """
        Subscribe to market data for a specific venue, instrument, and data type
        
        Returns subscription ID
        """
        subscription_id = f"{venue.value}_{instrument_id}_{data_type.value}"
        
        subscription = MarketDataSubscription(
            venue=venue, instrument_id=instrument_id, data_type=data_type, subscription_id=subscription_id
        )
        
        self._subscriptions[subscription_id] = subscription
        
        self.logger.info(f"Subscribed to {venue.value} {instrument_id} {data_type.value}")
        return subscription_id
        
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from market data"""
        if subscription_id in self._subscriptions:
            subscription = self._subscriptions[subscription_id]
            subscription.active = False
            del self._subscriptions[subscription_id]
            
            self.logger.info(f"Unsubscribed from {subscription_id}")
            return True
            
        return False
        
    def get_active_subscriptions(self) -> list[MarketDataSubscription]:
        """Get list of active subscriptions"""
        return [sub for sub in self._subscriptions.values() if sub.active]
        
    async def _setup_subscriptions(self) -> None:
        """Setup default subscriptions for common market data"""
        # Subscribe to common data types from major venues
        default_subscriptions = [
            (Venue.BINANCE, "BTCUSDT", DataType.TICK), (Venue.BINANCE, "ETHUSDT", DataType.TICK), (Venue.COINBASE, "BTC-USD", DataType.QUOTE), (Venue.COINBASE, "ETH-USD", DataType.QUOTE), ]
        
        for venue, instrument, data_type in default_subscriptions:
            await self.subscribe(venue, instrument, data_type)
            
    async def _handle_messagebus_message(self, message: MessageBusMessage) -> None:
        """Handle incoming MessageBus messages with rate limiting"""
        try:
            # Check if message is market data related
            if not self._is_market_data_topic(message.topic):
                return
                
            # Parse venue and instrument from topic
            venue, instrument_id, data_type = self._parse_topic(message.topic)
            
            if not venue or not instrument_id or not data_type:
                return
                
            # Check if we have an active subscription
            subscription_id = f"{venue}_{instrument_id}_{data_type}"
            if subscription_id not in self._subscriptions:
                return
                
            venue_enum = Venue(venue)
            
            # Check rate limiting
            allowed, reason = await rate_limiter.should_allow_request(
                venue_enum, message_data={
                    "topic": message.topic, "payload": message.payload, "timestamp": message.timestamp
                }
            )
            
            if not allowed:
                self.logger.debug(f"Rate limited message from {venue}: {reason}")
                return
                
            # Parse venue-specific data
            if venue_enum in self._venue_parsers:
                try:
                    normalized_data = self._venue_parsers[venue_enum](message.payload)
                    normalized_data.venue = venue
                    normalized_data.instrument_id = instrument_id
                    normalized_data.data_type = data_type
                    normalized_data.timestamp = message.timestamp
                    
                    # Record success for circuit breaker
                    rate_limiter.record_success(venue_enum)
                    
                    # Call data handlers
                    for handler in self._data_handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(normalized_data)
                            else:
                                handler(normalized_data)
                        except Exception as e:
                            self.logger.error(f"Error in data handler: {e}")
                            rate_limiter.record_failure(venue_enum, e)
                            
                except Exception as e:
                    self.logger.error(f"Error parsing {venue} data: {e}")
                    rate_limiter.record_failure(venue_enum, e)
                        
        except Exception as e:
            self.logger.error(f"Error handling MessageBus message: {e}")
            
    def _is_market_data_topic(self, topic: str) -> bool:
        """Check if topic is market data related"""
        market_data_prefixes = [
            "data.quotes", "data.trades", "data.ticks", "data.bars", "data.orderbook", "data.instrument", "data.status"
        ]
        
        return any(topic.startswith(prefix) for prefix in market_data_prefixes)
        
    def _parse_topic(self, topic: str) -> tuple[str | None, str | None, str | None]:
        """Parse venue, instrument, and data type from topic"""
        try:
            # Expected format: data.{data_type}.{venue}.{instrument_id}
            parts = topic.split(".")
            if len(parts) >= 4:
                data_type = parts[1]
                venue = parts[2]
                instrument_id = ".".join(parts[3:])  # Handle instruments with dots
                return venue, instrument_id, data_type
        except Exception:
            pass
            
        return None, None, None
        
    def _setup_venue_parsers(self) -> None:
        """Setup venue-specific data parsers"""
        self._venue_parsers = {
            Venue.BINANCE: self._parse_binance_data, Venue.COINBASE: self._parse_coinbase_data, Venue.KRAKEN: self._parse_kraken_data, Venue.BYBIT: self._parse_bybit_data, Venue.BITMEX: self._parse_bitmex_data, Venue.OKX: self._parse_okx_data, Venue.HYPERLIQUID: self._parse_hyperliquid_data, Venue.DATABENTO: self._parse_databento_data, Venue.INTERACTIVE_BROKERS: self._parse_ib_data, Venue.DYDX: self._parse_dydx_data, Venue.POLYMARKET: self._parse_polymarket_data, Venue.BETFAIR: self._parse_betfair_data, }
        
    def _parse_binance_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Binance-specific data format"""
        return NormalizedMarketData(
            venue="", # Will be set by caller
            instrument_id="", # Will be set by caller  
            data_type="", # Will be set by caller
            timestamp=0, # Will be set by caller
            data={
                "price": payload.get("price"), "quantity": payload.get("quantity"), "bid": payload.get("bid"), "ask": payload.get("ask"), "volume": payload.get("volume"), }, raw_data=payload
        )
        
    def _parse_coinbase_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Coinbase-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "size": payload.get("size"), "bid": payload.get("best_bid"), "ask": payload.get("best_ask"), "volume": payload.get("volume_24h"), }, raw_data=payload
        )
        
    def _parse_kraken_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Kraken-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "volume": payload.get("volume"), "bid": payload.get("bid"), "ask": payload.get("ask"), }, raw_data=payload
        )
        
    def _parse_bybit_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Bybit-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "qty": payload.get("qty"), "bid1_price": payload.get("bid1Price"), "ask1_price": payload.get("ask1Price"), }, raw_data=payload
        )
        
    def _parse_bitmex_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse BitMEX-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "size": payload.get("size"), "bid": payload.get("bidPrice"), "ask": payload.get("askPrice"), }, raw_data=payload
        )
        
    def _parse_okx_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse OKX-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "px": payload.get("px"), # Price
                "sz": payload.get("sz"), # Size
                "bidPx": payload.get("bidPx"), "askPx": payload.get("askPx"), }, raw_data=payload
        )
        
    def _parse_hyperliquid_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Hyperliquid-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "size": payload.get("size"), "bid": payload.get("bid"), "ask": payload.get("ask"), }, raw_data=payload
        )
        
    def _parse_databento_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Databento-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "size": payload.get("size"), "bid_px": payload.get("bid_px"), "ask_px": payload.get("ask_px"), }, raw_data=payload
        )
        
    def _parse_ib_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Interactive Brokers-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "size": payload.get("size"), "bid": payload.get("bid"), "ask": payload.get("ask"), }, raw_data=payload
        )
        
    def _parse_dydx_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse dYdX-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "size": payload.get("size"), "bid": payload.get("bids"), "ask": payload.get("asks"), }, raw_data=payload
        )
        
    def _parse_polymarket_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Polymarket-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "price": payload.get("price"), "outcome": payload.get("outcome"), "bid": payload.get("bid"), "ask": payload.get("ask"), }, raw_data=payload
        )
        
    def _parse_betfair_data(self, payload: dict[str, Any]) -> NormalizedMarketData:
        """Parse Betfair-specific data format"""
        return NormalizedMarketData(
            venue="", instrument_id="", data_type="", timestamp=0, data={
                "back_price": payload.get("back_price"), "lay_price": payload.get("lay_price"), "back_size": payload.get("back_size"), "lay_size": payload.get("lay_size"), }, raw_data=payload
        )


# Global service instance
market_data_service = MarketDataService()
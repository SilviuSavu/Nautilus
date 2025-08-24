"""
MessageBus Client for Volatility Engine

Provides real-time market data streaming integration with the Nautilus MessageBus
for instant volatility model updates and forecasting.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict

# MessageBus integration
try:
    from enhanced_messagebus_client import (
        EnhancedMessageBusClient, 
        MessageBusConfig,
        MessagePriority
    )
    MESSAGEBUS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to basic messagebus
        from messagebus_client import MessageBusClient as EnhancedMessageBusClient
        from messagebus_client import MessagePriority
        MessageBusConfig = dict  # Fallback to dict
        MESSAGEBUS_AVAILABLE = True
    except ImportError:
        EnhancedMessageBusClient = None
        MessageBusConfig = None
        MessagePriority = None
        MESSAGEBUS_AVAILABLE = False

# Redis streams for high-frequency data
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MarketDataEvent:
    """Market data event for volatility calculations"""
    symbol: str
    timestamp: datetime
    data_type: str  # 'tick', 'bar', 'trade', 'quote'
    
    # OHLCV data
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    
    # Tick data
    price: Optional[float] = None
    size: Optional[float] = None
    
    # Additional metadata
    source: str = "unknown"
    sequence_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VolatilityMessageBusClient:
    """
    MessageBus client specifically designed for volatility forecasting.
    
    Handles real-time market data ingestion and triggers volatility model updates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MessageBus client for volatility processing"""
        self.config = config
        self.client: Optional[EnhancedMessageBusClient] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Event handlers
        self.data_handlers: Dict[str, List[Callable]] = {
            'market_data': [],
            'tick_data': [],
            'bar_data': [],
            'volatility_trigger': []
        }
        
        # Processing state
        self.active_symbols: set = set()
        self.data_buffer: Dict[str, List[MarketDataEvent]] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Performance tracking
        self.events_processed = 0
        self.volatility_updates_triggered = 0
        self.start_time = datetime.utcnow()
        
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        logger.info("Volatility MessageBus client initialized")
    
    async def initialize(self) -> None:
        """Initialize MessageBus and Redis connections"""
        if not MESSAGEBUS_AVAILABLE:
            logger.warning("MessageBus not available - volatility streaming disabled")
            return
        
        try:
            # Initialize MessageBus client
            if isinstance(MessageBusConfig, type):
                # New enhanced messagebus format
                messagebus_config = MessageBusConfig(
                    redis_host=self.config.get('redis_host', 'redis'),
                    redis_port=self.config.get('redis_port', 6379),
                    consumer_name="volatility-engine",
                    stream_key="nautilus-volatility-streams",
                    consumer_group="volatility-group",
                    buffer_interval_ms=50,  # Fast updates for volatility
                    max_buffer_size=10000,
                    heartbeat_interval_secs=15
                )
            else:
                # Fallback configuration
                messagebus_config = {
                    'redis_host': self.config.get('redis_host', 'redis'),
                    'redis_port': self.config.get('redis_port', 6379),
                    'consumer_name': 'volatility-engine',
                    'stream_key': 'nautilus-volatility-streams'
                }
            
            self.client = EnhancedMessageBusClient(messagebus_config)
            await self.client.initialize()
            
            # Initialize Redis for high-frequency data buffering
            if REDIS_AVAILABLE:
                redis_url = f"redis://{self.config.get('redis_host', 'redis')}:{self.config.get('redis_port', 6379)}"
                self.redis_client = redis.from_url(redis_url)
                await self.redis_client.ping()
                logger.info("✅ Redis connection established for volatility streaming")
            
            # Subscribe to market data streams
            await self._setup_subscriptions()
            
            logger.info("✅ Volatility MessageBus client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MessageBus client: {e}")
            raise
    
    async def _setup_subscriptions(self) -> None:
        """Setup MessageBus subscriptions for market data"""
        if not self.client:
            return
        
        # Subscribe to relevant market data topics
        subscriptions = [
            "market_data.bars.*",      # OHLCV bar data
            "market_data.ticks.*",     # Tick data
            "market_data.trades.*",    # Trade data
            "market_data.quotes.*",    # Quote data
            "ibkr.market_data.*",      # IBKR specific data
            "alpha_vantage.updates.*", # Alpha Vantage data
            "volatility.triggers.*"    # Volatility-specific triggers
        ]
        
        for topic in subscriptions:
            await self.client.subscribe(topic, self._handle_market_data_message)
            logger.info(f"📡 Subscribed to: {topic}")
    
    async def _handle_market_data_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming market data messages"""
        try:
            # Parse the message
            topic = message.get('topic', '')
            payload = message.get('payload', {})
            timestamp = datetime.fromisoformat(message.get('timestamp', datetime.utcnow().isoformat()))
            
            # Extract symbol and data type
            symbol = payload.get('symbol', '').upper()
            if not symbol:
                return
            
            # Create market data event
            event = MarketDataEvent(
                symbol=symbol,
                timestamp=timestamp,
                data_type=self._determine_data_type(topic),
                open=payload.get('open'),
                high=payload.get('high'), 
                low=payload.get('low'),
                close=payload.get('close'),
                volume=payload.get('volume'),
                price=payload.get('price'),
                size=payload.get('size'),
                source=payload.get('source', 'messagebus'),
                sequence_number=payload.get('sequence', self.events_processed)
            )
            
            # Process the event
            await self._process_market_data_event(event)
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing market data message: {e}")
    
    def _determine_data_type(self, topic: str) -> str:
        """Determine data type from message topic"""
        if 'bars' in topic or 'ohlc' in topic:
            return 'bar'
        elif 'ticks' in topic:
            return 'tick'
        elif 'trades' in topic:
            return 'trade'
        elif 'quotes' in topic:
            return 'quote'
        else:
            return 'unknown'
    
    async def _process_market_data_event(self, event: MarketDataEvent) -> None:
        """Process market data event for volatility calculations"""
        symbol = event.symbol
        
        # Only process active symbols
        if symbol not in self.active_symbols:
            return
        
        # Buffer the event
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(event)
        
        # Maintain buffer size (keep last 1000 events per symbol)
        if len(self.data_buffer[symbol]) > 1000:
            self.data_buffer[symbol] = self.data_buffer[symbol][-1000:]
        
        # Update last update time
        self.last_update[symbol] = event.timestamp
        
        # Trigger volatility update if conditions are met
        await self._check_volatility_trigger(symbol, event)
        
        # Call registered data handlers
        for handler in self.data_handlers['market_data']:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in market data handler: {e}")
    
    async def _check_volatility_trigger(self, symbol: str, event: MarketDataEvent) -> None:
        """Check if volatility update should be triggered"""
        try:
            # Get recent events for this symbol
            recent_events = self.data_buffer.get(symbol, [])
            if len(recent_events) < 2:
                return
            
            # Check for significant price movement
            current_price = event.close or event.price
            if current_price is None:
                return
            
            # Look at recent prices
            recent_prices = [
                e.close or e.price for e in recent_events[-10:]
                if (e.close is not None or e.price is not None)
            ]
            
            if len(recent_prices) < 5:
                return
            
            # Calculate short-term return
            if len(recent_prices) >= 2:
                recent_return = abs(recent_prices[-1] / recent_prices[-2] - 1)
                
                # Trigger if return exceeds threshold (e.g., 1% move)
                if recent_return > 0.01:  # 1% threshold
                    await self._trigger_volatility_update(symbol, event, recent_return)
            
            # Time-based triggers (every 5 minutes for active symbols)
            last_trigger = self.last_update.get(f"{symbol}_volatility_trigger", datetime.min)
            if (event.timestamp - last_trigger).total_seconds() > 300:  # 5 minutes
                await self._trigger_volatility_update(symbol, event, None, "time_based")
        
        except Exception as e:
            logger.error(f"Error checking volatility trigger for {symbol}: {e}")
    
    async def _trigger_volatility_update(self, symbol: str, event: MarketDataEvent, 
                                       return_magnitude: Optional[float] = None,
                                       trigger_type: str = "price_movement") -> None:
        """Trigger volatility model update"""
        try:
            # Create volatility trigger event
            trigger_event = {
                'event_type': 'volatility_update_trigger',
                'symbol': symbol,
                'timestamp': event.timestamp.isoformat(),
                'trigger_type': trigger_type,
                'current_price': event.close or event.price,
                'return_magnitude': return_magnitude,
                'data_source': event.source,
                'buffer_size': len(self.data_buffer.get(symbol, [])),
                'sequence': self.volatility_updates_triggered
            }
            
            # Publish trigger event
            if self.client:
                await self.client.publish(
                    f"volatility.updates.{symbol.lower()}",
                    trigger_event,
                    priority=MessagePriority.HIGH if return_magnitude and return_magnitude > 0.02 else MessagePriority.NORMAL
                )
            
            # Call registered volatility handlers
            for handler in self.data_handlers['volatility_trigger']:
                try:
                    await handler(trigger_event)
                except Exception as e:
                    logger.error(f"Error in volatility trigger handler: {e}")
            
            # Update trigger tracking
            self.volatility_updates_triggered += 1
            self.last_update[f"{symbol}_volatility_trigger"] = event.timestamp
            
            logger.debug(f"🔄 Triggered volatility update for {symbol} ({trigger_type})")
            
        except Exception as e:
            logger.error(f"Error triggering volatility update for {symbol}: {e}")
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to active volatility tracking"""
        symbol = symbol.upper()
        self.active_symbols.add(symbol)
        logger.info(f"📈 Added {symbol} to volatility tracking")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from active volatility tracking"""
        symbol = symbol.upper()
        self.active_symbols.discard(symbol)
        if symbol in self.data_buffer:
            del self.data_buffer[symbol]
        if symbol in self.last_update:
            del self.last_update[symbol]
        logger.info(f"📉 Removed {symbol} from volatility tracking")
    
    def register_data_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler for specific data types"""
        if event_type not in self.data_handlers:
            self.data_handlers[event_type] = []
        self.data_handlers[event_type].append(handler)
        logger.info(f"🔗 Registered handler for {event_type}")
    
    async def get_recent_data(self, symbol: str, limit: int = 100) -> List[MarketDataEvent]:
        """Get recent market data for a symbol"""
        symbol = symbol.upper()
        recent_data = self.data_buffer.get(symbol, [])
        return recent_data[-limit:] if recent_data else []
    
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'messagebus_connected': self.client is not None and self.is_running,
            'redis_connected': self.redis_client is not None,
            'active_symbols': list(self.active_symbols),
            'total_symbols': len(self.active_symbols),
            'events_processed': self.events_processed,
            'volatility_updates_triggered': self.volatility_updates_triggered,
            'uptime_seconds': uptime,
            'events_per_second': self.events_processed / max(1, uptime),
            'buffer_sizes': {symbol: len(events) for symbol, events in self.data_buffer.items()},
            'last_updates': {symbol: timestamp.isoformat() for symbol, timestamp in self.last_update.items()}
        }
    
    async def start_streaming(self) -> None:
        """Start the streaming data processing"""
        if not self.client:
            logger.warning("MessageBus client not initialized")
            return
        
        try:
            self.is_running = True
            logger.info("🚀 Started volatility streaming")
            
            # Start the client
            if hasattr(self.client, 'start'):
                await self.client.start()
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.is_running = False
            raise
    
    async def stop_streaming(self) -> None:
        """Stop the streaming data processing"""
        try:
            self.is_running = False
            
            if self.client:
                if hasattr(self.client, 'stop'):
                    await self.client.stop()
                await self.client.cleanup()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("🛑 Stopped volatility streaming")
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")


# Factory function
def create_volatility_messagebus_client(config: Dict[str, Any]) -> Optional[VolatilityMessageBusClient]:
    """Create a volatility MessageBus client if available"""
    if not MESSAGEBUS_AVAILABLE:
        logger.warning("MessageBus not available - real-time volatility updates disabled")
        return None
    
    return VolatilityMessageBusClient(config)
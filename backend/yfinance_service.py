"""
YFinance Data Service - NautilusTrader Integration
Provides YFinance market data integration for the web dashboard using NautilusTrader's professional YFinance adapter.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Use installed nautilus_trader package (no path modification needed)

from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.data import YFinanceDataClient
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
from nautilus_trader.adapters.yfinance.factories import create_yfinance_data_client, create_yfinance_instrument_provider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import RequestBars
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import BarSpecification


@dataclass
class YFinanceStatus:
    """YFinance service status"""
    connected: bool = False
    initialized: bool = False
    instruments_loaded: int = 0
    last_request_time: Optional[datetime] = None
    error_message: Optional[str] = None
    rate_limit_delay: float = 0.1
    cache_expiry_seconds: int = 3600


@dataclass  
class YFinanceHistoricalData:
    """YFinance historical data response"""
    symbol: str
    timeframe: str
    bars: List[Dict[str, Any]]
    total_bars: int
    start_date: Optional[str]
    end_date: Optional[str]
    data_source: str = "YFinance"


class YFinanceService:
    """
    YFinance data service using NautilusTrader's professional adapter.
    
    Provides market data from Yahoo Finance with proper rate limiting,
    caching, and error handling for web dashboard integration.
    """
    
    def __init__(self, messagebus_client=None):
        self.logger = logging.getLogger(__name__)
        self.messagebus_client = messagebus_client
        
        # NautilusTrader components
        self.loop = asyncio.get_event_loop()
        self.cache = Cache()
        self.clock = LiveClock()
        self.msgbus = MessageBus(trader_id="YFINANCE-SERVICE", clock=self.clock)
        
        # YFinance components
        self.data_client: Optional[YFinanceDataClient] = None
        self.instrument_provider: Optional[YFinanceInstrumentProvider] = None
        self.config: Optional[YFinanceDataClientConfig] = None
        
        # Service status
        self.status = YFinanceStatus()
        
        # Callbacks
        self.data_callbacks: List = []
        
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the YFinance service with NautilusTrader adapter.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for the service.
            
        Returns
        -------
        bool
            True if initialization successful, False otherwise.
        """
        try:
            self.logger.info("🚀 Initializing YFinance service with NautilusTrader adapter...")
            
            # Create configuration
            config_params = config or {}
            self.config = YFinanceDataClientConfig(
                cache_expiry_seconds=config_params.get('cache_expiry_seconds', 3600),
                rate_limit_delay=config_params.get('rate_limit_delay', 0.1),
                request_timeout=config_params.get('request_timeout', 30.0),
                max_retries=config_params.get('max_retries', 3),
                retry_delay=config_params.get('retry_delay', 1.0),
                default_period=config_params.get('default_period', '1y'),
                default_interval=config_params.get('default_interval', '1d'),
                symbols=config_params.get('symbols', ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN'])
            )
            
            # Create instrument provider
            self.instrument_provider = create_yfinance_instrument_provider()
            
            # Create data client
            self.data_client = create_yfinance_data_client(
                loop=self.loop,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
                config=self.config,
                name="YFINANCE"
            )
            
            # Connect the data client
            await self.data_client.connect()
            
            # Update status
            self.status.initialized = True
            self.status.connected = True
            self.status.rate_limit_delay = self.config.rate_limit_delay
            self.status.cache_expiry_seconds = self.config.cache_expiry_seconds
            
            self.logger.info("✅ YFinance service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YFinance service: {e}")
            self.status.error_message = str(e)
            self.status.initialized = False
            self.status.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect the YFinance service"""
        try:
            if self.data_client:
                await self.data_client.disconnect()
            
            self.status.connected = False
            self.logger.info("🔌 YFinance service disconnected")
            
        except Exception as e:
            self.logger.error(f"❌ Error disconnecting YFinance service: {e}")
    
    def is_connected(self) -> bool:
        """Check if the service is connected and ready"""
        return self.status.connected and self.status.initialized
    
    def get_status(self) -> YFinanceStatus:
        """Get current service status"""
        return self.status
    
    async def load_instrument(self, symbol: str) -> bool:
        """
        Load a single instrument definition.
        
        Parameters
        ----------
        symbol : str
            The Yahoo Finance symbol to load.
            
        Returns
        -------
        bool
            True if loaded successfully, False otherwise.
        """
        if not self.is_connected():
            self.logger.error("YFinance service not connected")
            return False
        
        try:
            # Create instrument ID
            instrument_id = InstrumentId(Symbol(symbol), Venue("YAHOO"))
            
            # Load instrument
            await self.instrument_provider.load_async(instrument_id)
            
            # Check if loaded
            instrument = self.cache.instrument(instrument_id)
            if instrument:
                self.status.instruments_loaded += 1
                self.logger.info(f"✅ Loaded instrument: {symbol}")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to load instrument: {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading instrument {symbol}: {e}")
            return False
    
    async def load_instruments(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Load multiple instrument definitions.
        
        Parameters
        ----------
        symbols : list of str
            The Yahoo Finance symbols to load.
            
        Returns
        -------
        dict
            Dictionary with symbol: success status.
        """
        if not self.is_connected():
            return {symbol: False for symbol in symbols}
        
        results = {}
        for symbol in symbols:
            results[symbol] = await self.load_instrument(symbol)
        
        return results
    
    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        period: str = "1y",
        limit: Optional[int] = None
    ) -> Optional[YFinanceHistoricalData]:
        """
        Get historical OHLCV bars for a symbol.
        
        Parameters
        ----------
        symbol : str
            The Yahoo Finance symbol.
        timeframe : str, default "1d"
            The bar timeframe (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo).
        period : str, default "1y"
            The data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
        limit : int, optional
            Maximum number of bars to return.
            
        Returns
        -------
        YFinanceHistoricalData or None
            The historical data or None if failed.
        """
        if not self.is_connected():
            self.logger.error("YFinance service not connected")
            return None
        
        try:
            self.logger.info(f"📊 Requesting {timeframe} bars for {symbol} (period: {period})")
            
            # Update last request time
            self.status.last_request_time = datetime.now()
            
            # Ensure instrument is loaded
            instrument_id = InstrumentId(Symbol(symbol), Venue("YAHOO"))
            instrument = self.cache.instrument(instrument_id)
            if not instrument:
                success = await self.load_instrument(symbol)
                if not success:
                    self.logger.error(f"Failed to load instrument {symbol}")
                    return None
                instrument = self.cache.instrument(instrument_id)
            
            # Convert timeframe to Nautilus BarType
            bar_type = self._create_bar_type(symbol, timeframe)
            if not bar_type:
                self.logger.error(f"Unsupported timeframe: {timeframe}")
                return None
            
            # Create request
            correlation_id = UUID4()
            request = RequestBars(
                client_id=self.data_client.id,
                venue=Venue("YAHOO"),
                bar_type=bar_type,
                limit=limit or 1000,
                correlation_id=correlation_id,
                ts_init=self.clock.timestamp_ns()
            )
            
            # Execute request - this will trigger the data client
            bars_data = []
            
            # Set up a callback to capture the bars
            original_handle_data = self.data_client._handle_data
            captured_bars = []
            
            def capture_bars(data, corr_id):
                if corr_id == correlation_id:
                    captured_bars.append(data)
                return original_handle_data(data, corr_id)
            
            self.data_client._handle_data = capture_bars
            
            try:
                # Send the request
                await self.data_client._request_bars(request)
                
                # Wait a moment for async processing
                await asyncio.sleep(0.1)
                
                # Convert captured bars to API format
                bars_list = []
                for bar in captured_bars:
                    if hasattr(bar, 'open'):  # It's a Bar object
                        bars_list.append({
                            "time": datetime.fromtimestamp(bar.ts_event / 1_000_000_000).isoformat() + "Z",
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume)
                        })
                
                # Sort by time
                bars_list.sort(key=lambda x: x['time'])
                
                return YFinanceHistoricalData(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars_list,
                    total_bars=len(bars_list),
                    start_date=bars_list[0]['time'] if bars_list else None,
                    end_date=bars_list[-1]['time'] if bars_list else None,
                    data_source="YFinance"
                )
                
            finally:
                # Restore original handler
                self.data_client._handle_data = original_handle_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting historical bars for {symbol}: {e}")
            self.status.error_message = str(e)
            return None
    
    def _create_bar_type(self, symbol: str, timeframe: str) -> Optional[BarType]:
        """
        Create Nautilus BarType from symbol and timeframe.
        
        Parameters
        ----------
        symbol : str
            The symbol.
        timeframe : str
            The timeframe string.
            
        Returns
        -------
        BarType or None
            The BarType or None if unsupported.
        """
        try:
            # Parse timeframe
            if timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                aggregation = BarAggregation.MINUTE
                step = minutes
            elif timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                aggregation = BarAggregation.HOUR
                step = hours
            elif timeframe == '1d':
                aggregation = BarAggregation.DAY
                step = 1
            elif timeframe == '1wk':
                aggregation = BarAggregation.WEEK
                step = 1
            elif timeframe == '1mo':
                aggregation = BarAggregation.MONTH
                step = 1
            else:
                return None
            
            # Create bar specification
            bar_spec = BarSpecification(
                step=step,
                aggregation=aggregation,
                price_type=PriceType.LAST
            )
            
            # Create bar type
            instrument_id = InstrumentId(Symbol(symbol), Venue("YAHOO"))
            return BarType(instrument_id, bar_spec)
            
        except Exception as e:
            self.logger.error(f"Error creating bar type for {symbol} {timeframe}: {e}")
            return None
    
    def add_data_callback(self, callback):
        """Add callback for data updates"""
        self.data_callbacks.append(callback)
    
    async def health_check(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "service": "YFinance",
            "status": "operational" if self.is_connected() else "disconnected",
            "initialized": self.status.initialized,
            "connected": self.status.connected,
            "instruments_loaded": self.status.instruments_loaded,
            "last_request": self.status.last_request_time.isoformat() if self.status.last_request_time else None,
            "rate_limit_delay": self.status.rate_limit_delay,
            "cache_expiry_seconds": self.status.cache_expiry_seconds,
            "error_message": self.status.error_message,
            "adapter_version": "NautilusTrader Professional"
        }


# Global service instance
_yfinance_service: Optional[YFinanceService] = None

def get_yfinance_service() -> YFinanceService:
    """Get or create the YFinance service singleton"""
    global _yfinance_service
    
    if _yfinance_service is None:
        _yfinance_service = YFinanceService()
    
    return _yfinance_service

def reset_yfinance_service():
    """Reset the service singleton (for testing)"""
    global _yfinance_service
    if _yfinance_service:
        asyncio.create_task(_yfinance_service.disconnect())
    _yfinance_service = None
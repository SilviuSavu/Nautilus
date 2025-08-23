# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any

from nautilus_trader.adapters.alpha_vantage.config import AlphaVantageDataClientConfig
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageBar
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageQuote
from nautilus_trader.adapters.alpha_vantage.data import AlphaVantageCompanyData
from nautilus_trader.adapters.alpha_vantage.http import AlphaVantageHttpClient
from nautilus_trader.adapters.alpha_vantage.parsing import parse_alpha_vantage_bars
from nautilus_trader.adapters.alpha_vantage.parsing import parse_alpha_vantage_company_data
from nautilus_trader.adapters.alpha_vantage.parsing import parse_alpha_vantage_quote
from nautilus_trader.adapters.alpha_vantage.providers import AlphaVantageInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import DataRequest
from nautilus_trader.data.messages import DataResponse
from nautilus_trader.data.messages import Subscribe
from nautilus_trader.data.messages import Unsubscribe
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId


class AlphaVantageDataClient(LiveMarketDataClient):
    """
    Provides a data client for Alpha Vantage market data API.

    This client provides access to real-time quotes, historical data, and company
    fundamentals through Alpha Vantage API. It implements intelligent caching,
    rate limiting, and MessageBus integration for real-time data distribution.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop for the client.
    msgbus : MessageBus
        The message bus for the client.
    cache : Cache
        The cache for the client.
    clock : LiveClock
        The clock for the client.
    instrument_provider : AlphaVantageInstrumentProvider
        The instrument provider for the client.
    config : AlphaVantageDataClientConfig, optional
        The configuration for the client.
    name : str, optional
        The custom client ID.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        instrument_provider: AlphaVantageInstrumentProvider,
        config: AlphaVantageDataClientConfig | None = None,
        name: str | None = None,
    ) -> None:
        if config is None:
            config = AlphaVantageDataClientConfig()

        PyCondition.type(config, AlphaVantageDataClientConfig, "config")

        super().__init__(
            loop=loop,
            client_id=ClientId(name or "ALPHA_VANTAGE"),
            venue=None,  # Multi-venue support for market data
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
        )

        # Configuration
        self._config = config

        # HTTP client for Alpha Vantage API  
        self._http_client = AlphaVantageHttpClient(
            api_key=config.api_key,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
            rate_limit_delay=config.rate_limit_delay,
            logger=self._log,
        )

        # Subscriptions and data management
        self._subscriptions: dict[DataType, set[InstrumentId]] = defaultdict(set)
        self._update_tasks: dict[InstrumentId, asyncio.Task] = {}
        self._last_data_timestamps: dict[str, int] = {}
        
        # Enhanced market data capabilities
        self._company_cache: dict[str, AlphaVantageCompanyData] = {}
        self._fundamentals_cache: dict[str, dict[str, Any]] = {}
        
        # Common symbols for market monitoring
        self._major_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
            "NVDA", "META", "SPY", "QQQ", "IWM"
        ]

    # -- CONNECTION MANAGEMENT ---------------------------------------------------------------

    async def _connect(self) -> None:
        """Connect to the Alpha Vantage API."""
        await self._http_client.connect()

        # Load configured instruments
        if self._config.instrument_ids:
            await self._instrument_provider.load_ids_async(self._config.instrument_ids)
            
        if self._config.symbols:
            from nautilus_trader.model.identifiers import Symbol, Venue
            instrument_ids = [
                InstrumentId(Symbol(symbol), Venue("ALPHA_VANTAGE"))
                for symbol in self._config.symbols
            ]
            await self._instrument_provider.load_ids_async(instrument_ids)

        # Auto-subscribe if configured
        if self._config.auto_subscribe and self._config.instrument_ids:
            for instrument_id in self._config.instrument_ids:
                # Subscribe to quotes by default
                quote_data_type = DataType(AlphaVantageQuote, metadata={"instrument_id": instrument_id})
                await self._subscribe(quote_data_type)

        self._log.info("Connected to Alpha Vantage API", LogColor.GREEN)

    async def _disconnect(self) -> None:
        """Disconnect from the Alpha Vantage API."""
        # Cancel all update tasks
        for task in self._update_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._update_tasks:
            await asyncio.gather(*self._update_tasks.values(), return_exceptions=True)
        
        self._update_tasks.clear()
        self._subscriptions.clear()

        await self._http_client.disconnect()
        self._log.info("Disconnected from Alpha Vantage API", LogColor.GREEN)

    # -- SUBSCRIPTIONS ------------------------------------------------------------------------

    async def _subscribe(self, data_type: DataType) -> None:
        """
        Subscribe to Alpha Vantage data updates.

        Parameters
        ----------
        data_type : DataType
            The data type to subscribe to.
        """
        supported_types = (AlphaVantageQuote, AlphaVantageBar, AlphaVantageCompanyData)
        if data_type.type not in supported_types:
            self._log.error(f"Cannot subscribe to {data_type.type}, supported types: {supported_types}")
            return

        # Extract instrument ID from metadata
        metadata = data_type.metadata or {}
        instrument_id = metadata.get("instrument_id")
        
        if not instrument_id:
            self._log.error("instrument_id required in DataType metadata for Alpha Vantage subscriptions")
            return

        if not isinstance(instrument_id, InstrumentId):
            self._log.error(f"instrument_id must be InstrumentId, got {type(instrument_id)}")
            return

        # Add to subscriptions
        self._subscriptions[data_type].add(instrument_id)
        
        # Start update task if auto-subscribe enabled
        if self._config.auto_subscribe:
            await self._start_update_task(instrument_id, data_type)

        self._log.info(f"Subscribed to Alpha Vantage data: {data_type.type.__name__} for {instrument_id}")

    async def _unsubscribe(self, data_type: DataType) -> None:
        """
        Unsubscribe from Alpha Vantage data updates.

        Parameters
        ----------
        data_type : DataType
            The data type to unsubscribe from.
        """
        metadata = data_type.metadata or {}
        instrument_id = metadata.get("instrument_id")
        
        if instrument_id and instrument_id in self._subscriptions[data_type]:
            self._subscriptions[data_type].remove(instrument_id)
            
            # Stop update task
            if instrument_id in self._update_tasks:
                task = self._update_tasks[instrument_id]
                if not task.done():
                    task.cancel()
                del self._update_tasks[instrument_id]

        self._log.info(f"Unsubscribed from Alpha Vantage data: {data_type.type.__name__} for {instrument_id}")

    # -- REQUESTS -----------------------------------------------------------------------------

    async def _request(self, data_type: DataType, correlation_id: UUID4) -> None:
        """
        Handle a data request for Alpha Vantage data.

        Parameters
        ----------
        data_type : DataType
            The data type being requested.
        correlation_id : UUID4
            The correlation ID for the request.
        """
        supported_types = (AlphaVantageQuote, AlphaVantageBar, AlphaVantageCompanyData)
        if data_type.type not in supported_types:
            self._log.error(f"Cannot request {data_type.type}, supported types: {supported_types}")
            return

        metadata = data_type.metadata or {}
        instrument_id = metadata.get("instrument_id")
        
        if not instrument_id:
            self._log.error("instrument_id required in DataType metadata for Alpha Vantage requests")
            return

        try:
            symbol = instrument_id.symbol.value
            
            # Handle different data types
            if data_type.type == AlphaVantageQuote:
                await self._handle_quote_request(symbol, instrument_id, data_type, correlation_id)
            elif data_type.type == AlphaVantageBar:
                await self._handle_bars_request(symbol, instrument_id, metadata, data_type, correlation_id)
            elif data_type.type == AlphaVantageCompanyData:
                await self._handle_company_request(symbol, instrument_id, data_type, correlation_id)

        except Exception as e:
            self._log.error(f"Failed to handle Alpha Vantage data request for {instrument_id}: {e}")

    async def _handle_quote_request(
        self,
        symbol: str,
        instrument_id: InstrumentId,
        data_type: DataType,
        correlation_id: UUID4,
    ) -> None:
        """Handle real-time quote request."""
        quote_data = await self._http_client.get_quote(symbol)
        if not quote_data:
            return

        quote = parse_alpha_vantage_quote(
            symbol=symbol,
            quote_data=quote_data,
            ts_init=self._clock.timestamp_ns(),
        )

        if quote:
            # Send data response
            response = DataResponse(
                client_id=self.id,
                venue=instrument_id.venue,
                data_type=data_type,
                data=[quote],
                correlation_id=correlation_id,
                response_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
            )
            
            self._handle_data_response(response)
            
            # **CRITICAL: MessageBus Integration**
            # Publish the quote to MessageBus for real-time distribution
            self._handle_data(quote)
            
            self._log.info(f"Provided Alpha Vantage quote for {symbol}", LogColor.GREEN)

    async def _handle_bars_request(
        self,
        symbol: str,
        instrument_id: InstrumentId,
        metadata: dict[str, Any],
        data_type: DataType,
        correlation_id: UUID4,
    ) -> None:
        """Handle historical bars request."""
        interval = metadata.get("interval", "daily")
        adjusted = metadata.get("adjusted", True)
        
        if interval == "daily":
            bars_data = await self._http_client.get_daily_prices(symbol, adjusted)
        else:
            # Intraday data
            bars_data = await self._http_client.get_intraday_prices(
                symbol=symbol,
                interval=interval,
                adjusted=adjusted,
            )
            
        if not bars_data:
            return

        bars = parse_alpha_vantage_bars(
            symbol=symbol,
            time_series_data=bars_data,
            adjusted=adjusted,
            ts_init=self._clock.timestamp_ns(),
        )

        if bars:
            # Send data response
            response = DataResponse(
                client_id=self.id,
                venue=instrument_id.venue,
                data_type=data_type,
                data=bars,
                correlation_id=correlation_id,
                response_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
            )
            
            self._handle_data_response(response)
            
            # **CRITICAL: MessageBus Integration**  
            # Publish bars to MessageBus for real-time distribution
            for bar in bars:
                self._handle_data(bar)
            
            self._log.info(f"Provided {len(bars)} Alpha Vantage bars for {symbol}", LogColor.GREEN)

    async def _handle_company_request(
        self,
        symbol: str,
        instrument_id: InstrumentId,
        data_type: DataType,
        correlation_id: UUID4,
    ) -> None:
        """Handle company fundamentals request."""
        company_data = await self._http_client.get_company_overview(symbol)
        if not company_data:
            return

        company = parse_alpha_vantage_company_data(
            symbol=symbol,
            overview_data=company_data,
            ts_init=self._clock.timestamp_ns(),
        )

        if company:
            # Send data response
            response = DataResponse(
                client_id=self.id,
                venue=instrument_id.venue,
                data_type=data_type,
                data=[company],
                correlation_id=correlation_id,
                response_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
            )
            
            self._handle_data_response(response)
            
            # **CRITICAL: MessageBus Integration**
            # Publish company data to MessageBus for real-time distribution
            self._handle_data(company)
            
            self._log.info(f"Provided Alpha Vantage company data for {symbol}", LogColor.GREEN)

    # -- INTERNAL METHODS ---------------------------------------------------------------------

    async def _start_update_task(self, instrument_id: InstrumentId, data_type: DataType) -> None:
        """
        Start an update task for an instrument subscription.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to update.
        data_type : DataType
            The data type for updates.
        """
        task_key = f"{instrument_id}_{data_type.type.__name__}"
        if task_key in self._update_tasks:
            return  # Task already running
            
        task = self._loop.create_task(
            self._update_data_task(instrument_id, data_type)
        )
        self._update_tasks[task_key] = task

    async def _update_data_task(self, instrument_id: InstrumentId, data_type: DataType) -> None:
        """
        Periodically update data for a subscribed instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to update.
        data_type : DataType
            The data type for updates.
        """
        symbol = instrument_id.symbol.value
        
        while True:
            try:
                await asyncio.sleep(self._config.update_interval)
                
                # Update data based on type
                if data_type.type == AlphaVantageQuote:
                    await self._update_quote_data(symbol, instrument_id, data_type)
                elif data_type.type == AlphaVantageBar:
                    await self._update_bar_data(symbol, instrument_id, data_type)
                elif data_type.type == AlphaVantageCompanyData:
                    await self._update_company_data(symbol, instrument_id, data_type)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Error updating Alpha Vantage data for {symbol}: {e}")

    async def _update_quote_data(
        self,
        symbol: str,
        instrument_id: InstrumentId,
        data_type: DataType,
    ) -> None:
        """Update quote data and publish if new."""
        quote_data = await self._http_client.get_quote(symbol)
        if not quote_data:
            return
            
        quote = parse_alpha_vantage_quote(
            symbol=symbol,
            quote_data=quote_data,
            ts_init=self._clock.timestamp_ns(),
        )
        
        if quote:
            # Check if this is new data
            cached_ts = self._last_data_timestamps.get(f"{symbol}_quote", 0)
            if quote.ts_event > cached_ts:
                # **CRITICAL: MessageBus Integration for real-time updates**
                self._handle_data(quote)
                self._last_data_timestamps[f"{symbol}_quote"] = quote.ts_event
                self._log.debug(f"Published new Alpha Vantage quote for {symbol}")

    async def _update_bar_data(
        self,
        symbol: str,
        instrument_id: InstrumentId,
        data_type: DataType,
    ) -> None:
        """Update bar data and publish if new."""
        # For subscriptions, typically get recent daily data
        bars_data = await self._http_client.get_daily_prices(symbol, adjusted=True)
        if not bars_data:
            return
            
        bars = parse_alpha_vantage_bars(
            symbol=symbol,
            time_series_data=bars_data,
            adjusted=True,
            ts_init=self._clock.timestamp_ns(),
        )
        
        if bars:
            # Get the latest bar
            latest_bar = max(bars, key=lambda x: x.ts_event)
            
            # Check if this is new data
            cached_ts = self._last_data_timestamps.get(f"{symbol}_bars", 0)
            if latest_bar.ts_event > cached_ts:
                # **CRITICAL: MessageBus Integration for real-time updates**
                self._handle_data(latest_bar)
                self._last_data_timestamps[f"{symbol}_bars"] = latest_bar.ts_event
                self._log.debug(f"Published new Alpha Vantage bar for {symbol}")

    async def _update_company_data(
        self,
        symbol: str,
        instrument_id: InstrumentId,
        data_type: DataType,
    ) -> None:
        """Update company data (less frequent updates)."""
        # Company data updates are less frequent, maybe weekly
        company_data = await self._http_client.get_company_overview(symbol)
        if not company_data:
            return
            
        company = parse_alpha_vantage_company_data(
            symbol=symbol,
            overview_data=company_data,
            ts_init=self._clock.timestamp_ns(),
        )
        
        if company:
            # **CRITICAL: MessageBus Integration for company updates**
            self._handle_data(company)
            self._company_cache[symbol] = company
            self._log.debug(f"Published Alpha Vantage company data for {symbol}")

    # -- ENHANCED MARKET DATA METHODS --------------------------------------------------------

    async def get_multiple_quotes(self, symbols: list[str]) -> dict[str, AlphaVantageQuote]:
        """Get real-time quotes for multiple symbols efficiently."""
        quotes = {}
        
        # Rate limiting: process symbols in batches
        batch_size = 5  # Alpha Vantage free tier: 5 requests/minute
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Process batch in parallel with rate limiting
            tasks = [self._get_quote_for_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if not isinstance(result, Exception) and result:
                    quotes[symbol] = result
                    
            # Rate limiting delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(self._config.rate_limit_delay)
        
        return quotes

    async def _get_quote_for_symbol(self, symbol: str) -> AlphaVantageQuote | None:
        """Get quote for a single symbol."""
        try:
            quote_data = await self._http_client.get_global_quote(symbol)
            if not quote_data:
                return None
                
            quote = parse_alpha_vantage_quote(
                symbol=symbol,
                quote_data=quote_data,
                ts_init=self._clock.timestamp_ns(),
            )
            
            if quote:
                # Publish to message bus
                self._handle_data(quote)
                
            return quote
            
        except Exception as e:
            self._log.error(f"Error fetching quote for {symbol}: {e}")
            return None

    async def search_symbols(self, keywords: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search for symbols by keywords."""
        try:
            search_data = await self._http_client.search_symbols(keywords)
            if not search_data or "bestMatches" not in search_data:
                return []
                
            results = []
            for match in search_data["bestMatches"][:max_results]:
                results.append({
                    "symbol": match.get("1. symbol", ""),
                    "name": match.get("2. name", ""),
                    "type": match.get("3. type", ""),
                    "region": match.get("4. region", ""),
                    "currency": match.get("8. currency", ""),
                    "match_score": float(match.get("9. matchScore", 0.0))
                })
                
            return results
            
        except Exception as e:
            self._log.error(f"Error searching symbols for '{keywords}': {e}")
            return []

    async def get_company_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        """Get comprehensive company fundamentals."""
        try:
            # Check cache first
            if symbol in self._fundamentals_cache:
                return self._fundamentals_cache[symbol]
            
            # Get company overview
            overview_data = await self._http_client.get_company_overview(symbol)
            if not overview_data:
                return None
                
            # Get earnings data
            earnings_data = await self._http_client.get_earnings(symbol)
            
            # Combine fundamental data
            fundamentals = {
                "symbol": symbol,
                "overview": overview_data,
                "earnings": earnings_data,
                "last_updated": datetime.now().isoformat()
            }
            
            # Cache for future use
            self._fundamentals_cache[symbol] = fundamentals
            
            return fundamentals
            
        except Exception as e:
            self._log.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    async def get_market_overview(self) -> dict[str, Any]:
        """Get market overview using major indices and stocks."""
        try:
            overview = {
                "timestamp": datetime.now().isoformat(),
                "quotes": {},
                "market_status": "open"  # Could be enhanced with market hours check
            }
            
            # Get quotes for major symbols
            quotes = await self.get_multiple_quotes(self._major_symbols)
            overview["quotes"] = {
                symbol: {
                    "price": quote.price,
                    "change": quote.change_percent,
                    "volume": quote.volume
                }
                for symbol, quote in quotes.items()
                if quote is not None
            }
            
            return overview
            
        except Exception as e:
            self._log.error(f"Error fetching market overview: {e}")
            return {}

    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1min",
        adjusted: bool = True
    ) -> list[AlphaVantageBar]:
        """Get historical bar data for a symbol."""
        try:
            if period in ["1min", "5min", "15min", "30min", "60min"]:
                # Intraday data
                bar_data = await self._http_client.get_time_series_intraday(
                    symbol=symbol,
                    interval=period,
                    adjusted=adjusted
                )
            else:
                # Daily data
                bar_data = await self._http_client.get_time_series_daily(
                    symbol=symbol,
                    adjusted=adjusted
                )
            
            if not bar_data:
                return []
                
            bars = parse_alpha_vantage_bars(
                symbol=symbol,
                bar_data=bar_data,
                ts_init=self._clock.timestamp_ns(),
            )
            
            # Publish bars to message bus
            for bar in bars:
                self._handle_data(bar)
            
            return bars
            
        except Exception as e:
            self._log.error(f"Error fetching historical data for {symbol}: {e}")
            return []

    def get_cached_company_data(self, symbol: str) -> AlphaVantageCompanyData | None:
        """Get cached company data for a symbol."""
        return self._company_cache.get(symbol)

    def get_cached_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        """Get cached fundamentals for a symbol."""
        return self._fundamentals_cache.get(symbol)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._company_cache.clear()
        self._fundamentals_cache.clear()
        self._log.info("Cleared Alpha Vantage cache")

    async def health_check(self) -> dict[str, Any]:
        """Check health of Alpha Vantage integration."""
        try:
            # Test with a simple quote request
            test_quote = await self._get_quote_for_symbol("AAPL")
            
            return {
                "status": "healthy" if test_quote else "degraded",
                "api_connected": self._http_client.is_connected,
                "cached_companies": len(self._company_cache),
                "cached_fundamentals": len(self._fundamentals_cache),
                "active_subscriptions": sum(len(instruments) for instruments in self._subscriptions.values()),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
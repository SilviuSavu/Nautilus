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
import time
from collections import defaultdict
from typing import Any

import pandas as pd
import pytz
import yfinance as yf

from nautilus_trader.adapters.yfinance.config import YFinanceDataClientConfig
from nautilus_trader.adapters.yfinance.providers import YFinanceInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import RequestBars
from nautilus_trader.data.messages import RequestInstrument
from nautilus_trader.data.messages import RequestInstruments
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class YFinanceDataClient(LiveMarketDataClient):
    """
    Provides a data client for the Yahoo Finance API.

    This client provides historical market data access through the yfinance library.
    It implements intelligent caching, rate limiting, and robust error handling.

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
    instrument_provider : YFinanceInstrumentProvider
        The instrument provider for the client.
    config : YFinanceDataClientConfig, optional
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
        instrument_provider: YFinanceInstrumentProvider,
        config: YFinanceDataClientConfig | None = None,
        name: str | None = None,
    ) -> None:
        if config is None:
            config = YFinanceDataClientConfig()

        PyCondition.type(config, YFinanceDataClientConfig, "config")

        super().__init__(
            loop=loop,
            client_id=ClientId(name or "YFINANCE"),
            venue=None,  # Multi-venue support
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
        )

        # Configuration
        self._config = config
        self._cache_expiry_seconds = config.cache_expiry_seconds
        self._rate_limit_delay = config.rate_limit_delay
        self._request_timeout = config.request_timeout
        self._max_retries = config.max_retries
        self._retry_delay = config.retry_delay
        self._default_period = config.default_period
        self._default_interval = config.default_interval

        # Rate limiting
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()

        # Ticker cache with expiry
        self._ticker_cache: dict[str, tuple[yf.Ticker, float]] = {}

        # Log configuration
        self._log.info(f"Cache expiry: {self._cache_expiry_seconds}s", LogColor.BLUE)
        self._log.info(f"Rate limit delay: {self._rate_limit_delay}s", LogColor.BLUE)
        self._log.info(f"Request timeout: {self._request_timeout}s", LogColor.BLUE)
        self._log.info(f"Max retries: {self._max_retries}", LogColor.BLUE)

    async def _connect(self) -> None:
        """Connect to Yahoo Finance (no persistent connection needed)."""
        self._log.info("YFinance data client connected")

        # Load configured instruments if any
        if self._config.instrument_ids:
            await self._instrument_provider.load_ids_async(self._config.instrument_ids)
        elif self._config.symbols:
            # Convert symbols to InstrumentIds
            instrument_ids = [
                InstrumentId.from_str(f"{symbol}.YAHOO") for symbol in self._config.symbols
            ]
            await self._instrument_provider.load_ids_async(instrument_ids)

    async def _disconnect(self) -> None:
        """Disconnect from Yahoo Finance."""
        self._log.info("YFinance data client disconnected")
        
        # Clear ticker cache
        self._ticker_cache.clear()

    def reset(self) -> None:
        """Reset the client state."""
        self._ticker_cache.clear()
        self._last_request_time = 0.0

    def dispose(self) -> None:
        """Dispose of the client resources."""
        self._ticker_cache.clear()

    # -- REQUESTS ---------------------------------------------------------------------------------

    async def _request_instrument(self, request: RequestInstrument) -> None:
        """
        Handle an instrument definition request.

        Parameters
        ----------
        request : RequestInstrument
            The request to process.

        """
        self._log.debug(f"Received request for {request.instrument_id}")

        try:
            await self._instrument_provider.load_async(request.instrument_id)
            instrument = self._cache.instrument(request.instrument_id)
            
            if instrument is not None:
                self._handle_data(instrument, request.correlation_id)
            else:
                self._log.error(f"Failed to load instrument {request.instrument_id}")
                
        except Exception as e:
            self._log.error(f"Error requesting instrument {request.instrument_id}: {e}")

    async def _request_instruments(self, request: RequestInstruments) -> None:
        """
        Handle an instruments definition request.

        Parameters
        ----------
        request : RequestInstruments
            The request to process.

        """
        self._log.debug(f"Received request for {request.venue} instruments")

        try:
            # Load all instruments from cache for the venue
            instruments = []
            for instrument_id in request.instrument_ids or []:
                instrument = self._cache.instrument(instrument_id)
                if instrument is not None:
                    instruments.append(instrument)

            self._handle_data(instruments, request.correlation_id)
            
        except Exception as e:
            self._log.error(f"Error requesting instruments for {request.venue}: {e}")

    async def _request_bars(self, request: RequestBars) -> None:
        """
        Handle a historical bars request.

        Parameters
        ----------
        request : RequestBars
            The request to process.

        """
        self._log.debug(f"Received request for {request.bar_type} bars")

        try:
            # Extract symbol from instrument ID
            symbol = request.bar_type.instrument_id.symbol.value
            
            # Get or create ticker
            ticker = await self._get_ticker(symbol)
            
            # Convert bar spec to yfinance parameters
            interval = self._convert_bar_spec_to_interval(request.bar_type)
            period = self._determine_period(request.start, request.end)
            
            # Make the request with rate limiting
            bars_data = await self._fetch_historical_data(
                ticker, period, interval, request.start, request.end
            )
            
            if bars_data is not None and not bars_data.empty:
                # Convert to Nautilus Bar objects
                bars = self._convert_to_nautilus_bars(bars_data, request.bar_type)
                
                # Apply limit if specified
                if request.limit > 0:
                    bars = bars[-request.limit:]
                
                # Send bars to handler
                for bar in bars:
                    self._handle_data(bar, request.correlation_id)
                    
                self._log.info(
                    f"Sent {len(bars)} bars for {request.bar_type.instrument_id}",
                    LogColor.GREEN,
                )
            else:
                self._log.warning(f"No bar data received for {request.bar_type.instrument_id}")
                
        except Exception as e:
            self._log.error(f"Error requesting bars for {request.bar_type}: {e}")

    async def _get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get a cached ticker or create a new one.

        Parameters
        ----------
        symbol : str
            The Yahoo Finance symbol.

        Returns
        -------
        yf.Ticker
            The yfinance Ticker object.

        """
        current_time = time.time()
        
        # Check cache
        if symbol in self._ticker_cache:
            ticker, cached_time = self._ticker_cache[symbol]
            if current_time - cached_time < self._cache_expiry_seconds:
                return ticker
        
        # Create new ticker and cache it
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
        self._ticker_cache[symbol] = (ticker, current_time)
        
        return ticker

    async def _fetch_historical_data(
        self,
        ticker: yf.Ticker,
        period: str,
        interval: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame | None:
        """
        Fetch historical data with rate limiting and retries.

        Parameters
        ----------
        ticker : yf.Ticker
            The yfinance Ticker object.
        period : str
            The period for the request.
        interval : str
            The interval for the request.
        start : pd.Timestamp, optional
            The start time for the request.
        end : pd.Timestamp, optional
            The end time for the request.

        Returns
        -------
        pd.DataFrame or None
            The historical data or None if failed.

        """
        async with self._request_lock:
            # Apply rate limiting
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._rate_limit_delay:
                await asyncio.sleep(self._rate_limit_delay - time_since_last)
            
            self._last_request_time = time.time()

        # Prepare parameters
        kwargs = {"period": period, "interval": interval}
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end

        # Retry logic
        for attempt in range(self._max_retries + 1):
            try:
                loop = asyncio.get_event_loop()
                
                # Execute with timeout
                data = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: ticker.history(**kwargs)),
                    timeout=self._request_timeout,
                )
                
                if data is not None and not data.empty:
                    return data
                else:
                    self._log.warning(f"Empty data returned for {ticker.ticker}")
                    return None
                    
            except asyncio.TimeoutError:
                self._log.warning(f"Request timeout for {ticker.ticker} (attempt {attempt + 1})")
            except Exception as e:
                self._log.warning(f"Request failed for {ticker.ticker}: {e} (attempt {attempt + 1})")
            
            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(delay)
        
        return None

    def _convert_bar_spec_to_interval(self, bar_type: BarType) -> str:
        """
        Convert Nautilus bar specification to yfinance interval.

        Parameters
        ----------
        bar_type : BarType
            The Nautilus bar type.

        Returns
        -------
        str
            The yfinance interval string.

        """
        spec = bar_type.spec
        
        if spec.aggregation == BarAggregation.MINUTE:
            if spec.step == 1:
                return "1m"
            elif spec.step == 2:
                return "2m"
            elif spec.step == 5:
                return "5m"
            elif spec.step == 15:
                return "15m"
            elif spec.step == 30:
                return "30m"
            elif spec.step == 60:
                return "1h"
            else:
                self._log.warning(f"Unsupported minute step {spec.step}, using 1m")
                return "1m"
        elif spec.aggregation == BarAggregation.HOUR:
            return "1h"
        elif spec.aggregation == BarAggregation.DAY:
            return "1d"
        elif spec.aggregation == BarAggregation.WEEK:
            return "1wk"
        elif spec.aggregation == BarAggregation.MONTH:
            return "1mo"
        else:
            self._log.warning(f"Unsupported aggregation {spec.aggregation}, using daily")
            return "1d"

    def _determine_period(
        self,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> str:
        """
        Determine the appropriate period for yfinance request.

        Parameters
        ----------
        start : pd.Timestamp, optional
            The start time.
        end : pd.Timestamp, optional
            The end time.

        Returns
        -------
        str
            The period string.

        """
        if start is None and end is None:
            return self._default_period
        
        if start is not None and end is not None:
            # Calculate the time range and map to period
            duration = end - start
            days = duration.days
            
            if days <= 5:
                return "5d"
            elif days <= 30:
                return "1mo"
            elif days <= 90:
                return "3mo"
            elif days <= 180:
                return "6mo"
            elif days <= 365:
                return "1y"
            elif days <= 730:
                return "2y"
            elif days <= 1825:
                return "5y"
            else:
                return "max"
        
        return self._default_period

    def _convert_to_nautilus_bars(
        self,
        data: pd.DataFrame,
        bar_type: BarType,
    ) -> list[Bar]:
        """
        Convert yfinance DataFrame to Nautilus Bar objects.

        Parameters
        ----------
        data : pd.DataFrame
            The yfinance historical data.
        bar_type : BarType
            The Nautilus bar type.

        Returns
        -------
        list[Bar]
            The converted Nautilus bars.

        """
        bars = []
        
        for timestamp, row in data.iterrows():
            # Ensure timestamp is timezone-aware (UTC)
            if timestamp.tz is None:
                timestamp = timestamp.tz_localize("UTC")
            elif timestamp.tz != pytz.UTC:
                timestamp = timestamp.tz_convert("UTC")
            
            # Extract OHLCV data
            open_price = float(row["Open"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            close_price = float(row["Close"])
            volume = int(row["Volume"]) if pd.notna(row["Volume"]) else 0
            
            # Create Nautilus Bar
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.8f}"),
                high=Price.from_str(f"{high_price:.8f}"),
                low=Price.from_str(f"{low_price:.8f}"),
                close=Price.from_str(f"{close_price:.8f}"),
                volume=Quantity.from_int(volume),
                ts_event=int(timestamp.value),  # Nanoseconds since epoch
                ts_init=int(timestamp.value),
            )
            
            bars.append(bar)
        
        return bars
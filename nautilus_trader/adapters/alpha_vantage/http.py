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
import json
from typing import Any

import aiohttp

from nautilus_trader.common.logging import Logger


class AlphaVantageHttpClient:
    """
    HTTP client for Alpha Vantage API with rate limiting and error handling.
    
    Parameters
    ----------
    api_key : str
        The Alpha Vantage API key.
    base_url : str, default "https://www.alphavantage.co"
        The base URL for the Alpha Vantage API.
    request_timeout : float, default 30.0
        The request timeout in seconds.
    rate_limit_delay : float, default 12.0
        The delay between requests in seconds.
    logger : Logger
        The logger for the client.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.alphavantage.co",
        request_timeout: float = 30.0,
        rate_limit_delay: float = 12.0,
        logger: Logger | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._rate_limit_delay = rate_limit_delay
        self._logger = logger or Logger(name=self.__class__.__name__)
        
        self._session: aiohttp.ClientSession | None = None
        self._last_request_time = 0.0

    async def connect(self) -> None:
        """Connect the HTTP client."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._logger.info("Alpha Vantage HTTP client connected")

    async def disconnect(self) -> None:
        """Disconnect the HTTP client."""
        if self._session:
            await self._session.close()
            self._session = None
            self._logger.info("Alpha Vantage HTTP client disconnected")

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        import time
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        
        if elapsed < self._rate_limit_delay:
            delay = self._rate_limit_delay - elapsed
            self._logger.debug(f"Rate limiting: waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        
        self._last_request_time = time.time()

    async def _request(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """
        Make an HTTP request to Alpha Vantage API.
        
        Parameters
        ----------
        params : dict[str, Any]
            The request parameters.
            
        Returns
        -------
        dict[str, Any] | None
            The response data or None if request failed.
        """
        if not self._session:
            await self.connect()
        
        # Apply rate limiting
        await self._rate_limit()
        
        # Add API key to parameters
        params["apikey"] = self._api_key
        
        try:
            url = f"{self._base_url}/query"
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for API error messages
                    if "Error Message" in data:
                        self._logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                        return None
                    elif "Note" in data:
                        self._logger.warning(f"Alpha Vantage API note: {data['Note']}")
                        return None
                    elif "Information" in data:
                        self._logger.warning(f"Alpha Vantage API info: {data['Information']}")
                        return None
                    
                    return data
                else:
                    self._logger.error(f"HTTP error {response.status}: {await response.text()}")
                    return None
                    
        except asyncio.TimeoutError:
            self._logger.error("Request timeout")
            return None
        except Exception as e:
            self._logger.error(f"Request failed: {e}")
            return None

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """
        Get real-time quote for a symbol.
        
        Parameters
        ----------
        symbol : str
            The stock symbol.
            
        Returns
        -------
        dict[str, Any] | None
            The quote data or None if request failed.
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        }
        return await self._request(params)

    async def get_daily_prices(
        self,
        symbol: str,
        adjusted: bool = True,
    ) -> dict[str, Any] | None:
        """
        Get daily OHLCV data for a symbol.
        
        Parameters
        ----------
        symbol : str
            The stock symbol.
        adjusted : bool, default True
            Whether to get adjusted prices.
            
        Returns
        -------
        dict[str, Any] | None
            The daily price data or None if request failed.
        """
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": "compact",  # compact = latest 100 data points
        }
        return await self._request(params)

    async def get_intraday_prices(
        self,
        symbol: str,
        interval: str = "5min",
        adjusted: bool = True,
        extended_hours: bool = True,
        month: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get intraday OHLCV data for a symbol.
        
        Parameters
        ----------
        symbol : str
            The stock symbol.
        interval : str, default "5min"
            The time interval (1min, 5min, 15min, 30min, 60min).
        adjusted : bool, default True
            Whether to get adjusted prices.
        extended_hours : bool, default True
            Whether to include extended trading hours.
        month : str | None, default None
            The month to get data for (YYYY-MM format).
            
        Returns
        -------
        dict[str, Any] | None
            The intraday price data or None if request failed.
        """
        function = "TIME_SERIES_INTRADAY"
        params = {
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "adjusted": "true" if adjusted else "false",
            "extended_hours": "true" if extended_hours else "false",
            "outputsize": "compact",
        }
        
        if month:
            params["month"] = month
            
        return await self._request(params)

    async def search_symbols(self, keywords: str) -> dict[str, Any] | None:
        """
        Search for symbols matching keywords.
        
        Parameters
        ----------
        keywords : str
            The search keywords.
            
        Returns
        -------
        dict[str, Any] | None
            The search results or None if request failed.
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
        }
        return await self._request(params)

    async def get_company_overview(self, symbol: str) -> dict[str, Any] | None:
        """
        Get company fundamental data.
        
        Parameters
        ----------
        symbol : str
            The stock symbol.
            
        Returns
        -------
        dict[str, Any] | None
            The company data or None if request failed.
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
        }
        return await self._request(params)

    async def get_earnings(self, symbol: str) -> dict[str, Any] | None:
        """
        Get earnings data for a symbol.
        
        Parameters
        ----------
        symbol : str
            The stock symbol.
            
        Returns
        -------
        dict[str, Any] | None
            The earnings data or None if request failed.
        """
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
        }
        return await self._request(params)
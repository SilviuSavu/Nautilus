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
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from nautilus_trader.common.component import Logger


class FREDHttpClient:
    """
    HTTP client for interacting with the Federal Reserve Economic Data (FRED) API.
    
    This client handles all HTTP communications with the FRED API, including proper
    error handling, rate limiting, and connection management.
    
    Parameters
    ----------
    api_key : str
        The FRED API key for authentication.
    base_url : str, default "https://api.stlouisfed.org/fred"
        The base URL for the FRED API.
    request_timeout : float, default 30.0
        The timeout for HTTP requests in seconds.
    rate_limit_delay : float, default 1.0
        The minimum delay between consecutive API requests in seconds.
    logger : Logger, optional
        The logger for the client.
        
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.stlouisfed.org/fred",
        request_timeout: float = 30.0,
        rate_limit_delay: float = 1.0,
        logger: Logger | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._rate_limit_delay = rate_limit_delay
        self._logger = logger
        self._session: aiohttp.ClientSession | None = None
        self._last_request_time: float = 0.0
        
    async def connect(self) -> None:
        """
        Connect to the FRED API by creating an HTTP session.
        
        """
        if self._session is None:
            timeout = ClientTimeout(total=self._request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            if self._logger:
                self._logger.info("Connected to FRED API")
                
    async def disconnect(self) -> None:
        """
        Disconnect from the FRED API by closing the HTTP session.
        
        """
        if self._session:
            await self._session.close()
            self._session = None
            
            if self._logger:
                self._logger.info("Disconnected from FRED API")

    async def get_series_info(self, series_id: str) -> dict[str, Any]:
        """
        Get metadata information for a specific FRED series.
        
        Parameters
        ----------
        series_id : str
            The FRED series ID (e.g., "GDP", "UNRATE").
            
        Returns
        -------
        dict[str, Any]
            The series metadata from FRED API.
            
        """
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        
        response = await self._request("GET", "/series", params)
        return response.get("seriess", [{}])[0] if response else {}
        
    async def get_series_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort_order: str = "asc",
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> dict[str, Any]:
        """
        Get observations (data points) for a specific FRED series.
        
        Parameters
        ----------
        series_id : str
            The FRED series ID.
        start_date : str, optional
            The start date for observations (YYYY-MM-DD format).
        end_date : str, optional
            The end date for observations (YYYY-MM-DD format).
        limit : int, optional
            The maximum number of observations to return (max 100,000).
        offset : int, optional
            The offset for paginated results.
        sort_order : str, default "asc"
            The sort order ("asc" or "desc").
        observation_start : str, optional
            The observation start date (YYYY-MM-DD format).
        observation_end : str, optional
            The observation end date (YYYY-MM-DD format).
            
        Returns
        -------
        dict[str, Any]
            The series observations from FRED API.
            
        """
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": sort_order,
        }
        
        # Add optional parameters if provided
        if start_date:
            params["realtime_start"] = start_date
        if end_date:
            params["realtime_end"] = end_date
        if limit:
            params["limit"] = str(limit)
        if offset:
            params["offset"] = str(offset)
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
            
        response = await self._request("GET", "/series/observations", params)
        return response or {}
        
    async def get_categories(self, category_id: int | None = None) -> dict[str, Any]:
        """
        Get FRED categories information.
        
        Parameters
        ----------
        category_id : int, optional
            The category ID to retrieve. If None, gets root categories.
            
        Returns
        -------
        dict[str, Any]
            The categories information from FRED API.
            
        """
        params = {
            "api_key": self._api_key,
            "file_type": "json",
        }
        
        if category_id is not None:
            params["category_id"] = str(category_id)
            
        response = await self._request("GET", "/category", params)
        return response or {}
        
    async def get_category_series(
        self,
        category_id: int,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        Get series in a specific FRED category.
        
        Parameters
        ----------
        category_id : int
            The category ID.
        limit : int, optional
            The maximum number of series to return.
        offset : int, optional
            The offset for paginated results.
            
        Returns
        -------
        dict[str, Any]
            The series in the category from FRED API.
            
        """
        params = {
            "category_id": str(category_id),
            "api_key": self._api_key,
            "file_type": "json",
        }
        
        if limit:
            params["limit"] = str(limit)
        if offset:
            params["offset"] = str(offset)
            
        response = await self._request("GET", "/category/series", params)
        return response or {}
        
    async def search_series(
        self,
        search_text: str,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str = "search_rank",
        sort_order: str = "asc",
    ) -> dict[str, Any]:
        """
        Search for FRED series by text.
        
        Parameters
        ----------
        search_text : str
            The search text.
        limit : int, optional
            The maximum number of results to return.
        offset : int, optional
            The offset for paginated results.
        order_by : str, default "search_rank"
            The field to order results by.
        sort_order : str, default "asc"
            The sort order ("asc" or "desc").
            
        Returns
        -------
        dict[str, Any]
            The search results from FRED API.
            
        """
        params = {
            "search_text": search_text,
            "api_key": self._api_key,
            "file_type": "json",
            "order_by": order_by,
            "sort_order": sort_order,
        }
        
        if limit:
            params["limit"] = str(limit)
        if offset:
            params["offset"] = str(offset)
            
        response = await self._request("GET", "/series/search", params)
        return response or {}
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Make an HTTP request to the FRED API with rate limiting and error handling.
        
        Parameters
        ----------
        method : str
            The HTTP method (GET, POST, etc.).
        endpoint : str
            The API endpoint path.
        params : dict[str, Any], optional
            The request parameters.
            
        Returns
        -------
        dict[str, Any] | None
            The parsed JSON response or None if request failed.
            
        """
        if self._session is None:
            await self.connect()
            
        # Apply rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
            
        url = f"{self._base_url}{endpoint}"
        
        try:
            if self._logger:
                self._logger.debug(f"FRED API request: {method} {url} with params: {params}")
                
            async with self._session.request(method, url, params=params) as response:
                self._last_request_time = asyncio.get_event_loop().time()
                
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 400:
                    error_text = await response.text()
                    if self._logger:
                        self._logger.error(f"FRED API bad request (400): {error_text}")
                    raise ValueError(f"Bad request to FRED API: {error_text}")
                elif response.status == 429:
                    if self._logger:
                        self._logger.warning("FRED API rate limit exceeded, backing off")
                    # Wait longer and retry once
                    await asyncio.sleep(5.0)
                    return await self._request(method, endpoint, params)
                else:
                    error_text = await response.text()
                    if self._logger:
                        self._logger.error(f"FRED API error ({response.status}): {error_text}")
                    raise Exception(f"FRED API error ({response.status}): {error_text}")
                    
        except asyncio.TimeoutError:
            if self._logger:
                self._logger.error("FRED API request timeout")
            raise
        except Exception as e:
            if self._logger:
                self._logger.error(f"FRED API request failed: {e}")
            raise
"""
Alpha Vantage HTTP Client
========================

HTTP client with rate limiting and error handling for Alpha Vantage API.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from cachetools import TTLCache
from .config import AlphaVantageConfig
from .models import AlphaVantageError

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, calls_per_period: int, period_seconds: int):
        self.calls_per_period = calls_per_period
        self.period_seconds = period_seconds
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.period_seconds]
            
            if len(self.calls) >= self.calls_per_period:
                # Need to wait
                oldest_call = min(self.calls)
                wait_time = self.period_seconds - (now - oldest_call) + 1
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                
                # Clean up again after waiting
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time < self.period_seconds]
            
            self.calls.append(now)


class AlphaVantageHTTPClient:
    """HTTP client for Alpha Vantage API with rate limiting and caching."""
    
    def __init__(self, config: AlphaVantageConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(config.rate_limit_calls, config.rate_limit_period)
        self.cache = TTLCache(maxsize=1000, ttl=config.cache_ttl)
        self.last_successful_request: Optional[datetime] = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited request to Alpha Vantage API."""
        if not self.config.api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        await self._ensure_session()
        await self.rate_limiter.acquire()
        
        # Add API key to parameters
        params = params.copy()
        params['apikey'] = self.config.api_key
        
        # Check cache
        cache_key = str(sorted(params.items()))
        if cache_key in self.cache:
            logger.debug("Returning cached response")
            return self.cache[cache_key]
        
        retries = 0
        while retries <= self.config.max_retries:
            try:
                async with self.session.get(self.config.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for Alpha Vantage error responses
                        if 'Error Message' in data:
                            error_msg = data['Error Message']
                            logger.error(f"Alpha Vantage API error: {error_msg}")
                            raise Exception(f"Alpha Vantage API error: {error_msg}")
                        
                        if 'Note' in data:
                            # Rate limit message
                            note = data['Note']
                            logger.warning(f"Alpha Vantage note: {note}")
                            if 'call frequency' in note.lower():
                                # Rate limited, wait and retry
                                await asyncio.sleep(60)  # Wait 1 minute
                                retries += 1
                                continue
                        
                        # Cache successful response
                        self.cache[cache_key] = data
                        self.last_successful_request = datetime.now()
                        return data
                        
                    elif response.status == 429:
                        # Rate limited
                        wait_time = 2 ** retries * self.config.backoff_factor
                        logger.warning(f"Rate limited, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                        retries += 1
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP error {response.status}: {error_text}")
                        raise Exception(f"HTTP error {response.status}: {error_text}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, retry {retries + 1}/{self.config.max_retries + 1}")
                retries += 1
                if retries <= self.config.max_retries:
                    wait_time = 2 ** retries * self.config.backoff_factor
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception("Request timeout after maximum retries")
                    
            except Exception as e:
                if retries < self.config.max_retries:
                    wait_time = 2 ** retries * self.config.backoff_factor
                    logger.warning(f"Request failed, retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    raise
        
        raise Exception(f"Request failed after {self.config.max_retries + 1} attempts")
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol."""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol.upper()
        }
        return await self._make_request(params)
    
    async def get_intraday(self, symbol: str, interval: str = '5min', extended_hours: bool = True) -> Dict[str, Any]:
        """Get intraday time series data."""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol.upper(),
            'interval': interval,
            'extended_hours': 'true' if extended_hours else 'false',
            'outputsize': 'compact'  # Latest 100 data points
        }
        return await self._make_request(params)
    
    async def get_daily(self, symbol: str, outputsize: str = 'compact') -> Dict[str, Any]:
        """Get daily time series data."""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol.upper(),
            'outputsize': outputsize  # 'compact' or 'full'
        }
        return await self._make_request(params)
    
    async def search_symbols(self, keywords: str) -> Dict[str, Any]:
        """Search for symbols by keywords."""
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': keywords
        }
        return await self._make_request(params)
    
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company overview and fundamental data."""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol.upper()
        }
        return await self._make_request(params)
    
    async def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get quarterly and annual earnings data."""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol.upper()
        }
        return await self._make_request(params)
    
    async def get_supported_functions(self) -> List[str]:
        """Get list of supported Alpha Vantage functions."""
        return [
            'TIME_SERIES_INTRADAY',
            'TIME_SERIES_DAILY',
            'TIME_SERIES_DAILY_ADJUSTED',
            'TIME_SERIES_WEEKLY',
            'TIME_SERIES_WEEKLY_ADJUSTED',
            'TIME_SERIES_MONTHLY',
            'TIME_SERIES_MONTHLY_ADJUSTED',
            'GLOBAL_QUOTE',
            'SYMBOL_SEARCH',
            'OVERVIEW',
            'INCOME_STATEMENT',
            'BALANCE_SHEET',
            'CASH_FLOW',
            'EARNINGS',
            'LISTING_STATUS',
            'EARNINGS_CALENDAR',
            'IPO_CALENDAR'
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check by making a test request."""
        try:
            # Test with a simple quote request
            await self.get_quote('AAPL')
            return {
                'status': 'operational',
                'last_successful_request': self.last_successful_request.isoformat() if self.last_successful_request else None,
                'cache_size': len(self.cache)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'cache_size': len(self.cache)
            }
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
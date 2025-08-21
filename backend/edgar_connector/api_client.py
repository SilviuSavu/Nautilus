"""
EDGAR API Client
===============

HTTP client for SEC EDGAR API with rate limiting and proper user-agent handling.
"""

import asyncio
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

import httpx
from edgar_connector.config import EDGARConfig
from edgar_connector.data_types import FilingType, SECEntity, SECFiling


logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache, None if not found or expired."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            # Check if expired
            timestamp = self.timestamps[key]
            if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.timestamps.move_to_end(key)
            return self.cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set item in cache, evicting oldest if necessary."""
        async with self._lock:
            # Remove if exists (to update)
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            
            # Evict oldest items if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
    
    async def clear(self) -> None:
        """Clear all cached items."""
        async with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class RateLimiter:
    """Rate limiter for SEC API compliance (10 requests/sec max)."""
    
    def __init__(self, max_requests_per_second: float = 10.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit token, blocking if necessary."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = asyncio.get_event_loop().time()


class EDGARAPIClient:
    """
    SEC EDGAR API client with rate limiting and caching.
    
    Provides methods to fetch SEC filings, company facts, and entity data
    while respecting SEC API rate limits and best practices.
    """
    
    def __init__(self, config: EDGARConfig):
        self.config = config
        self.base_url = config.base_url
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_second)
        
        # HTTP client with proper headers and connection pooling
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": config.user_agent,
                "Accept": "application/json",
                "Host": "data.sec.gov"
            },
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
        )
        
        # Bounded LRU cache with TTL
        self._cache = LRUCache(
            max_size=1000,  # Limit memory usage
            ttl_seconds=config.cache_ttl_seconds
        )
        
        logger.info(f"EDGAR API Client initialized with user-agent: {config.user_agent}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client and clean up resources."""
        await self.client.aclose()
        await self._cache.clear()
        logger.info("EDGAR API Client closed and resources cleaned up")
    
    def _cache_key(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for request."""
        key = endpoint
        if params:
            # Sort params for consistent key generation
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            key = f"{endpoint}?{param_str}"
        return key
    
    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if available and not expired."""
        if not self.config.enable_cache:
            return None
        return await self._cache.get(key)
    
    async def _set_cache(self, key: str, data: Any) -> None:
        """Store data in cache."""
        if self.config.enable_cache:
            await self._cache.set(key, data)
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retry logic."""
        cache_key = self._cache_key(endpoint, params)
        
        # Check cache first
        cached_data = await self._get_cached(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_data
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        if retries is None:
            retries = self.config.max_retries
        
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Requesting {method} {url} (attempt {attempt + 1})")
                
                response = await self.client.request(method, url, params=params)
                response.raise_for_status()
                
                data = response.json()
                await self._set_cache(cache_key, data)
                
                logger.debug(f"Successfully fetched {endpoint}")
                return data
                
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code} for {url}")
                if e.response.status_code == 429:  # Rate limited
                    if attempt < retries:
                        wait_time = (2 ** attempt) * self.config.backoff_factor
                        logger.info(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                raise
                
            except httpx.RequestError as e:
                logger.warning(f"Request error for {url}: {e}")
                if attempt < retries:
                    wait_time = (2 ** attempt) * self.config.backoff_factor
                    logger.info(f"Request failed, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        raise Exception(f"Failed to fetch {url} after {retries + 1} attempts")
    
    async def get_company_tickers(self) -> Dict[str, Any]:
        """
        Get company ticker mapping from SEC.
        
        Returns:
            Dict containing CIK to ticker mappings
        """
        logger.info("Fetching company ticker mappings")
        return await self._request("GET", "/files/company_tickers.json")
    
    async def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """
        Get company facts (financial data) for a specific CIK.
        
        Args:
            cik: Central Index Key (e.g., "0000320193" for Apple)
            
        Returns:
            Dict containing company facts data
        """
        cik = cik.zfill(10)  # Pad CIK to 10 digits
        endpoint = f"/api/xbrl/companyfacts/CIK{cik}.json"
        logger.info(f"Fetching company facts for CIK {cik}")
        return await self._request("GET", endpoint)
    
    async def get_submissions(self, cik: str) -> Dict[str, Any]:
        """
        Get all submissions for a specific CIK.
        
        Args:
            cik: Central Index Key
            
        Returns:
            Dict containing submission data
        """
        cik = cik.zfill(10)
        endpoint = f"/submissions/CIK{cik}.json"
        logger.info(f"Fetching submissions for CIK {cik}")
        return await self._request("GET", endpoint)
    
    async def get_company_concept(
        self,
        cik: str,
        taxonomy: str,
        tag: str
    ) -> Dict[str, Any]:
        """
        Get specific financial concept for a company.
        
        Args:
            cik: Central Index Key
            taxonomy: Accounting taxonomy (e.g., "us-gaap")
            tag: Financial concept tag (e.g., "Revenue")
            
        Returns:
            Dict containing concept data
        """
        cik = cik.zfill(10)
        endpoint = f"/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
        logger.info(f"Fetching {taxonomy}:{tag} for CIK {cik}")
        return await self._request("GET", endpoint)
    
    async def search_companies(
        self,
        query: str,
        limit: int = 100
    ) -> List[SECEntity]:
        """
        Search for companies by name or ticker.
        
        Note: This uses the ticker mapping as SEC doesn't have a direct search API.
        
        Args:
            query: Search term (company name or ticker)
            limit: Maximum results to return
            
        Returns:
            List of matching SECEntity objects
        """
        logger.info(f"Searching companies for query: {query}")
        
        # Get ticker mappings
        ticker_data = await self.get_company_tickers()
        
        results = []
        query_lower = query.lower()
        
        for cik_key, company_info in ticker_data.items():
            if not isinstance(company_info, dict):
                continue
                
            company_name = company_info.get("title", "").lower()
            ticker = company_info.get("ticker", "").lower()
            
            # Search in company name or ticker
            if query_lower in company_name or query_lower == ticker:
                entity = SECEntity(
                    cik=str(company_info.get("cik_str", cik_key)).zfill(10),
                    name=company_info.get("title", ""),
                    ticker=company_info.get("ticker"),
                    exchange=company_info.get("exchange")
                )
                results.append(entity)
                
                if len(results) >= limit:
                    break
        
        logger.info(f"Found {len(results)} companies matching '{query}'")
        return results
    
    async def get_recent_filings(
        self,
        filing_types: Optional[List[FilingType]] = None,
        limit: int = 100,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get recent filings from all companies.
        
        Note: SEC doesn't provide a direct recent filings API, so this is a placeholder
        for the implementation pattern. In practice, you'd need to aggregate from
        multiple company submissions or use a third-party service.
        
        Args:
            filing_types: List of filing types to include
            limit: Maximum number of filings to return
            days_back: Number of days to look back
            
        Returns:
            List of filing dictionaries
        """
        logger.info(f"Getting recent filings (last {days_back} days)")
        
        # This is a simplified implementation - in practice you'd need to:
        # 1. Track which companies to monitor
        # 2. Check their submissions regularly
        # 3. Filter by date and filing type
        
        # For now, return empty list as placeholder
        return []
    
    async def resolve_ticker_to_cik(self, ticker: str) -> Optional[str]:
        """
        Resolve stock ticker to CIK.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK string if found, None otherwise
        """
        logger.info(f"Resolving ticker {ticker} to CIK")
        
        ticker_data = await self.get_company_tickers()
        ticker_upper = ticker.upper()
        
        for cik_key, company_info in ticker_data.items():
            if isinstance(company_info, dict) and company_info.get("ticker") == ticker_upper:
                cik = str(company_info.get("cik_str", cik_key)).zfill(10)
                logger.info(f"Resolved {ticker} to CIK {cik}")
                return cik
        
        logger.warning(f"Could not resolve ticker {ticker} to CIK")
        return None
    
    async def health_check(self) -> bool:
        """
        Check if SEC EDGAR API is accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Simple request to test API availability
            await self._request("GET", "/files/company_tickers.json")
            logger.info("EDGAR API health check passed")
            return True
        except Exception as e:
            logger.error(f"EDGAR API health check failed: {e}")
            return False
"""
Data.gov API Client
===================

CKAN-based API client for Data.gov with rate limiting and caching.
Follows the same patterns as EDGAR and FRED integrations.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

import aiohttp
from cachetools import TTLCache

from .config import DatagovAPIConfig, API_ENDPOINTS
from .data_types import DatagovDataset, DatasetSearchResult, OrganizationInfo, DatagovError
from .utils import validate_dataset_id, validate_organization, assess_trading_relevance

logger = logging.getLogger(__name__)


class DatagovAPIClient:
    """
    Async API client for Data.gov CKAN API with rate limiting and caching.
    
    Features:
    - Rate limiting (1000 req/hour limit)
    - TTL caching for expensive operations  
    - Async request handling
    - Proper error handling and retries
    - Trading-relevance filtering
    """
    
    def __init__(self, config: DatagovAPIConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = TTLCache(maxsize=config.cache_max_size, ttl=config.cache_ttl_seconds)
        
        # Rate limiting
        self._request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'rate_limited': 0,
            'errors': 0,
            'last_request_time': None
        }
        
        logger.info(f"Data.gov API client initialized with {config.rate_limit_requests_per_second:.2f} req/sec limit")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': self.config.user_agent,
                    'X-Api-Key': self.config.api_key
                }
            )
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._rate_limit_lock:
            current_time = time.time()
            
            # Remove requests older than 1 hour
            hour_ago = current_time - 3600
            self._request_times = [t for t in self._request_times if t > hour_ago]
            
            # Check if we're at the hourly limit
            if len(self._request_times) >= self.config.rate_limit_requests_per_hour:
                sleep_time = 3600 - (current_time - self._request_times[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                    self._stats['rate_limited'] += 1
                    await asyncio.sleep(sleep_time)
                    current_time = time.time()
            
            # Check per-second rate limit
            recent_requests = [t for t in self._request_times if t > current_time - 1.0]
            if len(recent_requests) >= self.config.rate_limit_requests_per_second:
                sleep_time = 1.0 / self.config.rate_limit_requests_per_second
                await asyncio.sleep(sleep_time)
                current_time = time.time()
            
            # Record this request
            self._request_times.append(current_time)
            self._stats['last_request_time'] = current_time
    
    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make rate-limited API request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use cache
            
        Returns:
            API response data
            
        Raises:
            DatagovError: If request fails
        """
        await self._ensure_session()
        await self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        params = params or {}
        
        # Create cache key
        cache_key = f"{endpoint}_{hash(str(sorted(params.items())))}"
        
        # Check cache first
        if use_cache and self.config.enable_cache and cache_key in self.cache:
            self._stats['cache_hits'] += 1
            logger.debug(f"Cache hit for {endpoint}")
            return self.cache[cache_key]
        
        self._stats['total_requests'] += 1
        
        # Make request with retries
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                async with self.session.get(url, params=params) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache successful responses
                        if use_cache and self.config.enable_cache:
                            self.cache[cache_key] = data
                        
                        logger.debug(f"API request successful: {endpoint} ({response_time:.1f}ms)")
                        return data
                    
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    else:
                        error_text = await response.text()
                        raise DatagovError(
                            f"API request failed: {response.status} - {error_text}",
                            status_code=response.status
                        )
                        
            except aiohttp.ClientError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    sleep_time = self.config.backoff_factor * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {sleep_time:.1f}s: {e}")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    break
        
        # All retries failed
        self._stats['errors'] += 1
        error_msg = f"Request failed after {self.config.max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        
        raise DatagovError(error_msg)
    
    async def health_check(self) -> bool:
        """
        Check API health.
        
        Returns:
            True if API is accessible
        """
        try:
            # Simple package search to test connectivity
            await self._make_request(API_ENDPOINTS["package_search"], {"rows": 1})
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def search_datasets(
        self,
        query: str = "*:*",
        organization: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start: int = 0,
        rows: int = 100,
        sort: str = "score desc, metadata_modified desc",
        trading_relevant_only: bool = False
    ) -> DatasetSearchResult:
        """
        Search for datasets.
        
        Args:
            query: Search query (CKAN query syntax)
            organization: Filter by organization
            tags: Filter by tags
            start: Result offset
            rows: Number of results
            sort: Sort order
            trading_relevant_only: Only return trading-relevant datasets
            
        Returns:
            Search results
        """
        params = {
            "q": query,
            "start": start,
            "rows": min(rows, self.config.max_rows_per_request),
            "sort": sort,
            "include_private": "false"
        }
        
        # Add filters
        filter_queries = []
        if organization:
            org_name = validate_organization(organization)
            filter_queries.append(f"organization:{org_name}")
        
        if tags:
            tag_filter = " OR ".join(f"tags:{tag}" for tag in tags)
            filter_queries.append(f"({tag_filter})")
        
        if filter_queries:
            params["fq"] = " AND ".join(filter_queries)
        
        try:
            response = await self._make_request(API_ENDPOINTS["package_search"], params)
            
            if not response.get("success", False):
                raise DatagovError(f"Search failed: {response.get('error', 'Unknown error')}")
            
            result_data = response.get("result", {})
            datasets = []
            
            for dataset_dict in result_data.get("results", []):
                try:
                    dataset = DatagovDataset(**dataset_dict)
                    
                    # Apply trading relevance filter
                    if trading_relevant_only:
                        relevance_score = assess_trading_relevance(dataset_dict)
                        if relevance_score < 0.3:  # Minimum relevance threshold
                            continue
                    
                    datasets.append(dataset)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse dataset: {e}")
                    continue
            
            return DatasetSearchResult(
                count=result_data.get("count", len(datasets)),
                results=datasets,
                facets=result_data.get("search_facets", {})
            )
            
        except DatagovError:
            raise
        except Exception as e:
            raise DatagovError(f"Search request failed: {e}")
    
    async def get_dataset(self, dataset_id: str) -> Optional[DatagovDataset]:
        """
        Get detailed dataset information.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset object or None if not found
        """
        try:
            dataset_id = validate_dataset_id(dataset_id)
            
            params = {"id": dataset_id}
            response = await self._make_request(API_ENDPOINTS["package_show"], params)
            
            if not response.get("success", False):
                logger.warning(f"Dataset not found: {dataset_id}")
                return None
            
            dataset_dict = response.get("result", {})
            return DatagovDataset(**dataset_dict)
            
        except DatagovError:
            raise
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e}")
            return None
    
    async def list_organizations(self, all_fields: bool = True) -> List[OrganizationInfo]:
        """
        Get list of organizations.
        
        Args:
            all_fields: Include all organization fields
            
        Returns:
            List of organizations
        """
        try:
            params = {"all_fields": all_fields}
            response = await self._make_request(API_ENDPOINTS["organization_list"], params)
            
            if not response.get("success", False):
                raise DatagovError("Failed to list organizations")
            
            organizations = []
            for org_dict in response.get("result", []):
                try:
                    if isinstance(org_dict, dict):
                        organizations.append(OrganizationInfo(**org_dict))
                    else:
                        # String format - create minimal org info
                        organizations.append(OrganizationInfo(
                            id=org_dict,
                            name=org_dict,
                            title=org_dict.replace('-', ' ').title()
                        ))
                except Exception as e:
                    logger.warning(f"Failed to parse organization: {e}")
                    continue
            
            return organizations
            
        except DatagovError:
            raise
        except Exception as e:
            raise DatagovError(f"Failed to list organizations: {e}")
    
    async def get_organization(self, org_id: str) -> Optional[OrganizationInfo]:
        """
        Get detailed organization information.
        
        Args:
            org_id: Organization ID
            
        Returns:
            Organization info or None if not found
        """
        try:
            org_id = validate_organization(org_id)
            
            params = {"id": org_id}
            response = await self._make_request(API_ENDPOINTS["organization_show"], params)
            
            if not response.get("success", False):
                return None
            
            org_dict = response.get("result", {})
            return OrganizationInfo(**org_dict)
            
        except Exception as e:
            logger.error(f"Failed to get organization {org_id}: {e}")
            return None
    
    async def get_trading_relevant_datasets(
        self,
        limit: int = 100,
        min_relevance: float = 0.3
    ) -> List[DatagovDataset]:
        """
        Get datasets most relevant for trading strategies.
        
        Args:
            limit: Maximum number of datasets
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of relevant datasets
        """
        # Search for datasets with trading-relevant keywords
        trading_keywords = [
            "economic", "financial", "market", "price", "commodity", "energy",
            "agriculture", "employment", "inflation", "gdp", "trade"
        ]
        
        query = " OR ".join(trading_keywords)
        
        # Focus on priority organizations
        relevant_datasets = []
        
        for org in self.config.priority_organizations[:5]:  # Limit to top 5 to avoid rate limits
            try:
                results = await self.search_datasets(
                    query=query,
                    organization=org,
                    rows=20,  # Small batch per org
                    trading_relevant_only=True
                )
                
                for dataset in results.results:
                    if len(relevant_datasets) >= limit:
                        break
                    
                    # Double-check relevance
                    relevance = assess_trading_relevance(dataset.dict())
                    if relevance >= min_relevance:
                        relevant_datasets.append(dataset)
                
                if len(relevant_datasets) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to search organization {org}: {e}")
                continue
        
        # Sort by estimated relevance
        relevant_datasets.sort(
            key=lambda d: assess_trading_relevance(d.dict()),
            reverse=True
        )
        
        return relevant_datasets[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            'cache_size': len(self.cache),
            'cache_maxsize': self.cache.maxsize,
            'cache_ttl': self.cache.ttl,
            'rate_limit_per_hour': self.config.rate_limit_requests_per_hour,
            'rate_limit_per_second': self.config.rate_limit_requests_per_second
        }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logger.info("Cache cleared")
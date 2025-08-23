"""
Data.gov Configuration
======================

Configuration classes and factory functions for Data.gov integration.
Following the same patterns as EDGAR connector config.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class DatagovAPIConfig:
    """Data.gov API client configuration."""
    
    # API Configuration
    api_key: str
    base_url: str = "https://api.gsa.gov/technology/datagov/v3/"
    user_agent: str = "NautilusTrader-Backend trading@nautilus-trader.com"
    
    # Rate Limiting (Data.gov limits: 1000 req/hour for registered keys)
    rate_limit_requests_per_hour: float = 950.0  # Leave buffer
    rate_limit_requests_per_second: float = 0.25  # ~900 req/hour to be safe
    rate_limit_burst: int = 5  # Allow small bursts
    
    # Caching Configuration
    enable_cache: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    cache_max_size: int = 1000
    
    # Request Configuration
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.3
    
    # Dataset Configuration
    default_rows_per_request: int = 100
    max_rows_per_request: int = 1000
    
    # Organization Filters (for focused data collection)
    priority_organizations: List[str] = None
    
    def __post_init__(self):
        if self.priority_organizations is None:
            self.priority_organizations = [
                "federal-reserve-economic-data",
                "bureau-of-labor-statistics",
                "bureau-of-economic-analysis", 
                "department-of-agriculture",
                "department-of-energy",
                "environmental-protection-agency",
                "department-of-commerce",
                "treasury-department",
                "securities-and-exchange-commission"
            ]


class DatagovInstrumentConfig(BaseModel):
    """Configuration for Data.gov instrument provider."""
    
    # Instrument Loading
    load_datasets_on_startup: bool = Field(True, description="Load dataset catalog on startup")
    dataset_cache_ttl: int = Field(3600, description="Dataset metadata cache TTL (seconds)")
    
    # Dataset Filtering
    include_categories: List[str] = Field(
        default=["economic", "financial", "agricultural", "energy"],
        description="Dataset categories to include"
    )
    
    exclude_formats: List[str] = Field(
        default=["pdf", "zip"],
        description="Resource formats to exclude"
    )
    
    # Trading Relevance
    trading_relevant_only: bool = Field(True, description="Only include trading-relevant datasets")
    min_resources: int = Field(1, description="Minimum number of data resources required")
    
    # Update Monitoring
    monitor_updates: bool = Field(True, description="Monitor dataset updates")
    update_check_interval: int = Field(3600, description="Update check interval (seconds)")
    
    class Config:
        arbitrary_types_allowed = True


def create_default_config(
    api_key: Optional[str] = None,
    user_agent: str = "NautilusTrader-Backend trading@nautilus-trader.com",
    rate_limit_requests_per_second: float = 0.25,
    cache_ttl_seconds: int = 1800,
    priority_organizations: Optional[List[str]] = None
) -> DatagovAPIConfig:
    """
    Create default Data.gov API configuration.
    
    Args:
        api_key: Data.gov API key (from environment if not provided)
        user_agent: User agent string for API requests
        rate_limit_requests_per_second: Rate limit for API calls
        cache_ttl_seconds: Cache TTL in seconds
        priority_organizations: List of priority government organizations
        
    Returns:
        DatagovAPIConfig instance
        
    Raises:
        ValueError: If no API key is available
    """
    if api_key is None:
        api_key = os.environ.get('DATAGOV_API_KEY')
    
    if not api_key:
        raise ValueError("Data.gov API key is required. Set DATAGOV_API_KEY environment variable or pass api_key parameter.")
    
    return DatagovAPIConfig(
        api_key=api_key,
        user_agent=user_agent,
        rate_limit_requests_per_second=rate_limit_requests_per_second,
        cache_ttl_seconds=cache_ttl_seconds,
        priority_organizations=priority_organizations
    )


def create_instrument_config(
    trading_relevant_only: bool = True,
    include_categories: Optional[List[str]] = None,
    monitor_updates: bool = True
) -> DatagovInstrumentConfig:
    """
    Create default instrument provider configuration.
    
    Args:
        trading_relevant_only: Only include trading-relevant datasets
        include_categories: Categories to include (defaults to trading-relevant)
        monitor_updates: Enable update monitoring
        
    Returns:
        DatagovInstrumentConfig instance
    """
    if include_categories is None:
        include_categories = ["economic", "financial", "agricultural", "energy"]
        
    return DatagovInstrumentConfig(
        trading_relevant_only=trading_relevant_only,
        include_categories=include_categories,
        monitor_updates=monitor_updates
    )


# Trading-focused dataset mapping
TRADING_DATASET_CATEGORIES = {
    "economic": [
        "GDP", "employment", "inflation", "interest rates", "money supply",
        "consumer spending", "business investment", "trade balance"
    ],
    "financial": [
        "banking", "securities", "derivatives", "commodities", "currencies",
        "market data", "credit", "bonds", "equities"
    ],
    "agricultural": [
        "crop reports", "livestock", "commodity prices", "weather", 
        "agricultural trade", "food prices", "supply chain"
    ],
    "energy": [
        "oil prices", "natural gas", "electricity", "renewable energy",
        "energy consumption", "petroleum stocks", "energy trade"
    ]
}

# Common API endpoints for different data types
API_ENDPOINTS = {
    "package_search": "action/package_search",
    "package_show": "action/package_show", 
    "organization_list": "action/organization_list",
    "organization_show": "action/organization_show",
    "group_list": "action/group_list",
    "tag_list": "action/tag_list",
    "resource_show": "action/resource_show",
    "package_list": "action/package_list"
}
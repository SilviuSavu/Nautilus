"""
EDGAR API Configuration
======================

Configuration classes for SEC EDGAR API integration with NautilusTrader.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class EDGARConfig(BaseModel):
    """Configuration for EDGAR API client."""
    
    # API Settings
    base_url: str = Field(
        default="https://data.sec.gov",
        description="Base URL for SEC EDGAR API"
    )
    
    user_agent: str = Field(
        description="Required user-agent string for SEC API requests"
    )
    
    # Rate Limiting
    rate_limit_requests_per_second: float = Field(
        default=10.0,
        description="Max requests per second (SEC limit: 10/sec)"
    )
    
    request_timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds"
    )
    
    # Caching
    enable_cache: bool = Field(
        default=True,
        description="Enable local caching of API responses"
    )
    
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache time-to-live in seconds"
    )
    
    cache_directory: Optional[str] = Field(
        default=None,
        description="Directory for cache files (None for memory cache)"
    )
    
    # Data Processing
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    
    backoff_factor: float = Field(
        default=1.0,
        description="Backoff factor for retry attempts"
    )
    
    # Entity Filtering
    include_company_tickers: Optional[list] = Field(
        default=None,
        description="List of specific tickers to include (None for all)"
    )
    
    include_filing_types: list = Field(
        default=["10-K", "10-Q", "8-K", "DEF 14A"],
        description="List of filing types to track"
    )
    
    # API Keys (for third-party wrappers if needed)
    sec_api_key: Optional[str] = Field(
        default=None,
        description="API key for sec-api.io service (optional)"
    )
    
    @validator("user_agent")
    def validate_user_agent(cls, v):
        """Validate user agent includes contact information."""
        if not v or "@" not in v:
            raise ValueError("user_agent must include contact email address")
        return v
    
    @validator("rate_limit_requests_per_second")
    def validate_rate_limit(cls, v):
        """Validate rate limit doesn't exceed SEC limits."""
        if v > 10.0:
            raise ValueError("rate_limit_requests_per_second cannot exceed 10.0")
        return v
    
    class Config:
        env_prefix = "EDGAR_"
        case_sensitive = False


class EDGARInstrumentConfig(BaseModel):
    """Configuration for EDGAR instrument provider."""
    
    # Entity Data
    update_entities_on_startup: bool = Field(
        default=True,
        description="Download latest entity data on startup"
    )
    
    entities_cache_path: str = Field(
        default="edgar_entities.json",
        description="Local cache file for entity mappings"
    )
    
    # CIK/Ticker Mapping
    enable_ticker_resolution: bool = Field(
        default=True,
        description="Enable automatic ticker to CIK resolution"
    )
    
    ticker_cache_ttl: int = Field(
        default=86400,  # 24 hours
        description="Ticker mapping cache TTL in seconds"
    )


class EDGARDataClientConfig(BaseModel):
    """Configuration for EDGAR data client."""
    
    # Subscriptions
    auto_subscribe_filings: bool = Field(
        default=True,
        description="Automatically subscribe to new filings"
    )
    
    subscription_check_interval: int = Field(
        default=300,  # 5 minutes
        description="Interval to check for new filings in seconds"
    )
    
    # Data Processing
    parse_financial_data: bool = Field(
        default=True,
        description="Parse financial data from XBRL filings"
    )
    
    include_raw_filing_text: bool = Field(
        default=False,
        description="Include raw filing text in data"
    )
    
    max_filing_age_days: int = Field(
        default=365,
        description="Maximum age of filings to include"
    )
    
    # Performance
    concurrent_requests: int = Field(
        default=5,
        description="Number of concurrent API requests"
    )
    
    batch_size: int = Field(
        default=10,
        description="Batch size for bulk requests"
    )


def create_default_config(user_agent: str, **overrides: Any) -> EDGARConfig:
    """Create default EDGAR configuration with user agent."""
    config_dict = {
        "user_agent": user_agent,
        **overrides
    }
    return EDGARConfig(**config_dict)
"""
Alpha Vantage Configuration
===========================

Configuration settings for Alpha Vantage API integration.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlphaVantageConfig:
    """Configuration for Alpha Vantage API integration."""
    
    api_key: Optional[str] = None
    base_url: str = "https://www.alphavantage.co/query"
    request_timeout: int = 30
    rate_limit_calls: int = 5  # calls per minute for free tier
    rate_limit_period: int = 60  # seconds
    cache_ttl: int = 300  # 5 minutes cache TTL
    max_retries: int = 3
    backoff_factor: float = 1.0
    
    def __post_init__(self):
        """Initialize configuration from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        if len(self.api_key) < 10:
            raise ValueError("Invalid Alpha Vantage API key format.")


# Global configuration instance
alpha_vantage_config = AlphaVantageConfig()
"""
Trading Economics Integration Service
====================================

Service layer for Trading Economics API integration with caching and rate limiting.
Provides access to 300,000+ economic indicators across 196 countries.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import os
import aiohttp
from dataclasses import dataclass
from enum import Enum

try:
    import tradingeconomics as te
    TRADINGECONOMICS_AVAILABLE = True
except ImportError:
    TRADINGECONOMICS_AVAILABLE = False
    logging.warning("tradingeconomics package not available. Using mock data.")

logger = logging.getLogger(__name__)


class TradingEconomicsConfig:
    """Trading Economics configuration."""
    
    def __init__(self):
        self.api_key = os.getenv("TRADING_ECONOMICS_API_KEY", "guest")
        self.use_guest = self.api_key == "guest"
        self.cache_ttl = 300  # 5 minutes cache
        self.rate_limit_requests = 500  # requests per minute
        self.rate_limit_window = 60  # 1 minute window
        self.timeout = 30.0
        
        # Initialize Trading Economics client
        if TRADINGECONOMICS_AVAILABLE:
            if not self.use_guest:
                te.login(self.api_key)
            else:
                te.login()  # Use guest access
        
        logger.info(f"Trading Economics initialized with {'API key' if not self.use_guest else 'guest access'}")


@dataclass
class EconomicIndicator:
    """Economic indicator data structure."""
    country: str
    indicator: str
    category: str
    title: str
    latest_value: Optional[float]
    previous_value: Optional[float]
    forecast: Optional[float]
    unit: str
    frequency: str
    last_update: datetime
    
    @classmethod
    def from_te_data(cls, data: Dict[str, Any]) -> 'EconomicIndicator':
        """Create from Trading Economics data."""
        return cls(
            country=data.get('Country', ''),
            indicator=data.get('Category', ''),
            category=data.get('Group', ''),
            title=data.get('Title', ''),
            latest_value=data.get('LatestValue'),
            previous_value=data.get('PreviousValue'),
            forecast=data.get('Forecast'),
            unit=data.get('Unit', ''),
            frequency=data.get('Frequency', ''),
            last_update=datetime.now()
        )


@dataclass
class EconomicCalendarEvent:
    """Economic calendar event data structure."""
    event: str
    country: str
    category: str
    date: datetime
    importance: str
    actual: Optional[float]
    previous: Optional[float]
    forecast: Optional[float]
    revised: Optional[float]
    unit: str
    
    @classmethod
    def from_te_data(cls, data: Dict[str, Any]) -> 'EconomicCalendarEvent':
        """Create from Trading Economics calendar data."""
        return cls(
            event=data.get('Event', ''),
            country=data.get('Country', ''),
            category=data.get('Category', ''),
            date=datetime.fromisoformat(data.get('Date', datetime.now().isoformat())),
            importance=data.get('Importance', 'medium'),
            actual=data.get('Actual'),
            previous=data.get('Previous'),
            forecast=data.get('Forecast'),
            revised=data.get('Revised'),
            unit=data.get('Unit', '')
        )


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    name: str
    country: str
    group: str
    last: Optional[float]
    daily_change: Optional[float]
    daily_change_percent: Optional[float]
    weekly_change: Optional[float]
    monthly_change: Optional[float]
    yearly_change: Optional[float]
    date: datetime
    
    @classmethod
    def from_te_data(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from Trading Economics market data."""
        return cls(
            symbol=data.get('Symbol', ''),
            name=data.get('Name', ''),
            country=data.get('Country', ''),
            group=data.get('Group', ''),
            last=data.get('Last'),
            daily_change=data.get('DailyChange'),
            daily_change_percent=data.get('DailyPercentualChange'),
            weekly_change=data.get('WeeklyChange'),
            monthly_change=data.get('MonthlyChange'),
            yearly_change=data.get('YearlyChange'),
            date=datetime.now()
        )


class TradingEconomicsIntegration:
    """Trading Economics integration service."""
    
    def __init__(self):
        self.config = TradingEconomicsConfig()
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._request_count = 0
        self._last_reset = datetime.now()
        
        # Mock data for development/testing
        self.mock_countries = [
            {"Country": "United States", "CountryGroup": "North America"},
            {"Country": "United Kingdom", "CountryGroup": "Europe"},
            {"Country": "Germany", "CountryGroup": "Europe"},
            {"Country": "Japan", "CountryGroup": "Asia"},
            {"Country": "China", "CountryGroup": "Asia"}
        ]
        
        self.mock_indicators = [
            {"Country": "United States", "Category": "GDP", "Title": "GDP Growth Rate", "LatestValue": 2.1},
            {"Country": "United States", "Category": "Inflation", "Title": "Consumer Price Index", "LatestValue": 3.2},
            {"Country": "United States", "Category": "Employment", "Title": "Unemployment Rate", "LatestValue": 3.7}
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Trading Economics service health."""
        try:
            if TRADINGECONOMICS_AVAILABLE:
                # Try a simple API call
                test_data = await self._safe_te_call(lambda: te.getCountries())
                status = "healthy" if test_data else "degraded"
            else:
                status = "mock_mode"
            
            return {
                "service": "trading_economics",
                "status": status,
                "package_available": TRADINGECONOMICS_AVAILABLE,
                "using_guest_access": self.config.use_guest,
                "last_check": datetime.now().isoformat(),
                "rate_limit_status": self._get_rate_limit_status()
            }
        except Exception as e:
            logger.error(f"Trading Economics health check failed: {e}")
            return {
                "service": "trading_economics",
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def get_countries(self) -> List[Dict[str, Any]]:
        """Get available countries."""
        cache_key = "countries"
        
        # Check cache first
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        
        try:
            if TRADINGECONOMICS_AVAILABLE:
                countries = await self._safe_te_call(lambda: te.getCountries())
                if countries:
                    result = [dict(country) for country in countries]
                else:
                    result = self.mock_countries
            else:
                result = self.mock_countries
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get countries: {e}")
            return self.mock_countries
    
    async def get_indicators(self) -> List[Dict[str, Any]]:
        """Get available economic indicators."""
        cache_key = "indicators"
        
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        
        try:
            if TRADINGECONOMICS_AVAILABLE:
                indicators = await self._safe_te_call(lambda: te.getIndicatorData())
                if indicators:
                    result = [dict(indicator) for indicator in indicators]
                else:
                    result = self.mock_indicators
            else:
                result = self.mock_indicators
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get indicators: {e}")
            return self.mock_indicators
    
    async def get_indicator_data(
        self, 
        country: str, 
        indicator: Optional[str] = None,
        category: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get economic indicator data for a specific country."""
        cache_key = f"indicator_{country}_{indicator}_{category}"
        
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        
        try:
            if TRADINGECONOMICS_AVAILABLE:
                if indicator:
                    data = await self._safe_te_call(
                        lambda: te.getIndicatorData(country=country, indicator=indicator)
                    )
                else:
                    data = await self._safe_te_call(
                        lambda: te.getIndicatorData(country=country)
                    )
                
                if data:
                    result = [dict(item) for item in data]
                else:
                    result = [item for item in self.mock_indicators if item['Country'].lower() == country.lower()]
            else:
                result = [item for item in self.mock_indicators if item['Country'].lower() == country.lower()]
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get indicator data for {country}: {e}")
            return [item for item in self.mock_indicators if item['Country'].lower() == country.lower()]
    
    async def get_calendar(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get economic calendar events."""
        cache_key = f"calendar_{country}_{category}_{importance}"
        
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        
        try:
            if TRADINGECONOMICS_AVAILABLE:
                params = {}
                if country:
                    params['country'] = country
                if category:
                    params['category'] = category
                
                calendar_data = await self._safe_te_call(
                    lambda: te.getCalendarData(**params)
                )
                
                if calendar_data:
                    result = [dict(event) for event in calendar_data]
                else:
                    result = self._get_mock_calendar()
            else:
                result = self._get_mock_calendar()
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get calendar data: {e}")
            return self._get_mock_calendar()
    
    async def get_markets(
        self,
        market_type: str,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get market data by type."""
        cache_key = f"markets_{market_type}_{country}"
        
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        
        try:
            if TRADINGECONOMICS_AVAILABLE:
                market_data = await self._safe_te_call(
                    lambda: te.getMarketsData(marketsField=market_type)
                )
                
                if market_data:
                    result = [dict(item) for item in market_data]
                else:
                    result = self._get_mock_markets(market_type)
            else:
                result = self._get_mock_markets(market_type)
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get market data for {market_type}: {e}")
            return self._get_mock_markets(market_type)
    
    async def get_forecast(
        self,
        country: str,
        indicator: str
    ) -> Dict[str, Any]:
        """Get forecast data for specific indicator."""
        cache_key = f"forecast_{country}_{indicator}"
        
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        
        try:
            if TRADINGECONOMICS_AVAILABLE:
                forecast_data = await self._safe_te_call(
                    lambda: te.getForecastData(country=country, indicator=indicator)
                )
                
                if forecast_data:
                    result = dict(forecast_data[0]) if forecast_data else {}
                else:
                    result = self._get_mock_forecast(country, indicator)
            else:
                result = self._get_mock_forecast(country, indicator)
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get forecast for {indicator} in {country}: {e}")
            return self._get_mock_forecast(country, indicator)
    
    async def search(
        self,
        term: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search economic indicators by term."""
        try:
            if TRADINGECONOMICS_AVAILABLE:
                search_results = await self._safe_te_call(
                    lambda: te.getSearch(term=term, category=category or 'indicators')
                )
                
                if search_results:
                    result = [dict(item) for item in search_results]
                else:
                    result = self._get_mock_search(term)
            else:
                result = self._get_mock_search(term)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to search for '{term}': {e}")
            return self._get_mock_search(term)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "total_requests": self._request_count,
            "cache_size": len(self._cache),
            "rate_limit_status": self._get_rate_limit_status(),
            "package_available": TRADINGECONOMICS_AVAILABLE,
            "using_guest_access": self.config.use_guest,
            "last_reset": self._last_reset.isoformat()
        }
    
    async def refresh_cache(self) -> Dict[str, Any]:
        """Refresh data cache."""
        old_cache_size = len(self._cache)
        self._cache.clear()
        self._cache_timestamps.clear()
        
        return {
            "cache_refreshed": True,
            "old_cache_size": old_cache_size,
            "new_cache_size": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_supported_functions(self) -> List[str]:
        """Get list of supported API functions."""
        return [
            "getCountries",
            "getIndicatorData",
            "getCalendarData",
            "getMarketsData",
            "getForecastData",
            "getSearch",
            "getCreditRating",
            "getEarnings",
            "getComtradeCountries",
            "getComtradeCategories"
        ]
    
    # Private helper methods
    
    async def _safe_te_call(self, func):
        """Safely execute Trading Economics API call with rate limiting."""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        try:
            # Execute in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func)
            self._request_count += 1
            return result
        except Exception as e:
            logger.error(f"Trading Economics API call failed: {e}")
            return None
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        if (now - self._last_reset).seconds >= self.config.rate_limit_window:
            self._request_count = 0
            self._last_reset = now
        
        return self._request_count < self.config.rate_limit_requests
    
    def _get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "requests_made": self._request_count,
            "requests_limit": self.config.rate_limit_requests,
            "window_seconds": self.config.rate_limit_window,
            "time_until_reset": max(0, self.config.rate_limit_window - (datetime.now() - self._last_reset).seconds)
        }
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is in cache and still valid."""
        if key not in self._cache:
            return False
        
        timestamp = self._cache_timestamps.get(key, datetime.min)
        return (datetime.now() - timestamp).seconds < self.config.cache_ttl
    
    def _get_cached(self, key: str) -> Any:
        """Get cached data."""
        return self._cache.get(key)
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with timestamp."""
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()
    
    def _get_mock_calendar(self) -> List[Dict[str, Any]]:
        """Get mock calendar data."""
        return [
            {
                "Event": "GDP Growth Rate",
                "Country": "United States",
                "Category": "GDP",
                "Date": datetime.now().isoformat(),
                "Importance": "high",
                "Actual": 2.1,
                "Previous": 2.4,
                "Forecast": 2.0
            }
        ]
    
    def _get_mock_markets(self, market_type: str) -> List[Dict[str, Any]]:
        """Get mock market data."""
        return [
            {
                "Symbol": "USD/EUR",
                "Name": "US Dollar vs Euro",
                "Country": "United States",
                "Group": market_type,
                "Last": 0.85,
                "DailyChange": -0.01,
                "DailyPercentualChange": -1.2
            }
        ]
    
    def _get_mock_forecast(self, country: str, indicator: str) -> Dict[str, Any]:
        """Get mock forecast data."""
        return {
            "Country": country,
            "Category": indicator,
            "YearEnd": 2024,
            "q1": 2.1,
            "q2": 2.3,
            "q3": 2.5,
            "q4": 2.7
        }
    
    def _get_mock_search(self, term: str) -> List[Dict[str, Any]]:
        """Get mock search results."""
        return [
            {
                "Country": "United States",
                "Category": "GDP",
                "Title": f"GDP Growth Rate (matches: {term})",
                "LatestValue": 2.1,
                "Unit": "Percent"
            }
        ]


# Global instance
trading_economics_integration = TradingEconomicsIntegration()
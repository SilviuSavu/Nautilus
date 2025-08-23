"""
Comprehensive test suite for Trading Economics integration
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import date, datetime

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app
from trading_economics_integration import TradingEconomicsIntegration

client = TestClient(app)


class TestTradingEconomicsHealthEndpoints:
    """Test Trading Economics health and status endpoints"""
    
    def test_health_endpoint(self):
        """Test Trading Economics health check"""
        response = client.get("/api/v1/trading-economics/health")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "trading_economics"
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "error", "mock_mode"]
    
    def test_health_endpoint_content_type(self):
        """Test health endpoint returns JSON"""
        response = client.get("/api/v1/trading-economics/health")
        assert response.headers["content-type"] == "application/json"
    
    def test_health_endpoint_structure(self):
        """Test health endpoint returns expected structure"""
        response = client.get("/api/v1/trading-economics/health")
        data = response.json()
        required_fields = ["service", "status", "package_available", "using_guest_access", "last_check"]
        for field in required_fields:
            assert field in data


class TestTradingEconomicsDataEndpoints:
    """Test Trading Economics data retrieval endpoints"""
    
    def test_countries_endpoint(self):
        """Test countries data retrieval"""
        response = client.get("/api/v1/trading-economics/countries")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "countries" in data
        assert "count" in data
        assert isinstance(data["countries"], list)
    
    def test_indicators_endpoint(self):
        """Test indicators data retrieval"""
        response = client.get("/api/v1/trading-economics/indicators")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "indicators" in data
        assert "count" in data
        assert isinstance(data["indicators"], list)
    
    def test_country_indicators_endpoint(self):
        """Test country-specific indicators"""
        country = "united states"
        response = client.get(f"/api/v1/trading-economics/indicators/{country}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "country" in data
        assert data["country"] == country
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_country_indicators_with_category(self):
        """Test country indicators with category filter"""
        country = "united states"
        category = "gdp"
        response = client.get(f"/api/v1/trading-economics/indicators/{country}?category={category}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["country"] == country
        assert data["category"] == category
    
    def test_specific_indicator_endpoint(self):
        """Test specific indicator data retrieval"""
        country = "united states"
        indicator = "gdp"
        response = client.get(f"/api/v1/trading-economics/indicator/{country}/{indicator}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["country"] == country
        assert data["indicator"] == indicator
        assert "data" in data
    
    def test_specific_indicator_with_dates(self):
        """Test specific indicator with date filters"""
        country = "united states"
        indicator = "gdp"
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        response = client.get(
            f"/api/v1/trading-economics/indicator/{country}/{indicator}"
            f"?start_date={start_date}&end_date={end_date}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
    
    def test_calendar_endpoint(self):
        """Test economic calendar data retrieval"""
        response = client.get("/api/v1/trading-economics/calendar")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "events" in data
        assert "count" in data
        assert isinstance(data["events"], list)
    
    def test_calendar_with_filters(self):
        """Test economic calendar with filters"""
        params = {
            "country": "united states",
            "importance": "high"
        }
        response = client.get("/api/v1/trading-economics/calendar", params=params)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["filters"]["country"] == params["country"]
        assert data["filters"]["importance"] == params["importance"]
    
    def test_markets_endpoint(self):
        """Test market data retrieval"""
        market_type = "currencies"
        response = client.get(f"/api/v1/trading-economics/markets/{market_type}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "market_type" in data
        assert data["market_type"] == market_type
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_markets_with_country_filter(self):
        """Test market data with country filter"""
        market_type = "currencies"
        country = "united states"
        response = client.get(f"/api/v1/trading-economics/markets/{market_type}?country={country}")
        assert response.status_code == 200
        data = response.json()
        assert data["country"] == country
    
    def test_forecast_endpoint(self):
        """Test forecast data retrieval"""
        country = "united states"
        indicator = "gdp"
        response = client.get(f"/api/v1/trading-economics/forecast/{country}/{indicator}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["country"] == country
        assert data["indicator"] == indicator
    
    def test_search_endpoint(self):
        """Test search functionality"""
        term = "gdp"
        response = client.get(f"/api/v1/trading-economics/search?term={term}")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["search_term"] == term
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_search_with_category(self):
        """Test search with category filter"""
        term = "inflation"
        category = "indicators"
        response = client.get(f"/api/v1/trading-economics/search?term={term}&category={category}")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == category
    
    def test_statistics_endpoint(self):
        """Test statistics retrieval"""
        response = client.get("/api/v1/trading-economics/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "statistics" in data
    
    def test_supported_functions_endpoint(self):
        """Test supported functions list"""
        response = client.get("/api/v1/trading-economics/supported-functions")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "functions" in data
        assert isinstance(data["functions"], list)


class TestTradingEconomicsServiceMethods:
    """Test Trading Economics service layer methods"""
    
    @pytest.fixture
    def service(self):
        """Create Trading Economics service instance"""
        return TradingEconomicsIntegration()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, service):
        """Test successful health check"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_call.return_value = [{"Country": "United States"}]
            result = await service.health_check()
            assert result["service"] == "trading_economics"
            assert result["status"] in ["healthy", "mock_mode"]
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, service):
        """Test health check failure"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_call.side_effect = Exception("API Error")
            result = await service.health_check()
            assert result["status"] == "error"
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_get_countries(self, service):
        """Test get countries method"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_data = [
                {"Country": "United States", "CountryGroup": "North America"},
                {"Country": "United Kingdom", "CountryGroup": "Europe"}
            ]
            mock_call.return_value = mock_data
            result = await service.get_countries()
            assert len(result) >= 2
            assert any(country["Country"] == "United States" for country in result)
    
    @pytest.mark.asyncio
    async def test_get_indicator_data(self, service):
        """Test get indicator data method"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_data = [{
                "Country": "United States",
                "Category": "GDP",
                "Title": "GDP Growth Rate",
                "LatestValue": 2.1,
                "Unit": "Percent"
            }]
            mock_call.return_value = mock_data
            result = await service.get_indicator_data("united states", "gdp")
            assert len(result) > 0
            assert result[0]["Country"] == "United States"
    
    @pytest.mark.asyncio
    async def test_get_calendar(self, service):
        """Test get calendar method"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_data = [{
                "Event": "GDP Growth Rate",
                "Country": "United States",
                "Date": "2023-10-26",
                "Importance": "high"
            }]
            mock_call.return_value = mock_data
            result = await service.get_calendar(country="united states")
            assert len(result) > 0
            assert result[0]["Country"] == "United States"
    
    @pytest.mark.asyncio
    async def test_get_markets(self, service):
        """Test get markets method"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_data = [{
                "Symbol": "USD/EUR",
                "Name": "US Dollar vs Euro",
                "Last": 0.85,
                "Group": "currencies"
            }]
            mock_call.return_value = mock_data
            result = await service.get_markets("currencies")
            assert len(result) > 0
            assert result[0]["Symbol"] == "USD/EUR"
    
    @pytest.mark.asyncio
    async def test_search(self, service):
        """Test search method"""
        with patch.object(service, '_safe_te_call') as mock_call:
            mock_data = [{
                "Country": "United States",
                "Category": "GDP",
                "Title": "GDP Growth Rate",
                "LatestValue": 2.1
            }]
            mock_call.return_value = mock_data
            result = await service.search("gdp")
            assert len(result) > 0
            assert "GDP" in result[0]["Title"]


class TestTradingEconomicsErrorHandling:
    """Test Trading Economics error handling"""
    
    def test_invalid_country_code(self):
        """Test handling of invalid country code"""
        invalid_country = "invalid_country_123"
        response = client.get(f"/api/v1/trading-economics/indicators/{invalid_country}")
        assert response.status_code == 200  # Should return empty results, not error
        data = response.json()
        assert "data" in data
    
    def test_invalid_market_type(self):
        """Test handling of invalid market type"""
        invalid_market = "invalid_market_123"
        response = client.get(f"/api/v1/trading-economics/markets/{invalid_market}")
        assert response.status_code == 200  # Should return empty results, not error
        data = response.json()
        assert "data" in data
    
    def test_missing_search_term(self):
        """Test search without required term parameter"""
        response = client.get("/api/v1/trading-economics/search")
        assert response.status_code == 422  # Validation error for missing required parameter


class TestTradingEconomicsRateLimiting:
    """Test Trading Economics rate limiting"""
    
    @pytest.fixture
    def service(self):
        """Create Trading Economics service instance"""
        return TradingEconomicsIntegration()
    
    def test_rate_limit_status(self, service):
        """Test rate limit status tracking"""
        status = service._get_rate_limit_status()
        assert "requests_made" in status
        assert "requests_limit" in status
        assert "window_seconds" in status
        assert "time_until_reset" in status
    
    def test_rate_limit_check(self, service):
        """Test rate limit check functionality"""
        # Should pass initially
        assert service._check_rate_limit() is True
        
        # Simulate hitting the limit
        service._request_count = service.config.rate_limit_requests
        assert service._check_rate_limit() is False


class TestTradingEconomicsCaching:
    """Test Trading Economics caching functionality"""
    
    @pytest.fixture
    def service(self):
        """Create Trading Economics service instance"""
        return TradingEconomicsIntegration()
    
    def test_cache_functionality(self, service):
        """Test basic caching operations"""
        # Test cache miss
        assert service._is_cached("test_key") is False
        
        # Test cache set
        service._cache_data("test_key", {"test": "data"})
        assert service._is_cached("test_key") is True
        
        # Test cache get
        cached_data = service._get_cached("test_key")
        assert cached_data == {"test": "data"}
    
    def test_cache_expiration(self, service):
        """Test cache expiration"""
        # Set a very short TTL for testing
        service.config.cache_ttl = -1  # Expired immediately
        service._cache_data("test_key", {"test": "data"})
        
        # Should be expired
        assert service._is_cached("test_key") is False
    
    @pytest.mark.asyncio
    async def test_cache_refresh(self, service):
        """Test cache refresh functionality"""
        # Add some data to cache
        service._cache_data("test_key", {"test": "data"})
        assert len(service._cache) > 0
        
        # Refresh cache
        result = await service.refresh_cache()
        assert result["cache_refreshed"] is True
        assert len(service._cache) == 0


class TestTradingEconomicsIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def test_complete_workflow_gdp_analysis(self):
        """Test complete workflow for GDP analysis"""
        # 1. Check health
        health_response = client.get("/api/v1/trading-economics/health")
        assert health_response.status_code == 200
        
        # 2. Get GDP data for US
        gdp_response = client.get("/api/v1/trading-economics/indicator/united states/gdp")
        assert gdp_response.status_code == 200
        
        # 3. Search for related indicators
        search_response = client.get("/api/v1/trading-economics/search?term=gdp")
        assert search_response.status_code == 200
    
    def test_complete_workflow_market_analysis(self):
        """Test complete workflow for market analysis"""
        # 1. Get currency markets
        currencies_response = client.get("/api/v1/trading-economics/markets/currencies")
        assert currencies_response.status_code == 200
        
        # 2. Get commodities markets
        commodities_response = client.get("/api/v1/trading-economics/markets/commodities")
        assert commodities_response.status_code == 200
        
        # 3. Get calendar events
        calendar_response = client.get("/api/v1/trading-economics/calendar?importance=high")
        assert calendar_response.status_code == 200
    
    def test_complete_workflow_forecasting(self):
        """Test complete workflow for economic forecasting"""
        # 1. Get forecast data
        forecast_response = client.get("/api/v1/trading-economics/forecast/united states/gdp")
        assert forecast_response.status_code == 200
        
        # 2. Get related indicators for context
        indicators_response = client.get("/api/v1/trading-economics/indicators/united states?category=gdp")
        assert indicators_response.status_code == 200


class TestTradingEconomicsOperationalEndpoints:
    """Test operational endpoints for monitoring and management"""
    
    def test_cache_refresh_endpoint(self):
        """Test cache refresh endpoint"""
        response = client.post("/api/v1/trading-economics/cache/refresh")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "cache_refresh" in data
    
    def test_statistics_monitoring(self):
        """Test statistics for operational monitoring"""
        response = client.get("/api/v1/trading-economics/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data
        
        if data["statistics"]:
            stats = data["statistics"]
            assert "total_requests" in stats
            assert "rate_limit_status" in stats
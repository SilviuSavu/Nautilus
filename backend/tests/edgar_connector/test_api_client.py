"""
Tests for EDGAR API Client
==========================
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from edgar_connector.api_client import EDGARAPIClient, RateLimiter
from edgar_connector.config import create_default_config
from edgar_connector.data_types import SECEntity


class TestRateLimiter:
    """Tests for rate limiter."""
    
    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests_per_second=5.0)
        assert limiter.max_requests_per_second == 5.0
        assert limiter.min_interval == 0.2
        assert limiter.last_request_time == 0.0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test rate limiter acquire method."""
        limiter = RateLimiter(max_requests_per_second=10.0)
        
        # First request should be immediate
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should be very quick (less than 10ms)
        assert elapsed < 0.01
        
        # Second request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should be at least min_interval (0.1 seconds for 10 req/sec)
        assert elapsed >= 0.09  # Allow some tolerance


class TestEDGARAPIClient:
    """Tests for EDGAR API client."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_default_config(
            user_agent="TestAgent test@example.com",
            rate_limit_requests_per_second=10.0,
            cache_ttl_seconds=60
        )
        
        # Mock HTTP client
        self.mock_client = AsyncMock()
        
    @pytest.fixture
    def api_client(self):
        """Create API client fixture."""
        client = EDGARAPIClient(self.config)
        client.client = self.mock_client  # Replace with mock
        return client
    
    def test_api_client_init(self, api_client):
        """Test API client initialization."""
        assert api_client.config == self.config
        assert api_client.base_url == self.config.base_url
        assert api_client.rate_limiter.max_requests_per_second == 10.0
        assert isinstance(api_client._cache, dict)
        assert isinstance(api_client._cache_timestamps, dict)
    
    def test_cache_key_generation(self, api_client):
        """Test cache key generation."""
        # Simple endpoint
        key1 = api_client._cache_key("/api/test")
        assert key1 == "/api/test"
        
        # With parameters
        params = {"cik": "123", "format": "json"}
        key2 = api_client._cache_key("/api/test", params)
        assert key2 == "/api/test?cik=123&format=json"
    
    def test_cache_operations(self, api_client):
        """Test cache set/get operations."""
        key = "test_key"
        data = {"test": "data"}
        
        # Initially not cached
        assert not api_client._is_cached(key)
        
        # Set cache
        api_client._set_cache(key, data)
        
        # Should be cached now
        assert api_client._is_cached(key)
        assert api_client._cache[key] == data
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, api_client):
        """Test successful health check."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        result = await api_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, api_client):
        """Test health check failure."""
        # Mock exception
        self.mock_client.request.side_effect = Exception("Connection failed")
        
        result = await api_client.health_check()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_company_tickers(self, api_client):
        """Test getting company ticker mappings."""
        expected_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = expected_data
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        result = await api_client.get_company_tickers()
        assert result == expected_data
        
        # Verify correct API call
        self.mock_client.request.assert_called_once_with(
            "GET",
            "https://data.sec.gov/files/company_tickers.json",
            params=None
        )
    
    @pytest.mark.asyncio
    async def test_get_company_facts(self, api_client):
        """Test getting company facts."""
        cik = "320193"
        expected_data = {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {}
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = expected_data
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        result = await api_client.get_company_facts(cik)
        assert result == expected_data
        
        # Verify CIK padding and correct endpoint
        expected_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
        self.mock_client.request.assert_called_once_with(
            "GET",
            expected_url,
            params=None
        )
    
    @pytest.mark.asyncio
    async def test_get_submissions(self, api_client):
        """Test getting company submissions."""
        cik = "320193"
        expected_data = {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "filings": {}
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = expected_data
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        result = await api_client.get_submissions(cik)
        assert result == expected_data
    
    @pytest.mark.asyncio
    async def test_resolve_ticker_to_cik_success(self, api_client):
        """Test successful ticker to CIK resolution."""
        ticker_data = {
            "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": "1652044", "ticker": "GOOGL", "title": "Alphabet Inc."}
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = ticker_data
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        result = await api_client.resolve_ticker_to_cik("AAPL")
        assert result == "0000320193"
        
        # Test case insensitive
        result = await api_client.resolve_ticker_to_cik("aapl")
        assert result == "0000320193"
    
    @pytest.mark.asyncio
    async def test_resolve_ticker_to_cik_not_found(self, api_client):
        """Test ticker to CIK resolution when ticker not found."""
        ticker_data = {
            "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = ticker_data
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        result = await api_client.resolve_ticker_to_cik("NONEXISTENT")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_search_companies(self, api_client):
        """Test company search functionality."""
        ticker_data = {
            "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": "1652044", "ticker": "GOOGL", "title": "Alphabet Inc."},
            "2": {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corporation"}
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = ticker_data
        mock_response.raise_for_status.return_value = None
        self.mock_client.request.return_value = mock_response
        
        # Search by ticker
        results = await api_client.search_companies("AAPL")
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
        assert results[0].cik == "0000320193"
        
        # Search by company name
        results = await api_client.search_companies("apple")
        assert len(results) == 1
        assert results[0].name == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_request_with_retry(self, api_client):
        """Test request retry logic."""
        # Mock rate limited response first, then success
        responses = [
            Exception("Rate limited"),
            MagicMock()
        ]
        responses[1].json.return_value = {"success": True}
        responses[1].raise_for_status.return_value = None
        
        self.mock_client.request.side_effect = responses
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await api_client._request("GET", "/test")
            
            # Should have retried
            assert self.mock_client.request.call_count == 2
            assert mock_sleep.called
            assert result == {"success": True}
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        config = create_default_config(user_agent="Test test@example.com")
        
        async with EDGARAPIClient(config) as client:
            assert client is not None
            # Client should be initialized properly
            assert hasattr(client, 'client')
    
    @pytest.mark.asyncio
    async def test_close_client(self, api_client):
        """Test client cleanup."""
        await api_client.close()
        self.mock_client.aclose.assert_called_once()


@pytest.mark.integration
class TestEDGARAPIClientIntegration:
    """Integration tests that hit the real SEC API."""
    
    @pytest.fixture
    def real_api_client(self):
        """Create real API client for integration tests."""
        config = create_default_config(
            user_agent="NautilusTrader-Test test@nautilus-trader.com",
            rate_limit_requests_per_second=1.0  # Be conservative in tests
        )
        return EDGARAPIClient(config)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_health_check(self, real_api_client):
        """Test health check against real API."""
        async with real_api_client as client:
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_company_tickers(self, real_api_client):
        """Test getting real company tickers."""
        async with real_api_client as client:
            tickers = await client.get_company_tickers()
            
            assert isinstance(tickers, dict)
            assert len(tickers) > 0
            
            # Check that we have some known companies
            found_apple = False
            for key, company in tickers.items():
                if isinstance(company, dict) and company.get("ticker") == "AAPL":
                    found_apple = True
                    assert "Apple" in company.get("title", "")
                    assert company.get("cik_str") is not None
                    break
            
            assert found_apple, "Should find Apple (AAPL) in ticker data"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_resolve_ticker(self, real_api_client):
        """Test real ticker resolution."""
        async with real_api_client as client:
            # Test Apple
            cik = await client.resolve_ticker_to_cik("AAPL")
            assert cik is not None
            assert cik == "0000320193"  # Apple's known CIK
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_company_facts(self, real_api_client):
        """Test getting real company facts."""
        async with real_api_client as client:
            # Get Apple's company facts
            facts = await client.get_company_facts("0000320193")
            
            assert isinstance(facts, dict)
            assert facts.get("cik") == 320193
            assert "Apple" in facts.get("entityName", "")
            assert "facts" in facts
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rate_limiting_in_practice(self, real_api_client):
        """Test that rate limiting works with real API."""
        async with real_api_client as client:
            # Make multiple requests quickly
            start_time = asyncio.get_event_loop().time()
            
            tasks = []
            for _ in range(3):
                task = asyncio.create_task(client.get_company_tickers())
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # With 1 req/sec limit, 3 requests should take at least 2 seconds
            assert elapsed >= 2.0
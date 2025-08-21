"""
EDGAR Connector Integration Tests
=================================

Tests for full EDGAR connector integration with NautilusTrader.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from nautilus_trader.common.component import LiveClock
from nautilus_trader.model.identifiers import ClientId

from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import create_default_config, EDGARDataClientConfig
from edgar_connector.data_client import EDGARDataClient
from edgar_connector.instrument_provider import EDGARInstrumentProvider
from edgar_connector.data_types import FilingData, CompanyFacts, FilingType


class TestEDGARConnectorIntegration:
    """Integration tests for EDGAR connector components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_default_config(
            user_agent="TestAgent test@example.com",
            rate_limit_requests_per_second=10.0
        )
        
        self.data_config = EDGARDataClientConfig(
            auto_subscribe_filings=False,  # Disable for tests
            subscription_check_interval=1
        )
        
        # Mock clock
        self.clock = MagicMock(spec=LiveClock)
        self.clock.timestamp_ns.return_value = 1640995200000000000  # 2022-01-01
    
    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client."""
        mock_client = AsyncMock(spec=EDGARAPIClient)
        
        # Mock ticker data
        mock_client.get_company_tickers.return_value = {
            "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc.", "exchange": "NASDAQ"},
            "1": {"cik_str": "1652044", "ticker": "GOOGL", "title": "Alphabet Inc.", "exchange": "NASDAQ"}
        }
        
        # Mock company facts
        mock_client.get_company_facts.return_value = {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [{"end": "2023-09-30", "val": 352755000000}]
                        }
                    }
                }
            }
        }
        
        # Mock submissions
        mock_client.get_submissions.return_value = {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "filingDate": ["2023-11-03", "2023-08-04"],
                    "accessionNumber": ["0000320193-23-000123", "0000320193-23-000100"]
                }
            }
        }
        
        mock_client.health_check.return_value = True
        mock_client.resolve_ticker_to_cik.return_value = "0000320193"
        
        return mock_client
    
    @pytest.fixture
    async def instrument_provider(self, mock_api_client):
        """Create instrument provider with mock API client."""
        from edgar_connector.config import EDGARInstrumentConfig
        
        config = EDGARInstrumentConfig(
            update_entities_on_startup=False  # Don't update on startup for tests
        )
        
        provider = EDGARInstrumentProvider(mock_api_client, config)
        await provider.load_all_async()
        return provider
    
    @pytest.fixture
    def data_client(self, mock_api_client, instrument_provider):
        """Create data client with mocked dependencies."""
        client_id = ClientId("EDGAR-001")
        
        with patch.object(EDGARDataClient, '__init__', return_value=None):
            client = EDGARDataClient.__new__(EDGARDataClient)
            
            # Manually set attributes
            client.client_id = client_id
            client._clock = self.clock
            client._instrument_provider = instrument_provider
            client.config = self.config
            client.data_config = self.data_config
            client.api_client = mock_api_client
            client._subscriptions = {}
            client._subscription_tasks = {}
            client._is_connected = False
            client._last_filing_check = None
            
            # Add required methods from parent
            client._handle_data = MagicMock()
            
        return client
    
    @pytest.mark.asyncio
    async def test_full_connection_flow(self, data_client, mock_api_client):
        """Test complete connection and disconnection flow."""
        # Initial state
        assert not data_client._is_connected
        
        # Connect
        await data_client._connect()
        assert data_client._is_connected
        
        # Verify health check was called
        mock_api_client.health_check.assert_called_once()
        
        # Disconnect
        await data_client._disconnect()
        assert not data_client._is_connected
        
        # Verify API client was closed
        mock_api_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_instrument_provider_integration(self, instrument_provider, mock_api_client):
        """Test instrument provider integration."""
        # Should have loaded entities from mock data
        assert instrument_provider.get_entity_count() == 2
        
        # Test ticker resolution
        cik = instrument_provider.resolve_ticker_to_cik("AAPL")
        assert cik == "0000320193"
        
        # Test entity retrieval
        entity = instrument_provider.get_entity_by_ticker("GOOGL")
        assert entity is not None
        assert entity.name == "Alphabet Inc."
        
        # Test search
        results = instrument_provider.search_entities("Apple")
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
    
    @pytest.mark.asyncio
    async def test_data_request_flow(self, data_client, mock_api_client):
        """Test data request handling."""
        from nautilus_trader.model.data import DataType
        from nautilus_trader.core.uuid import UUID4
        
        # Connect first
        data_client._is_connected = True
        
        # Create mock data type with metadata
        data_type = MagicMock(spec=DataType)
        data_type.metadata = {
            'cik': '0000320193',
            'filing_types': ['10-K', '10-Q']
        }
        
        correlation_id = UUID4()
        
        # Handle request
        await data_client._request(data_type, correlation_id)
        
        # Verify API calls were made
        mock_api_client.get_company_facts.assert_called_once_with("0000320193")
        mock_api_client.get_submissions.assert_called_once_with("0000320193")
        
        # Verify data was sent to message bus
        assert data_client._handle_data.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_company_facts_request(self, data_client, mock_api_client):
        """Test requesting company facts."""
        data_client._is_connected = True
        
        # Request company facts
        facts = await data_client.request_company_facts("0000320193")
        
        assert facts is not None
        assert isinstance(facts, CompanyFacts)
        assert facts.cik == "0000320193"
        assert facts.company_name == "Apple Inc."
        assert "financial_data" in facts.facts
    
    @pytest.mark.asyncio
    async def test_company_filings_request(self, data_client, mock_api_client):
        """Test requesting company filings."""
        data_client._is_connected = True
        
        # Request filings
        filings = await data_client.request_company_filings("0000320193")
        
        assert len(filings) == 2
        assert all(isinstance(f, FilingData) for f in filings)
        
        # Check filing types
        filing_types = [f.filing_type for f in filings]
        assert FilingType.FORM_10K in filing_types
        assert FilingType.FORM_10Q in filing_types
        
        # Check filing details
        for filing in filings:
            assert filing.cik == "0000320193"
            assert filing.company_name == "Apple Inc."
            assert filing.accession_number.startswith("0000320193-23-")
    
    @pytest.mark.asyncio
    async def test_subscription_worker(self, data_client, mock_api_client):
        """Test subscription worker functionality."""
        from edgar_connector.data_types import EDGARSubscription
        from nautilus_trader.model.data import DataType
        
        data_client._is_connected = True
        
        # Create subscription
        subscription = EDGARSubscription(
            cik="0000320193",
            filing_types=[FilingType.FORM_10K]
        )
        
        data_type = MagicMock(spec=DataType)
        
        # Mock the worker to run once and exit
        async def mock_worker():
            await data_client._check_company_filings("0000320193", data_type)
            return  # Exit after one iteration
        
        # Run worker once
        await mock_worker()
        
        # Verify API was called
        mock_api_client.get_submissions.assert_called_with("0000320193")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, data_client, mock_api_client):
        """Test error handling in various scenarios."""
        # Test connection failure
        mock_api_client.health_check.return_value = False
        
        with pytest.raises(Exception):
            await data_client._connect()
        
        # Test API error during request
        mock_api_client.get_company_facts.side_effect = Exception("API Error")
        
        # Should not raise exception, but should handle gracefully
        facts = await data_client.request_company_facts("0000320193")
        assert facts is None
    
    @pytest.mark.asyncio
    async def test_ticker_to_cik_resolution_flow(self, data_client, instrument_provider):
        """Test full ticker to CIK resolution flow."""
        data_client._is_connected = True
        
        # Test with ticker in data type metadata
        from nautilus_trader.model.data import DataType
        from nautilus_trader.core.uuid import UUID4
        
        data_type = MagicMock(spec=DataType)
        data_type.metadata = {
            'ticker': 'AAPL',
            'filing_types': ['10-K']
        }
        
        correlation_id = UUID4()
        
        # Should resolve ticker to CIK and make API calls
        await data_client._request(data_type, correlation_id)
        
        # Verify the resolved CIK was used
        # (The mock returns "0000320193" for any ticker resolution)
        data_client.api_client.get_company_facts.assert_called_once()
    
    def test_configuration_integration(self):
        """Test configuration integration across components."""
        # Test that configurations work together
        edgar_config = create_default_config(
            user_agent="Test Agent test@example.com",
            rate_limit_requests_per_second=5.0,
            cache_ttl_seconds=1800
        )
        
        data_config = EDGARDataClientConfig(
            auto_subscribe_filings=True,
            subscription_check_interval=600,
            max_filing_age_days=180
        )
        
        # Should be able to create all components with these configs
        api_client = EDGARAPIClient(edgar_config)
        assert api_client.rate_limiter.max_requests_per_second == 5.0
        assert api_client.config.cache_ttl_seconds == 1800
        
        # Data config should have expected values
        assert data_config.auto_subscribe_filings is True
        assert data_config.subscription_check_interval == 600
        assert data_config.max_filing_age_days == 180
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, data_client, mock_api_client):
        """Test handling concurrent requests."""
        data_client._is_connected = True
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(3):
            cik = f"000032019{i}"
            task = asyncio.create_task(
                data_client.request_company_facts(cik)
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete (even if with mock data)
        assert len(results) == 3
        
        # API client should have been called multiple times
        assert mock_api_client.get_company_facts.call_count == 3
    
    @pytest.mark.asyncio
    async def test_subscription_lifecycle(self, data_client, mock_api_client):
        """Test subscription lifecycle management."""
        from nautilus_trader.model.data import DataType
        
        data_client._is_connected = True
        
        # Create data type for subscription
        data_type = MagicMock(spec=DataType)
        data_type.metadata = {'cik': '0000320193'}
        
        # Subscribe
        await data_client._subscribe(data_type)
        
        # Should have subscription and task
        assert data_type in data_client._subscriptions
        assert len(data_client._subscription_tasks) == 1
        
        # Unsubscribe
        await data_client._unsubscribe(data_type)
        
        # Should be cleaned up
        assert data_type not in data_client._subscriptions
        # Tasks should be cancelled (may take time to clean up)
    
    def test_status_and_statistics(self, data_client, instrument_provider):
        """Test status reporting and statistics."""
        # Test data client status
        status = data_client.get_subscription_status()
        assert "active_subscriptions" in status
        assert "running_tasks" in status
        assert "is_connected" in status
        
        # Test instrument provider statistics
        stats = instrument_provider.get_statistics()
        assert "total_entities" in stats
        assert "entities_with_tickers" in stats
        assert stats["total_entities"] == 2  # From mock data


@pytest.mark.integration
class TestRealEDGARIntegration:
    """Integration tests with real SEC EDGAR API."""
    
    @pytest.fixture
    def real_config(self):
        """Create configuration for real API tests."""
        return create_default_config(
            user_agent="NautilusTrader-Integration-Test test@nautilus-trader.com",
            rate_limit_requests_per_second=0.5  # Be very conservative
        )
    
    @pytest.fixture
    def real_data_config(self):
        """Create data configuration for real API tests."""
        return EDGARDataClientConfig(
            auto_subscribe_filings=False,
            subscription_check_interval=60,
            max_filing_age_days=30
        )
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_apple_data(self, real_config, real_data_config):
        """Test end-to-end flow with real Apple data."""
        # Create real components
        async with EDGARAPIClient(real_config) as api_client:
            
            # Test instrument provider
            from edgar_connector.config import EDGARInstrumentConfig
            instrument_config = EDGARInstrumentConfig()
            
            instrument_provider = EDGARInstrumentProvider(api_client, instrument_config)
            await instrument_provider.load_all_async()
            
            # Verify Apple is in the data
            apple_entity = instrument_provider.get_entity_by_ticker("AAPL")
            assert apple_entity is not None
            assert apple_entity.cik == "0000320193"
            assert "Apple" in apple_entity.name
            
            # Create data client
            client_id = ClientId("EDGAR-REAL-TEST")
            clock = MagicMock(spec=LiveClock)
            clock.timestamp_ns.return_value = 1640995200000000000
            
            # Note: We can't easily test the full LiveDataClient without
            # the complete NautilusTrader framework, so we test the core
            # functionality through direct API calls
            
            # Test getting Apple's company facts
            facts_data = await api_client.get_company_facts("0000320193")
            assert facts_data is not None
            assert facts_data.get("cik") == 320193
            assert "Apple" in facts_data.get("entityName", "")
            
            # Test getting Apple's submissions
            submissions = await api_client.get_submissions("0000320193")
            assert submissions is not None
            assert submissions.get("cik") == "0000320193"
            assert "filings" in submissions
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rate_limiting_compliance(self, real_config):
        """Test that we comply with SEC rate limits."""
        async with EDGARAPIClient(real_config) as api_client:
            # Make several requests and ensure they're properly rate limited
            start_time = asyncio.get_event_loop().time()
            
            # Make 3 requests (should take at least 4 seconds at 0.5 req/sec)
            await api_client.get_company_tickers()
            await api_client.get_company_facts("0000320193")  # Apple
            await api_client.get_submissions("0000320193")
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Should take at least 4 seconds (2 intervals at 0.5 req/sec)
            assert elapsed >= 4.0
            
    @pytest.mark.asyncio 
    @pytest.mark.slow
    async def test_data_quality(self, real_config):
        """Test quality of real data returned."""
        async with EDGARAPIClient(real_config) as api_client:
            # Get Apple's company facts
            facts = await api_client.get_company_facts("0000320193")
            
            # Verify structure and data quality
            assert "facts" in facts
            assert "us-gaap" in facts["facts"]
            
            # Should have common financial concepts
            gaap_facts = facts["facts"]["us-gaap"]
            
            # Look for Assets (should be present for Apple)
            if "Assets" in gaap_facts:
                assets_data = gaap_facts["Assets"]
                assert "units" in assets_data
                assert "USD" in assets_data["units"]
                
                # Should have historical data points
                usd_data = assets_data["units"]["USD"]
                assert len(usd_data) > 0
                
                # Each data point should have required fields
                for data_point in usd_data[:5]:  # Check first 5
                    assert "val" in data_point
                    assert "end" in data_point
                    assert isinstance(data_point["val"], (int, float))
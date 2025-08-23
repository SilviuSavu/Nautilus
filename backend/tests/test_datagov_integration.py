"""
Test Data.gov Integration
========================

Comprehensive test suite for Data.gov CKAN API integration.
Following pytest patterns used in other integration tests.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from datagov_integration import DatagovIntegrationService
from datagov_connector import (
    DatagovAPIClient,
    DatagovInstrumentProvider,
    DatagovDataset,
    DatasetCategory,
    DatasetFrequency,
    DatagovError
)


# Test fixtures
@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def mock_dataset_data():
    """Mock dataset data from CKAN API."""
    return {
        "id": "test-dataset-001",
        "name": "federal-reserve-interest-rates",
        "title": "Federal Reserve Interest Rate Data",
        "notes": "Daily interest rate data from the Federal Reserve",
        "author": "Federal Reserve",
        "organization": {
            "id": "federal-reserve",
            "name": "Federal Reserve",
            "title": "Federal Reserve System"
        },
        "tags": ["economics", "finance", "interest-rates", "monetary-policy"],
        "resources": [
            {
                "id": "resource-001",
                "name": "Interest Rates CSV",
                "url": "https://api.example.com/rates.csv",
                "format": "csv",
                "description": "CSV file with daily interest rates"
            },
            {
                "id": "resource-002", 
                "name": "Interest Rates API",
                "url": "https://api.example.com/rates/api",
                "format": "api",
                "description": "REST API for interest rate data"
            }
        ],
        "metadata_created": "2023-01-01T00:00:00.000Z",
        "metadata_modified": "2024-01-01T00:00:00.000Z"
    }


@pytest.fixture
def mock_search_response():
    """Mock CKAN search response."""
    return {
        "success": True,
        "result": {
            "count": 1,
            "results": [
                {
                    "id": "test-dataset-001",
                    "name": "federal-reserve-interest-rates",
                    "title": "Federal Reserve Interest Rate Data",
                    "notes": "Daily interest rate data from the Federal Reserve",
                    "tags": [{"name": "economics"}, {"name": "finance"}],
                    "organization": {
                        "id": "federal-reserve",
                        "name": "Federal Reserve",
                        "title": "Federal Reserve System"
                    },
                    "resources": []
                }
            ]
        }
    }


@pytest.fixture
def mock_organizations_response():
    """Mock organizations list response."""
    return {
        "success": True,
        "result": [
            {
                "id": "federal-reserve",
                "name": "Federal Reserve", 
                "title": "Federal Reserve System",
                "description": "Central banking system of the United States"
            },
            {
                "id": "department-of-agriculture",
                "name": "Department of Agriculture",
                "title": "U.S. Department of Agriculture",
                "description": "Federal executive department responsible for agriculture"
            }
        ]
    }


class TestDatagovAPIClient:
    """Test Data.gov API client functionality."""

    @pytest.mark.asyncio
    async def test_api_client_initialization(self, mock_api_key):
        """Test API client initialization."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        assert client.config.api_key == mock_api_key
        assert client.config.base_url == "https://api.gsa.gov/technology/datagov/v3/"
        assert client.cache.maxsize == config.cache_max_size
        assert client._request_times == []
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_api_key):
        """Test health check functionality."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        # Mock successful response
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {"success": True, "result": {"count": 100}}
            
            health = await client.health_check()
            assert health is True
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_datasets(self, mock_api_key, mock_search_response):
        """Test dataset search functionality."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = mock_search_response
            
            results = await client.search_datasets(query="economics")
            
            assert results.count == 1
            assert len(results.results) == 1
            assert results.results[0].title == "Federal Reserve Interest Rate Data"
            assert results.results[0].category == DatasetCategory.ECONOMIC
    
    @pytest.mark.asyncio
    async def test_get_dataset(self, mock_api_key, mock_dataset_data):
        """Test getting individual dataset."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {"success": True, "result": mock_dataset_data}
            
            dataset = await client.get_dataset("test-dataset-001")
            
            assert dataset is not None
            assert dataset.id == "test-dataset-001"
            assert dataset.title == "Federal Reserve Interest Rate Data"
            assert len(dataset.resources) == 2
            assert dataset.category == DatasetCategory.ECONOMIC
    
    @pytest.mark.asyncio
    async def test_list_organizations(self, mock_api_key, mock_organizations_response):
        """Test listing organizations."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = mock_organizations_response
            
            orgs = await client.list_organizations()
            
            assert len(orgs) == 2
            assert orgs[0].name == "Federal Reserve"
            assert orgs[1].id == "department-of-agriculture"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_api_key):
        """Test rate limiting functionality."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        config.rate_limit_requests_per_second = 2.0  # 2 requests per second for testing
        
        client = DatagovAPIClient(config)
        
        # Mock sleep to track rate limiting
        with patch('asyncio.sleep') as mock_sleep:
            with patch.object(client, '_ensure_session'):
                with patch.object(client.session, 'get') as mock_get:
                    mock_response = Mock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={"success": True})
                    mock_get.return_value.__aenter__.return_value = mock_response
                    
                    # Make multiple requests rapidly
                    tasks = []
                    for i in range(3):
                        task = client._make_request("test/endpoint", {})
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks)
                    
                    # Should have triggered rate limiting (sleep)
                    assert len(client._request_times) == 3
    
    @pytest.mark.asyncio
    async def test_caching(self, mock_api_key):
        """Test caching functionality."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_ensure_session'):
            with patch.object(client.session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"success": True, "data": "test"})
                mock_get.return_value.__aenter__.return_value = mock_response
                
                # First request - should hit API
                result1 = await client._make_request("test/endpoint", {"param": "value"})
                
                # Second request - should hit cache
                result2 = await client._make_request("test/endpoint", {"param": "value"})
                
                assert result1 == result2
                assert mock_get.call_count == 1  # Only one actual HTTP request
                assert client._stats['cache_hits'] == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_api_key):
        """Test error handling and retries."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        config.max_retries = 2
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_ensure_session'):
            with patch.object(client.session, 'get') as mock_get:
                # Mock 500 error response
                mock_response = Mock()
                mock_response.status = 500
                mock_response.text = AsyncMock(return_value="Internal Server Error")
                mock_get.return_value.__aenter__.return_value = mock_response
                
                with pytest.raises(DatagovError) as exc_info:
                    await client._make_request("test/endpoint", {})
                
                assert "API request failed" in str(exc_info.value)
                assert exc_info.value.status_code == 500


class TestDatagovInstrumentProvider:
    """Test Data.gov instrument provider."""

    @pytest.mark.asyncio
    async def test_instrument_provider_initialization(self, mock_api_key):
        """Test instrument provider initialization."""
        from datagov_connector.config import create_default_config, create_instrument_config
        
        api_config = create_default_config(mock_api_key)
        api_client = DatagovAPIClient(api_config)
        instrument_config = create_instrument_config()
        
        provider = DatagovInstrumentProvider(api_client, instrument_config)
        
        assert provider.api_client == api_client
        assert provider.config == instrument_config
        assert provider._datasets == {}
        assert not provider._is_loaded
    
    @pytest.mark.asyncio
    async def test_load_datasets(self, mock_api_key, mock_search_response):
        """Test loading datasets."""
        from datagov_connector.config import create_default_config, create_instrument_config
        
        api_config = create_default_config(mock_api_key)
        api_client = DatagovAPIClient(api_config)
        instrument_config = create_instrument_config()
        
        provider = DatagovInstrumentProvider(api_client, instrument_config)
        
        with patch.object(api_client, 'search_datasets') as mock_search:
            from datagov_connector.data_types import DatasetSearchResult, DatagovDataset
            
            # Create mock dataset
            dataset = DatagovDataset(
                id="test-001",
                name="test-dataset",
                title="Test Economic Dataset",
                notes="Test dataset for economics",
                resources=[],
                tags=["economics", "finance"]
            )
            
            search_result = DatasetSearchResult(count=1, results=[dataset])
            mock_search.return_value = search_result
            
            await provider.load_all_async()
            
            assert provider._is_loaded
            assert len(provider._datasets) == 1
            assert "test-001" in provider._datasets
    
    @pytest.mark.asyncio
    async def test_search_datasets(self, mock_api_key):
        """Test searching datasets."""
        from datagov_connector.config import create_default_config, create_instrument_config
        from datagov_connector.data_types import DatagovDataset
        
        api_config = create_default_config(mock_api_key)
        api_client = DatagovAPIClient(api_config)
        instrument_config = create_instrument_config()
        
        provider = DatagovInstrumentProvider(api_client, instrument_config)
        
        # Add test dataset
        test_dataset = DatagovDataset(
            id="economics-001",
            name="federal-economic-data",
            title="Federal Economic Data",
            notes="Economic indicators from federal agencies",
            resources=[]
        )
        
        provider._datasets["economics-001"] = test_dataset
        provider._dataset_index["federal economic data"] = "economics-001"
        provider._is_loaded = True
        
        # Test search
        results = provider.search_datasets("economic")
        
        assert len(results) == 1
        assert results[0].title == "Federal Economic Data"
    
    def test_create_instrument_id(self, mock_api_key):
        """Test instrument ID creation."""
        from datagov_connector.config import create_default_config, create_instrument_config
        from datagov_connector.data_types import DatagovDataset
        
        api_config = create_default_config(mock_api_key)
        api_client = DatagovAPIClient(api_config)
        instrument_config = create_instrument_config()
        
        provider = DatagovInstrumentProvider(api_client, instrument_config)
        
        dataset = DatagovDataset(
            id="test-001",
            name="federal-reserve-rates",
            title="Federal Reserve Interest Rates",
            resources=[]
        )
        
        instrument_id = provider.create_instrument_id(dataset)
        
        assert instrument_id.endswith(".DATAGOV")
        assert "federal_reserve_rates" in instrument_id.lower()


class TestDatagovIntegrationService:
    """Test main Data.gov integration service."""

    @patch.dict('os.environ', {'DATAGOV_API_KEY': 'test-key-123'})
    def test_service_initialization_with_api_key(self):
        """Test service initialization with API key."""
        service = DatagovIntegrationService()
        
        assert service.api_key == 'test-key-123'
        assert service.api_client is not None
        assert service.instrument_provider is not None
    
    @patch.dict('os.environ', {}, clear=True)  # Clear environment
    def test_service_initialization_without_api_key(self):
        """Test service initialization without API key."""
        service = DatagovIntegrationService()
        
        assert service.api_key is None
        assert service.api_client is None
        assert service.instrument_provider is None
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'DATAGOV_API_KEY': 'test-key-123'})
    async def test_health_check_configured(self):
        """Test health check with API key configured."""
        service = DatagovIntegrationService()
        
        with patch.object(service.api_client, 'health_check') as mock_health:
            mock_health.return_value = True
            
            health = await service.health_check()
            
            assert health['service'] == "Data.gov Integration"
            assert health['api_key_configured'] is True
            assert health['status'] == "operational"
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {}, clear=True)
    async def test_health_check_not_configured(self):
        """Test health check without API key."""
        service = DatagovIntegrationService()
        
        health = await service.health_check()
        
        assert health['status'] == "not_configured"
        assert health['api_key_configured'] is False
        assert "DATAGOV_API_KEY" in health['error_message']
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'DATAGOV_API_KEY': 'test-key-123'})
    async def test_search_datasets(self, mock_search_response):
        """Test dataset search through service."""
        service = DatagovIntegrationService()
        
        with patch.object(service.api_client, 'search_datasets') as mock_search:
            from datagov_connector.data_types import DatasetSearchResult, DatagovDataset
            
            dataset = DatagovDataset(
                id="test-001",
                name="test-dataset", 
                title="Test Dataset",
                resources=[]
            )
            
            search_result = DatasetSearchResult(count=1, results=[dataset])
            mock_search.return_value = search_result
            
            result = await service.search_datasets(query="economics")
            
            assert result['success'] is True
            assert result['total_count'] == 1
            assert len(result['datasets']) == 1


class TestDatagovRoutes:
    """Test Data.gov API routes (integration tests)."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from datagov_routes import datagov_health_check
        
        with patch('datagov_routes.datagov_integration') as mock_integration:
            mock_integration.health_check.return_value = {
                'service': 'Data.gov Integration',
                'status': 'operational',
                'api_key_configured': True,
                'api_accessible': True,
                'datasets_loaded': 100,
                'timestamp': '2024-01-01T00:00:00'
            }
            
            response = await datagov_health_check()
            
            assert response.service == 'Data.gov Integration'
            assert response.status == 'operational'
            assert response.api_key_configured is True
    
    @pytest.mark.asyncio
    async def test_search_endpoint(self):
        """Test dataset search endpoint."""
        from datagov_routes import search_datasets
        
        with patch('datagov_routes.datagov_integration') as mock_integration:
            mock_integration.search_datasets.return_value = {
                'success': True,
                'query': 'economics',
                'total_count': 1,
                'returned_count': 1,
                'datasets': [
                    {
                        'id': 'test-001',
                        'name': 'test-dataset',
                        'title': 'Test Dataset',
                        'description': 'Test description',
                        'category': 'economic',
                        'organization': {'name': 'Test Org'},
                        'tags': ['economics'],
                        'resource_count': 1,
                        'data_resources': 1,
                        'api_resources': 0,
                        'estimated_frequency': 'daily',
                        'trading_relevant': True,
                        'created': '2024-01-01T00:00:00',
                        'modified': '2024-01-01T00:00:00',
                        'url': 'https://example.com/dataset'
                    }
                ],
                'facets': {},
                'timestamp': '2024-01-01T00:00:00'
            }
            
            response = await search_datasets(q="economics")
            
            assert response.success is True
            assert response.query == 'economics'
            assert len(response.datasets) == 1


# Performance tests
class TestDatagovPerformance:
    """Performance and load testing."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_api_key):
        """Test handling concurrent requests."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {"success": True, "result": {"count": 1}}
            
            # Run 10 concurrent health checks
            tasks = [client.health_check() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(results)
            assert mock_request.call_count == 10
    
    @pytest.mark.asyncio
    async def test_large_dataset_loading(self, mock_api_key):
        """Test loading large numbers of datasets."""
        from datagov_connector.config import create_default_config, create_instrument_config
        from datagov_connector.data_types import DatasetSearchResult, DatagovDataset
        
        api_config = create_default_config(mock_api_key)
        api_client = DatagovAPIClient(api_config)
        instrument_config = create_instrument_config()
        
        provider = DatagovInstrumentProvider(api_client, instrument_config)
        
        # Mock large dataset response
        datasets = []
        for i in range(100):  # 100 datasets
            dataset = DatagovDataset(
                id=f"dataset-{i:03d}",
                name=f"dataset-{i}",
                title=f"Dataset {i}",
                resources=[]
            )
            datasets.append(dataset)
        
        with patch.object(api_client, 'search_datasets') as mock_search:
            search_result = DatasetSearchResult(count=100, results=datasets)
            mock_search.return_value = search_result
            
            start_time = datetime.now()
            await provider.load_all_async()
            end_time = datetime.now()
            
            # Should complete within reasonable time
            assert (end_time - start_time).total_seconds() < 5
            assert len(provider._datasets) == 100


# Error handling tests
class TestDatagovErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, mock_api_key):
        """Test handling of API timeouts."""
        from datagov_connector.config import create_default_config
        import aiohttp
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_ensure_session'):
            with patch.object(client.session, 'get') as mock_get:
                mock_get.side_effect = asyncio.TimeoutError("Request timeout")
                
                with pytest.raises(DatagovError) as exc_info:
                    await client._make_request("test/endpoint", {})
                
                assert "failed after" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, mock_api_key):
        """Test handling of malformed API responses."""
        from datagov_connector.config import create_default_config
        
        config = create_default_config(mock_api_key)
        client = DatagovAPIClient(config)
        
        with patch.object(client, '_make_request') as mock_request:
            # Malformed response missing required fields
            mock_request.return_value = {"success": True, "result": {"malformed": True}}
            
            with pytest.raises(Exception):  # Should handle gracefully
                await client.search_datasets(query="test")
    
    def test_invalid_dataset_id_validation(self):
        """Test dataset ID validation."""
        from datagov_connector.utils import validate_dataset_id
        
        # Valid IDs
        assert validate_dataset_id("valid-dataset-id") == "valid-dataset-id"
        assert validate_dataset_id("dataset_with_underscores") == "dataset_with_underscores"
        
        # Invalid IDs
        with pytest.raises(ValueError):
            validate_dataset_id("")
        
        with pytest.raises(ValueError):
            validate_dataset_id("invalid@dataset#id")
    
    def test_trading_relevance_edge_cases(self):
        """Test trading relevance assessment with edge cases."""
        from datagov_connector.utils import assess_trading_relevance
        
        # Empty dataset
        empty_dataset = {}
        relevance = assess_trading_relevance(empty_dataset)
        assert 0.0 <= relevance <= 1.0
        
        # Dataset with minimal information
        minimal_dataset = {"title": "Test", "notes": "", "tags": []}
        relevance = assess_trading_relevance(minimal_dataset)
        assert 0.0 <= relevance <= 1.0
        
        # Highly relevant dataset
        trading_dataset = {
            "title": "Federal Reserve Interest Rates and Monetary Policy Data",
            "notes": "Daily interest rates, inflation data, and GDP indicators",
            "tags": ["economics", "finance", "trading", "market-data"],
            "organization": {"name": "federal-reserve"},
            "resources": [{"format": "json"}, {"format": "api"}]
        }
        relevance = assess_trading_relevance(trading_dataset)
        assert relevance > 0.5  # Should be highly relevant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
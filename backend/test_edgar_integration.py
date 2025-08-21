#!/usr/bin/env python3
"""
EDGAR Integration Test
=====================

Test script to validate EDGAR connector integration with the main FastAPI application.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_edgar_imports():
    """Test that all EDGAR components can be imported."""
    print("=== Testing EDGAR Imports ===")
    
    try:
        from edgar_connector import (
            EDGARAPIClient,
            EDGARConfig,
            EDGARDataClient, 
            EDGARInstrumentProvider,
            FilingData,
            CompanyFacts,
            SECFiling,
            FilingType
        )
        print("✓ EDGAR connector components imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    try:
        from edgar_routes import router
        print("✓ EDGAR routes imported successfully")
    except ImportError as e:
        print(f"✗ EDGAR routes import failed: {e}")
        return False
    
    return True


async def test_edgar_configuration():
    """Test EDGAR configuration creation."""
    print("\n=== Testing EDGAR Configuration ===")
    
    try:
        from edgar_connector.config import create_default_config, EDGARDataClientConfig
        
        # Test basic config creation
        config = create_default_config(
            user_agent="TestAgent test@example.com"
        )
        print(f"✓ Basic config created: rate_limit={config.rate_limit_requests_per_second}")
        
        # Test data client config
        data_config = EDGARDataClientConfig()
        print(f"✓ Data client config created: check_interval={data_config.subscription_check_interval}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


async def test_edgar_data_types():
    """Test EDGAR data type creation."""
    print("\n=== Testing EDGAR Data Types ===")
    
    try:
        from edgar_connector.data_types import (
            FilingType, SECEntity, create_filing_data, create_company_facts
        )
        from datetime import datetime
        
        # Test FilingType enum
        assert FilingType.FORM_10K == "10-K"
        print("✓ FilingType enum works correctly")
        
        # Test SECEntity
        entity = SECEntity(
            cik="0000320193",
            name="Apple Inc.",
            ticker="AAPL"
        )
        print(f"✓ SECEntity created: {entity.name} ({entity.ticker})")
        
        # Test FilingData creation
        filing = create_filing_data(
            filing_type="10-K",
            cik="0000320193",
            company_name="Apple Inc.",
            accession_number="0000320193-23-000123",
            filing_date=datetime.now()
        )
        print(f"✓ FilingData created: {filing.filing_type} for {filing.company_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data types test failed: {e}")
        return False


async def test_edgar_utilities():
    """Test EDGAR utility functions.""" 
    print("\n=== Testing EDGAR Utilities ===")
    
    try:
        from edgar_connector.utils import (
            normalize_cik, normalize_ticker, format_financial_value,
            XBRLParser, DataCache
        )
        from decimal import Decimal
        
        # Test normalization functions
        assert normalize_cik("123") == "0000000123"
        assert normalize_ticker("  aapl  ") == "AAPL"
        print("✓ Normalization functions work correctly")
        
        # Test financial formatting
        formatted = format_financial_value(1234567890, "USD")
        assert "B" in formatted  # Should format as billions
        print(f"✓ Financial formatting works: {formatted}")
        
        # Test XBRL parser
        parser = XBRLParser()
        assert parser is not None
        print("✓ XBRL parser initialized")
        
        # Test cache
        cache = DataCache()
        cache.set("test_key", {"test": "data"})
        cached_data = cache.get("test_key")
        assert cached_data == {"test": "data"}
        print("✓ Data cache works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


async def test_edgar_api_client():
    """Test EDGAR API client initialization (no actual API calls)."""
    print("\n=== Testing EDGAR API Client ===")
    
    try:
        from edgar_connector.api_client import EDGARAPIClient, RateLimiter
        from edgar_connector.config import create_default_config
        
        # Test rate limiter
        limiter = RateLimiter(10.0)
        assert limiter.max_requests_per_second == 10.0
        print("✓ Rate limiter initialized")
        
        # Test API client initialization
        config = create_default_config(
            user_agent="TestAgent test@example.com"
        )
        
        client = EDGARAPIClient(config)
        assert client.base_url == config.base_url
        assert client.config == config
        print("✓ API client initialized")
        
        # Clean up
        await client.close()
        print("✓ API client closed properly")
        
        return True
        
    except Exception as e:
        print(f"✗ API client test failed: {e}")
        return False


async def test_edgar_instrument_provider():
    """Test EDGAR instrument provider initialization."""
    print("\n=== Testing EDGAR Instrument Provider ===")
    
    try:
        from edgar_connector.instrument_provider import EDGARInstrumentProvider
        from edgar_connector.config import EDGARInstrumentConfig
        from unittest.mock import AsyncMock
        
        # Mock API client
        mock_api_client = AsyncMock()
        mock_api_client.get_company_tickers.return_value = {
            "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}
        }
        
        # Test instrument provider
        config = EDGARInstrumentConfig(
            update_entities_on_startup=False  # Don't make real API calls
        )
        
        provider = EDGARInstrumentProvider(mock_api_client, config)
        assert provider is not None
        print("✓ Instrument provider initialized")
        
        # Test utility methods
        assert provider.get_entity_count() == 0  # No entities loaded yet
        assert provider.is_valid_ticker("AAPL") is False  # No entities loaded
        print("✓ Instrument provider utility methods work")
        
        return True
        
    except Exception as e:
        print(f"✗ Instrument provider test failed: {e}")
        return False


async def test_route_registration():
    """Test that EDGAR routes can be registered with FastAPI."""
    print("\n=== Testing Route Registration ===")
    
    try:
        from fastapi import FastAPI
        from edgar_routes import router
        
        # Create test app
        app = FastAPI()
        app.include_router(router)
        
        # Check that routes were added
        edgar_routes = [route for route in app.routes if hasattr(route, 'path') and '/edgar' in route.path]
        print(f"✓ {len(edgar_routes)} EDGAR routes registered")
        
        # Check for key endpoints
        route_paths = [route.path for route in edgar_routes]
        expected_routes = [
            "/api/v1/edgar/health",
            "/api/v1/edgar/companies/search", 
            "/api/v1/edgar/ticker/{ticker}/resolve"
        ]
        
        for expected_route in expected_routes:
            if any(expected_route.replace('{ticker}', 'AAPL') in path or expected_route in path for path in route_paths):
                print(f"✓ Found route: {expected_route}")
            else:
                print(f"? Route pattern not found: {expected_route}")
        
        return True
        
    except Exception as e:
        print(f"✗ Route registration test failed: {e}")
        return False


async def test_dependencies_available():
    """Test that required dependencies are available."""
    print("\n=== Testing Dependencies ===")
    
    dependencies = [
        ("httpx", "HTTP client"),
        ("pydantic", "Data validation"),
        ("fastapi", "Web framework")
    ]
    
    all_available = True
    
    for dep_name, desc in dependencies:
        try:
            __import__(dep_name)
            print(f"✓ {desc} ({dep_name}) available")
        except ImportError:
            print(f"✗ {desc} ({dep_name}) NOT available")
            all_available = False
    
    return all_available


async def main():
    """Run all tests."""
    print("EDGAR API Connector Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_edgar_imports),
        ("Configuration Tests", test_edgar_configuration),
        ("Data Types Tests", test_edgar_data_types),
        ("Utilities Tests", test_edgar_utilities),
        ("API Client Tests", test_edgar_api_client),
        ("Instrument Provider Tests", test_edgar_instrument_provider),
        ("Route Registration Tests", test_route_registration),
        ("Dependencies Tests", test_dependencies_available),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! EDGAR connector is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
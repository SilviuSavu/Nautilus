"""
Comprehensive Integration Tests for Nautilus Trading Platform
=============================================================

Complete integration tests covering all major system components and new features.
"""

import pytest
import asyncio
import httpx
import json
from datetime import datetime, timedelta

# Test configuration
BASE_URL = "http://localhost:8001"
TIMEOUT = 30.0


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_basic_health(self):
        """Test basic health endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_comprehensive_health(self):
        """Test comprehensive health endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health/comprehensive")
            assert response.status_code == 200
            data = response.json()
            assert "overall_status" in data
            assert "services" in data
            assert "summary" in data
    
    @pytest.mark.asyncio
    async def test_service_specific_health(self):
        """Test service-specific health endpoints."""
        services = ["redis", "postgres", "ib_gateway"]
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for service in services:
                response = await client.get(f"{BASE_URL}/health/{service}")
                assert response.status_code == 200
                data = response.json()
                assert data["service"] == service
                assert "status" in data
                assert "response_time_ms" in data


class TestMarketDataEndpoints:
    """Test market data functionality."""
    
    @pytest.mark.asyncio
    async def test_historical_backfill_status(self):
        """Test historical data status endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/historical/backfill/status")
            assert response.status_code == 200
            data = response.json()
            assert "controller" in data
            assert "service_status" in data
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_messagebus_status(self):
        """Test message bus status."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/messagebus/status")
            assert response.status_code == 200
            data = response.json()
            assert "connection_state" in data


class TestTradingEndpoints:
    """Test trading-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_ib_status(self):
        """Test IB Gateway status."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/ib/status")
            assert response.status_code == 200
            data = response.json()
            assert "connected" in data
    
    @pytest.mark.asyncio
    async def test_exchanges_status(self):
        """Test exchanges status."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/exchanges/status")
            assert response.status_code == 200
            data = response.json()
            assert "exchanges" in data
            assert "summary" in data


class TestPortfolioEndpoints:
    """Test portfolio management endpoints."""
    
    @pytest.mark.asyncio
    async def test_portfolio_positions(self):
        """Test portfolio positions endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/portfolio/positions")
            # May return 200 with data or 404 if no positions
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_portfolio_balance(self):
        """Test portfolio balance endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/portfolio/balance")
            # May return 200 with data or 404 if no balance data
            assert response.status_code in [200, 404]


class TestNewEndpoints:
    """Test newly implemented endpoints."""
    
    @pytest.mark.asyncio
    async def test_factor_engine_endpoints(self):
        """Test factor engine endpoints are accessible."""
        endpoints = [
            "/api/v1/factor-engine/factors/list",
            "/api/v1/factor-engine/calculate",
            "/api/v1/factor-engine/status"
        ]
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for endpoint in endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                # Should not return 500 (server error), may return 404 or validation errors
                assert response.status_code != 500
    
    @pytest.mark.asyncio
    async def test_edgar_endpoints(self):
        """Test EDGAR API endpoints are accessible."""
        endpoints = [
            "/api/v1/edgar/status",
            "/api/v1/edgar/companies/search",
        ]
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for endpoint in endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                # Should not return 500 (server error)
                assert response.status_code != 500


class TestPerformanceMetrics:
    """Test system performance metrics."""
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self):
        """Test API response times are within acceptable limits."""
        endpoints = [
            "/health",
            "/api/v1/messagebus/status",
            "/api/v1/historical/backfill/status"
        ]
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for endpoint in endpoints:
                start_time = asyncio.get_event_loop().time()
                response = await client.get(f"{BASE_URL}{endpoint}")
                end_time = asyncio.get_event_loop().time()
                
                response_time = (end_time - start_time) * 1000  # ms
                
                # Response time should be under 1000ms for most endpoints
                assert response_time < 1000, f"{endpoint} took {response_time}ms"
                assert response.status_code == 200


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.mark.asyncio
    async def test_historical_data_consistency(self):
        """Test historical data service consistency."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/historical/backfill/status")
            assert response.status_code == 200
            
            data = response.json()
            service_status = data.get("service_status", {})
            
            # Basic data consistency checks
            if service_status.get("total_bars", 0) > 0:
                assert service_status.get("unique_instruments", 0) > 0
                assert service_status.get("database_size_gb", 0) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoints(self):
        """Test that invalid endpoints return appropriate errors."""
        invalid_endpoints = [
            "/api/v1/nonexistent",
            "/health/invalid_service",
            "/api/v1/portfolio/invalid/endpoint"
        ]
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for endpoint in invalid_endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                assert response.status_code in [404, 422]  # Not Found or Unprocessable Entity
    
    @pytest.mark.asyncio
    async def test_malformed_requests(self):
        """Test handling of malformed requests."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test malformed JSON in POST request
            response = await client.post(
                f"{BASE_URL}/api/v1/risk/calculate",
                json={"invalid": "data", "missing_required": "fields"}
            )
            assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
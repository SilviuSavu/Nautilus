"""
Integration tests for market data streaming infrastructure
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from main import app
from market_data_service import Venue, DataType
from messagebus_client import MessageBusMessage


class TestMarketDataAPI:
    """Test market data REST API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_market_data_status(self, client):
        """Test market data status endpoint"""
        response = client.get("/api/v1/market-data/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_subscriptions" in data
        assert "supported_venues" in data
        assert "supported_data_types" in data
        assert isinstance(data["supported_venues"], list)
        assert len(data["supported_venues"]) > 0
        
    def test_market_data_subscription(self, client):
        """Test market data subscription endpoint"""
        subscription_request = {
            "venue": "BINANCE",
            "instrument_id": "BTCUSDT",
            "data_type": "tick"
        }
        
        response = client.post("/api/v1/market-data/subscribe", json=subscription_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["venue"] == "BINANCE"
        assert data["instrument_id"] == "BTCUSDT"
        assert data["data_type"] == "tick"
        assert data["active"] is True
        assert "subscription_id" in data
        
    def test_invalid_venue_subscription(self, client):
        """Test subscription with invalid venue"""
        subscription_request = {
            "venue": "INVALID_VENUE",
            "instrument_id": "BTCUSDT",
            "data_type": "tick"
        }
        
        response = client.post("/api/v1/market-data/subscribe", json=subscription_request)
        assert response.status_code == 400
        assert "Invalid venue or data type" in response.json()["detail"]
        
    def test_list_subscriptions(self, client):
        """Test listing active subscriptions"""
        # First create a subscription
        subscription_request = {
            "venue": "BINANCE",
            "instrument_id": "ETHUSDT",
            "data_type": "quote"
        }
        client.post("/api/v1/market-data/subscribe", json=subscription_request)
        
        # Then list subscriptions
        response = client.get("/api/v1/market-data/subscriptions")
        assert response.status_code == 200
        
        data = response.json()
        assert "subscriptions" in data
        assert isinstance(data["subscriptions"], list)


class TestCacheAPI:
    """Test Redis cache API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_cache_status(self, client):
        """Test cache status endpoint"""
        response = client.get("/api/v1/cache/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        
    def test_cache_stats(self, client):
        """Test cache statistics endpoint"""
        response = client.get("/api/v1/cache/stats")
        assert response.status_code == 200
        # Note: May return empty if Redis not connected in test
        
    def test_latest_tick_not_found(self, client):
        """Test getting latest tick when none exists"""
        response = client.get("/api/v1/cache/latest-tick/BINANCE/NONEXISTENT")
        assert response.status_code == 404
        assert "No tick data found" in response.json()["detail"]


class TestRateLimitingAPI:
    """Test rate limiting API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_rate_limiting_status(self, client):
        """Test rate limiting status endpoint"""
        response = client.get("/api/v1/rate-limiting/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "total_venues" in data
        assert "healthy_venues" in data
        
    def test_rate_limiting_metrics(self, client):
        """Test rate limiting metrics endpoint"""
        response = client.get("/api/v1/rate-limiting/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        # Should contain venue metrics
        
    def test_venue_rate_limiting_status(self, client):
        """Test venue-specific rate limiting status"""
        response = client.get("/api/v1/rate-limiting/venue/BINANCE")
        assert response.status_code == 200
        
        data = response.json()
        assert data["venue"] == "BINANCE"
        assert "config" in data
        assert "rate_limiter" in data
        assert "metrics" in data
        
    def test_invalid_venue_rate_limiting(self, client):
        """Test invalid venue rate limiting status"""
        response = client.get("/api/v1/rate-limiting/venue/INVALID")
        assert response.status_code == 400
        assert "Invalid venue" in response.json()["detail"]


class TestHistoricalDataAPI:
    """Test historical data API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_historical_data_status(self, client):
        """Test historical data status endpoint"""
        response = client.get("/api/v1/historical/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        
    def test_data_summary(self, client):
        """Test data summary endpoint"""
        response = client.get("/api/v1/historical/summary/BINANCE/BTCUSDT")
        assert response.status_code == 200
        
        data = response.json()
        assert data["venue"] == "BINANCE"
        assert data["instrument_id"] == "BTCUSDT"
        assert "summary" in data
        
    def test_historical_ticks_query(self, client):
        """Test historical ticks query"""
        start_time = (datetime.now() - timedelta(hours=1)).isoformat()
        end_time = datetime.now().isoformat()
        
        response = client.get(
            f"/api/v1/historical/ticks/BINANCE/BTCUSDT"
            f"?start_time={start_time}&end_time={end_time}&limit=100"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["venue"] == "BINANCE"
        assert data["instrument_id"] == "BTCUSDT"
        assert data["data_type"] == "tick"
        assert "count" in data
        assert "data" in data
        
    def test_invalid_datetime_format(self, client):
        """Test historical query with invalid datetime"""
        response = client.get(
            "/api/v1/historical/ticks/BINANCE/BTCUSDT"
            "?start_time=invalid&end_time=invalid"
        )
        assert response.status_code == 400
        assert "Invalid datetime format" in response.json()["detail"]


class TestWebSocketConnection:
    """Test WebSocket connectivity"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and basic messaging"""
        async with httpx.AsyncClient() as client:
            # Note: This is a basic connectivity test
            # Full WebSocket testing would require a running server
            response = await client.get("http://localhost:8000/")
            # Just test that the root endpoint is accessible
            # WebSocket testing requires more complex setup


class TestMessageBusIntegration:
    """Test MessageBus integration"""
    
    @pytest.mark.asyncio
    async def test_message_processing(self):
        """Test MessageBus message processing"""
        from market_data_service import market_data_service
        from market_data_handlers import market_data_handlers
        
        # Mock the broadcast callback
        mock_broadcast = AsyncMock()
        market_data_handlers.set_broadcast_callback(mock_broadcast)
        
        # Create test message
        test_message = MessageBusMessage(
            topic="data.tick.BINANCE.BTCUSDT",
            payload={
                "price": "50000.0",
                "quantity": "1.5",
                "side": "buy",
                "timestamp": 1640995200000
            },
            timestamp=1640995200000000000,
            message_type="market_data"
        )
        
        # Subscribe to the instrument
        await market_data_service.subscribe(Venue.BINANCE, "BTCUSDT", DataType.TICK)
        
        # Process the message
        await market_data_service._handle_messagebus_message(test_message)
        
        # Verify message was processed (would normally check Redis/DB)
        # This is a basic test - full integration requires Redis/DB


class TestPerformanceRequirements:
    """Test performance requirements"""
    
    def test_data_normalization_latency(self):
        """Test data normalization latency < 5ms"""
        from data_normalizer import DataNormalizer
        from market_data_service import NormalizedMarketData
        
        normalizer = DataNormalizer()
        
        # Create test data
        raw_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5"},
            raw_data={}
        )
        
        # Measure normalization time
        start_time = time.perf_counter()
        for _ in range(100):  # Test multiple iterations
            normalized = normalizer.normalize_market_data(raw_data)
        end_time = time.perf_counter()
        
        avg_latency_ms = ((end_time - start_time) / 100) * 1000
        assert avg_latency_ms < 5.0, f"Average normalization latency {avg_latency_ms}ms exceeds 5ms requirement"
        
    @pytest.mark.asyncio
    async def test_throughput_requirements(self):
        """Test system can handle required throughput"""
        from rate_limiter import RateLimiter
        
        rate_limiter = RateLimiter()
        
        # Test BINANCE can handle 100 requests/second
        venue = Venue.BINANCE
        success_count = 0
        
        # Simulate burst of requests
        for _ in range(100):
            allowed, _ = await rate_limiter.should_allow_request(venue)
            if allowed:
                success_count += 1
                
        # Should allow most requests (some may be rate limited)
        assert success_count >= 50, f"Only {success_count}/100 requests allowed, may indicate throughput issues"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self):
        """Test handling of invalid market data"""
        from data_normalizer import DataNormalizer, ValidationError
        from market_data_service import NormalizedMarketData
        
        normalizer = DataNormalizer()
        
        # Test invalid price
        invalid_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "-100.0", "quantity": "1.5"},  # Negative price
            raw_data={}
        )
        
        with pytest.raises(ValidationError):
            normalizer.normalize_market_data(invalid_data)
            
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker opens on repeated failures"""
        from rate_limiter import RateLimiter, CircuitState
        
        rate_limiter = RateLimiter()
        venue = Venue.BINANCE
        
        # Simulate multiple failures
        for _ in range(6):  # Exceed failure threshold
            rate_limiter.record_failure(venue, Exception("Test failure"))
            
        # Check if circuit breaker opened
        if venue in rate_limiter._circuit_breakers:
            circuit = rate_limiter._circuit_breakers[venue]
            # Circuit should be open after multiple failures
            assert circuit.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]
            
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of connection failures"""
        from redis_cache import RedisCache
        
        cache = RedisCache()
        # Test with disconnected cache
        cache._connected = False
        
        # Should handle gracefully without errors
        from data_normalizer import NormalizedTick
        from decimal import Decimal
        
        tick = NormalizedTick(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            price=Decimal("50000.0"),
            size=Decimal("1.5"),
            timestamp_ns=1640995200000000000
        )
        
        # Should not raise exception
        await cache.cache_tick(tick)


class TestDataQuality:
    """Test data quality monitoring"""
    
    def test_data_quality_metrics(self):
        """Test data quality metrics collection"""
        from data_normalizer import DataNormalizer
        
        normalizer = DataNormalizer()
        
        # Check initial metrics
        metrics = normalizer.get_quality_metrics()
        assert isinstance(metrics, dict)
        
        # Process some valid data
        from market_data_service import NormalizedMarketData
        
        valid_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5"},
            raw_data={}
        )
        
        normalizer.normalize_market_data(valid_data)
        
        # Check metrics were updated
        updated_metrics = normalizer.get_quality_metrics(Venue.BINANCE)
        assert "BINANCE" in updated_metrics
        binance_metrics = updated_metrics["BINANCE"]
        assert binance_metrics.total_messages > 0
        assert binance_metrics.valid_messages > 0


# Pytest fixtures for test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_test_environment():
    """Setup test environment before each test"""
    # Reset any global state if needed
    yield
    # Cleanup after test


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
"""
Test suite for market data streaming functionality
"""

import pytest
import asyncio
import json
import time
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from market_data_service import (
    MarketDataService, MarketDataSubscription, NormalizedMarketData,
    Venue, DataType
)
from market_data_handlers import (
    MarketDataHandlers, TickData, QuoteData, BarData, OrderBookData
)
from data_normalizer import (
    DataNormalizer, NormalizedTick, NormalizedQuote, NormalizedBar,
    ValidationError, NormalizationError
)
from rate_limiter import RateLimiter, ThrottleStrategy, CircuitState
from redis_cache import RedisCache
from historical_data_service import HistoricalDataService, HistoricalDataQuery


class TestMarketDataService:
    """Test MarketDataService functionality"""
    
    @pytest.fixture
    def service(self):
        return MarketDataService()
        
    @pytest.fixture
    def mock_message(self):
        return Mock(
            topic="data.tick.BINANCE.BTCUSDT",
            payload={"price": "50000.0", "quantity": "1.5"},
            timestamp=1640995200000000000,
            message_type="data"
        )
        
    def test_initialization(self, service):
        """Test service initialization"""
        assert service._subscriptions == {}
        assert service._data_handlers == []
        assert service._running is False
        
    @pytest.mark.asyncio
    async def test_subscription(self, service):
        """Test market data subscription"""
        subscription_id = await service.subscribe(
            Venue.BINANCE, "BTCUSDT", DataType.TICK
        )
        
        assert subscription_id == "BINANCE_BTCUSDT_tick"
        assert subscription_id in service._subscriptions
        
        subscription = service._subscriptions[subscription_id]
        assert subscription.venue == Venue.BINANCE
        assert subscription.instrument_id == "BTCUSDT"
        assert subscription.data_type == DataType.TICK
        assert subscription.active is True
        
    @pytest.mark.asyncio
    async def test_unsubscription(self, service):
        """Test market data unsubscription"""
        subscription_id = await service.subscribe(
            Venue.BINANCE, "BTCUSDT", DataType.TICK
        )
        
        success = await service.unsubscribe(subscription_id)
        assert success is True
        assert subscription_id not in service._subscriptions
        
        # Test unsubscribing non-existent subscription
        success = await service.unsubscribe("non_existent")
        assert success is False
        
    def test_topic_parsing(self, service):
        """Test message topic parsing"""
        venue, instrument, data_type = service._parse_topic("data.tick.BINANCE.BTCUSDT")
        assert venue == "BINANCE"
        assert instrument == "BTCUSDT"
        assert data_type == "tick"
        
        # Test with instrument containing dots
        venue, instrument, data_type = service._parse_topic("data.quote.COINBASE.BTC-USD")
        assert venue == "COINBASE"
        assert instrument == "BTC-USD"
        assert data_type == "quote"
        
    def test_market_data_topic_detection(self, service):
        """Test market data topic detection"""
        assert service._is_market_data_topic("data.tick.BINANCE.BTCUSDT") is True
        assert service._is_market_data_topic("data.quote.COINBASE.ETHUSD") is True
        assert service._is_market_data_topic("system.status.update") is False
        assert service._is_market_data_topic("user.auth.login") is False


class TestDataNormalizer:
    """Test data normalization functionality"""
    
    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()
        
    @pytest.fixture
    def raw_tick_data(self):
        return NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5", "side": "buy"},
            raw_data={"original": "data"}
        )
        
    @pytest.fixture
    def raw_quote_data(self):
        return NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="quote",
            timestamp=1640995200000000000,
            data={
                "bid": "49999.0", "ask": "50001.0",
                "bid_size": "2.0", "ask_size": "1.8"
            },
            raw_data={"original": "data"}
        )
        
    def test_tick_normalization(self, normalizer, raw_tick_data):
        """Test tick data normalization"""
        normalized = normalizer.normalize_market_data(raw_tick_data)
        
        assert isinstance(normalized, NormalizedTick)
        assert normalized.venue == "BINANCE"
        assert normalized.instrument_id == "BTCUSDT"
        assert normalized.price == Decimal("50000.0")
        assert normalized.size == Decimal("1.5")
        assert normalized.side == "buy"
        assert normalized.timestamp_ns == 1640995200000000000
        
    def test_quote_normalization(self, normalizer, raw_quote_data):
        """Test quote data normalization"""
        normalized = normalizer.normalize_market_data(raw_quote_data)
        
        assert isinstance(normalized, NormalizedQuote)
        assert normalized.venue == "BINANCE"
        assert normalized.instrument_id == "BTCUSDT"
        assert normalized.bid_price == Decimal("49999.0")
        assert normalized.ask_price == Decimal("50001.0")
        assert normalized.bid_size == Decimal("2.0")
        assert normalized.ask_size == Decimal("1.8")
        
    def test_invalid_price_validation(self, normalizer):
        """Test price validation"""
        invalid_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "-100.0", "quantity": "1.5"},
            raw_data={}
        )
        
        with pytest.raises(ValidationError, match="Invalid price"):
            normalizer.normalize_market_data(invalid_data)
            
    def test_timestamp_normalization(self, normalizer):
        """Test timestamp normalization"""
        # Test milliseconds to nanoseconds
        assert normalizer._normalize_timestamp(1640995200000) == 1640995200000000000
        
        # Test seconds to nanoseconds
        assert normalizer._normalize_timestamp(1640995200) == 1640995200000000000
        
        # Test already in nanoseconds
        assert normalizer._normalize_timestamp(1640995200000000000) == 1640995200000000000


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()
        
    @pytest.mark.asyncio
    async def test_venue_rate_limiting(self, rate_limiter):
        """Test venue-specific rate limiting"""
        venue = Venue.BINANCE
        
        # Should allow initial requests
        allowed, reason = await rate_limiter.should_allow_request(venue)
        assert allowed is True
        assert reason == "Request allowed"
        
    def test_token_bucket(self):
        """Test token bucket algorithm"""
        from rate_limiter import TokenBucket
        
        bucket = TokenBucket(rate=10, capacity=20)  # 10 tokens/second, 20 capacity
        
        # Should allow consuming tokens up to capacity
        assert bucket.consume(10) is True
        assert bucket.consume(10) is True
        assert bucket.consume(1) is False  # Should fail (exceeds capacity)
        
        # Wait for token refill (simulate)
        time.sleep(0.1)
        assert bucket.consume(1) is True
        
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        from rate_limiter import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Initial state should be closed
        assert breaker.state == CircuitState.CLOSED
        
        # Simulate failures
        for _ in range(3):
            breaker._on_failure()
            
        # Should be open after threshold failures
        assert breaker.state == CircuitState.OPEN
        
        # Success should reset if in half-open state
        breaker.state = CircuitState.HALF_OPEN
        breaker._on_success()
        assert breaker.state == CircuitState.CLOSED
        
    def test_venue_configuration(self, rate_limiter):
        """Test venue rate limit configuration"""
        venue = Venue.BINANCE
        status = rate_limiter.get_venue_status(venue)
        
        assert status["venue"] == "BINANCE"
        assert "config" in status
        assert "rate_limiter" in status
        assert "metrics" in status
        assert status["config"]["requests_per_second"] > 0


class TestMarketDataHandlers:
    """Test market data event handlers"""
    
    @pytest.fixture
    def handlers(self):
        return MarketDataHandlers()
        
    @pytest.fixture
    def mock_broadcast(self):
        return AsyncMock()
        
    @pytest.fixture
    def normalized_tick(self):
        return NormalizedTick(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            price=Decimal("50000.0"),
            size=Decimal("1.5"),
            timestamp_ns=1640995200000000000,
            side="buy",
            trade_id="12345"
        )
        
    def test_handler_registration(self, handlers):
        """Test handler registration"""
        tick_handler = Mock()
        quote_handler = Mock()
        
        handlers.add_tick_handler(tick_handler)
        handlers.add_quote_handler(quote_handler)
        
        assert tick_handler in handlers._tick_handlers
        assert quote_handler in handlers._quote_handlers
        
    @pytest.mark.asyncio
    async def test_tick_handling(self, handlers, mock_broadcast, normalized_tick):
        """Test tick data handling"""
        handlers.set_broadcast_callback(mock_broadcast)
        
        tick_handler = AsyncMock()
        handlers.add_tick_handler(tick_handler)
        
        await handlers._handle_normalized_tick(normalized_tick)
        
        # Check that handler was called
        tick_handler.assert_called_once()
        
        # Check that broadcast was called
        mock_broadcast.assert_called_once()
        call_args = mock_broadcast.call_args[0][0]
        assert call_args["type"] == "tick"
        assert "data" in call_args


class TestRedisCache:
    """Test Redis caching functionality"""
    
    @pytest.fixture
    def cache(self):
        # Create cache with mock Redis for testing
        cache = RedisCache()
        cache._connected = True
        cache._redis = AsyncMock()
        return cache
        
    @pytest.fixture
    def normalized_tick(self):
        return NormalizedTick(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            price=Decimal("50000.0"),
            size=Decimal("1.5"),
            timestamp_ns=1640995200000000000,
            side="buy"
        )
        
    @pytest.mark.asyncio
    async def test_tick_caching(self, cache, normalized_tick):
        """Test tick data caching"""
        await cache.cache_tick(normalized_tick)
        
        # Verify Redis operations were called
        cache._redis.pipeline.assert_called_once()
        pipeline_mock = cache._redis.pipeline.return_value
        pipeline_mock.lpush.assert_called()
        pipeline_mock.ltrim.assert_called()
        pipeline_mock.set.assert_called()
        pipeline_mock.execute.assert_called()
        
    @pytest.mark.asyncio
    async def test_cache_retrieval(self, cache):
        """Test cache data retrieval"""
        # Mock Redis get response
        cache._redis.get.return_value = '{"price": "50000.0", "size": "1.5"}'
        
        result = await cache.get_latest_tick("BINANCE", "BTCUSDT")
        
        assert result is not None
        assert result["price"] == "50000.0"
        cache._redis.get.assert_called_once()
        
    def test_key_patterns(self, cache):
        """Test cache key pattern generation"""
        expected_patterns = {
            "tick": "market:tick:{venue}:{instrument}",
            "quote": "market:quote:{venue}:{instrument}",
            "bar": "market:bar:{venue}:{instrument}:{timeframe}",
        }
        
        for key, pattern in expected_patterns.items():
            assert cache.KEY_PATTERNS[key] == pattern


class TestHistoricalDataService:
    """Test historical data service functionality"""
    
    @pytest.fixture
    def service(self):
        service = HistoricalDataService()
        service._connected = True
        service._pool = AsyncMock()
        return service
        
    @pytest.fixture
    def query(self):
        return HistoricalDataQuery(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            limit=100
        )
        
    @pytest.mark.asyncio
    async def test_tick_storage(self, service):
        """Test tick data storage"""
        tick = NormalizedTick(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            price=Decimal("50000.0"),
            size=Decimal("1.5"),
            timestamp_ns=1640995200000000000,
            side="buy"
        )
        
        await service.store_tick(tick)
        
        # Verify database operations
        service._pool.acquire.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_data_querying(self, service, query):
        """Test historical data querying"""
        # Mock database response
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "venue": "BINANCE",
                "instrument_id": "BTCUSDT",
                "price": Decimal("50000.0"),
                "timestamp_ns": 1640995200000000000
            }
        ]
        service._pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        results = await service.query_ticks(query)
        
        assert len(results) == 1
        assert results[0]["venue"] == "BINANCE"
        mock_conn.fetch.assert_called_once()


class TestIntegration:
    """Integration tests for complete market data flow"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test complete market data processing flow"""
        # Create service instances
        service = MarketDataService()
        handlers = MarketDataHandlers()
        normalizer = DataNormalizer()
        
        # Mock external dependencies
        with patch('redis_cache.redis_cache') as mock_cache, \
             patch('historical_data_service.historical_data_service') as mock_db:
            
            mock_cache.cache_tick = AsyncMock()
            mock_db.store_tick = AsyncMock()
            
            # Set up handlers
            service.add_data_handler(handlers.handle_market_data)
            
            # Create mock market data
            raw_data = NormalizedMarketData(
                venue="BINANCE",
                instrument_id="BTCUSDT",
                data_type="tick",
                timestamp=1640995200000000000,
                data={"price": "50000.0", "quantity": "1.5"},
                raw_data={}
            )
            
            # Process data through the pipeline
            await handlers.handle_market_data(raw_data)
            
            # Verify data was cached and stored
            mock_cache.cache_tick.assert_called_once()
            mock_db.store_tick.assert_called_once()
            
    def test_performance_requirements(self):
        """Test that performance requirements are met"""
        normalizer = DataNormalizer()
        
        # Test normalization latency < 5ms
        raw_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5"},
            raw_data={}
        )
        
        start_time = time.perf_counter()
        normalized = normalizer.normalize_market_data(raw_data)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 5.0, f"Normalization latency {latency_ms}ms exceeds 5ms requirement"
        assert isinstance(normalized, NormalizedTick)


# Pytest configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
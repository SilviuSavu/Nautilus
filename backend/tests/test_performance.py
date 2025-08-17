"""
Performance tests for market data streaming infrastructure
Tests performance requirements and benchmarks system components.
"""

import pytest
import asyncio
import time
import statistics
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from market_data_service import NormalizedMarketData, Venue, DataType
from data_normalizer import DataNormalizer, NormalizedTick, NormalizedQuote, NormalizedBar
from rate_limiter import RateLimiter
from market_data_handlers import MarketDataHandlers


class TestNormalizationPerformance:
    """Test data normalization performance requirements"""
    
    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()
        
    @pytest.fixture
    def sample_tick_data(self):
        return NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5", "side": "buy"},
            raw_data={}
        )
        
    def test_single_normalization_latency(self, normalizer, sample_tick_data):
        """Test single normalization latency < 5ms requirement"""
        start_time = time.perf_counter()
        normalized = normalizer.normalize_market_data(sample_tick_data)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 5.0, f"Normalization latency {latency_ms:.3f}ms exceeds 5ms requirement"
        assert isinstance(normalized, NormalizedTick)
        
    def test_batch_normalization_performance(self, normalizer, sample_tick_data):
        """Test batch normalization performance"""
        batch_size = 1000
        latencies = []
        
        for _ in range(batch_size):
            start_time = time.perf_counter()
            normalized = normalizer.normalize_market_data(sample_tick_data)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        print(f"\nNormalization Performance (n={batch_size}):")
        print(f"  Average latency: {avg_latency:.3f}ms")
        print(f"  95th percentile: {p95_latency:.3f}ms")
        print(f"  Max latency: {max_latency:.3f}ms")
        
        assert avg_latency < 5.0, f"Average latency {avg_latency:.3f}ms exceeds 5ms requirement"
        assert p95_latency < 10.0, f"95th percentile latency {p95_latency:.3f}ms exceeds 10ms threshold"
        
    def test_throughput_10k_ticks_per_second(self, normalizer):
        """Test processing 10,000+ ticks per second per venue"""
        tick_count = 10000
        test_duration = 1.0  # 1 second
        
        # Create sample data
        sample_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5"},
            raw_data={}
        )
        
        # Measure throughput
        start_time = time.perf_counter()
        processed_count = 0
        
        while time.perf_counter() - start_time < test_duration:
            normalizer.normalize_market_data(sample_data)
            processed_count += 1
            
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        throughput = processed_count / actual_duration
        
        print(f"\nThroughput Test:")
        print(f"  Processed: {processed_count} ticks")
        print(f"  Duration: {actual_duration:.3f}s")
        print(f"  Throughput: {throughput:.0f} ticks/second")
        
        assert throughput >= 10000, f"Throughput {throughput:.0f} ticks/sec below 10,000 requirement"


class TestRateLimiterPerformance:
    """Test rate limiter performance"""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()
        
    @pytest.mark.asyncio
    async def test_rate_limiter_latency(self, rate_limiter):
        """Test rate limiter check latency"""
        venue = Venue.BINANCE
        
        # Warm up
        for _ in range(100):
            await rate_limiter.should_allow_request(venue)
            
        # Measure latency
        latencies = []
        for _ in range(1000):
            start_time = time.perf_counter()
            await rate_limiter.should_allow_request(venue)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        print(f"\nRate Limiter Performance:")
        print(f"  Average latency: {avg_latency:.3f}ms")
        print(f"  95th percentile: {p95_latency:.3f}ms")
        
        assert avg_latency < 1.0, f"Rate limiter latency {avg_latency:.3f}ms exceeds 1ms threshold"
        
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, rate_limiter):
        """Test rate limiter under concurrent load"""
        venue = Venue.BINANCE
        concurrent_requests = 100
        
        async def make_request():
            return await rate_limiter.should_allow_request(venue)
            
        start_time = time.perf_counter()
        
        # Create concurrent tasks
        tasks = [make_request() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000
        
        successful_requests = sum(1 for allowed, _ in results if allowed)
        
        print(f"\nConcurrent Rate Limiting:")
        print(f"  Requests: {concurrent_requests}")
        print(f"  Duration: {duration:.3f}ms")
        print(f"  Successful: {successful_requests}")
        print(f"  Throughput: {concurrent_requests / (duration / 1000):.0f} req/sec")
        
        assert duration < 100, f"Concurrent processing took {duration:.3f}ms, too slow"


class TestHandlerPerformance:
    """Test market data handler performance"""
    
    @pytest.fixture
    def handlers(self):
        handlers = MarketDataHandlers()
        # Mock the broadcast callback to avoid WebSocket overhead
        handlers.set_broadcast_callback(AsyncMock())
        return handlers
        
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
    async def test_handler_processing_latency(self, handlers, normalized_tick):
        """Test handler processing latency"""
        # Mock external dependencies
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("redis_cache.redis_cache.cache_tick", AsyncMock())
            mp.setattr("historical_data_service.historical_data_service.store_tick", AsyncMock())
            
            latencies = []
            
            for _ in range(100):
                start_time = time.perf_counter()
                await handlers._handle_normalized_tick(normalized_tick)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)
                
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]
            
            print(f"\nHandler Performance:")
            print(f"  Average latency: {avg_latency:.3f}ms")
            print(f"  95th percentile: {p95_latency:.3f}ms")
            
            assert avg_latency < 10.0, f"Handler latency {avg_latency:.3f}ms exceeds 10ms threshold"
            
    @pytest.mark.asyncio
    async def test_handler_throughput(self, handlers, normalized_tick):
        """Test handler throughput capability"""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("redis_cache.redis_cache.cache_tick", AsyncMock())
            mp.setattr("historical_data_service.historical_data_service.store_tick", AsyncMock())
            
            batch_size = 1000
            start_time = time.perf_counter()
            
            for _ in range(batch_size):
                await handlers._handle_normalized_tick(normalized_tick)
                
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = batch_size / duration
            
            print(f"\nHandler Throughput:")
            print(f"  Processed: {batch_size} ticks")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {throughput:.0f} ticks/sec")
            
            assert throughput >= 1000, f"Handler throughput {throughput:.0f} ticks/sec below 1000 threshold"


class TestMemoryUsage:
    """Test memory usage and efficiency"""
    
    def test_normalizer_memory_efficiency(self):
        """Test normalizer doesn't leak memory"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        normalizer = DataNormalizer()
        
        # Process large batch of data
        for i in range(10000):
            sample_data = NormalizedMarketData(
                venue="BINANCE",
                instrument_id=f"SYMBOL{i % 100}USDT",
                data_type="tick",
                timestamp=1640995200000000000 + i,
                data={"price": f"{50000 + i}.0", "quantity": "1.5"},
                raw_data={}
            )
            normalizer.normalize_market_data(sample_data)
            
            # Force garbage collection periodically
            if i % 1000 == 0:
                gc.collect()
                
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"  Final: {final_memory / 1024 / 1024:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable (less than 50MB for 10k messages)
        assert memory_increase < 50, f"Memory increase {memory_increase:.2f}MB too high"


class TestCachePerformance:
    """Test cache performance requirements"""
    
    @pytest.mark.asyncio
    async def test_cache_access_latency(self):
        """Test Redis cache access < 1ms requirement"""
        from redis_cache import RedisCache
        
        # Mock Redis for consistent testing
        cache = RedisCache()
        cache._connected = True
        cache._redis = AsyncMock()
        
        # Mock get operation with minimal delay
        async def mock_get(key):
            await asyncio.sleep(0.0001)  # 0.1ms simulated delay
            return '{"price": "50000.0", "size": "1.5"}'
            
        cache._redis.get = mock_get
        
        latencies = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            await cache.get_latest_tick("BINANCE", "BTCUSDT")
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        print(f"\nCache Performance:")
        print(f"  Average latency: {avg_latency:.3f}ms")
        print(f"  95th percentile: {p95_latency:.3f}ms")
        
        # Note: This tests the code path, actual Redis latency depends on network/server
        assert avg_latency < 5.0, f"Cache access latency {avg_latency:.3f}ms too high for code path"


class TestSystemLoadTest:
    """End-to-end system load testing"""
    
    @pytest.mark.asyncio
    async def test_multi_venue_load(self):
        """Test system under multi-venue load"""
        from market_data_service import MarketDataService
        from market_data_handlers import MarketDataHandlers
        
        # Setup services
        service = MarketDataService()
        handlers = MarketDataHandlers()
        handlers.set_broadcast_callback(AsyncMock())
        service.add_data_handler(handlers.handle_market_data)
        
        # Mock external dependencies
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("redis_cache.redis_cache.cache_tick", AsyncMock())
            mp.setattr("redis_cache.redis_cache.cache_quote", AsyncMock())
            mp.setattr("historical_data_service.historical_data_service.store_tick", AsyncMock())
            mp.setattr("historical_data_service.historical_data_service.store_quote", AsyncMock())
            mp.setattr("rate_limiter.rate_limiter.should_allow_request", AsyncMock(return_value=(True, "allowed")))
            
            # Subscribe to multiple venues/instruments
            venues = ["BINANCE", "COINBASE", "KRAKEN"]
            instruments = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            
            for venue in venues:
                for instrument in instruments:
                    await service.subscribe(Venue(venue), instrument, DataType.TICK)
                    
            # Generate load
            message_count = 1000
            start_time = time.perf_counter()
            
            tasks = []
            for i in range(message_count):
                venue = venues[i % len(venues)]
                instrument = instruments[i % len(instruments)]
                
                # Create mock message
                raw_data = NormalizedMarketData(
                    venue=venue,
                    instrument_id=instrument,
                    data_type="tick",
                    timestamp=1640995200000000000 + i,
                    data={"price": f"{50000 + i}.0", "quantity": "1.5"},
                    raw_data={}
                )
                
                # Process message
                task = asyncio.create_task(handlers.handle_market_data(raw_data))
                tasks.append(task)
                
            # Wait for all messages to process
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = message_count / duration
            
            print(f"\nMulti-Venue Load Test:")
            print(f"  Messages: {message_count}")
            print(f"  Venues: {len(venues)}")
            print(f"  Instruments: {len(instruments)}")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {throughput:.0f} msg/sec")
            
            assert throughput >= 500, f"System throughput {throughput:.0f} msg/sec below 500 threshold"
            assert duration < 10.0, f"Load test took {duration:.3f}s, too slow"


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests"""
    
    def test_normalization_regression(self):
        """Ensure normalization performance doesn't regress"""
        # This would typically compare against baseline metrics
        # For now, just ensure it meets minimum requirements
        normalizer = DataNormalizer()
        
        sample_data = NormalizedMarketData(
            venue="BINANCE",
            instrument_id="BTCUSDT",
            data_type="tick",
            timestamp=1640995200000000000,
            data={"price": "50000.0", "quantity": "1.5"},
            raw_data={}
        )
        
        # Baseline: Should process 1000 normalizations in under 100ms
        start_time = time.perf_counter()
        for _ in range(1000):
            normalizer.normalize_market_data(sample_data)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        
        print(f"\nRegression Test - 1000 normalizations: {duration_ms:.3f}ms")
        assert duration_ms < 100, f"Normalization regression: {duration_ms:.3f}ms exceeds 100ms baseline"


if __name__ == "__main__":
    # Run performance tests with timing information
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "not slow"])
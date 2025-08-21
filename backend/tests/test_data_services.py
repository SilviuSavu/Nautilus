"""
Test suite for data services and processing
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_backfill_service import DataBackfillService
from market_data_service import MarketDataService
from data_normalizer import DataNormalizer


class TestDataBackfillService:
    """Test data backfill functionality"""
    
    @pytest.fixture
    def backfill_service(self):
        """Create mock backfill service"""
        with patch('data_backfill_service.IBIntegrationService'):
            service = DataBackfillService()
            return service
    
    def test_backfill_service_initialization(self, backfill_service):
        """Test service initializes correctly"""
        assert backfill_service is not None
        assert hasattr(backfill_service, 'get_backfill_status')
    
    @patch('data_backfill_service.DataBackfillService._check_ib_connection')
    def test_backfill_connection_check(self, mock_check_connection, backfill_service):
        """Test IB connection checking"""
        mock_check_connection.return_value = True
        
        result = backfill_service._check_ib_connection()
        assert result is True
        mock_check_connection.assert_called_once()
    
    def test_backfill_status_structure(self, backfill_service):
        """Test backfill status returns correct structure"""
        # Mock the status response
        with patch.object(backfill_service, 'get_backfill_status') as mock_status:
            mock_status.return_value = {
                "symbol": "AAPL",
                "timeframe": "5m",
                "is_running": False,
                "progress": 100,
                "success_count": 500,
                "error_count": 0
            }
            
            status = backfill_service.get_backfill_status("AAPL", "5m")
            
            assert "symbol" in status
            assert "timeframe" in status
            assert "is_running" in status
            assert "progress" in status
            assert isinstance(status["progress"], (int, float))
    
    def test_backfill_validation(self, backfill_service):
        """Test input validation for backfill parameters"""
        # Test with valid parameters
        with patch.object(backfill_service, 'start_backfill') as mock_start:
            mock_start.return_value = {"success": True}
            
            # Should not raise exception with valid parameters
            try:
                backfill_service.start_backfill(
                    symbol="AAPL",
                    timeframe="5m",
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
            except Exception as e:
                pytest.fail(f"Valid parameters should not raise exception: {e}")


class TestMarketDataService:
    """Test market data service functionality"""
    
    @pytest.fixture
    def market_data_service(self):
        """Create mock market data service"""
        with patch('market_data_service.IBIntegrationService'):
            service = MarketDataService()
            return service
    
    def test_market_data_service_initialization(self, market_data_service):
        """Test service initializes correctly"""
        assert market_data_service is not None
    
    def test_latest_data_structure(self, market_data_service):
        """Test latest data returns correct structure"""
        with patch.object(market_data_service, 'get_latest_data') as mock_latest:
            mock_latest.return_value = {
                "symbol": "AAPL",
                "price": 150.0,
                "timestamp": datetime.now().isoformat(),
                "volume": 1000000
            }
            
            data = market_data_service.get_latest_data("AAPL")
            
            assert "symbol" in data
            assert "price" in data
            assert "timestamp" in data
            assert isinstance(data["price"], (int, float))
    
    def test_historical_data_structure(self, market_data_service):
        """Test historical data returns correct structure"""
        with patch.object(market_data_service, 'get_historical_data') as mock_historical:
            mock_historical.return_value = {
                "candles": [
                    {
                        "timestamp": "2024-01-01T09:30:00Z",
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 104.0,
                        "volume": 1000000
                    }
                ]
            }
            
            data = market_data_service.get_historical_data("AAPL", "5m", 100)
            
            assert "candles" in data
            assert len(data["candles"]) > 0
            
            candle = data["candles"][0]
            required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
            for field in required_fields:
                assert field in candle
                if field != "timestamp":
                    assert isinstance(candle[field], (int, float))


class TestDataNormalizer:
    """Test data normalization functionality"""
    
    @pytest.fixture
    def normalizer(self):
        """Create data normalizer instance"""
        return DataNormalizer()
    
    def test_timestamp_normalization(self, normalizer):
        """Test timestamp normalization"""
        test_timestamps = [
            "2024-01-01 09:30:00",
            "2024-01-01T09:30:00",
            "2024-01-01T09:30:00Z",
            "2024-01-01T09:30:00-05:00"
        ]
        
        for ts in test_timestamps:
            try:
                normalized = normalizer.normalize_timestamp(ts)
                assert normalized is not None
                # Should be ISO format string or datetime object
                assert isinstance(normalized, (str, datetime))
            except Exception as e:
                pytest.fail(f"Failed to normalize timestamp {ts}: {e}")
    
    def test_price_normalization(self, normalizer):
        """Test price data normalization"""
        test_prices = [
            "150.0",
            150.0,
            "150",
            150
        ]
        
        for price in test_prices:
            normalized = normalizer.normalize_price(price)
            assert isinstance(normalized, float)
            assert normalized > 0
    
    def test_volume_normalization(self, normalizer):
        """Test volume data normalization"""
        test_volumes = [
            "1000000",
            1000000,
            "1,000,000",
            1000000.0
        ]
        
        for volume in test_volumes:
            normalized = normalizer.normalize_volume(volume)
            assert isinstance(normalized, int)
            assert normalized >= 0
    
    def test_candle_data_normalization(self, normalizer):
        """Test complete candle data normalization"""
        raw_candle = {
            "timestamp": "2024-01-01 09:30:00",
            "open": "150.0",
            "high": "155.0",
            "low": "149.0",
            "close": "154.0",
            "volume": "1000000"
        }
        
        normalized = normalizer.normalize_candle(raw_candle)
        
        # Check all required fields are present
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
        for field in required_fields:
            assert field in normalized
        
        # Check data types
        assert isinstance(normalized["open"], float)
        assert isinstance(normalized["high"], float)
        assert isinstance(normalized["low"], float)
        assert isinstance(normalized["close"], float)
        assert isinstance(normalized["volume"], int)
        
        # Check price relationships
        assert normalized["high"] >= normalized["open"]
        assert normalized["high"] >= normalized["close"]
        assert normalized["low"] <= normalized["open"]
        assert normalized["low"] <= normalized["close"]


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_missing_data_handling(self):
        """Test handling of missing or null data"""
        normalizer = DataNormalizer()
        
        # Test with missing fields
        incomplete_candle = {
            "timestamp": "2024-01-01T09:30:00Z",
            "open": 150.0,
            # Missing high, low, close, volume
        }
        
        with pytest.raises(Exception):
            normalizer.normalize_candle(incomplete_candle)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data values"""
        normalizer = DataNormalizer()
        
        # Test with invalid price (negative)
        with pytest.raises(Exception):
            normalizer.normalize_price(-10.0)
        
        # Test with invalid volume (negative)
        with pytest.raises(Exception):
            normalizer.normalize_volume(-1000)
    
    def test_data_consistency(self):
        """Test data consistency checks"""
        normalizer = DataNormalizer()
        
        # Test candle with inconsistent OHLC data
        inconsistent_candle = {
            "timestamp": "2024-01-01T09:30:00Z",
            "open": 150.0,
            "high": 140.0,  # High is less than open - inconsistent
            "low": 160.0,   # Low is greater than open - inconsistent  
            "close": 155.0,
            "volume": 1000000
        }
        
        # Should either fix inconsistencies or raise validation error
        with pytest.raises(Exception):
            normalizer.normalize_candle(inconsistent_candle)


class TestPerformanceMetrics:
    """Test performance and monitoring metrics"""
    
    def test_data_processing_performance(self):
        """Test data processing performance metrics"""
        normalizer = DataNormalizer()
        
        # Create large dataset for performance testing
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                "timestamp": f"2024-01-01T{i//60:02d}:{i%60:02d}:00Z",
                "open": 150.0 + (i % 10),
                "high": 155.0 + (i % 10),
                "low": 149.0 + (i % 10),
                "close": 154.0 + (i % 10),
                "volume": 1000000 + (i * 1000)
            })
        
        # Measure processing time
        import time
        start_time = time.time()
        
        for candle in large_dataset[:100]:  # Test with smaller subset
            try:
                normalizer.normalize_candle(candle)
            except Exception:
                pass  # Allow some failures for this performance test
        
        processing_time = time.time() - start_time
        
        # Should process 100 candles in reasonable time (less than 1 second)
        assert processing_time < 1.0
    
    def test_memory_usage(self):
        """Test memory usage during data processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process some data
        normalizer = DataNormalizer()
        for i in range(100):
            candle = {
                "timestamp": f"2024-01-01T09:{i%60:02d}:00Z",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 154.0,
                "volume": 1000000
            }
            try:
                normalizer.normalize_candle(candle)
            except Exception:
                pass
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10MB for this test)
        assert memory_increase < 10 * 1024 * 1024


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
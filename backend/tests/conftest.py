"""
Pytest configuration and shared fixtures for backend tests
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ib_service():
    """Mock Interactive Brokers service"""
    with patch('ib_integration_service.IBIntegrationService') as mock_ib:
        mock_instance = Mock()
        mock_ib.return_value = mock_instance
        
        # Setup default mock behaviors
        mock_instance.is_connected.return_value = True
        mock_instance.get_client_id.return_value = 1
        mock_instance.search_instruments.return_value = []
        mock_instance.get_historical_data.return_value = {"candles": []}
        
        yield mock_instance


@pytest.fixture
def mock_database():
    """Mock database connections"""
    with patch('psycopg2.connect') as mock_conn:
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_conn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None
        
        yield mock_connection


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "symbol": "AAPL",
        "candles": [
            {
                "timestamp": "2024-01-01T09:30:00Z",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 154.0,
                "volume": 1000000
            },
            {
                "timestamp": "2024-01-01T09:35:00Z", 
                "open": 154.0,
                "high": 156.0,
                "low": 153.0,
                "close": 155.5,
                "volume": 800000
            }
        ]
    }


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_price": 150.0,
                "current_price": 155.0,
                "unrealized_pnl": 500.0
            },
            {
                "symbol": "MSFT",
                "quantity": 50,
                "avg_price": 300.0,
                "current_price": 305.0,
                "unrealized_pnl": 250.0
            }
        ],
        "summary": {
            "total_value": 100000.0,
            "cash": 10000.0,
            "unrealized_pnl": 750.0,
            "realized_pnl": 1500.0
        }
    }


@pytest.fixture
def mock_redis():
    """Mock Redis connection"""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis_instance = Mock()
        mock_redis_class.return_value = mock_redis_instance
        
        # Setup default Redis behaviors
        mock_redis_instance.get.return_value = None
        mock_redis_instance.set.return_value = True
        mock_redis_instance.delete.return_value = 1
        mock_redis_instance.ping.return_value = True
        
        yield mock_redis_instance


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test"""
    # Store original values
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test'
    os.environ['REDIS_URL'] = 'redis://localhost:6379/1'
    os.environ['IB_CLIENT_ID'] = '999'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_config():
    """Test configuration settings"""
    return {
        "database": {
            "url": "postgresql://test:test@localhost:5432/test",
            "pool_size": 5
        },
        "redis": {
            "url": "redis://localhost:6379/1",
            "timeout": 10
        },
        "ib": {
            "client_id": 999,
            "host": "127.0.0.1",
            "port": 7497
        },
        "api": {
            "rate_limit": 1000,
            "timeout": 30
        }
    }


# Test data generators
def generate_candle_data(symbol="AAPL", count=100, start_price=150.0):
    """Generate sample candle data for testing"""
    import random
    from datetime import datetime, timedelta
    
    candles = []
    current_price = start_price
    base_time = datetime.now() - timedelta(minutes=count * 5)
    
    for i in range(count):
        # Generate realistic OHLC data
        open_price = current_price
        close_price = open_price + random.uniform(-2.0, 2.0)
        high_price = max(open_price, close_price) + random.uniform(0, 1.0)
        low_price = min(open_price, close_price) - random.uniform(0, 1.0)
        volume = random.randint(500000, 2000000)
        
        candle = {
            "timestamp": (base_time + timedelta(minutes=i * 5)).isoformat() + "Z",
            "open": round(open_price, 2),
            "high": round(high_price, 2), 
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        }
        
        candles.append(candle)
        current_price = close_price
    
    return {
        "symbol": symbol,
        "candles": candles
    }


@pytest.fixture
def generate_test_data():
    """Fixture that provides test data generation functions"""
    return {
        "candles": generate_candle_data
    }


# Error handling fixtures
@pytest.fixture
def mock_api_error():
    """Mock API error responses"""
    def _create_error(status_code=500, message="Internal Server Error"):
        from fastapi import HTTPException
        return HTTPException(status_code=status_code, detail=message)
    
    return _create_error


@pytest.fixture
def mock_ib_connection_error():
    """Mock IB connection errors"""
    def _create_ib_error(message="IB Gateway not connected"):
        class IBConnectionError(Exception):
            pass
        return IBConnectionError(message)
    
    return _create_ib_error
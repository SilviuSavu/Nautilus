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


# Sprint 3 Specific Fixtures and Mocks

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for Sprint 3 WebSocket infrastructure"""
    from unittest.mock import AsyncMock
    
    mock_ws = Mock()
    mock_ws.send = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.ping = AsyncMock()
    mock_ws.wait_for_message = AsyncMock()
    mock_ws.accept = AsyncMock()
    mock_ws.receive_text = AsyncMock()
    mock_ws.receive_json = AsyncMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.send_json = AsyncMock()
    
    # WebSocket states
    mock_ws.client_state = "CONNECTED"
    mock_ws.application_state = "CONNECTED"
    
    return mock_ws


@pytest.fixture
def mock_redis_pubsub():
    """Mock Redis pub/sub for Sprint 3 real-time messaging"""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis_instance = Mock()
        mock_pubsub = Mock()
        
        # Setup Redis pub/sub behaviors
        mock_redis_instance.pubsub.return_value = mock_pubsub
        mock_pubsub.subscribe = Mock()
        mock_pubsub.unsubscribe = Mock()
        mock_pubsub.publish = Mock(return_value=1)
        mock_pubsub.get_message = Mock(return_value=None)
        mock_pubsub.listen = Mock(return_value=iter([]))
        
        mock_redis_class.return_value = mock_redis_instance
        
        yield {
            'redis': mock_redis_instance,
            'pubsub': mock_pubsub
        }


@pytest.fixture
def sample_websocket_messages():
    """Sample WebSocket messages for Sprint 3 testing"""
    return {
        'market_data': {
            'type': 'market_data',
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'timestamp': '2024-01-01T10:30:00Z'
        },
        'portfolio_update': {
            'type': 'portfolio_update',
            'portfolio_id': 'portfolio_1',
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'market_value': 15025.0},
                {'symbol': 'MSFT', 'quantity': 50, 'market_value': 15250.0}
            ],
            'total_value': 30275.0,
            'timestamp': '2024-01-01T10:30:00Z'
        },
        'risk_alert': {
            'type': 'risk_alert',
            'alert_id': 'risk_001',
            'portfolio_id': 'portfolio_1',
            'alert_type': 'position_limit_breach',
            'severity': 'HIGH',
            'message': 'Position limit exceeded for AAPL',
            'timestamp': '2024-01-01T10:30:00Z'
        },
        'strategy_update': {
            'type': 'strategy_update',
            'strategy_id': 'momentum_001',
            'status': 'RUNNING',
            'performance': {'pnl': 1500.0, 'return': 0.025},
            'timestamp': '2024-01-01T10:30:00Z'
        },
        'heartbeat': {
            'type': 'heartbeat',
            'timestamp': '2024-01-01T10:30:00Z',
            'status': 'ok'
        }
    }


@pytest.fixture
def sample_analytics_data():
    """Sample analytics data for Sprint 3 analytics testing"""
    return {
        'trades': [
            {
                'trade_id': 'trade_001',
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 150.0,
                'side': 'BUY',
                'timestamp': '2024-01-01T09:30:00Z',
                'portfolio_id': 'portfolio_1'
            },
            {
                'trade_id': 'trade_002',
                'symbol': 'AAPL',
                'quantity': 50,
                'price': 155.0,
                'side': 'SELL',
                'timestamp': '2024-01-01T10:30:00Z',
                'portfolio_id': 'portfolio_1'
            }
        ],
        'positions': [
            {
                'symbol': 'AAPL',
                'quantity': 50,
                'avg_price': 152.5,
                'market_price': 155.0,
                'unrealized_pnl': 125.0,
                'portfolio_id': 'portfolio_1'
            },
            {
                'symbol': 'MSFT',
                'quantity': 100,
                'avg_price': 300.0,
                'market_price': 305.0,
                'unrealized_pnl': 500.0,
                'portfolio_id': 'portfolio_1'
            }
        ],
        'performance_metrics': {
            'total_pnl': 625.0,
            'total_return': 0.0208,
            'sharpe_ratio': 1.85,
            'max_drawdown': -0.025,
            'volatility': 0.15,
            'win_rate': 0.75,
            'avg_win': 250.0,
            'avg_loss': -125.0
        }
    }


@pytest.fixture
def sample_risk_data():
    """Sample risk data for Sprint 3 risk management testing"""
    return {
        'risk_limits': [
            {
                'limit_id': 'limit_001',
                'portfolio_id': 'portfolio_1',
                'limit_type': 'max_position_value',
                'value': 50000.0,
                'current_value': 30275.0,
                'utilization': 0.6055
            },
            {
                'limit_id': 'limit_002',
                'portfolio_id': 'portfolio_1',
                'limit_type': 'max_portfolio_risk',
                'value': 0.1,
                'current_value': 0.075,
                'utilization': 0.75
            },
            {
                'limit_id': 'limit_003',
                'portfolio_id': 'portfolio_1',
                'limit_type': 'max_concentration',
                'value': 0.25,
                'current_value': 0.20,
                'utilization': 0.80
            }
        ],
        'risk_metrics': {
            'portfolio_var_1d': 2500.0,
            'portfolio_var_10d': 7500.0,
            'portfolio_beta': 1.15,
            'tracking_error': 0.05,
            'information_ratio': 1.25,
            'sector_exposures': {
                'Technology': 0.60,
                'Healthcare': 0.25,
                'Finance': 0.15
            }
        },
        'breaches': [
            {
                'breach_id': 'breach_001',
                'limit_id': 'limit_002',
                'portfolio_id': 'portfolio_1',
                'breach_type': 'SOFT',
                'current_value': 0.085,
                'limit_value': 0.10,
                'severity': 'MEDIUM',
                'timestamp': '2024-01-01T10:30:00Z',
                'message': 'Portfolio risk approaching limit'
            }
        ]
    }


@pytest.fixture
def sample_strategy_data():
    """Sample strategy data for Sprint 3 strategy framework testing"""
    return {
        'strategies': [
            {
                'strategy_id': 'momentum_001',
                'name': 'Momentum Strategy',
                'version': 'v1.2.0',
                'status': 'RUNNING',
                'config': {
                    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                    'lookback_period': 20,
                    'rebalance_frequency': 'DAILY',
                    'max_position_size': 0.1
                },
                'performance': {
                    'total_return': 0.125,
                    'sharpe_ratio': 1.75,
                    'max_drawdown': -0.08,
                    'trades_count': 45,
                    'win_rate': 0.67
                }
            },
            {
                'strategy_id': 'mean_reversion_001',
                'name': 'Mean Reversion Strategy',
                'version': 'v2.1.0',
                'status': 'PAUSED',
                'config': {
                    'symbols': ['SPY', 'QQQ', 'IWM'],
                    'lookback_period': 50,
                    'z_score_threshold': 2.0,
                    'max_position_size': 0.15
                },
                'performance': {
                    'total_return': 0.085,
                    'sharpe_ratio': 1.25,
                    'max_drawdown': -0.12,
                    'trades_count': 32,
                    'win_rate': 0.72
                }
            }
        ],
        'deployments': [
            {
                'deployment_id': 'deploy_001',
                'strategy_id': 'momentum_001',
                'version': 'v1.2.0',
                'environment': 'PRODUCTION',
                'status': 'ACTIVE',
                'deployed_at': '2024-01-01T08:00:00Z',
                'container_id': 'container_momentum_001',
                'health_status': 'HEALTHY'
            }
        ],
        'backtests': [
            {
                'backtest_id': 'bt_001',
                'strategy_id': 'momentum_001',
                'version': 'v1.2.0',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'status': 'COMPLETED',
                'results': {
                    'total_return': 0.155,
                    'sharpe_ratio': 1.82,
                    'max_drawdown': -0.095,
                    'trades_count': 234,
                    'win_rate': 0.68
                }
            }
        ]
    }


@pytest.fixture
def mock_docker_service():
    """Mock Docker service for Sprint 3 strategy deployment testing"""
    with patch('strategy_pipeline.deployment_manager.DockerService') as mock_docker:
        mock_instance = Mock()
        mock_docker.return_value = mock_instance
        
        # Setup Docker service behaviors
        mock_instance.build_image = AsyncMock(return_value={'id': 'image_12345'})
        mock_instance.run_container = AsyncMock(return_value={'id': 'container_12345'})
        mock_instance.stop_container = AsyncMock(return_value=True)
        mock_instance.remove_container = AsyncMock(return_value=True)
        mock_instance.get_container_logs = AsyncMock(return_value="Container logs...")
        mock_instance.get_container_stats = AsyncMock(return_value={
            'cpu_percent': '5.2%',
            'memory_usage': '256MB',
            'network_io': '1.2MB'
        })
        
        yield mock_instance


@pytest.fixture
def mock_git_service():
    """Mock Git service for Sprint 3 version control testing"""
    with patch('strategy_pipeline.version_control.GitService') as mock_git:
        mock_instance = Mock()
        mock_git.return_value = mock_instance
        
        # Setup Git service behaviors
        mock_instance.clone_repository = AsyncMock(return_value=True)
        mock_instance.checkout_version = AsyncMock(return_value=True)
        mock_instance.get_version_info = AsyncMock(return_value={
            'version': 'v1.2.0',
            'commit_hash': 'abc123',
            'author': 'test_user',
            'timestamp': '2024-01-01T10:00:00Z'
        })
        mock_instance.create_tag = AsyncMock(return_value=True)
        mock_instance.list_versions = AsyncMock(return_value=['v1.0.0', 'v1.1.0', 'v1.2.0'])
        
        yield mock_instance


@pytest.fixture
def mock_database_service():
    """Enhanced mock database service for Sprint 3 components"""
    with patch('database.database_service.DatabaseService') as mock_db:
        mock_instance = Mock()
        mock_db.return_value = mock_instance
        
        # Setup database behaviors for Sprint 3
        mock_instance.get_trades = AsyncMock(return_value=[])
        mock_instance.get_positions = AsyncMock(return_value=[])
        mock_instance.get_portfolio_summary = AsyncMock(return_value={})
        mock_instance.get_risk_limits = AsyncMock(return_value=[])
        mock_instance.get_risk_metrics = AsyncMock(return_value={})
        mock_instance.get_strategy_configs = AsyncMock(return_value=[])
        mock_instance.get_strategy_performance = AsyncMock(return_value={})
        mock_instance.save_analytics_result = AsyncMock(return_value=True)
        mock_instance.save_risk_calculation = AsyncMock(return_value=True)
        mock_instance.save_strategy_update = AsyncMock(return_value=True)
        
        yield mock_instance


@pytest.fixture(scope="session")
def load_test_config():
    """Configuration for load testing"""
    return {
        'websocket': {
            'max_connections': 1000,
            'message_rate_per_second': 100,
            'test_duration_seconds': 60
        },
        'analytics': {
            'max_calculations_per_second': 50,
            'dataset_size': 10000,
            'concurrent_calculations': 10
        },
        'risk': {
            'max_risk_checks_per_second': 100,
            'portfolios_count': 50,
            'positions_per_portfolio': 100
        },
        'strategy': {
            'max_concurrent_deployments': 20,
            'deployment_timeout_seconds': 30,
            'max_strategies_per_test': 100
        },
        'performance_thresholds': {
            'websocket_throughput_min': 50000,  # messages per second
            'analytics_latency_max': 2.0,       # seconds
            'risk_calculation_rate_min': 500,   # calculations per second
            'strategy_deployment_rate_min': 0.5, # deployments per second
            'memory_usage_max': 500,            # MB
            'cpu_usage_max': 85                 # percentage
        }
    }


@pytest.fixture
def websocket_test_client():
    """Test client for WebSocket connections in Sprint 3"""
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    
    class MockWebSocketTestClient:
        def __init__(self):
            self.sent_messages = []
            self.received_messages = []
            self.is_connected = True
            
        async def connect(self, url):
            """Mock WebSocket connection"""
            self.is_connected = True
            return True
            
        async def send_json(self, data):
            """Mock send JSON message"""
            self.sent_messages.append(data)
            
        async def receive_json(self):
            """Mock receive JSON message"""
            if self.received_messages:
                return self.received_messages.pop(0)
            return {'type': 'heartbeat', 'timestamp': '2024-01-01T10:30:00Z'}
            
        async def close(self):
            """Mock close connection"""
            self.is_connected = False
            
        def add_mock_message(self, message):
            """Add a mock message to be received"""
            self.received_messages.append(message)
    
    return MockWebSocketTestClient()


# Pytest markers for Sprint 3 test categories
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "websocket: mark test as WebSocket infrastructure test"
    )
    config.addinivalue_line(
        "markers", "analytics: mark test as analytics component test"
    )
    config.addinivalue_line(
        "markers", "risk: mark test as risk management test"
    )
    config.addinivalue_line(
        "markers", "strategy: mark test as strategy framework test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as load/performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
    )
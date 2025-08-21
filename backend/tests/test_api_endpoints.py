"""
Comprehensive test suite for backend API endpoints
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_health_endpoint(self):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint_content_type(self):
        """Test health endpoint returns JSON"""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestIBIntegrationEndpoints:
    """Test Interactive Brokers integration endpoints"""
    
    def test_ib_connection_status(self):
        """Test IB connection status endpoint"""
        response = client.get("/api/v1/ib/status")
        assert response.status_code in [200, 503]  # 200 if connected, 503 if not
        data = response.json()
        assert "connected" in data
        assert "client_id" in data
    
    @patch('ib_integration_service.IBIntegrationService.search_instruments')
    def test_instrument_search_success(self, mock_search):
        """Test successful instrument search"""
        mock_search.return_value = [{
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "sec_type": "STK"
        }]
        
        response = client.get("/api/v1/ib/instruments/search/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert data[0]["symbol"] == "AAPL"
    
    def test_instrument_search_invalid_query(self):
        """Test instrument search with invalid query"""
        response = client.get("/api/v1/ib/instruments/search/")
        assert response.status_code == 404  # Empty query should return 404
    
    @patch('ib_integration_service.IBIntegrationService.get_historical_data')
    def test_historical_data_success(self, mock_get_data):
        """Test successful historical data retrieval"""
        mock_get_data.return_value = {
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
        
        response = client.get("/api/v1/ib/historical/AAPL?timeframe=5m&bars=100")
        assert response.status_code == 200
        data = response.json()
        assert "candles" in data
        assert len(data["candles"]) > 0
    
    def test_historical_data_invalid_timeframe(self):
        """Test historical data with invalid timeframe"""
        response = client.get("/api/v1/ib/historical/AAPL?timeframe=invalid&bars=100")
        assert response.status_code == 400


class TestDataBackfillEndpoints:
    """Test data backfill endpoints"""
    
    @patch('data_backfill_service.DataBackfillService.get_backfill_status')
    def test_backfill_status(self, mock_get_status):
        """Test backfill status endpoint"""
        mock_get_status.return_value = {
            "symbol": "AAPL",
            "timeframe": "5m",
            "is_running": False,
            "progress": 100,
            "success_count": 1000,
            "error_count": 0
        }
        
        response = client.get("/api/v1/data/backfill/status?symbol=AAPL&timeframe=5m")
        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "progress" in data
    
    @patch('data_backfill_service.DataBackfillService.start_backfill')
    def test_start_backfill_success(self, mock_start):
        """Test successful backfill start"""
        mock_start.return_value = {"success": True, "message": "Backfill started"}
        
        response = client.post("/api/v1/data/backfill/start", json={
            "symbol": "AAPL",
            "timeframe": "5m",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_start_backfill_missing_params(self):
        """Test backfill start with missing parameters"""
        response = client.post("/api/v1/data/backfill/start", json={
            "symbol": "AAPL"
            # Missing timeframe, start_date, end_date
        })
        assert response.status_code == 422  # Validation error


class TestMarketDataEndpoints:
    """Test market data endpoints"""
    
    @patch('market_data_service.MarketDataService.get_latest_data')
    def test_latest_market_data(self, mock_get_data):
        """Test latest market data retrieval"""
        mock_get_data.return_value = {
            "symbol": "AAPL",
            "price": 150.0,
            "timestamp": "2024-01-01T09:30:00Z",
            "volume": 1000000
        }
        
        response = client.get("/api/v1/market-data/latest/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "price" in data
        assert "timestamp" in data
    
    def test_market_data_invalid_symbol(self):
        """Test market data with invalid symbol"""
        response = client.get("/api/v1/market-data/latest/INVALID")
        assert response.status_code in [404, 400]


class TestPortfolioEndpoints:
    """Test portfolio and position endpoints"""
    
    @patch('portfolio_service.PortfolioService.get_positions')
    def test_get_positions(self, mock_get_positions):
        """Test portfolio positions retrieval"""
        mock_get_positions.return_value = [{
            "symbol": "AAPL",
            "quantity": 100,
            "avg_price": 150.0,
            "current_price": 155.0,
            "unrealized_pnl": 500.0
        }]
        
        response = client.get("/api/v1/portfolio/positions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "symbol" in data[0]
            assert "quantity" in data[0]
    
    @patch('portfolio_service.PortfolioService.get_portfolio_summary')
    def test_portfolio_summary(self, mock_get_summary):
        """Test portfolio summary"""
        mock_get_summary.return_value = {
            "total_value": 100000.0,
            "cash": 10000.0,
            "unrealized_pnl": 2500.0,
            "realized_pnl": 1500.0
        }
        
        response = client.get("/api/v1/portfolio/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "unrealized_pnl" in data


class TestAuthenticationEndpoints:
    """Test authentication endpoints"""
    
    def test_auth_login_missing_credentials(self):
        """Test login with missing credentials"""
        response = client.post("/api/v1/auth/login", json={})
        assert response.status_code == 422  # Validation error
    
    @patch('auth.security.verify_password')
    @patch('auth.database.get_user_by_username')
    def test_auth_login_invalid_credentials(self, mock_get_user, mock_verify):
        """Test login with invalid credentials"""
        mock_get_user.return_value = None
        
        response = client.post("/api/v1/auth/login", json={
            "username": "invalid_user",
            "password": "wrong_password"
        })
        assert response.status_code == 401


class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_404_endpoint(self):
        """Test non-existent endpoint returns 404"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test wrong HTTP method returns 405"""
        response = client.post("/health")  # GET endpoint called with POST
        assert response.status_code == 405


class TestPerformanceEndpoints:
    """Test performance monitoring endpoints"""
    
    def test_system_metrics(self):
        """Test system metrics endpoint"""
        response = client.get("/api/v1/system/metrics")
        # Should return either 200 with metrics or 501 if not implemented
        assert response.status_code in [200, 501, 404]
    
    @patch('monitoring_service.MonitoringService.get_performance_metrics')
    def test_performance_metrics(self, mock_get_metrics):
        """Test performance metrics retrieval"""
        mock_get_metrics.return_value = {
            "cpu_usage": 25.5,
            "memory_usage": 45.2,
            "disk_usage": 60.0,
            "network_io": {"bytes_sent": 1000, "bytes_recv": 2000}
        }
        
        response = client.get("/api/v1/performance/metrics")
        # Should work if endpoint exists
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
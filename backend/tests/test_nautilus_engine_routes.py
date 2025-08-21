"""
Tests for NautilusTrader Engine Management API Routes
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from main import app


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_auth_token():
    """Mock JWT token for authentication"""
    return "Bearer test_token"


class TestNautilusEngineRoutes:
    """Test suite for Nautilus engine API routes"""

    def test_health_endpoint(self, client):
        """Test engine health endpoint (no auth required)"""
        response = client.get("/api/v1/nautilus/engine/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "nautilus-engine-api"

    @patch('nautilus_engine_routes.get_current_user')
    @patch('nautilus_engine_routes.get_nautilus_engine_manager')
    def test_get_engine_status(self, mock_manager, mock_auth, client):
        """Test get engine status endpoint"""
        # Mock authentication
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock engine manager
        mock_engine = AsyncMock()
        mock_engine.get_engine_status.return_value = {
            "state": "stopped",
            "config": None,
            "started_at": None,
            "last_error": None,
            "resource_usage": {},
            "container_info": {},
            "active_backtests": 0,
            "health_check": {"status": "healthy"}
        }
        mock_manager.return_value = mock_engine

        response = client.get(
            "/api/v1/nautilus/engine/status",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "status" in data

    @patch('nautilus_engine_routes.get_current_user')
    @patch('nautilus_engine_routes.get_nautilus_engine_manager')
    def test_start_engine_paper_mode(self, mock_manager, mock_auth, client):
        """Test start engine in paper trading mode"""
        # Mock authentication
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock engine manager
        mock_engine = AsyncMock()
        mock_engine.start_engine.return_value = {
            "success": True,
            "message": "Engine started",
            "state": "starting"
        }
        mock_manager.return_value = mock_engine

        config = {
            "config": {
                "engine_type": "live",
                "trading_mode": "paper",
                "log_level": "INFO",
                "instance_id": "test-001",
                "max_memory": "2g",
                "max_cpu": "2.0",
                "risk_engine_enabled": True
            },
            "confirm_live_trading": False
        }

        response = client.post(
            "/api/v1/nautilus/engine/start",
            json=config,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Engine started"

    @patch('nautilus_engine_routes.get_current_user')
    def test_start_engine_live_mode_no_confirmation(self, mock_auth, client):
        """Test start engine in live mode without confirmation"""
        mock_auth.return_value = {"user_id": "test_user"}

        config = {
            "config": {
                "engine_type": "live",
                "trading_mode": "live",
                "log_level": "INFO",
                "instance_id": "test-001",
                "max_memory": "2g",
                "max_cpu": "2.0",
                "risk_engine_enabled": True
            },
            "confirm_live_trading": False
        }

        response = client.post(
            "/api/v1/nautilus/engine/start",
            json=config,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 400
        assert "Live trading mode requires explicit confirmation" in response.json()["detail"]

    @patch('nautilus_engine_routes.get_current_user')
    @patch('nautilus_engine_routes.get_nautilus_engine_manager')
    def test_stop_engine(self, mock_manager, mock_auth, client):
        """Test stop engine endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        mock_engine = AsyncMock()
        mock_engine.stop_engine.return_value = {
            "success": True,
            "message": "Engine stopped",
            "state": "stopping"
        }
        mock_manager.return_value = mock_engine

        response = client.post(
            "/api/v1/nautilus/engine/stop",
            json={"force": False},
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Engine stopped"

    @patch('nautilus_engine_routes.get_current_user')
    @patch('nautilus_engine_routes.get_nautilus_engine_manager')
    def test_emergency_stop(self, mock_manager, mock_auth, client):
        """Test emergency stop endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        mock_engine = AsyncMock()
        mock_engine.stop_engine.return_value = {
            "success": True,
            "message": "Emergency stop executed",
            "state": "stopped"
        }
        mock_manager.return_value = mock_engine

        response = client.post(
            "/api/v1/nautilus/engine/emergency-stop",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Emergency stop executed"

    def test_unauthorized_access(self, client):
        """Test unauthorized access to protected endpoints"""
        response = client.get("/api/v1/nautilus/engine/status")
        
        # Should return 401 or redirect to login
        assert response.status_code in [401, 422]  # 422 for missing auth header

    @patch('nautilus_engine_routes.get_current_user')
    @patch('nautilus_engine_routes.get_nautilus_engine_manager')
    def test_get_engine_logs(self, mock_manager, mock_auth, client):
        """Test get engine logs endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock engine logs (simplified for this implementation)
        response = client.get(
            "/api/v1/nautilus/engine/logs?lines=10",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "logs" in data

    @patch('nautilus_engine_routes.get_current_user')
    def test_get_logs_too_many_lines(self, mock_auth, client):
        """Test get logs with too many lines requested"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        response = client.get(
            "/api/v1/nautilus/engine/logs?lines=2000",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 400
        assert "Maximum 1000 lines allowed" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
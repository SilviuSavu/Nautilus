"""
Unit tests for FastAPI main application with MessageBus integration.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

# Import after setting up the mock
with patch('messagebus_client.messagebus_client'):
    from main import app, handle_messagebus_message
    from messagebus_client import MessageBusMessage, ConnectionState, ConnectionStatus


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def mock_messagebus():
    """Mock messagebus client"""
    with patch('main.messagebus_client') as mock:
        mock.is_connected = True
        mock.connection_status = ConnectionStatus(
            state=ConnectionState.CONNECTED,
            messages_received=10,
            reconnect_attempts=0
        )
        yield mock


class TestHealthEndpoints:
    """Test health and status endpoints"""

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data
        assert "debug" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Nautilus Trader API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_api_status_endpoint(self, client, mock_messagebus):
        """Test API status endpoint"""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["api_version"] == "1.0.0"
        assert data["status"] == "operational"
        assert data["features"]["websocket"] is True
        assert data["features"]["messagebus"] is True

    def test_messagebus_status_endpoint(self, client, mock_messagebus):
        """Test MessageBus status endpoint"""
        response = client.get("/api/v1/messagebus/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["connection_state"] == "connected"
        assert data["messages_received"] == 10
        assert data["reconnect_attempts"] == 0


class TestWebSocketEndpoint:
    """Test WebSocket functionality"""

    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "connection"
            assert message["status"] == "connected"

    def test_websocket_echo(self, client):
        """Test WebSocket echo functionality"""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_text()
            
            # Send test message
            test_message = "Hello WebSocket"
            websocket.send_text(test_message)
            
            # Receive echo
            response = websocket.receive_text()
            assert response == f"Echo: {test_message}"


class TestMessageBusIntegration:
    """Test MessageBus message handling integration"""

    @pytest.mark.asyncio
    async def test_handle_messagebus_message(self):
        """Test MessageBus message handler"""
        # Create test message
        message = MessageBusMessage(
            topic="test.data",
            payload={"symbol": "EURUSD", "price": 1.0850},
            timestamp=1630000000000000000,
            message_type="market_data"
        )
        
        # Mock connection manager
        mock_manager = MagicMock()
        mock_manager.broadcast = AsyncMock()
        
        with patch('main.manager', mock_manager):
            await handle_messagebus_message(message)
            
            # Verify broadcast was called
            mock_manager.broadcast.assert_called_once()
            
            # Verify message format
            call_args = mock_manager.broadcast.call_args[0][0]
            broadcast_data = json.loads(call_args)
            
            assert broadcast_data["type"] == "messagebus"
            assert broadcast_data["topic"] == "test.data"
            assert broadcast_data["payload"]["symbol"] == "EURUSD"
            assert broadcast_data["timestamp"] == 1630000000000000000

    @pytest.mark.asyncio
    async def test_handle_messagebus_message_error(self):
        """Test MessageBus message handler with error"""
        message = MessageBusMessage(
            topic="test.data",
            payload={"key": "value"},
            timestamp=1630000000000000000
        )
        
        # Mock connection manager that raises exception
        mock_manager = MagicMock()
        mock_manager.broadcast.side_effect = Exception("Broadcast failed")
        
        with patch('main.manager', mock_manager):
            with patch('main.logging.error') as mock_log:
                # Should not raise exception
                await handle_messagebus_message(message)
                
                # Should log error
                mock_log.assert_called_once()


class TestApplicationLifespan:
    """Test application lifecycle events"""

    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test application startup lifecycle"""
        mock_messagebus = MagicMock()
        mock_messagebus.start = AsyncMock()
        mock_messagebus.add_message_handler = MagicMock()
        
        with patch('main.messagebus_client', mock_messagebus):
            from main import lifespan
            
            # Test startup
            async with lifespan(app):
                pass
            
            # Verify MessageBus was configured and started
            mock_messagebus.add_message_handler.assert_called_once()
            mock_messagebus.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self):
        """Test application shutdown lifecycle"""
        mock_messagebus = MagicMock()
        mock_messagebus.start = AsyncMock()
        mock_messagebus.stop = AsyncMock()
        mock_messagebus.add_message_handler = MagicMock()
        
        with patch('main.messagebus_client', mock_messagebus):
            from main import lifespan
            
            # Test full lifecycle
            async with lifespan(app):
                pass
            
            # Verify MessageBus was stopped
            mock_messagebus.stop.assert_called_once()


class TestCORSConfiguration:
    """Test CORS middleware configuration"""

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.get("/health")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    def test_options_request(self, client):
        """Test OPTIONS preflight request"""
        response = client.options("/api/v1/status")
        
        # Should return 200 for OPTIONS
        assert response.status_code == 200
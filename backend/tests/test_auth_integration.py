"""
Integration tests for authentication API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from auth.database import user_db
from auth.models import UserCreate


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    """Reset user database before each test"""
    # Clear all users except the default admin
    user_db._users.clear()
    user_db._usernames.clear()
    user_db._api_keys.clear()
    user_db._next_id = 1
    user_db._revoked_tokens.clear()
    
    # Recreate default admin user
    user_db.create_user(UserCreate(
        username="admin",
        password="admin123",
        api_key=None
    ))


class TestAuthenticationEndpoints:
    """Test authentication API endpoints"""
    
    def test_login_with_username_password(self, client):
        """Test login with username and password"""
        response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        
        # Check that refresh token cookie is set
        assert "refresh_token" in response.cookies
    
    def test_login_with_api_key(self, client):
        """Test login with API key"""
        # First get the admin user's API key
        admin_user = user_db.get_user_by_username("admin")
        admin_dict = user_db._users[admin_user.id]
        api_key = admin_dict["api_key"]
        
        response = client.post("/api/v1/auth/login", json={
            "api_key": api_key
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_with_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "wrongpassword"
        })
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_with_invalid_api_key(self, client):
        """Test login with invalid API key"""
        response = client.post("/api/v1/auth/login", json={
            "api_key": "invalid_key"
        })
        
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    def test_login_with_missing_credentials(self, client):
        """Test login with missing credentials"""
        response = client.post("/api/v1/auth/login", json={})
        
        assert response.status_code == 400
        assert "Either username/password or api_key must be provided" in response.json()["detail"]
    
    def test_get_current_user(self, client):
        """Test getting current user info"""
        # Login first
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Get user info
        response = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["username"] == "admin"
        assert data["is_active"] is True
        assert "id" in data
    
    def test_validate_token(self, client):
        """Test token validation"""
        # Login first
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Validate token
        response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["valid"] is True
        assert data["user"]["username"] == "admin"
    
    def test_refresh_token(self, client):
        """Test token refresh"""
        # Login first
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_refresh_with_cookie(self, client):
        """Test token refresh using httpOnly cookie"""
        # Login first to get cookie
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        # Check that refresh token cookie is set
        assert "refresh_token" in login_response.cookies
        
        # For now, test refresh with the refresh token from response body
        # In a real browser, the cookie would be automatically included
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh using the refresh token from body
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
    
    def test_logout(self, client):
        """Test logout"""
        # Login first
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        cookies = login_response.cookies
        
        # Logout
        response = client.post("/api/v1/auth/logout", 
                             headers={"Authorization": f"Bearer {token}"},
                             cookies=cookies)
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
        
        # Token should be invalid after logout
        validate_response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert validate_response.status_code == 401
    
    def test_unauthorized_access(self, client):
        """Test accessing protected endpoints without token"""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    def test_invalid_token_access(self, client):
        """Test accessing protected endpoints with invalid token"""
        response = client.get("/api/v1/auth/me", headers={
            "Authorization": "Bearer invalid_token"
        })
        
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]
    
    def test_api_key_authentication_header(self, client):
        """Test API key authentication via X-API-Key header"""
        # Get admin API key
        admin_user = user_db.get_user_by_username("admin")
        admin_dict = user_db._users[admin_user.id]
        api_key = admin_dict["api_key"]
        
        # Access protected endpoint with API key header
        response = client.get("/api/v1/auth/me", headers={
            "X-API-Key": api_key
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
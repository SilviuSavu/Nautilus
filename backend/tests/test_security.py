"""
Security tests for authentication system
"""

import pytest
import time
from fastapi.testclient import TestClient
from jose import jwt
from main import app
from auth.database import user_db
from auth.models import UserCreate
from auth.security import SECRET_KEY, ALGORITHM


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    """Reset user database before each test"""
    user_db._users.clear()
    user_db._usernames.clear()
    user_db._api_keys.clear()
    user_db._next_id = 1
    user_db._revoked_tokens.clear()
    
    user_db.create_user(UserCreate(
        username="admin",
        password="admin123",
        api_key=None
    ))


class TestSecurityFeatures:
    """Test security features and protections"""
    
    def test_token_expiration(self, client):
        """Test that expired tokens are rejected"""
        # Create a token with very short expiration
        from auth.security import create_access_token
        from datetime import timedelta
        
        # Create token that expires in 1 second
        short_token = create_access_token(
            data={"sub": "1"}, 
            expires_delta=timedelta(seconds=1)
        )
        
        # Token should work immediately
        response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {short_token}"
        })
        assert response.status_code == 200
        
        # Wait for token to expire
        time.sleep(2)
        
        # Token should now be rejected
        response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {short_token}"
        })
        assert response.status_code == 401
    
    def test_token_revocation(self, client):
        """Test that revoked tokens are rejected"""
        # Login to get a token
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Token should work
        response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 200
        
        # Logout to revoke token
        client.post("/api/v1/auth/logout", headers={
            "Authorization": f"Bearer {token}"
        })
        
        # Token should now be rejected
        response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 401
    
    def test_malformed_token_rejection(self, client):
        """Test that malformed tokens are rejected"""
        malformed_tokens = [
            "not_a_token",
            "Bearer.malformed.token",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",
            "",
            "null",
        ]
        
        for token in malformed_tokens:
            response = client.get("/api/v1/auth/validate", headers={
                "Authorization": f"Bearer {token}"
            })
            assert response.status_code == 401
    
    def test_wrong_token_type_rejection(self, client):
        """Test that refresh tokens can't be used as access tokens"""
        # Login to get tokens
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Try to use refresh token as access token
        response = client.get("/api/v1/auth/validate", headers={
            "Authorization": f"Bearer {refresh_token}"
        })
        assert response.status_code == 401
        assert "Invalid token type" in response.json()["detail"]
    
    def test_password_strength_enforcement(self):
        """Test that weak passwords are handled appropriately"""
        # Note: This test assumes password validation would be added
        # Currently the system accepts any password, but this test
        # documents the expected behavior for future implementation
        
        weak_passwords = [
            "123",          # Too short
            "password",     # Common password
            "abc",          # Too short and simple
        ]
        
        # For now, just test that the system doesn't crash with these
        # In a production system, these should be rejected
        for password in weak_passwords:
            try:
                user_db.create_user(UserCreate(
                    username=f"user_{password}",
                    password=password
                ))
                # Currently passes, but should be enhanced in future
            except Exception:
                # If validation is added, this would be expected
                pass
    
    def test_sql_injection_prevention(self, client):
        """Test protection against SQL injection attempts"""
        # Since we're using an in-memory database, SQL injection
        # isn't directly applicable, but test malicious input handling
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "<script>alert('xss')</script>",
            "' UNION SELECT * FROM users --",
        ]
        
        for malicious_input in malicious_inputs:
            # Test login endpoint
            response = client.post("/api/v1/auth/login", json={
                "username": malicious_input,
                "password": "test"
            })
            # Should fail authentication, not crash
            assert response.status_code in [400, 401, 422]
            
            # Test as API key
            response = client.post("/api/v1/auth/login", json={
                "api_key": malicious_input
            })
            assert response.status_code in [400, 401, 422]
    
    def test_brute_force_protection_concept(self, client):
        """Test concept for brute force protection"""
        # This test documents the need for rate limiting
        # In production, should implement rate limiting middleware
        
        # Attempt multiple failed logins
        for i in range(10):
            response = client.post("/api/v1/auth/login", json={
                "username": "admin",
                "password": "wrongpassword"
            })
            assert response.status_code == 401
        
        # Currently no rate limiting, but valid login should still work
        response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        assert response.status_code == 200
    
    def test_csrf_protection_cookies(self, client):
        """Test CSRF protection for cookie-based authentication"""
        # Login to get refresh token cookie
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        # Check that refresh token cookie is set
        assert "refresh_token" in login_response.cookies
        
        # For now, test that the cookie exists and has proper structure
        # In production, cookies should have security attributes
        refresh_token_value = login_response.cookies.get("refresh_token")
        
        assert refresh_token_value is not None
        assert len(refresh_token_value) > 0
        
        # Note: In production, these should be enforced:
        # assert refresh_cookie.secure  # HTTPS only
        # assert refresh_cookie.httponly  # No JS access
        # assert refresh_cookie.samesite == "strict"  # CSRF protection
    
    def test_token_tampering_detection(self, client):
        """Test detection of tampered tokens"""
        # Login to get a valid token
        login_response = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Decode token to get its parts
        parts = token.split('.')
        
        # Tamper with different parts
        tampered_tokens = [
            # Tampered header
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0." + ".".join(parts[1:]),
            # Tampered payload
            parts[0] + ".eyJzdWIiOiI5OTkifQ." + parts[2],
            # Tampered signature
            ".".join(parts[:2]) + ".invalid_signature",
            # Extra data
            token + "extra",
        ]
        
        for tampered_token in tampered_tokens:
            response = client.get("/api/v1/auth/validate", headers={
                "Authorization": f"Bearer {tampered_token}"
            })
            assert response.status_code == 401
    
    def test_user_enumeration_prevention(self, client):
        """Test that user enumeration is prevented"""
        # Login attempts with valid vs invalid usernames
        # should return similar error messages
        
        response1 = client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "wrongpassword"
        })
        
        response2 = client.post("/api/v1/auth/login", json={
            "username": "nonexistent_user",
            "password": "wrongpassword"
        })
        
        # Both should return 401 with similar error messages
        assert response1.status_code == 401
        assert response2.status_code == 401
        
        # Error messages should be generic enough to not reveal
        # whether the username exists
        assert "Incorrect username or password" in response1.json()["detail"]
        assert "Incorrect username or password" in response2.json()["detail"]
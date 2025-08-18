"""
Unit tests for authentication functionality
"""

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from auth.security import (
    hash_password, 
    verify_password, 
    create_access_token, 
    create_refresh_token,
    verify_token,
    generate_api_key
)
from auth.database import UserDatabase
from auth.models import UserCreate


class TestSecurity:
    """Test security utilities"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = hash_password(password)
        
        # Hash should not be the same as original password
        assert hashed != password
        
        # Verification should work
        assert verify_password(password, hashed) is True
        
        # Wrong password should fail
        assert verify_password("wrong_password", hashed) is False
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        data = {"sub": "123"}
        
        # Create access token
        access_token = create_access_token(data)
        assert access_token is not None
        
        # Verify access token
        token_data = verify_token(access_token, "access")
        assert token_data.sub == "123"
        assert token_data.type == "access"
        
        # Create refresh token
        refresh_token = create_refresh_token(data)
        assert refresh_token is not None
        
        # Verify refresh token
        token_data = verify_token(refresh_token, "refresh")
        assert token_data.sub == "123"
        assert token_data.type == "refresh"
    
    def test_token_type_validation(self):
        """Test that tokens can't be used for wrong type"""
        data = {"sub": "123"}
        
        access_token = create_access_token(data)
        refresh_token = create_refresh_token(data)
        
        # Access token should not work as refresh token
        with pytest.raises(Exception):
            verify_token(access_token, "refresh")
        
        # Refresh token should not work as access token
        with pytest.raises(Exception):
            verify_token(refresh_token, "access")
    
    def test_api_key_generation(self):
        """Test API key generation"""
        api_key1 = generate_api_key()
        api_key2 = generate_api_key()
        
        # Should generate different keys
        assert api_key1 != api_key2
        
        # Should be reasonable length
        assert len(api_key1) > 32


class TestUserDatabase:
    """Test user database functionality"""
    
    def test_user_creation(self):
        """Test user creation"""
        db = UserDatabase()
        
        user_data = UserCreate(
            username="testuser",
            password="testpass123"
        )
        
        user = db.create_user(user_data)
        
        assert user.username == "testuser"
        assert user.id > 0
        assert user.is_active is True
        assert user.created_at is not None
    
    def test_duplicate_username_fails(self):
        """Test that duplicate usernames are rejected"""
        db = UserDatabase()
        
        user_data = UserCreate(
            username="duplicate",
            password="testpass123"
        )
        
        # First creation should succeed
        db.create_user(user_data)
        
        # Second creation should fail
        with pytest.raises(ValueError, match="Username already exists"):
            db.create_user(user_data)
    
    def test_user_authentication(self):
        """Test user authentication with password"""
        db = UserDatabase()
        
        user_data = UserCreate(
            username="authtest",
            password="mypassword123"
        )
        
        created_user = db.create_user(user_data)
        
        # Correct credentials should work
        auth_user = db.authenticate_user("authtest", "mypassword123")
        assert auth_user is not None
        assert auth_user.id == created_user.id
        assert auth_user.last_login is not None
        
        # Wrong password should fail
        auth_user = db.authenticate_user("authtest", "wrongpassword")
        assert auth_user is None
        
        # Wrong username should fail
        auth_user = db.authenticate_user("wronguser", "mypassword123")
        assert auth_user is None
    
    def test_api_key_authentication(self):
        """Test API key authentication"""
        db = UserDatabase()
        
        user_data = UserCreate(
            username="apitest",
            password="testpass123",
            api_key="test_api_key_12345"
        )
        
        created_user = db.create_user(user_data)
        
        # Correct API key should work
        auth_user = db.authenticate_api_key("test_api_key_12345")
        assert auth_user is not None
        assert auth_user.id == created_user.id
        
        # Wrong API key should fail
        auth_user = db.authenticate_api_key("wrong_api_key")
        assert auth_user is None
    
    def test_token_revocation(self):
        """Test token revocation functionality"""
        db = UserDatabase()
        
        jti = "test_token_123"
        
        # Token should not be revoked initially
        assert db.is_token_revoked(jti) is False
        
        # Revoke token
        db.revoke_token(jti)
        
        # Token should now be revoked
        assert db.is_token_revoked(jti) is True
    
    def test_user_retrieval(self):
        """Test user retrieval methods"""
        db = UserDatabase()
        
        user_data = UserCreate(
            username="retrievetest",
            password="testpass123"
        )
        
        created_user = db.create_user(user_data)
        
        # Get by ID
        user_by_id = db.get_user_by_id(created_user.id)
        assert user_by_id is not None
        assert user_by_id.username == "retrievetest"
        
        # Get by username
        user_by_username = db.get_user_by_username("retrievetest")
        assert user_by_username is not None
        assert user_by_username.id == created_user.id
        
        # Non-existent user should return None
        assert db.get_user_by_id(99999) is None
        assert db.get_user_by_username("nonexistent") is None
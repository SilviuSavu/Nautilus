"""
Simple in-memory database for authentication
In production, this would be replaced with a proper database like PostgreSQL
"""

from typing import List
from datetime import datetime, timezone
from auth.models import User, UserCreate
from auth.security import hash_password, verify_password, generate_api_key


class UserDatabase:
    """Simple in-memory user database"""
    
    def __init__(self):
        self._users: dict[int, Dict] = {}
        self._usernames: dict[str, int] = {}
        self._api_keys: dict[str, int] = {}
        self._next_id = 1
        self._revoked_tokens: set = set()
        
        # Create default admin user for testing
        self.create_user(UserCreate(
            username="admin", password="admin123", api_key=None
        ))
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        if user_data.username in self._usernames:
            raise ValueError("Username already exists")
        
        user_id = self._next_id
        self._next_id += 1
        
        # Generate API key if not provided
        api_key = user_data.api_key or generate_api_key()
        
        user_dict = {
            "id": user_id, "username": user_data.username, "password_hash": hash_password(user_data.password), "api_key": api_key, "is_active": True, "created_at": datetime.now(timezone.utc), "last_login": None
        }
        
        self._users[user_id] = user_dict
        self._usernames[user_data.username] = user_id
        self._api_keys[api_key] = user_id
        
        return User(
            id=user_id, username=user_data.username, is_active=True, created_at=user_dict["created_at"]
        )
    
    def get_user_by_id(self, user_id: int) -> User | None:
        """Get user by ID"""
        user_dict = self._users.get(user_id)
        if not user_dict:
            return None
        
        return User(
            id=user_dict["id"], username=user_dict["username"], is_active=user_dict["is_active"], created_at=user_dict["created_at"], last_login=user_dict["last_login"]
        )
    
    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username"""
        user_id = self._usernames.get(username)
        if not user_id:
            return None
        return self.get_user_by_id(user_id)
    
    def get_user_by_api_key(self, api_key: str) -> User | None:
        """Get user by API key"""
        user_id = self._api_keys.get(api_key)
        if not user_id:
            return None
        return self.get_user_by_id(user_id)
    
    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user with username and password"""
        user_id = self._usernames.get(username)
        if not user_id:
            return None
        
        user_dict = self._users[user_id]
        if not user_dict["is_active"]:
            return None
        
        if not verify_password(password, user_dict["password_hash"]):
            return None
        
        # Update last login
        user_dict["last_login"] = datetime.now(timezone.utc)
        
        return User(
            id=user_dict["id"], username=user_dict["username"], is_active=user_dict["is_active"], created_at=user_dict["created_at"], last_login=user_dict["last_login"]
        )
    
    def authenticate_api_key(self, api_key: str) -> User | None:
        """Authenticate user with API key"""
        user_id = self._api_keys.get(api_key)
        if not user_id:
            return None
        
        user_dict = self._users[user_id]
        if not user_dict["is_active"]:
            return None
        
        # Update last login
        user_dict["last_login"] = datetime.now(timezone.utc)
        
        return User(
            id=user_dict["id"], username=user_dict["username"], is_active=user_dict["is_active"], created_at=user_dict["created_at"], last_login=user_dict["last_login"]
        )
    
    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI"""
        self._revoked_tokens.add(jti)
    
    def is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked"""
        return jti in self._revoked_tokens
    
    def list_users(self) -> list[User]:
        """List all users"""
        users = []
        for user_dict in self._users.values():
            users.append(User(
                id=user_dict["id"], username=user_dict["username"], is_active=user_dict["is_active"], created_at=user_dict["created_at"], last_login=user_dict["last_login"]
            ))
        return users


# Global database instance
user_db = UserDatabase()
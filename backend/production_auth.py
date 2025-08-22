"""
Production-Ready Authentication System
=====================================

Comprehensive authentication and authorization system for production deployment.
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import redis
from dataclasses import dataclass
from enum import Enum


class UserRole(str, Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst" 
    VIEWER = "viewer"


class PermissionLevel(str, Enum):
    """Permission levels for different operations."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class User:
    """User model for authentication."""
    id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None


class AuthConfig:
    """Authentication configuration."""
    
    def __init__(self):
        # JWT Configuration
        self.secret_key = secrets.token_urlsafe(32)  # Generate secure secret
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Redis for token blacklist and sessions
        self.redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [
                "read:all", "write:all", "execute:all", "admin:all"
            ],
            UserRole.TRADER: [
                "read:market_data", "read:portfolio", "read:trading",
                "write:orders", "write:portfolio", "execute:trades"
            ],
            UserRole.ANALYST: [
                "read:market_data", "read:portfolio", "read:analytics",
                "write:strategies", "write:reports"
            ],
            UserRole.VIEWER: [
                "read:market_data", "read:portfolio", "read:analytics"
            ]
        }


class ProductionAuthService:
    """Production-grade authentication service."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.security = HTTPBearer()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return self.config.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.config.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.config.access_token_expire_minutes
        )
        
        to_encode = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": user.permissions,
            "exp": expire,
            "type": "access"
        }
        
        return jwt.encode(
            to_encode, 
            self.config.secret_key, 
            algorithm=self.config.algorithm
        )
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(
            days=self.config.refresh_token_expire_days
        )
        
        to_encode = {
            "sub": user.id,
            "exp": expire,
            "type": "refresh"
        }
        
        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            if self.is_token_blacklisted(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist."""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}
            )
            
            # Set expiry time for blacklist entry
            exp = payload.get("exp", 0)
            if exp > 0:
                ttl = max(0, exp - int(datetime.now(timezone.utc).timestamp()))
                self.config.redis_client.setex(
                    f"blacklist:{token}",
                    ttl,
                    "revoked"
                )
        except jwt.JWTError:
            # If we can't decode, add to permanent blacklist
            self.config.redis_client.set(f"blacklist:{token}", "revoked")
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        return self.config.redis_client.exists(f"blacklist:{token}")
    
    def get_user_permissions(self, role: UserRole) -> List[str]:
        """Get permissions for user role."""
        return self.config.role_permissions.get(role, [])
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        # Admin has all permissions
        if "admin:all" in user_permissions:
            return True
        
        # Check exact permission match
        if required_permission in user_permissions:
            return True
        
        # Check wildcard permissions
        permission_parts = required_permission.split(":")
        if len(permission_parts) == 2:
            action, resource = permission_parts
            wildcard = f"{action}:all"
            if wildcard in user_permissions:
                return True
        
        return False


# Global instances
auth_config = AuthConfig()
auth_service = ProductionAuthService(auth_config)


# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    # In a real application, you'd fetch user from database
    # For now, create user from token payload
    user = User(
        id=payload["sub"],
        username=payload["username"],
        email=f"{payload['username']}@example.com",  # Would come from DB
        role=UserRole(payload["role"]),
        permissions=payload["permissions"]
    )
    
    return user


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(user: User = Depends(get_current_user)) -> User:
        if not auth_service.check_permission(user.permissions, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return user
    
    return permission_checker


def require_role(required_role: UserRole):
    """Decorator to require specific role."""
    def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {required_role.value}"
            )
        return user
    
    return role_checker


# Optional authentication for development
async def get_current_user_optional() -> Optional[User]:
    """Optional authentication for development mode."""
    try:
        # In development, create a mock admin user
        return User(
            id="dev-admin",
            username="dev-admin", 
            email="admin@dev.local",
            role=UserRole.ADMIN,
            permissions=auth_service.get_user_permissions(UserRole.ADMIN)
        )
    except:
        return None
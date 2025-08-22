"""
Production Authentication Routes
===============================

Authentication endpoints for production deployment.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

from production_auth import (
    auth_service, 
    User, 
    UserRole, 
    get_current_user,
    require_permission,
    require_role
)


router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# Request/Response models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class UserCreateRequest(BaseModel):
    """User creation request model."""
    username: str
    email: EmailStr
    password: str
    role: UserRole


class UserResponse(BaseModel):
    """User response model."""
    id: str
    username: str
    email: str
    role: str
    permissions: list[str]
    is_active: bool
    created_at: Optional[datetime]
    last_login: Optional[datetime]


# Mock user database (in production, use real database)
MOCK_USERS = {
    "admin": {
        "id": "admin-001",
        "username": "admin",
        "email": "admin@nautilus.com",
        "password": auth_service.hash_password("admin123"),
        "role": UserRole.ADMIN,
        "is_active": True,
        "created_at": datetime.now()
    },
    "trader": {
        "id": "trader-001", 
        "username": "trader",
        "email": "trader@nautilus.com",
        "password": auth_service.hash_password("trader123"),
        "role": UserRole.TRADER,
        "is_active": True,
        "created_at": datetime.now()
    }
}


@router.post("/login", response_model=LoginResponse, summary="User Login")
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT tokens.
    
    **Test Credentials:**
    - Username: `admin` / Password: `admin123` (Admin access)
    - Username: `trader` / Password: `trader123` (Trader access)
    """
    
    # Find user in mock database
    user_data = MOCK_USERS.get(request.username)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Verify password
    if not auth_service.verify_password(request.password, user_data["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Check if user is active
    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    
    # Create user object
    user = User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        permissions=auth_service.get_user_permissions(user_data["role"]),
        is_active=user_data["is_active"],
        created_at=user_data["created_at"]
    )
    
    # Generate tokens
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=auth_service.config.access_token_expire_minutes * 60,
        user={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": user.permissions
        }
    )


@router.post("/refresh", response_model=dict, summary="Refresh Access Token")
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    payload = auth_service.verify_token(request.refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload["sub"]
    
    # Find user (in production, fetch from database)
    user_data = None
    for user in MOCK_USERS.values():
        if user["id"] == user_id:
            user_data = user
            break
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create user object
    user = User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        permissions=auth_service.get_user_permissions(user_data["role"]),
        is_active=user_data["is_active"],
        created_at=user_data["created_at"]
    )
    
    # Generate new access token
    access_token = auth_service.create_access_token(user)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": auth_service.config.access_token_expire_minutes * 60
    }


@router.post("/logout", summary="User Logout")
async def logout(user: User = Depends(get_current_user)):
    """
    Logout user and invalidate token.
    """
    # In a real implementation, you'd blacklist the current token
    # For now, we'll just return success
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse, summary="Get Current User")
async def get_me(user: User = Depends(get_current_user)):
    """
    Get current user information.
    """
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.value,
        permissions=user.permissions,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.get("/users", response_model=list[UserResponse], summary="List Users")
async def list_users(user: User = Depends(require_role(UserRole.ADMIN))):
    """
    List all users (Admin only).
    """
    users = []
    for user_data in MOCK_USERS.values():
        users.append(UserResponse(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"].value,
            permissions=auth_service.get_user_permissions(user_data["role"]),
            is_active=user_data["is_active"],
            created_at=user_data["created_at"],
            last_login=None
        ))
    
    return users


@router.get("/permissions", summary="Get Available Permissions")
async def get_permissions(user: User = Depends(get_current_user)):
    """
    Get available permissions for current user role.
    """
    return {
        "user_permissions": user.permissions,
        "role": user.role.value,
        "all_permissions": auth_service.config.role_permissions
    }
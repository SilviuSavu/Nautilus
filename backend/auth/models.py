"""
Authentication models and schemas
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class UserCreate(BaseModel):
    """User creation schema"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    api_key: Optional[str] = Field(None, max_length=256)


class UserLogin(BaseModel):
    """User login schema"""
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None


class User(BaseModel):
    """User model"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    username: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request"""
    refresh_token: str


class TokenPayload(BaseModel):
    """JWT token payload"""
    sub: str  # subject (user_id)
    exp: int  # expiration time
    iat: int  # issued at
    jti: str  # JWT ID
    type: str  # token type: 'access' or 'refresh'
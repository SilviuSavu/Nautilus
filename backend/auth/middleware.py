"""
Authentication middleware and dependencies
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth.security import verify_token
from auth.database import user_db
from auth.models import User, TokenPayload

security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """Get current authenticated user from JWT token or API key"""
    
    # Check for API key in headers first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = user_db.authenticate_api_key(api_key)
        if user:
            return user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Check for JWT token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify JWT token
    token_data = verify_token(credentials.credentials, "access")
    
    # Check if token is revoked
    if user_db.is_token_revoked(token_data.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = user_db.get_user_by_id(int(token_data.sub))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def verify_refresh_token(token: str) -> TokenPayload:
    """Verify refresh token"""
    token_data = verify_token(token, "refresh")
    
    # Check if token is revoked
    if user_db.is_token_revoked(token_data.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )
    
    return token_data
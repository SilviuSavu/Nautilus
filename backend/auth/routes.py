"""
Authentication API routes
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPAuthorizationCredentials
from auth.models import UserLogin, Token, TokenRefresh, User
from auth.security import (
    create_access_token, 
    create_refresh_token, 
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from auth.database import user_db
from auth.middleware import get_current_user, verify_refresh_token, security

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, response: Response):
    """
    Login with username/password or API key
    Returns JWT access and refresh tokens
    """
    user = None
    
    # Try API key authentication first
    if user_credentials.api_key:
        user = user_db.authenticate_api_key(user_credentials.api_key)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Try username/password authentication
    elif user_credentials.username and user_credentials.password:
        user = user_db.authenticate_user(
            user_credentials.username, 
            user_credentials.password
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either username/password or api_key must be provided"
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, 
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    # Set secure httpOnly cookie for refresh token
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,  # Use HTTPS in production
        samesite="strict",
        max_age=7 * 24 * 60 * 60  # 7 days
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: Request, 
    response: Response
):
    """
    Refresh access token using refresh token
    Accepts refresh token from request body or httpOnly cookie
    """
    refresh_token = None
    
    # Try to get refresh token from request body first
    try:
        body = await request.json()
        if "refresh_token" in body:
            refresh_token = body["refresh_token"]
    except:
        # No JSON body or invalid JSON, that's okay
        pass
    
    # Fallback to httpOnly cookie
    if not refresh_token:
        refresh_token = request.cookies.get("refresh_token")
    
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not provided"
        )
    
    # Verify refresh token
    try:
        token_data = verify_refresh_token(refresh_token)
    except HTTPException:
        # Clear invalid refresh token cookie
        response.delete_cookie("refresh_token")
        raise
    
    # Get user
    user = user_db.get_user_by_id(int(token_data.sub))
    if not user or not user.is_active:
        response.delete_cookie("refresh_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"sub": str(user.id)}, 
        expires_delta=access_token_expires
    )
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    # Revoke old refresh token
    user_db.revoke_token(token_data.jti)
    
    # Set new refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=7 * 24 * 60 * 60
    )
    
    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Logout user by revoking current access token and refresh token
    """
    # Revoke access token
    if credentials:
        from auth.security import verify_token
        try:
            token_data = verify_token(credentials.credentials, "access")
            user_db.revoke_token(token_data.jti)
        except HTTPException:
            pass  # Token already invalid
    
    # Revoke refresh token from cookie
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        try:
            token_data = verify_refresh_token(refresh_token)
            user_db.revoke_token(token_data.jti)
        except HTTPException:
            pass  # Token already invalid
    
    # Clear refresh token cookie
    response.delete_cookie("refresh_token")
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information
    """
    return current_user


@router.get("/validate")
async def validate_token(current_user: User = Depends(get_current_user)):
    """
    Validate current token and return user info
    """
    return {
        "valid": True,
        "user": current_user,
        "message": "Token is valid"
    }


# DEBUG ENDPOINT REMOVED FOR PRODUCTION
# The debug endpoint /debug/admin-api-key has been removed for security
# In production, API keys should be managed through proper user management system
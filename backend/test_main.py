"""
Minimal FastAPI backend for testing authentication and frontend integration
Bypasses PostgreSQL and Redis dependencies for browser testing
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings

from auth.routes import router as auth_router

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", 8002))
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002"
    
    model_config = {"env_file": ".env"}

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Minimal application lifespan events"""
    print(f"Starting Minimal Nautilus Trader Backend - Environment: {settings.environment}")
    print(f"CORS Origins: {settings.cors_origins}")
    yield
    print("Shutting down Minimal Nautilus Trader Backend")

# Create FastAPI application
app = FastAPI(
    title="Nautilus Trader API (Test Mode)",
    description="Minimal REST API for testing authentication",
    version="1.0.0-test",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes (router already has prefix)
app.include_router(auth_router, tags=["authentication"])

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "mode": "test",
        "message": "Backend ready for authentication testing"
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "API is working", "status": "ok"}

@app.get("/api/v1/auth/debug/admin-api-key")
async def get_admin_api_key():
    """Debug endpoint to get admin API key for testing"""
    from auth.database import user_db
    admin_user = user_db.get_user_by_username("admin")
    if not admin_user:
        return {"error": "Admin user not found"}
    
    # Get API key from internal database
    user_dict = user_db._users.get(admin_user.id)
    if not user_dict:
        return {"error": "Admin user data not found"}
    
    return {
        "username": "admin",
        "user_id": admin_user.id,
        "api_key": user_dict["api_key"],
        "created_at": admin_user.created_at.isoformat(),
        "warning": "This is a debug endpoint - remove in production!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "test_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )
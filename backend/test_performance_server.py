#!/usr/bin/env python3
"""
Simple test server for Performance routes only
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from performance_routes import router as performance_router

# Create FastAPI app
app = FastAPI(
    title="Performance API Test Server",
    description="Test server for Performance monitoring endpoints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include performance routes
app.include_router(performance_router)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Performance API server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
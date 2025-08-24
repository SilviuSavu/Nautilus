#!/usr/bin/env python3
"""
Risk Engine - Main entry point (now modularized)
This file maintains backward compatibility while using the new modular architecture

FIXED: Reduced from 27,492 tokens to manageable size by modularizing into:
- models.py: Data classes and enums
- services.py: Business logic 
- routes.py: FastAPI routes
- engine.py: Main orchestrator
"""

# Import the modular engine
try:
    from engine import RiskEngine
    from enhanced_risk_api import router as enhanced_api_router
except ImportError:
    # Handle Docker container environment where module path might be different
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from engine import RiskEngine
    try:
        from enhanced_risk_api import router as enhanced_api_router
    except ImportError:
        print("Warning: Enhanced risk API not available - continuing with basic functionality")
        enhanced_api_router = None

# Create the global risk engine instance
risk_engine = RiskEngine()

# For backward compatibility, expose the app directly
app = risk_engine.app

# Add enhanced risk API endpoints if available
if enhanced_api_router is not None:
    app.include_router(enhanced_api_router, prefix="", tags=["Enhanced Risk"])
    print("✅ Enhanced Risk API endpoints added successfully")
else:
    print("⚠️  Enhanced Risk API endpoints not available - running in basic mode")

# Main execution point
if __name__ == "__main__":
    import os
    import uvicorn
    
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    
    print(f"Starting Risk Engine on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
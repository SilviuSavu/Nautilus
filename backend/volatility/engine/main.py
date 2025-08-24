#!/usr/bin/env python3
"""
Nautilus Volatility Forecasting Engine - Main Entry Point

This is the main entry point for the containerized volatility forecasting service.
Runs on port 9000 with M4 Max hardware acceleration and real-time streaming.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

# Add the volatility package to Python path
sys.path.insert(0, '/app')

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import volatility engine components
from volatility.engine.volatility_engine import get_engine, shutdown_engine
from volatility.api.routes import volatility_router
from volatility.api.websocket import websocket_router
from volatility.config import VolatilityConfig

# Configure logging
def setup_logging():
    """Configure logging for the volatility engine"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/app/logs/volatility_engine.log', mode='a')
        ]
    )
    
    # Set specific log levels for external libraries
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown of the volatility engine.
    """
    try:
        # Startup
        logger.info("🚀 Starting Nautilus Volatility Forecasting Engine...")
        
        # Initialize engine
        engine = await get_engine()
        logger.info(f"✅ Volatility engine initialized with {len(engine.config.get_enabled_models())} model types")
        
        # Log hardware acceleration status
        hw_config = engine.config.hardware
        logger.info(f"🔧 Hardware acceleration - GPU: {hw_config.use_metal_gpu}, "
                   f"Neural Engine: {hw_config.use_neural_engine}, "
                   f"CPU Opt: {hw_config.use_cpu_optimization}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start volatility engine: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Nautilus Volatility Forecasting Engine...")
        await shutdown_engine()
        logger.info("✅ Volatility engine shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Nautilus Volatility Forecasting Engine",
    description="Advanced volatility forecasting with M4 Max hardware acceleration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(volatility_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    config = VolatilityConfig.from_environment()
    
    return {
        "service": "Nautilus Volatility Forecasting Engine",
        "version": "1.0.0",
        "status": "operational",
        "port": config.api.container_port,
        "hardware_acceleration": {
            "metal_gpu": config.hardware.use_metal_gpu,
            "neural_engine": config.hardware.use_neural_engine,
            "cpu_optimization": config.hardware.use_cpu_optimization,
            "auto_routing": config.hardware.auto_hardware_routing
        },
        "capabilities": [
            "GARCH/EGARCH volatility models",
            "Real-time estimators (Garman-Klass, Yang-Zhang)", 
            "Stochastic volatility models (Heston, SABR)",
            "Ensemble forecasting with dynamic weighting",
            "WebSocket streaming",
            "M4 Max hardware acceleration"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/volatility/health",
            "websocket": "/ws/volatility"
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint"""
    return JSONResponse(content={}, status_code=204)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred in the volatility engine"
        }
    )


async def main():
    """Main function to run the volatility engine"""
    try:
        # Configuration
        config = VolatilityConfig.from_environment()
        
        # Server configuration
        server_config = {
            "host": config.api.host,
            "port": config.api.container_port,
            "workers": 1,  # Use 1 worker for now due to shared state
            "log_level": config.api.log_level.lower(),
            "access_log": True,
            "loop": "asyncio",
            "reload": False,
            "app": "main:app"
        }
        
        logger.info(f"🌐 Starting server on {server_config['host']}:{server_config['port']}")
        logger.info(f"📊 Configuration: {config.ensemble.method.value} ensemble, "
                   f"{len(config.get_enabled_models())} model types")
        
        # Run server
        await uvicorn.run(**server_config)
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("/app/logs", exist_ok=True)
    
    # Print startup banner
    print("=" * 60)
    print("🚀 NAUTILUS VOLATILITY FORECASTING ENGINE")
    print("   Advanced volatility modeling with M4 Max acceleration")
    print("=" * 60)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️  Received shutdown signal")
        print("\n✅ Volatility engine stopped")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
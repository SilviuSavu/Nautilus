"""
Phase 8 FastAPI Integration Example
Demonstrates how to integrate Phase 8 startup service with the main Nautilus FastAPI application.

This file shows the integration patterns and can be used to update main.py.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

# Import the Phase 8 startup service
from phase8_startup_service import (
    phase8_startup_service,
    phase8_lifespan_context,
    phase8_startup,
    phase8_shutdown,
    get_phase8_health,
    get_phase8_metrics,
    setup_phase8_messagebus_handlers,
    Phase8Settings,
    ServiceStatus,
    HealthStatus
)

# Import existing services
from messagebus_client import messagebus_client


@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """
    Enhanced lifespan context manager that includes Phase 8 initialization.
    This should replace or extend the existing lifespan function in main.py.
    """
    print("🚀 Starting Enhanced Nautilus Trader Backend with Phase 8 Security...")
    
    # Existing startup code from main.py goes here...
    # Configure MessageBus client with settings
    # messagebus_client.redis_host = settings.redis_host
    # messagebus_client.redis_port = settings.redis_port
    # ... existing initialization ...
    
    try:
        # Start Phase 8 services after core services are initialized
        print("🔐 Initializing Phase 8 Autonomous Security Operations...")
        await phase8_startup()
        
        # Setup Phase 8 message bus handlers
        await setup_phase8_messagebus_handlers()
        
        print("✅ Phase 8 Security Operations initialized successfully")
        
    except Exception as e:
        print(f"⚠ Phase 8 startup error: {e}")
        # Continue without Phase 8 if it fails (graceful degradation)
    
    # Application ready to serve requests
    yield
    
    # Shutdown Phase 8 services
    try:
        print("🛑 Stopping Phase 8 Security Operations...")
        await phase8_shutdown()
        print("✅ Phase 8 Security Operations stopped")
    except Exception as e:
        print(f"⚠ Phase 8 shutdown error: {e}")
    
    # Existing shutdown code from main.py goes here...
    # Stop other services...


# Health check endpoints for Phase 8
async def phase8_health_endpoint():
    """Health check endpoint for Phase 8 services"""
    try:
        health_status = get_phase8_health()
        
        # Return appropriate HTTP status based on health
        if health_status["overall_health"] == HealthStatus.HEALTHY.value:
            status_code = 200
        elif health_status["overall_health"] == HealthStatus.DEGRADED.value:
            status_code = 206  # Partial Content
        else:
            status_code = 503  # Service Unavailable
            
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Health check failed: {str(e)}"}, 
            status_code=500
        )


async def phase8_metrics_endpoint():
    """Detailed metrics endpoint for Phase 8 services"""
    try:
        metrics = get_phase8_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Metrics collection failed: {str(e)}"}, 
            status_code=500
        )


async def phase8_service_status_endpoint(service_name: str):
    """Get status of a specific Phase 8 service"""
    try:
        all_status = phase8_startup_service.get_service_status()
        
        if service_name not in all_status:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        return JSONResponse(content=all_status[service_name].__dict__)
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"error": f"Service status check failed: {str(e)}"}, 
            status_code=500
        )


# Example of how to create the FastAPI app with Phase 8 integration
def create_app_with_phase8() -> FastAPI:
    """
    Create FastAPI application with Phase 8 integration.
    This shows how to modify the existing app creation in main.py.
    """
    
    # Create FastAPI app with enhanced lifespan
    app = FastAPI(
        title="Nautilus Trader with Phase 8 Security",
        description="Enterprise Trading Platform with Autonomous Security Operations",
        version="2.0.0",
        lifespan=enhanced_lifespan  # Use enhanced lifespan with Phase 8
    )
    
    # Add existing middleware and routes...
    # app.add_middleware(CORSMiddleware, ...)
    # app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    # ... all existing routers ...
    
    # Add Phase 8 health and monitoring endpoints
    app.add_api_route(
        "/api/v1/phase8/health", 
        phase8_health_endpoint, 
        methods=["GET"], 
        tags=["Phase 8 Security"]
    )
    
    app.add_api_route(
        "/api/v1/phase8/metrics", 
        phase8_metrics_endpoint, 
        methods=["GET"], 
        tags=["Phase 8 Security"]
    )
    
    app.add_api_route(
        "/api/v1/phase8/service/{service_name}/status", 
        phase8_service_status_endpoint, 
        methods=["GET"], 
        tags=["Phase 8 Security"]
    )
    
    return app


# Integration steps for main.py:

"""
INTEGRATION STEPS FOR main.py:

1. Import Phase 8 services at the top:
   ```python
   from phase8_startup_service import (
       phase8_startup, phase8_shutdown, get_phase8_health, 
       get_phase8_metrics, setup_phase8_messagebus_handlers
   )
   ```

2. Add Phase 8 startup to the existing lifespan function:
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # ... existing startup code ...
       
       # Add Phase 8 startup after ML integration
       try:
           print("🔐 Starting Phase 8 Security Operations...")
           await phase8_startup()
           await setup_phase8_messagebus_handlers()
           print("✅ Phase 8 Security Operations started")
       except Exception as e:
           print(f"⚠ Phase 8 startup error: {e}")
       
       yield
       
       # Add Phase 8 shutdown before existing shutdown code
       try:
           print("🛑 Stopping Phase 8 Security Operations...")
           await phase8_shutdown()
           print("✅ Phase 8 Security Operations stopped")
       except Exception as e:
           print(f"⚠ Phase 8 shutdown error: {e}")
       
       # ... existing shutdown code ...
   ```

3. Add health check endpoints:
   ```python
   @app.get("/api/v1/phase8/health", tags=["Phase 8 Security"])
   async def phase8_health():
       return get_phase8_health()
   
   @app.get("/api/v1/phase8/metrics", tags=["Phase 8 Security"])
   async def phase8_metrics():
       return get_phase8_metrics()
   ```

4. Optional: Add environment configuration:
   Add to your Settings class:
   ```python
   class Settings(BaseSettings):
       # ... existing settings ...
       
       # Phase 8 Security Settings
       enable_phase8_security: bool = True
       phase8_redis_host: str = "localhost"
       phase8_redis_port: int = 6379
       phase8_log_level: str = "INFO"
   ```

5. Docker Compose Integration:
   Add to your docker-compose.yml:
   ```yaml
   environment:
     - ENABLE_PHASE8_SECURITY=true
     - PHASE8_REDIS_HOST=redis
     - PHASE8_LOG_LEVEL=INFO
     - ENABLE_COGNITIVE_SECURITY=true
     - ENABLE_THREAT_INTELLIGENCE=true
     - ENABLE_AUTONOMOUS_RESPONSE=true
     - ENABLE_FRAUD_DETECTION=true
     - ENABLE_SECURITY_ORCHESTRATION=true
   ```

TESTING THE INTEGRATION:

1. Start the application:
   ```bash
   docker-compose up
   ```

2. Check Phase 8 health:
   ```bash
   curl http://localhost:8001/api/v1/phase8/health
   ```

3. View Phase 8 metrics:
   ```bash
   curl http://localhost:8001/api/v1/phase8/metrics
   ```

4. Monitor logs:
   ```bash
   docker-compose logs -f backend | grep "Phase8"
   ```

ERROR HANDLING:

The Phase 8 startup service is designed with graceful degradation:
- If Phase 8 fails to start, the main application continues running
- Individual service failures don't affect other Phase 8 services
- Auto-recovery attempts to restore failed services
- Comprehensive health checks monitor service status
- All errors are logged for debugging

MONITORING:

Phase 8 provides multiple monitoring levels:
1. Service-level health checks
2. Component-level metrics
3. System-wide health aggregation
4. Performance metrics collection
5. Auto-recovery status tracking

This ensures your trading platform maintains security without affecting performance.
"""
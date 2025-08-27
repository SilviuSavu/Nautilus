#!/usr/bin/env python3
"""
Ultra-Fast Analytics Engine with Enhanced MessageBus Integration
FastAPI server with MessageBus background tasks for sub-5ms analytics processing.

Features:
- FastAPI HTTP endpoints for backward compatibility
- Enhanced MessageBus integration for real-time streaming
- Neural Engine hardware acceleration for <5ms calculations
- Background analytics processing tasks
- Real-time portfolio and risk analytics
- Streaming analytics results via MessageBus
"""

import asyncio
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Import the enhanced analytics engine
from enhanced_analytics_messagebus_integration import (
    EnhancedAnalyticsEngineMessageBus,
    AnalyticsResult,
    PerformanceMetrics
)

# Import clock for deterministic testing
try:
    from backend.engines.ml.clock import get_ml_clock, Clock
    CLOCK_AVAILABLE = True
except ImportError:
    import time
    class MockClock:
        def timestamp(self) -> float:
            return time.time()
    def get_ml_clock():
        return MockClock()
    CLOCK_AVAILABLE = False

# Import hardware routing
try:
    from backend.hardware_router import get_hardware_router
    HARDWARE_ROUTING_AVAILABLE = True
except ImportError:
    HARDWARE_ROUTING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global enhanced analytics engine instance
enhanced_analytics_engine: Optional[EnhancedAnalyticsEngineMessageBus] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management with MessageBus integration"""
    global enhanced_analytics_engine
    
    try:
        logger.info("🚀 Starting Ultra-Fast Analytics Engine with MessageBus...")
        
        # Initialize enhanced analytics engine
        enhanced_analytics_engine = EnhancedAnalyticsEngineMessageBus()
        await enhanced_analytics_engine.initialize()
        
        # Store in app state for access in endpoints
        app.state.analytics_engine = enhanced_analytics_engine
        
        # Start background analytics tasks
        background_tasks = [
            asyncio.create_task(real_time_analytics_processor()),
            asyncio.create_task(streaming_performance_monitor()),
            asyncio.create_task(analytics_health_reporter())
        ]
        app.state.background_tasks = background_tasks
        
        logger.info("✅ Ultra-Fast Analytics Engine started successfully")
        logger.info(f"   MessageBus: {'✅ CONNECTED' if enhanced_analytics_engine.messagebus_client else '❌ STANDALONE'}")
        logger.info(f"   Neural Engine: {'✅ ACTIVE' if enhanced_analytics_engine.neural_engine_available else '❌ CPU-only'}")
        logger.info(f"   Hardware Router: {'✅ ACTIVE' if enhanced_analytics_engine.hardware_router else '❌ DISABLED'}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Ultra-Fast Analytics Engine: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔄 Stopping Ultra-Fast Analytics Engine...")
        
        if enhanced_analytics_engine:
            await enhanced_analytics_engine.stop()
        
        # Cancel background tasks
        if hasattr(app.state, 'background_tasks'):
            for task in app.state.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(*app.state.background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some background tasks did not complete within timeout")
        
        logger.info("✅ Ultra-Fast Analytics Engine stopped")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Nautilus Ultra-Fast Analytics Engine",
    description="Analytics Engine with Enhanced MessageBus Integration and Neural Engine Acceleration",
    version="2.0.0",
    lifespan=lifespan
)


# ==================== HEALTH AND STATUS ENDPOINTS ====================

@app.get("/health")
async def health_check():
    """Enhanced health check with MessageBus and hardware status"""
    if not enhanced_analytics_engine:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": "Analytics engine not initialized"}
        )
    
    # Get comprehensive health metrics
    performance_summary = await enhanced_analytics_engine.get_performance_summary()
    
    return {
        "status": "healthy",
        "timestamp": get_ml_clock().timestamp(),
        "engine_info": {
            "calculations_processed": enhanced_analytics_engine.calculations_processed,
            "average_processing_time_ms": enhanced_analytics_engine.average_processing_time_ms,
            "neural_engine_available": enhanced_analytics_engine.neural_engine_available,
            "messagebus_connected": enhanced_analytics_engine.messagebus_client is not None
        },
        "performance_grade": "A+" if enhanced_analytics_engine.average_processing_time_ms < 5.0 else "A",
        "target_achieved": enhanced_analytics_engine.average_processing_time_ms < 5.0,
        "detailed_metrics": performance_summary
    }


@app.get("/metrics")
async def get_comprehensive_metrics():
    """Get comprehensive analytics engine metrics"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    performance_summary = await enhanced_analytics_engine.get_performance_summary()
    
    return {
        "timestamp": get_ml_clock().timestamp(),
        "uptime_seconds": time.time() - getattr(enhanced_analytics_engine, '_start_time', time.time()),
        "performance_metrics": performance_summary,
        "hardware_status": {
            "neural_engine_available": enhanced_analytics_engine.neural_engine_available,
            "hardware_router_active": enhanced_analytics_engine.hardware_router is not None,
            "routing_config": enhanced_analytics_engine.hardware_router.get_routing_config() if enhanced_analytics_engine.hardware_router else None
        },
        "analytics_specific": {
            "portfolio_analytics_processed": enhanced_analytics_engine.portfolio_analytics_processed,
            "risk_analytics_processed": enhanced_analytics_engine.risk_analytics_processed,
            "performance_attribution_processed": enhanced_analytics_engine.performance_attribution_processed,
            "correlation_analysis_processed": enhanced_analytics_engine.correlation_analysis_processed,
            "cached_portfolios": len(enhanced_analytics_engine.portfolio_cache),
            "cached_market_data": len(enhanced_analytics_engine.market_data_cache),
            "cached_risk_metrics": len(enhanced_analytics_engine.risk_metrics_cache)
        }
    }


@app.get("/messagebus/status")
async def get_messagebus_status():
    """Get MessageBus connection and performance status"""
    if not enhanced_analytics_engine or not enhanced_analytics_engine.messagebus_client:
        return {"status": "disconnected", "error": "MessageBus not available"}
    
    try:
        messagebus_metrics = await enhanced_analytics_engine.messagebus_client.get_performance_metrics()
        system_health = await enhanced_analytics_engine.messagebus_client.get_system_health()
        
        return {
            "status": "connected",
            "performance_metrics": messagebus_metrics,
            "system_health": system_health,
            "subscribed_topics": len(enhanced_analytics_engine.subscribed_topics)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==================== ANALYTICS CALCULATION ENDPOINTS ====================

@app.post("/analytics/performance/{portfolio_id}")
async def calculate_portfolio_performance(
    portfolio_id: str, 
    data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    priority: str = "high"
):
    """Calculate portfolio performance with hardware acceleration"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    try:
        # Convert priority string to enum
        from enhanced_analytics_messagebus_integration import MessagePriority
        message_priority = getattr(MessagePriority, priority.upper(), MessagePriority.HIGH)
        
        # Execute calculation with hardware acceleration
        result = await enhanced_analytics_engine.calculate_portfolio_performance(
            portfolio_id, data, message_priority
        )
        
        # Add background task for additional processing if needed
        background_tasks.add_task(log_calculation_result, "portfolio_performance", result)
        
        return {
            "result_id": result.result_id,
            "portfolio_id": portfolio_id,
            "calculation_type": "portfolio_performance",
            "processing_time_ms": result.processing_time_ms,
            "hardware_used": result.hardware_used,
            "confidence": result.confidence,
            "status": "completed",
            "result_data": result.result_data,
            "routing_info": result.routing_decision
        }
        
    except Exception as e:
        logger.error(f"Portfolio performance calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/risk/{portfolio_id}")
async def calculate_risk_analytics(
    portfolio_id: str, 
    data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Calculate risk analytics with Neural Engine acceleration"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    try:
        result = await enhanced_analytics_engine.calculate_risk_analytics(portfolio_id, data)
        
        background_tasks.add_task(log_calculation_result, "risk_analytics", result)
        
        return {
            "result_id": result.result_id,
            "portfolio_id": portfolio_id,
            "calculation_type": "risk_analytics",
            "processing_time_ms": result.processing_time_ms,
            "hardware_used": result.hardware_used,
            "status": "completed",
            "result_data": result.result_data,
            "routing_info": result.routing_decision
        }
        
    except Exception as e:
        logger.error(f"Risk analytics calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/attribution/{portfolio_id}")
async def calculate_performance_attribution(
    portfolio_id: str, 
    data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Calculate performance attribution analysis"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    try:
        result = await enhanced_analytics_engine.calculate_performance_attribution(portfolio_id, data)
        
        background_tasks.add_task(log_calculation_result, "performance_attribution", result)
        
        return {
            "result_id": result.result_id,
            "portfolio_id": portfolio_id,
            "calculation_type": "performance_attribution",
            "processing_time_ms": result.processing_time_ms,
            "hardware_used": result.hardware_used,
            "status": "completed",
            "result_data": result.result_data
        }
        
    except Exception as e:
        logger.error(f"Performance attribution calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/correlation")
async def calculate_correlation_analysis(
    symbols: List[str], 
    data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Calculate correlation analysis between assets"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    try:
        result = await enhanced_analytics_engine.calculate_correlation_analysis(symbols, data)
        
        background_tasks.add_task(log_calculation_result, "correlation_analysis", result)
        
        return {
            "result_id": result.result_id,
            "symbols": symbols,
            "calculation_type": "correlation_analysis",
            "processing_time_ms": result.processing_time_ms,
            "hardware_used": result.hardware_used,
            "status": "completed",
            "result_data": result.result_data
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/volatility/{symbol}")
async def calculate_volatility_analytics(
    symbol: str, 
    price_data: List[float],
    background_tasks: BackgroundTasks
):
    """Calculate volatility analytics for a symbol"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    try:
        result = await enhanced_analytics_engine.calculate_volatility_analytics(symbol, price_data)
        
        background_tasks.add_task(log_calculation_result, "volatility_analytics", result)
        
        return {
            "result_id": result.result_id,
            "symbol": symbol,
            "calculation_type": "volatility_analytics",
            "processing_time_ms": result.processing_time_ms,
            "hardware_used": result.hardware_used,
            "status": "completed",
            "result_data": result.result_data
        }
        
    except Exception as e:
        logger.error(f"Volatility analytics calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/execution")
async def calculate_execution_quality(
    execution_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Calculate execution quality analytics"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    try:
        result = await enhanced_analytics_engine.calculate_execution_quality_analytics(execution_data)
        
        background_tasks.add_task(log_calculation_result, "execution_quality", result)
        
        return {
            "result_id": result.result_id,
            "calculation_type": "execution_quality",
            "processing_time_ms": result.processing_time_ms,
            "hardware_used": result.hardware_used,
            "status": "completed",
            "result_data": result.result_data
        }
        
    except Exception as e:
        logger.error(f"Execution quality calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HARDWARE AND ROUTING ENDPOINTS ====================

@app.get("/hardware/status")
async def get_hardware_status():
    """Get hardware acceleration status"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    return {
        "neural_engine_available": enhanced_analytics_engine.neural_engine_available,
        "hardware_router_active": enhanced_analytics_engine.hardware_router is not None,
        "hardware_acceleration_ratio": enhanced_analytics_engine.hardware_acceleration_ratio,
        "neural_engine_calculations": enhanced_analytics_engine.neural_engine_calculations,
        "cpu_fallback_calculations": enhanced_analytics_engine.cpu_fallback_calculations,
        "routing_config": enhanced_analytics_engine.hardware_router.get_routing_config() if enhanced_analytics_engine.hardware_router else None
    }


@app.post("/hardware/routing/test")
async def test_hardware_routing():
    """Test hardware routing decisions for different analytics workloads"""
    if not enhanced_analytics_engine or not enhanced_analytics_engine.hardware_router:
        raise HTTPException(status_code=503, detail="Hardware router not available")
    
    try:
        test_results = {}
        
        # Test portfolio performance routing
        portfolio_routing = await enhanced_analytics_engine._get_analytics_routing_decision(
            "portfolio_performance", {"portfolio_value": 1000000, "positions": 50}
        )
        test_results["portfolio_performance"] = portfolio_routing
        
        # Test risk analytics routing
        risk_routing = await enhanced_analytics_engine._get_analytics_routing_decision(
            "risk_analytics", {"portfolio_value": 1000000, "risk_factors": 20}
        )
        test_results["risk_analytics"] = risk_routing
        
        # Test correlation analysis routing
        correlation_routing = await enhanced_analytics_engine._get_analytics_routing_decision(
            "correlation_analysis", {"symbols": ["AAPL", "GOOGL", "MSFT"], "data_points": 1000}
        )
        test_results["correlation_analysis"] = correlation_routing
        
        return {
            "status": "success",
            "test_results": test_results,
            "routing_summary": {
                "neural_engine_enabled": enhanced_analytics_engine.hardware_router.neural_engine_enabled,
                "metal_gpu_enabled": enhanced_analytics_engine.hardware_router.metal_gpu_enabled,
                "hybrid_acceleration": enhanced_analytics_engine.hardware_router.hybrid_acceleration
            }
        }
        
    except Exception as e:
        logger.error(f"Hardware routing test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REAL-TIME DATA ENDPOINTS ====================

@app.get("/analytics/cache/portfolios")
async def get_cached_portfolios():
    """Get cached portfolio data for analytics"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    return {
        "cached_portfolios": list(enhanced_analytics_engine.portfolio_cache.keys()),
        "cache_size": len(enhanced_analytics_engine.portfolio_cache),
        "last_updated": max(
            (data.get("last_updated", 0) for data in enhanced_analytics_engine.portfolio_cache.values()),
            default=0
        )
    }


@app.get("/analytics/cache/market_data")
async def get_cached_market_data():
    """Get cached market data for analytics"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    return {
        "cached_symbols": list(enhanced_analytics_engine.market_data_cache.keys()),
        "cache_size": len(enhanced_analytics_engine.market_data_cache),
        "sample_data": dict(list(enhanced_analytics_engine.market_data_cache.items())[:3])
    }


@app.get("/analytics/results/{result_id}")
async def get_calculation_result(result_id: str):
    """Get specific calculation result by ID"""
    if not enhanced_analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    if result_id not in enhanced_analytics_engine.active_calculations:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
    
    result = enhanced_analytics_engine.active_calculations[result_id]
    
    return {
        "result_id": result.result_id,
        "calculation_type": result.calculation_type,
        "portfolio_id": result.portfolio_id,
        "symbol": result.symbol,
        "result_data": result.result_data,
        "processing_time_ms": result.processing_time_ms,
        "hardware_used": result.hardware_used,
        "timestamp": result.timestamp,
        "confidence": result.confidence,
        "routing_decision": result.routing_decision
    }


# ==================== BACKGROUND TASKS ====================

async def real_time_analytics_processor():
    """Background task for real-time analytics processing"""
    logger.info("🔄 Starting real-time analytics processor...")
    
    while True:
        try:
            if enhanced_analytics_engine:
                # Process real-time analytics based on cached data
                current_time = get_ml_clock().timestamp()
                
                # Check for portfolios that need real-time analytics
                for portfolio_id, portfolio_data in enhanced_analytics_engine.portfolio_cache.items():
                    last_analytics = portfolio_data.get("last_analytics_time", 0)
                    
                    # Perform analytics every 60 seconds for active portfolios
                    if current_time - last_analytics > 60:
                        try:
                            result = await enhanced_analytics_engine.calculate_portfolio_performance(
                                portfolio_id, portfolio_data
                            )
                            portfolio_data["last_analytics_time"] = current_time
                            logger.debug(f"Real-time analytics for {portfolio_id}: {result.processing_time_ms:.2f}ms")
                        except Exception as e:
                            logger.debug(f"Real-time analytics failed for {portfolio_id}: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Real-time analytics processor error: {e}")
            await asyncio.sleep(60)


async def streaming_performance_monitor():
    """Background task for streaming performance metrics via MessageBus"""
    logger.info("📊 Starting streaming performance monitor...")
    
    while True:
        try:
            if enhanced_analytics_engine and enhanced_analytics_engine.messagebus_client:
                # Get performance summary
                performance_summary = await enhanced_analytics_engine.get_performance_summary()
                
                # Publish performance metrics via MessageBus
                from enhanced_analytics_messagebus_integration import MessageType, MessagePriority
                
                await enhanced_analytics_engine.messagebus_client.publish(
                    MessageType.PERFORMANCE_METRIC,
                    "analytics.performance.streaming",
                    {
                        "timestamp": get_ml_clock().timestamp(),
                        "engine_type": "analytics",
                        "performance_summary": performance_summary,
                        "real_time_stats": {
                            "calculations_processed": enhanced_analytics_engine.calculations_processed,
                            "average_processing_time_ms": enhanced_analytics_engine.average_processing_time_ms,
                            "neural_engine_utilization": enhanced_analytics_engine.neural_engine_calculations / max(1, enhanced_analytics_engine.calculations_processed)
                        }
                    },
                    MessagePriority.LOW
                )
                
                logger.debug("Streaming performance metrics published")
            
            await asyncio.sleep(60)  # Stream every minute
            
        except Exception as e:
            logger.error(f"Streaming performance monitor error: {e}")
            await asyncio.sleep(120)


async def analytics_health_reporter():
    """Background task for analytics health reporting"""
    logger.info("💊 Starting analytics health reporter...")
    
    while True:
        try:
            if enhanced_analytics_engine:
                # Check engine health
                health_status = "healthy"
                health_issues = []
                
                # Check processing times
                if enhanced_analytics_engine.average_processing_time_ms > 10.0:
                    health_issues.append("High processing times detected")
                    health_status = "degraded"
                
                # Check hardware utilization
                if enhanced_analytics_engine.neural_engine_available and enhanced_analytics_engine.neural_engine_calculations == 0:
                    health_issues.append("Neural Engine not being utilized")
                
                # Check MessageBus connectivity
                if enhanced_analytics_engine.messagebus_client is None:
                    health_issues.append("MessageBus disconnected")
                    health_status = "degraded"
                
                # Report health via MessageBus if available
                if enhanced_analytics_engine.messagebus_client:
                    from enhanced_analytics_messagebus_integration import MessageType, MessagePriority
                    
                    await enhanced_analytics_engine.messagebus_client.publish(
                        MessageType.ENGINE_HEALTH,
                        "analytics.health.report",
                        {
                            "timestamp": get_ml_clock().timestamp(),
                            "engine_type": "analytics",
                            "health_status": health_status,
                            "health_issues": health_issues,
                            "performance_grade": "A+" if enhanced_analytics_engine.average_processing_time_ms < 5.0 else "A",
                            "uptime_seconds": time.time() - getattr(enhanced_analytics_engine, '_start_time', time.time())
                        },
                        MessagePriority.NORMAL if health_status == "healthy" else MessagePriority.HIGH
                    )
                
                logger.debug(f"Analytics health status: {health_status}")
            
            await asyncio.sleep(120)  # Report every 2 minutes
            
        except Exception as e:
            logger.error(f"Analytics health reporter error: {e}")
            await asyncio.sleep(180)


async def log_calculation_result(calculation_type: str, result: AnalyticsResult):
    """Background task to log calculation results"""
    logger.info(f"📈 Analytics calculation completed: {calculation_type}")
    logger.info(f"   Result ID: {result.result_id}")
    logger.info(f"   Processing Time: {result.processing_time_ms:.2f}ms")
    logger.info(f"   Hardware Used: {result.hardware_used}")
    logger.info(f"   Confidence: {result.confidence:.3f}")
    
    if result.routing_decision:
        logger.debug(f"   Routing Decision: {result.routing_decision}")


# ==================== MAIN APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8100"))
    
    # Enhanced logging for development
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"🚀 Starting Ultra-Fast Analytics Engine on {host}:{port}")
    logger.info(f"   Clock Available: {'✅' if CLOCK_AVAILABLE else '❌'}")
    logger.info(f"   Hardware Routing Available: {'✅' if HARDWARE_ROUTING_AVAILABLE else '❌'}")
    
    uvicorn.run(
        "ultra_fast_analytics_engine:app",
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
        reload=False,  # Disable reload for production
        workers=1      # Single worker for shared state
    )
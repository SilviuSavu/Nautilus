"""
Volatility Forecasting Routes for Nautilus Main Backend

Native integration of volatility forecasting engine with the main Nautilus FastAPI app.
Provides full M4 Max hardware acceleration without containerization overhead.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator

from volatility_engine_service import get_volatility_service, NautilusVolatilityService

logger = logging.getLogger(__name__)

# Create router for volatility endpoints
volatility_routes = APIRouter(prefix="/api/v1/volatility", tags=["Volatility Forecasting"])


# Request/Response Models
class VolatilityForecastRequest(BaseModel):
    """Request for volatility forecast"""
    recent_data: Optional[List[Dict[str, Any]]] = None
    horizon: int = Field(default=5, ge=1, le=30)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)


class VolatilityTrainingRequest(BaseModel):
    """Request for model training"""
    training_data: List[Dict[str, Any]] = Field(..., min_items=50)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)
    async_training: bool = Field(default=False)
    
    @validator('training_data')
    def validate_training_data(cls, v):
        if not v or len(v) < 50:
            raise ValueError("At least 50 data points required for training")
        
        # Check for required columns in first item
        required_cols = ['close']
        if not all(col in v[0] for col in required_cols):
            raise ValueError("Training data must include 'close' prices")
        
        return v


class RealtimeUpdateRequest(BaseModel):
    """Request for real-time data update"""
    market_data: Dict[str, float] = Field(..., description="OHLCV market data")
    
    @validator('market_data')
    def validate_market_data(cls, v):
        if 'close' not in v:
            raise ValueError("Market data must include 'close' price")
        return v


# Main Volatility Routes

@volatility_routes.get("/health")
async def volatility_health_check():
    """Health check for volatility engine"""
    try:
        service = await get_volatility_service()
        status = service.get_volatility_service_status()
        return {
            "status": "healthy",
            "volatility_engine": status.get('status', 'unknown'),
            "hardware_acceleration": status.get('hardware_acceleration', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Volatility health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@volatility_routes.get("/status")
async def get_volatility_status():
    """Get detailed volatility engine status"""
    try:
        service = await get_volatility_service()
        return service.get_volatility_service_status()
    except Exception as e:
        logger.error(f"Failed to get volatility status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get volatility engine status")


@volatility_routes.get("/models")
async def get_available_volatility_models():
    """Get list of available volatility models"""
    try:
        service = await get_volatility_service()
        return service.get_available_models()
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available models")


@volatility_routes.post("/symbols/{symbol}/add")
async def add_symbol_to_volatility_engine(symbol: str):
    """Add a symbol to the volatility forecasting engine"""
    try:
        service = await get_volatility_service()
        result = await service.add_symbol_for_forecasting(symbol)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "result": result,
            "message": f"Symbol {symbol} added to volatility engine"
        }
        
    except Exception as e:
        logger.error(f"Failed to add symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add symbol {symbol}")


@volatility_routes.post("/symbols/{symbol}/train")
async def train_volatility_models(
    symbol: str,
    request: VolatilityTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train volatility models for a symbol with M4 Max acceleration"""
    try:
        service = await get_volatility_service()
        
        if request.async_training:
            # Train in background
            background_tasks.add_task(
                service.train_volatility_models,
                symbol,
                request.training_data
            )
            
            return {
                "status": "training_started",
                "symbol": symbol.upper(),
                "message": "Model training started in background",
                "async": True
            }
        else:
            # Train synchronously
            result = await service.train_volatility_models(symbol, request.training_data)
            
            return {
                "status": "training_completed",
                "symbol": symbol.upper(),
                "training_results": result,
                "async": False
            }
            
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed for {symbol}")


@volatility_routes.post("/symbols/{symbol}/forecast")
async def generate_volatility_forecast(
    symbol: str,
    request: VolatilityForecastRequest
):
    """Generate volatility forecast with M4 Max hardware acceleration"""
    try:
        service = await get_volatility_service()
        
        result = await service.generate_volatility_forecast(
            symbol=symbol,
            recent_data=request.recent_data,
            horizon=request.horizon,
            confidence_level=request.confidence_level
        )
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "forecast": result.get('forecast'),
            "generation_time_ms": result.get('generation_time_ms'),
            "hardware_accelerated": True
        }
        
    except Exception as e:
        logger.error(f"Forecast generation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed for {symbol}")


@volatility_routes.get("/symbols/{symbol}/forecast")
async def get_latest_volatility_forecast(symbol: str):
    """Get the latest volatility forecast for a symbol"""
    try:
        service = await get_volatility_service()
        result = await service.get_latest_volatility_forecast(symbol)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"No forecast found for {symbol}")
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "forecast": result.get('forecast'),
            "source": result.get('source', 'cache')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get forecast for {symbol}")


@volatility_routes.post("/symbols/{symbol}/realtime")
async def update_realtime_volatility(
    symbol: str,
    request: RealtimeUpdateRequest
):
    """Update volatility models with real-time market data"""
    try:
        service = await get_volatility_service()
        
        result = await service.update_realtime_volatility(symbol, request.market_data)
        
        if result is None:
            return {
                "status": "symbol_not_active",
                "message": f"Symbol {symbol} is not active in volatility engine"
            }
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "update_result": result
        }
        
    except Exception as e:
        logger.error(f"Real-time update failed for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol.upper(),
            "error": str(e)
        }


@volatility_routes.get("/symbols")
async def list_active_volatility_symbols():
    """List all symbols active in volatility engine"""
    try:
        service = await get_volatility_service()
        status = service.get_volatility_service_status()
        
        return {
            "active_symbols": status.get('active_symbols', []),
            "total_symbols": len(status.get('active_symbols', []))
        }
        
    except Exception as e:
        logger.error(f"Failed to list symbols: {e}")
        raise HTTPException(status_code=500, detail="Failed to list active symbols")


@volatility_routes.delete("/symbols/{symbol}")
async def remove_symbol_from_volatility_engine(symbol: str):
    """Remove a symbol from the volatility engine"""
    try:
        service = await get_volatility_service()
        
        # For now, just remove from active symbols
        # Full cleanup would be implemented in the service
        if hasattr(service, 'active_symbols') and symbol.upper() in service.active_symbols:
            service.active_symbols.discard(symbol.upper())
            
            return {
                "status": "success",
                "symbol": symbol.upper(),
                "message": f"Symbol {symbol} removed from volatility engine"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in volatility engine")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove symbol {symbol}")


@volatility_routes.post("/benchmark")
async def run_volatility_benchmark(
    symbols: List[str] = Query(default=["AAPL"]),
    iterations: int = Query(default=5, ge=1, le=20),
    horizon: int = Query(default=5, ge=1, le=10)
):
    """Run volatility forecasting benchmark with M4 Max acceleration"""
    try:
        service = await get_volatility_service()
        
        benchmark_results = {
            "benchmark_id": f"volatility_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "symbols_tested": symbols,
            "iterations": iterations,
            "horizon": horizon,
            "results": [],
            "summary": {},
            "hardware_acceleration": True
        }
        
        total_time = 0.0
        successful_forecasts = 0
        
        for symbol in symbols:
            symbol = symbol.upper()
            
            # Ensure symbol is active
            try:
                await service.add_symbol_for_forecasting(symbol)
            except:
                pass  # May already exist
            
            symbol_results = []
            
            for i in range(iterations):
                start_time = datetime.utcnow()
                
                try:
                    result = await service.generate_volatility_forecast(symbol, horizon=horizon)
                    end_time = datetime.utcnow()
                    
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    total_time += duration_ms
                    successful_forecasts += 1
                    
                    symbol_results.append({
                        "iteration": i + 1,
                        "duration_ms": duration_ms,
                        "success": True,
                        "volatility": result.get('forecast', {}).get('ensemble_volatility', 0.0)
                    })
                    
                except Exception as e:
                    symbol_results.append({
                        "iteration": i + 1,
                        "duration_ms": 0.0,
                        "success": False,
                        "error": str(e)
                    })
            
            # Calculate symbol statistics
            successful_runs = [r for r in symbol_results if r['success']]
            avg_time = sum(r['duration_ms'] for r in successful_runs) / max(1, len(successful_runs))
            success_rate = len(successful_runs) / len(symbol_results)
            
            benchmark_results["results"].append({
                "symbol": symbol,
                "iterations": symbol_results,
                "avg_time_ms": avg_time,
                "success_rate": success_rate,
                "successful_runs": len(successful_runs)
            })
        
        # Summary statistics
        benchmark_results["summary"] = {
            "total_iterations": len(symbols) * iterations,
            "successful_forecasts": successful_forecasts,
            "overall_success_rate": successful_forecasts / (len(symbols) * iterations),
            "avg_forecast_time_ms": total_time / max(1, successful_forecasts),
            "total_benchmark_time_ms": total_time,
            "forecasts_per_second": successful_forecasts / max(0.001, total_time / 1000)
        }
        
        logger.info(f"Volatility benchmark completed: {successful_forecasts}/{len(symbols) * iterations} successful")
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Volatility benchmark failed: {e}")
        raise HTTPException(status_code=500, detail="Benchmark execution failed")


@volatility_routes.get("/performance")
async def get_volatility_performance():
    """Get volatility engine performance metrics"""
    try:
        service = await get_volatility_service()
        status = service.get_volatility_service_status()
        
        return {
            "service_stats": status.get('service_stats', {}),
            "hardware_acceleration": status.get('hardware_acceleration', {}),
            "active_symbols_count": len(status.get('active_symbols', [])),
            "ensemble_config": status.get('ensemble_config', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


# MessageBus and Real-time Streaming Endpoints

@volatility_routes.get("/streaming/status")
async def get_streaming_status():
    """Get real-time streaming status from MessageBus integration"""
    try:
        service = await get_volatility_service()
        
        if not service.engine or not service.engine.messagebus_client:
            return {
                "status": "messagebus_unavailable",
                "message": "MessageBus client not initialized",
                "streaming_active": False
            }
        
        # Get streaming statistics
        streaming_stats = await service.engine.messagebus_client.get_streaming_stats()
        
        return {
            "status": "active" if streaming_stats.get('messagebus_connected') else "inactive",
            "streaming_stats": streaming_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get streaming status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get streaming status")


@volatility_routes.get("/streaming/symbols")
async def get_streaming_symbols():
    """Get list of symbols being tracked in real-time streams"""
    try:
        service = await get_volatility_service()
        
        if not service.engine or not service.engine.messagebus_client:
            return {"symbols": [], "total": 0, "streaming_active": False}
        
        streaming_stats = await service.engine.messagebus_client.get_streaming_stats()
        active_symbols = streaming_stats.get('active_symbols', [])
        
        return {
            "symbols": active_symbols,
            "total": len(active_symbols),
            "streaming_active": streaming_stats.get('messagebus_connected', False),
            "buffer_sizes": streaming_stats.get('buffer_sizes', {}),
            "last_updates": streaming_stats.get('last_updates', {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get streaming symbols: {e}")
        raise HTTPException(status_code=500, detail="Failed to get streaming symbols")


@volatility_routes.get("/streaming/events/stats")
async def get_streaming_events_stats():
    """Get detailed streaming events statistics"""
    try:
        service = await get_volatility_service()
        
        if not service.engine or not service.engine.messagebus_client:
            return {
                "events_processed": 0,
                "volatility_updates_triggered": 0,
                "events_per_second": 0.0,
                "uptime_seconds": 0
            }
        
        streaming_stats = await service.engine.messagebus_client.get_streaming_stats()
        
        return {
            "events_processed": streaming_stats.get('events_processed', 0),
            "volatility_updates_triggered": streaming_stats.get('volatility_updates_triggered', 0),
            "events_per_second": streaming_stats.get('events_per_second', 0.0),
            "uptime_seconds": streaming_stats.get('uptime_seconds', 0),
            "active_symbols_count": len(streaming_stats.get('active_symbols', [])),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get streaming events stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get streaming events stats")


@volatility_routes.get("/symbols/{symbol}/streaming/data")
async def get_symbol_streaming_data(symbol: str, limit: int = Query(default=50, ge=1, le=1000)):
    """Get recent streaming data for a specific symbol"""
    try:
        service = await get_volatility_service()
        
        if not service.engine or not service.engine.messagebus_client:
            raise HTTPException(status_code=503, detail="MessageBus streaming not available")
        
        # Get recent data from MessageBus client
        recent_data = await service.engine.messagebus_client.get_recent_data(symbol.upper(), limit=limit)
        
        # Convert events to JSON-serializable format
        data_points = []
        for event in recent_data:
            data_points.append({
                "timestamp": event.timestamp.isoformat(),
                "data_type": event.data_type,
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
                "price": event.price,
                "source": event.source,
                "sequence_number": event.sequence_number
            })
        
        return {
            "symbol": symbol.upper(),
            "data_points": data_points,
            "count": len(data_points),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get streaming data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get streaming data for {symbol}")


@volatility_routes.get("/engine/detailed-status")
async def get_detailed_engine_status():
    """Get comprehensive engine status including MessageBus and hardware details"""
    try:
        service = await get_volatility_service()
        
        if not service.engine:
            raise HTTPException(status_code=503, detail="Volatility engine not available")
        
        # Get full engine status
        engine_status = await service.engine.get_engine_status()
        
        return {
            "engine_status": engine_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get detailed engine status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get detailed engine status")


@volatility_routes.get("/deep-learning/availability")
async def get_deep_learning_availability():
    """Get deep learning models availability and hardware acceleration status"""
    try:
        # Check deep learning availability
        deep_learning_status = {
            "pytorch_available": False,
            "neural_engine_available": False,
            "metal_gpu_available": False,
            "lstm_available": False,
            "transformer_available": False,
            "models_info": {}
        }
        
        try:
            # Check PyTorch and hardware
            import torch
            deep_learning_status["pytorch_available"] = True
            deep_learning_status["metal_gpu_available"] = torch.backends.mps.is_available()
        except ImportError:
            pass
        
        try:
            # Check Core ML / Neural Engine
            import coremltools
            deep_learning_status["neural_engine_available"] = True
        except ImportError:
            pass
        
        try:
            # Check volatility deep learning models
            from volatility.models.deep_learning_models import (
                DEEP_LEARNING_AVAILABLE,
                NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
            )
            deep_learning_status["lstm_available"] = DEEP_LEARNING_AVAILABLE
            deep_learning_status["transformer_available"] = DEEP_LEARNING_AVAILABLE
            deep_learning_status["neural_engine_optimization"] = NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
        except ImportError:
            pass
        
        # Get model information from service
        service = await get_volatility_service()
        models_info = service.get_available_models()
        deep_learning_status["models_info"] = models_info.get('ml_models', {})
        
        return {
            "deep_learning_status": deep_learning_status,
            "hardware_acceleration": models_info.get('hardware_acceleration', 'Not available'),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get deep learning availability: {e}")
        raise HTTPException(status_code=500, detail="Failed to get deep learning availability")


@volatility_routes.get("/models/ensemble/weights/{symbol}")
async def get_ensemble_model_weights(symbol: str):
    """Get current ensemble model weights for a symbol"""
    try:
        service = await get_volatility_service()
        
        if not service.engine:
            raise HTTPException(status_code=503, detail="Volatility engine not available")
        
        # Get latest forecast which contains model weights
        latest_forecast = await service.get_latest_volatility_forecast(symbol)
        
        if not latest_forecast or latest_forecast.get('forecast') is None:
            raise HTTPException(status_code=404, detail=f"No forecast data available for {symbol}")
        
        forecast_data = latest_forecast['forecast']
        
        return {
            "symbol": symbol.upper(),
            "model_weights": forecast_data.get('model_weights', {}),
            "model_contributions": forecast_data.get('model_contributions', {}),
            "models_used": forecast_data.get('models_used', 0),
            "ensemble_method": forecast_data.get('ensemble_method', 'unknown'),
            "forecast_timestamp": forecast_data.get('forecast_timestamp'),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ensemble weights for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ensemble weights for {symbol}")


@volatility_routes.post("/streaming/trigger-update/{symbol}")
async def trigger_manual_volatility_update(symbol: str):
    """Manually trigger a volatility update for a symbol (useful for testing)"""
    try:
        service = await get_volatility_service()
        
        if not service.engine or not service.engine.messagebus_client:
            raise HTTPException(status_code=503, detail="MessageBus streaming not available")
        
        # Create manual trigger event
        trigger_event = {
            'symbol': symbol.upper(),
            'trigger_type': 'manual_trigger',
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'api_request'
        }
        
        # Handle the trigger
        await service.engine._handle_volatility_trigger(trigger_event)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "message": "Manual volatility update triggered",
            "trigger_event": trigger_event,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger manual update for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger manual update for {symbol}")


@volatility_routes.get("/hardware/acceleration-status")
async def get_hardware_acceleration_status():
    """Get detailed hardware acceleration status"""
    try:
        service = await get_volatility_service()
        
        # Get basic status
        status = service.get_volatility_service_status()
        hardware_config = status.get('hardware_acceleration', {})
        
        # Detailed hardware detection
        hardware_details = {
            "metal_gpu": {
                "enabled": hardware_config.get('metal_gpu_enabled', False),
                "available": False,
                "device_name": None,
                "memory_gb": None
            },
            "neural_engine": {
                "enabled": hardware_config.get('neural_engine_enabled', False),
                "available": False,
                "optimization_active": False
            },
            "cpu_optimization": {
                "enabled": hardware_config.get('cpu_optimization_enabled', False),
                "cores": None,
                "performance_cores": None,
                "efficiency_cores": None
            }
        }
        
        # Check Metal GPU
        try:
            import torch
            if torch.backends.mps.is_available():
                hardware_details["metal_gpu"]["available"] = True
                hardware_details["metal_gpu"]["device_name"] = "Apple M4 Max Metal GPU"
                hardware_details["metal_gpu"]["memory_gb"] = 16  # Estimated
        except ImportError:
            pass
        
        # Check Neural Engine
        try:
            import coremltools
            hardware_details["neural_engine"]["available"] = True
            try:
                from volatility.models.deep_learning_models import NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
                hardware_details["neural_engine"]["optimization_active"] = NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
            except ImportError:
                pass
        except ImportError:
            pass
        
        # Check CPU information
        import os
        hardware_details["cpu_optimization"]["cores"] = os.cpu_count()
        # For M4 Max: 12 performance + 4 efficiency cores
        if os.cpu_count() == 16:  # Likely M4 Max
            hardware_details["cpu_optimization"]["performance_cores"] = 12
            hardware_details["cpu_optimization"]["efficiency_cores"] = 4
        
        return {
            "hardware_acceleration": hardware_details,
            "auto_routing_enabled": hardware_config.get('auto_routing_enabled', False),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get hardware acceleration status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get hardware acceleration status")


@volatility_routes.get("/models/training/status/{symbol}")
async def get_model_training_status(symbol: str):
    """Get training status for models of a specific symbol"""
    try:
        service = await get_volatility_service()
        
        if not service.engine:
            raise HTTPException(status_code=503, detail="Volatility engine not available")
        
        # Get model status from orchestrator
        symbol = symbol.upper()
        if symbol not in service.engine.active_symbols:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in active symbols")
        
        model_status = service.engine.orchestrator.get_model_status(symbol)
        
        return {
            "symbol": symbol,
            "training_status": model_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status for {symbol}")


# WebSocket endpoint for real-time updates (if WebSocket support is enabled)
try:
    from fastapi import WebSocket, WebSocketDisconnect
    
    @volatility_routes.websocket("/ws/streaming/{symbol}")
    async def websocket_streaming_endpoint(websocket: WebSocket, symbol: str):
        """WebSocket endpoint for real-time volatility updates"""
        await websocket.accept()
        symbol = symbol.upper()
        
        try:
            service = await get_volatility_service()
            
            if not service.engine or not service.engine.messagebus_client:
                await websocket.send_json({
                    "error": "MessageBus streaming not available",
                    "symbol": symbol
                })
                return
            
            logger.info(f"WebSocket connected for symbol {symbol}")
            
            # Send initial status
            await websocket.send_json({
                "type": "connection_status",
                "symbol": symbol,
                "connected": True,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Main streaming loop
            while True:
                # Get recent data and send updates
                try:
                    streaming_stats = await service.engine.messagebus_client.get_streaming_stats()
                    recent_data = await service.engine.messagebus_client.get_recent_data(symbol, limit=5)
                    
                    if recent_data:
                        # Send latest data point
                        latest_event = recent_data[-1]
                        await websocket.send_json({
                            "type": "market_data",
                            "symbol": symbol,
                            "data": {
                                "timestamp": latest_event.timestamp.isoformat(),
                                "close": latest_event.close,
                                "price": latest_event.price,
                                "volume": latest_event.volume,
                                "source": latest_event.source
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    # Send streaming stats every few iterations
                    await websocket.send_json({
                        "type": "streaming_stats",
                        "symbol": symbol,
                        "stats": {
                            "events_processed": streaming_stats.get('events_processed', 0),
                            "events_per_second": streaming_stats.get('events_per_second', 0.0),
                            "buffer_size": streaming_stats.get('buffer_sizes', {}).get(symbol, 0)
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Wait before next update
                    await asyncio.sleep(2)  # 2-second updates
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    await asyncio.sleep(5)  # Back off on errors
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for symbol {symbol}")
        except Exception as e:
            logger.error(f"WebSocket error for {symbol}: {e}")
            
except ImportError:
    # WebSocket not available
    pass
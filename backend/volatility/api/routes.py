"""
Volatility Forecasting REST API Routes

This module defines all REST API endpoints for the volatility forecasting engine.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..engine.volatility_engine import get_engine, VolatilityEngine
from .models import (
    ForecastRequest, ForecastResponse, TrainingRequest, TrainingResponse,
    RealtimeDataRequest, EngineStatusResponse, ModelStatusResponse
)

logger = logging.getLogger(__name__)

# Create API router
volatility_router = APIRouter(prefix="/api/v1/volatility", tags=["Volatility Forecasting"])


@volatility_router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "volatility-forecasting-engine"}


@volatility_router.get("/status", response_model=EngineStatusResponse)
async def get_engine_status(engine: VolatilityEngine = Depends(get_engine)):
    """Get comprehensive engine status"""
    try:
        status = await engine.get_engine_status()
        return EngineStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get engine status")


@volatility_router.post("/symbols/{symbol}/add")
async def add_symbol(symbol: str, engine: VolatilityEngine = Depends(get_engine)):
    """Add a symbol to the volatility engine"""
    try:
        result = await engine.add_symbol(symbol.upper())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error adding symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add symbol {symbol}")


@volatility_router.post("/symbols/{symbol}/train", response_model=TrainingResponse)
async def train_symbol_models(
    symbol: str,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    engine: VolatilityEngine = Depends(get_engine)
):
    """Train models for a symbol with historical data"""
    try:
        symbol = symbol.upper()
        
        # Convert request data to DataFrame
        if request.data_format == "json":
            df = pd.DataFrame(request.training_data)
        elif request.data_format == "csv":
            # Assume CSV string in training_data[0] if it's a string
            import io
            csv_data = request.training_data if isinstance(request.training_data, str) else ""
            df = pd.read_csv(io.StringIO(csv_data))
        else:
            raise HTTPException(status_code=400, detail="Unsupported data format")
        
        # Validate required columns
        required_columns = ['close']
        if request.include_ohlc:
            required_columns.extend(['open', 'high', 'low'])
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Start training
        if request.async_training:
            # Train in background
            background_tasks.add_task(
                engine.train_symbol_models, symbol, df
            )
            return TrainingResponse(
                status="started",
                symbol=symbol,
                message="Training started in background",
                async_training=True
            )
        else:
            # Train synchronously
            result = await engine.train_symbol_models(symbol, df)
            return TrainingResponse(
                status=result['status'],
                symbol=symbol,
                training_results=result.get('training_results', {}),
                message="Training completed successfully",
                async_training=False
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed for {symbol}")


@volatility_router.post("/symbols/{symbol}/forecast", response_model=ForecastResponse)
async def generate_forecast(
    symbol: str,
    request: ForecastRequest,
    engine: VolatilityEngine = Depends(get_engine)
):
    """Generate volatility forecast for a symbol"""
    try:
        symbol = symbol.upper()
        
        # Convert recent data to DataFrame if provided
        recent_data = None
        if request.recent_data:
            recent_data = pd.DataFrame(request.recent_data)
        
        # Generate forecast
        result = await engine.generate_forecast(
            symbol=symbol,
            recent_data=recent_data,
            horizon=request.horizon,
            confidence_level=request.confidence_level
        )
        
        return ForecastResponse(
            status=result['status'],
            symbol=symbol,
            forecast=result['forecast'],
            generation_time_ms=result['generation_time_ms']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed for {symbol}")


@volatility_router.get("/symbols/{symbol}/forecast", response_model=ForecastResponse)
async def get_latest_forecast(
    symbol: str,
    engine: VolatilityEngine = Depends(get_engine)
):
    """Get the latest forecast for a symbol"""
    try:
        symbol = symbol.upper()
        result = await engine.get_latest_forecast(symbol)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"No forecast found for {symbol}")
        
        return ForecastResponse(
            status=result['status'],
            symbol=symbol,
            forecast=result['forecast'],
            source=result.get('source', 'unknown')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get forecast for {symbol}")


@volatility_router.post("/symbols/{symbol}/realtime")
async def update_realtime_data(
    symbol: str,
    request: RealtimeDataRequest,
    engine: VolatilityEngine = Depends(get_engine)
):
    """Update real-time market data for a symbol"""
    try:
        symbol = symbol.upper()
        
        result = await engine.update_real_time_data(symbol, request.market_data)
        
        if result is None:
            return JSONResponse(content={
                "status": "symbol_not_active",
                "message": f"Symbol {symbol} is not active in the engine"
            })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error updating real-time data for {symbol}: {e}")
        return JSONResponse(content={
            "status": "error",
            "message": f"Failed to update data for {symbol}"
        })


@volatility_router.get("/symbols/{symbol}/models", response_model=ModelStatusResponse)
async def get_model_status(
    symbol: str,
    engine: VolatilityEngine = Depends(get_engine)
):
    """Get model status for a specific symbol"""
    try:
        symbol = symbol.upper()
        
        if symbol not in engine.active_symbols:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        status = engine.orchestrator.get_model_status(symbol)
        return ModelStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status for {symbol}")


@volatility_router.get("/symbols")
async def list_active_symbols(engine: VolatilityEngine = Depends(get_engine)):
    """List all active symbols in the engine"""
    return {
        "active_symbols": list(engine.active_symbols),
        "total_symbols": len(engine.active_symbols)
    }


@volatility_router.delete("/symbols/{symbol}")
async def remove_symbol(
    symbol: str,
    engine: VolatilityEngine = Depends(get_engine)
):
    """Remove a symbol from the engine"""
    try:
        symbol = symbol.upper()
        
        if symbol not in engine.active_symbols:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Cleanup symbol
        await engine.orchestrator.cleanup(symbol)
        engine.active_symbols.discard(symbol)
        
        # Clear caches
        engine.forecast_cache.pop(symbol, None)
        engine.data_streams.pop(symbol, None)
        
        return {
            "status": "success",
            "symbol": symbol,
            "message": f"Symbol {symbol} removed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove symbol {symbol}")


@volatility_router.get("/models")
async def list_available_models():
    """List all available volatility models"""
    from ..config import ModelType
    
    models_info = {
        "econometric_models": {
            "GARCH": "Standard GARCH(p,q) model for volatility clustering",
            "EGARCH": "Exponential GARCH for asymmetric volatility effects",
            "GJR_GARCH": "GJR-GARCH for leverage effects",
        },
        "stochastic_models": {
            "HESTON": "Heston stochastic volatility model for options pricing",
            "SABR": "SABR model for interest rate derivatives"
        },
        "estimators": {
            "GARMAN_KLASS": "Garman-Klass estimator using OHLC data",
            "YANG_ZHANG": "Yang-Zhang estimator with overnight returns",
            "ROGERS_SATCHELL": "Rogers-Satchell drift-independent estimator"
        },
        "ml_models": {
            "LSTM": "LSTM neural networks for pattern recognition",
            "TRANSFORMER": "Transformer models for multi-horizon forecasting"
        },
        "total_models": len(ModelType)
    }
    
    return models_info


@volatility_router.get("/performance")
async def get_performance_stats(engine: VolatilityEngine = Depends(get_engine)):
    """Get engine performance statistics"""
    status = await engine.get_engine_status()
    
    return {
        "performance_stats": status.get('performance_stats', {}),
        "hardware_acceleration": status.get('hardware_acceleration', {}),
        "uptime_seconds": status.get('uptime_seconds', 0),
        "active_symbols_count": len(status.get('active_symbols', [])),
        "total_models": sum(status.get('models_per_symbol', {}).values())
    }


@volatility_router.post("/benchmark")
async def run_performance_benchmark(
    symbols: List[str] = Query(default=["AAPL"]),
    iterations: int = Query(default=10, ge=1, le=100),
    engine: VolatilityEngine = Depends(get_engine)
):
    """Run performance benchmark on specified symbols"""
    try:
        benchmark_results = {
            "symbols_tested": symbols,
            "iterations": iterations,
            "results": [],
            "summary": {}
        }
        
        total_time = 0.0
        successful_forecasts = 0
        
        for symbol in symbols:
            symbol = symbol.upper()
            
            # Ensure symbol is active
            if symbol not in engine.active_symbols:
                await engine.add_symbol(symbol)
            
            symbol_results = []
            
            for i in range(iterations):
                start_time = datetime.utcnow()
                
                try:
                    result = await engine.generate_forecast(symbol)
                    end_time = datetime.utcnow()
                    
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    total_time += duration_ms
                    successful_forecasts += 1
                    
                    symbol_results.append({
                        "iteration": i + 1,
                        "duration_ms": duration_ms,
                        "success": True,
                        "forecast_volatility": result['forecast']['forecast']['ensemble_volatility']
                    })
                    
                except Exception as e:
                    symbol_results.append({
                        "iteration": i + 1,
                        "duration_ms": 0.0,
                        "success": False,
                        "error": str(e)
                    })
            
            benchmark_results["results"].append({
                "symbol": symbol,
                "iterations": symbol_results,
                "avg_time_ms": sum(r["duration_ms"] for r in symbol_results if r["success"]) / max(1, sum(1 for r in symbol_results if r["success"])),
                "success_rate": sum(1 for r in symbol_results if r["success"]) / len(symbol_results)
            })
        
        # Summary statistics
        benchmark_results["summary"] = {
            "total_iterations": len(symbols) * iterations,
            "successful_forecasts": successful_forecasts,
            "success_rate": successful_forecasts / (len(symbols) * iterations),
            "avg_forecast_time_ms": total_time / max(1, successful_forecasts),
            "total_benchmark_time_ms": total_time
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail="Benchmark execution failed")
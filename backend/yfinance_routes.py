"""
YFinance API Routes - FastAPI Integration
Provides REST API endpoints for YFinance data access using NautilusTrader's professional adapter.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, Header
from pydantic import BaseModel

# from auth.middleware import get_current_user_optional
# from auth.models import User
from yfinance_service_simple import get_yfinance_service, YFinanceHistoricalData


# Simple API key authentication for YFinance
YFINANCE_API_KEY = os.getenv("YFINANCE_API_KEY", "nautilus-dev-key-123")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Simple API key verification for YFinance endpoints"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required in X-API-Key header")
    if x_api_key != YFINANCE_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# Create router
router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models

class YFinanceInstrumentRequest(BaseModel):
    """Request model for loading instruments"""
    symbols: List[str]


class YFinanceInstrumentResponse(BaseModel):
    """Response model for instrument loading"""
    symbol: str
    loaded: bool
    error_message: Optional[str] = None


class YFinanceStatusResponse(BaseModel):
    """Response model for service status"""
    service: str
    status: str
    initialized: bool
    connected: bool
    instruments_loaded: int
    last_request: Optional[str]
    rate_limit_delay: float
    cache_expiry_seconds: int
    error_message: Optional[str]
    adapter_version: str


class YFinanceHistoricalBarsResponse(BaseModel):
    """Response model for historical bars"""
    symbol: str
    timeframe: str
    bars: List[Dict[str, Any]]
    total_bars: int
    start_date: Optional[str]
    end_date: Optional[str]
    data_source: str


# API Endpoints

@router.get("/api/v1/yfinance/status", response_model=YFinanceStatusResponse)
async def get_yfinance_status():
    """Get YFinance service status and health information"""
    try:
        yfinance_service = get_yfinance_service()
        health_data = await yfinance_service.health_check()
        
        return YFinanceStatusResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Error getting YFinance status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get YFinance status: {str(e)}")


@router.post("/api/v1/yfinance/initialize")
async def initialize_yfinance_service(
    config: Optional[Dict[str, Any]] = None,
    api_key_valid: bool = Depends(verify_api_key)
):
    """Initialize the YFinance service with optional configuration"""
    
    try:
        yfinance_service = get_yfinance_service()
        
        # Use default config if none provided
        default_config = {
            'cache_expiry_seconds': 3600,
            'rate_limit_delay': 0.1,
            'request_timeout': 30.0,
            'max_retries': 3,
            'retry_delay': 1.0,
            'default_period': '1y',
            'default_interval': '1d',
            'symbols': ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'UNH', 'JNJ']
        }
        
        final_config = {**default_config, **(config or {})}
        
        success = await yfinance_service.initialize(final_config)
        
        if success:
            return {
                "message": "YFinance service initialized successfully",
                "status": "operational",
                "config": final_config
            }
        else:
            status = yfinance_service.get_status()
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize YFinance service: {status.error_message}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing YFinance service: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/api/v1/yfinance/instruments/load")
async def load_yfinance_instruments(
    request: YFinanceInstrumentRequest,
    api_key_valid: bool = Depends(verify_api_key)
):
    """Load instrument definitions for specified symbols"""
    
    try:
        yfinance_service = get_yfinance_service()
        
        if not yfinance_service.is_connected():
            raise HTTPException(status_code=503, detail="YFinance service not connected")
        
        results = await yfinance_service.load_instruments(request.symbols)
        
        response_data = []
        for symbol, success in results.items():
            response_data.append(YFinanceInstrumentResponse(
                symbol=symbol,
                loaded=success,
                error_message=None if success else "Failed to load instrument"
            ))
        
        return {
            "message": f"Processed {len(request.symbols)} instruments",
            "results": response_data,
            "summary": {
                "total": len(request.symbols),
                "loaded": sum(1 for r in response_data if r.loaded),
                "failed": sum(1 for r in response_data if not r.loaded)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading YFinance instruments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load instruments: {str(e)}")


@router.get("/api/v1/yfinance/instruments/{symbol}")
async def get_yfinance_instrument(
    symbol: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """Get instrument definition for a specific symbol"""
    
    try:
        yfinance_service = get_yfinance_service()
        
        if not yfinance_service.is_connected():
            raise HTTPException(status_code=503, detail="YFinance service not connected")
        
        # Load instrument if not already loaded
        success = await yfinance_service.load_instrument(symbol.upper())
        
        if success:
            # Get instrument from cache
            from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
            instrument_id = InstrumentId(Symbol(symbol.upper()), Venue("YAHOO"))
            instrument = yfinance_service.cache.instrument(instrument_id)
            
            if instrument:
                return {
                    "symbol": symbol.upper(),
                    "instrument_id": str(instrument.id),
                    "asset_class": str(instrument.asset_class),
                    "instrument_class": str(instrument.instrument_class),
                    "quote_currency": str(instrument.quote_currency),
                    "price_precision": instrument.price_precision,
                    "size_precision": instrument.size_precision,
                    "loaded": True
                }
        
        raise HTTPException(status_code=404, detail=f"Instrument {symbol} not found or failed to load")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting YFinance instrument {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get instrument: {str(e)}")


@router.get("/api/v1/yfinance/historical/bars", response_model=YFinanceHistoricalBarsResponse)
async def get_yfinance_historical_bars(
    symbol: str = Query(..., description="Yahoo Finance symbol"),
    timeframe: str = Query("1d", description="Bar timeframe (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)"),
    period: str = Query("1y", description="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    limit: Optional[int] = Query(None, description="Maximum number of bars to return"),
    api_key_valid: bool = Depends(verify_api_key)
):
    """Get historical OHLCV bars from YFinance"""
    
    try:
        yfinance_service = get_yfinance_service()
        
        if not yfinance_service.is_connected():
            raise HTTPException(status_code=503, detail="YFinance service not connected")
        
        # Validate timeframe
        valid_timeframes = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe. Supported: {', '.join(valid_timeframes)}"
            )
        
        # Validate period
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period. Supported: {', '.join(valid_periods)}"
            )
        
        # Get historical data
        logger.info(f"📊 Requesting YFinance data: {symbol} {timeframe} {period}")
        historical_data = await yfinance_service.get_historical_bars(
            symbol=symbol.upper(),
            timeframe=timeframe,
            period=period,
            limit=limit
        )
        
        if historical_data:
            return YFinanceHistoricalBarsResponse(
                symbol=historical_data.symbol,
                timeframe=historical_data.timeframe,
                bars=historical_data.bars,
                total_bars=historical_data.total_bars,
                start_date=historical_data.start_date,
                end_date=historical_data.end_date,
                data_source=historical_data.data_source
            )
        else:
            # Return empty response rather than error
            return YFinanceHistoricalBarsResponse(
                symbol=symbol.upper(),
                timeframe=timeframe,
                bars=[],
                total_bars=0,
                start_date=None,
                end_date=None,
                data_source="YFinance (No Data)"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting YFinance historical bars for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical bars: {str(e)}")


@router.get("/api/v1/yfinance/historical/bars/{symbol}")
async def get_yfinance_bars_by_symbol(
    symbol: str,
    timeframe: str = Query("1d", description="Bar timeframe"),
    period: str = Query("1y", description="Data period"),
    limit: Optional[int] = Query(None, description="Maximum number of bars"),
    api_key_valid: bool = Depends(verify_api_key)
):
    """Alternative endpoint for getting historical bars by symbol path parameter"""
    return await get_yfinance_historical_bars(
        symbol=symbol,
        timeframe=timeframe,
        period=period,
        limit=limit,
        api_key_valid=api_key_valid
    )


@router.post("/api/v1/yfinance/disconnect")
async def disconnect_yfinance_service(
    api_key_valid: bool = Depends(verify_api_key)
):
    """Disconnect the YFinance service"""
    
    try:
        yfinance_service = get_yfinance_service()
        await yfinance_service.disconnect()
        
        return {
            "message": "YFinance service disconnected successfully",
            "status": "disconnected"
        }
        
    except Exception as e:
        logger.error(f"Error disconnecting YFinance service: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disconnect: {str(e)}")


@router.get("/api/v1/yfinance/health")
async def yfinance_health_check():
    """Health check endpoint for YFinance service"""
    try:
        yfinance_service = get_yfinance_service()
        health_data = await yfinance_service.health_check()
        
        # Return appropriate HTTP status based on service health
        if health_data["status"] == "operational":
            return health_data
        else:
            # Service is not healthy but we can still return status
            return health_data
            
    except Exception as e:
        logger.error(f"YFinance health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service health check failed: {str(e)}")


@router.get("/api/v1/yfinance/test-data")
async def test_yfinance_data():
    """Test endpoint to get sample YFinance data"""
    try:
        yfinance_service = get_yfinance_service()
        
        if not yfinance_service.is_connected():
            return {"error": "YFinance service not connected"}
        
        # Test getting AAPL data
        data = await yfinance_service.get_historical_bars(
            symbol="AAPL",
            timeframe="1d",
            period="5d",
            limit=5
        )
        
        if data:
            return {
                "success": True,
                "symbol": data.symbol,
                "timeframe": data.timeframe,
                "total_bars": data.total_bars,
                "sample_bars": data.bars[:3] if data.bars else [],
                "data_source": data.data_source
            }
        else:
            return {"error": "No data returned"}
            
    except Exception as e:
        logger.error(f"Error in test data fetch: {e}")
        return {"error": str(e)}


@router.post("/api/v1/yfinance/test-init")
async def test_yfinance_init():
    """Test endpoint to initialize YFinance service without auth"""
    try:
        yfinance_service = get_yfinance_service()
        
        default_config = {
            'cache_expiry_seconds': 3600,
            'rate_limit_delay': 0.1,
            'symbols': ['AAPL', 'MSFT', 'TSLA']
        }
        
        success = await yfinance_service.initialize(default_config)
        
        if success:
            return {
                "message": "YFinance service test initialization successful",
                "status": "operational",
                "config": default_config
            }
        else:
            status = yfinance_service.get_status()
            raise HTTPException(
                status_code=500, 
                detail=f"Test initialization failed: {status.error_message}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Test initialization failed: {str(e)}")


# YFinance Backfill Models
class YFinanceBackfillRequest(BaseModel):
    """Request model for YFinance data backfill"""
    symbols: List[str]
    timeframes: Optional[List[str]] = ["1d", "1h", "30m", "15m", "5m"]
    periods: Optional[Dict[str, str]] = None  # timeframe -> period mapping
    store_to_database: Optional[bool] = True


class YFinanceBackfillProgress(BaseModel):
    """Progress tracking for YFinance backfill"""
    request_id: str
    symbols: List[str]
    timeframes: List[str]
    total_requests: int
    completed_requests: int
    failed_requests: int
    current_symbol: Optional[str] = None
    current_timeframe: Optional[str] = None
    status: str  # 'running', 'completed', 'failed'
    start_time: str
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    results: List[Dict[str, Any]] = []


# Global backfill tracking
_backfill_progress = {}


@router.post("/api/v1/yfinance/backfill/start")
async def start_yfinance_backfill(
    request: YFinanceBackfillRequest,
    api_key_valid: bool = Depends(verify_api_key)
):
    """Start YFinance data backfill for multiple symbols and timeframes"""
    try:
        yfinance_service = get_yfinance_service()
        
        if not yfinance_service.is_connected():
            raise HTTPException(status_code=503, detail="YFinance service not connected")
        
        # Create unique request ID
        import uuid
        request_id = f"yfinance_backfill_{uuid.uuid4().hex[:8]}"
        
        # Default periods for timeframes
        default_periods = {
            "1m": "1d",
            "5m": "5d", 
            "15m": "1mo",
            "30m": "3mo",
            "1h": "1y",
            "1d": "5y",
            "1wk": "10y",
            "1mo": "max"
        }
        
        periods = request.periods or default_periods
        
        # Calculate total requests
        total_requests = len(request.symbols) * len(request.timeframes)
        
        # Initialize progress tracking
        progress = YFinanceBackfillProgress(
            request_id=request_id,
            symbols=request.symbols,
            timeframes=request.timeframes,
            total_requests=total_requests,
            completed_requests=0,
            failed_requests=0,
            status="running",
            start_time=datetime.now().isoformat(),
            results=[]
        )
        
        _backfill_progress[request_id] = progress
        
        # Start backfill process in background
        import asyncio
        asyncio.create_task(_process_yfinance_backfill(request_id, request, periods, yfinance_service))
        
        return {
            "message": "YFinance backfill started",
            "request_id": request_id,
            "total_requests": total_requests,
            "symbols": request.symbols,
            "timeframes": request.timeframes,
            "status": "running"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting YFinance backfill: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start backfill: {str(e)}")


@router.get("/api/v1/yfinance/backfill/status")
async def get_yfinance_backfill_status():
    """Get YFinance backfill status for all active/recent requests"""
    try:
        if not _backfill_progress:
            return {
                "active_requests": 0,
                "total_requests": 0,
                "completed_requests": 0,
                "failed_requests": 0,
                "requests": []
            }
        
        active_requests = [p for p in _backfill_progress.values() if p.status == "running"]
        completed_requests = [p for p in _backfill_progress.values() if p.status == "completed"]
        failed_requests = [p for p in _backfill_progress.values() if p.status == "failed"]
        
        return {
            "active_requests": len(active_requests),
            "total_requests": len(_backfill_progress),
            "completed_requests": len(completed_requests),
            "failed_requests": len(failed_requests),
            "requests": [
                {
                    "request_id": p.request_id,
                    "symbols": p.symbols,
                    "timeframes": p.timeframes,
                    "total_requests": p.total_requests,
                    "completed_requests": p.completed_requests,
                    "failed_requests": p.failed_requests,
                    "current_symbol": p.current_symbol,
                    "current_timeframe": p.current_timeframe,
                    "status": p.status,
                    "start_time": p.start_time,
                    "end_time": p.end_time,
                    "progress_percentage": round((p.completed_requests / p.total_requests) * 100, 1) if p.total_requests > 0 else 0
                }
                for p in list(_backfill_progress.values())[-10:]  # Last 10 requests
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting YFinance backfill status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backfill status: {str(e)}")


@router.get("/api/v1/yfinance/backfill/status/{request_id}")
async def get_yfinance_backfill_request_status(request_id: str):
    """Get detailed status for a specific YFinance backfill request"""
    try:
        if request_id not in _backfill_progress:
            raise HTTPException(status_code=404, detail=f"Backfill request {request_id} not found")
        
        progress = _backfill_progress[request_id]
        
        return {
            "request_id": progress.request_id,
            "symbols": progress.symbols,
            "timeframes": progress.timeframes,
            "total_requests": progress.total_requests,
            "completed_requests": progress.completed_requests,
            "failed_requests": progress.failed_requests,
            "current_symbol": progress.current_symbol,
            "current_timeframe": progress.current_timeframe,
            "status": progress.status,
            "start_time": progress.start_time,
            "end_time": progress.end_time,
            "error_message": progress.error_message,
            "progress_percentage": round((progress.completed_requests / progress.total_requests) * 100, 1) if progress.total_requests > 0 else 0,
            "results": progress.results[-20:]  # Last 20 results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting YFinance backfill request status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")


async def _process_yfinance_backfill(request_id: str, request: YFinanceBackfillRequest, periods: Dict[str, str], yfinance_service):
    """Process YFinance backfill in background"""
    progress = _backfill_progress[request_id]
    
    try:
        logger.info(f"🚀 Starting YFinance backfill {request_id} for {len(request.symbols)} symbols, {len(request.timeframes)} timeframes")
        
        for symbol in request.symbols:
            progress.current_symbol = symbol
            
            for timeframe in request.timeframes:
                progress.current_timeframe = timeframe
                period = periods.get(timeframe, "1y")
                
                try:
                    logger.info(f"📊 Fetching {symbol} {timeframe} {period}")
                    
                    # Get data from YFinance
                    historical_data = await yfinance_service.get_historical_bars(
                        symbol=symbol,
                        timeframe=timeframe,
                        period=period,
                        limit=None
                    )
                    
                    if historical_data and historical_data.bars:
                        bars_count = len(historical_data.bars)
                        
                        # Store to database if requested
                        stored_count = 0
                        if request.store_to_database:
                            stored_count = await _store_yfinance_data_to_db(historical_data)
                        
                        progress.results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "period": period,
                            "bars_fetched": bars_count,
                            "bars_stored": stored_count,
                            "status": "success",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        logger.info(f"✅ {symbol} {timeframe}: {bars_count} bars fetched, {stored_count} stored")
                        
                    else:
                        progress.results.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "period": period,
                            "bars_fetched": 0,
                            "bars_stored": 0,
                            "status": "no_data",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        logger.warning(f"⚠️ {symbol} {timeframe}: No data available")
                    
                    progress.completed_requests += 1
                    
                    # Rate limiting
                    await asyncio.sleep(yfinance_service.rate_limit_delay)
                    
                except Exception as e:
                    progress.failed_requests += 1
                    progress.results.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "period": period,
                        "bars_fetched": 0,
                        "bars_stored": 0,
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.error(f"❌ Error fetching {symbol} {timeframe}: {e}")
        
        progress.status = "completed"
        progress.end_time = datetime.now().isoformat()
        progress.current_symbol = None
        progress.current_timeframe = None
        
        logger.info(f"🎉 YFinance backfill {request_id} completed: {progress.completed_requests} successful, {progress.failed_requests} failed")
        
    except Exception as e:
        progress.status = "failed"
        progress.error_message = str(e)
        progress.end_time = datetime.now().isoformat()
        progress.current_symbol = None
        progress.current_timeframe = None
        
        logger.error(f"💥 YFinance backfill {request_id} failed: {e}")


async def _store_yfinance_data_to_db(historical_data: YFinanceHistoricalData) -> int:
    """Store YFinance data to PostgreSQL database"""
    try:
        from historical_data_service import historical_data_service
        from data_normalizer import NormalizedBar
        
        stored_count = 0
        
        for bar in historical_data.bars:
            try:
                # Convert YFinance bar to NormalizedBar format
                normalized_bar = NormalizedBar(
                    venue="YAHOO",
                    instrument_id=f"{historical_data.symbol}.YAHOO",
                    timeframe=historical_data.timeframe,
                    timestamp_ns=int(bar["timestamp"] * 1_000_000_000),  # Convert to nanoseconds
                    open_price=float(bar["open"]),
                    high_price=float(bar["high"]),
                    low_price=float(bar["low"]),
                    close_price=float(bar["close"]),
                    volume=float(bar["volume"]),
                    is_final=True
                )
                
                await historical_data_service.store_bar(normalized_bar)
                stored_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to store bar for {historical_data.symbol}: {e}")
        
        return stored_count
        
    except Exception as e:
        logger.error(f"Error storing YFinance data to database: {e}")
        return 0


# Market Data Integration Endpoint - For frontend chart integration
@router.get("/api/v1/yfinance/market-data/bars")
async def get_market_data_bars_yfinance(
    symbol: str = Query(..., description="Symbol to fetch"),
    timeframe: str = Query("1h", description="Timeframe"),
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Market data endpoint compatible with frontend chart integration.
    Returns data in the same format as the main market data API.
    """
    
    try:
        yfinance_service = get_yfinance_service()
        
        if not yfinance_service.is_connected():
            raise HTTPException(status_code=503, detail="YFinance service not connected")
        
        # Map timeframes to periods for better data coverage
        timeframe_to_period = {
            "1m": "1d",    # 1 day of 1-minute data
            "5m": "5d",    # 5 days of 5-minute data
            "15m": "1mo",  # 1 month of 15-minute data
            "30m": "3mo",  # 3 months of 30-minute data
            "1h": "1y",    # 1 year of hourly data
            "1d": "5y",    # 5 years of daily data
            "1wk": "10y",  # 10 years of weekly data
            "1mo": "max"   # Maximum monthly data
        }
        
        period = timeframe_to_period.get(timeframe, "1y")
        
        # Get data from YFinance
        historical_data = await yfinance_service.get_historical_bars(
            symbol=symbol.upper(),
            timeframe=timeframe,
            period=period,
            limit=1000
        )
        
        if historical_data and historical_data.bars:
            # Convert to frontend-compatible format
            candles = []
            for bar in historical_data.bars:
                candles.append({
                    "time": bar["time"],
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": int(bar["volume"])
                })
            
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "candles": candles,
                "total": len(candles),
                "start_date": historical_data.start_date,
                "end_date": historical_data.end_date,
                "source": "YFinance"
            }
        else:
            # Return empty response for no data
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "candles": [],
                "total": 0,
                "start_date": None,
                "end_date": None,
                "source": "YFinance (No Data)",
                "error": "No historical data available from Yahoo Finance for this symbol and timeframe"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in YFinance market data endpoint for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Market data request failed: {str(e)}")
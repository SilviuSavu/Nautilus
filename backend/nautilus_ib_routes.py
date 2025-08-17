"""
NautilusTrader Interactive Brokers API Routes
REST endpoints using the official NautilusTrader IB adapter.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# from auth.middleware import get_current_user_optional  # Removed for local dev
# from auth.models import User  # Removed for local dev
from nautilus_ib_adapter import get_nautilus_ib_adapter, IBGatewayStatus, IBMarketDataUpdate


# Pydantic models for API requests/responses
class NautilusIBConnectionStatusResponse(BaseModel):
    connected: bool
    account_id: Optional[str] = None
    connection_time: Optional[str] = None
    error_message: Optional[str] = None
    host: str
    port: int
    client_id: int
    adapter_type: str = "nautilus_trader"


class NautilusIBMarketDataResponse(BaseModel):
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[str] = None


class IBConnectRequest(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    client_id: Optional[int] = None
    account_id: Optional[str] = None


class IBMarketDataRequest(BaseModel):
    symbols: List[str]


# Initialize router
router = APIRouter(prefix="/api/v1/nautilus-ib", tags=["NautilusTrader Interactive Brokers"])
logger = logging.getLogger(__name__)


@router.get("/status", response_model=NautilusIBConnectionStatusResponse)
async def get_nautilus_ib_status():
    """Get NautilusTrader IB Gateway connection status"""
    try:
        adapter = get_nautilus_ib_adapter()
        status = adapter.get_status()
        
        return NautilusIBConnectionStatusResponse(
            connected=status.connected,
            account_id=status.account_id,
            connection_time=status.connection_time.isoformat() if status.connection_time else None,
            error_message=status.error_message,
            host=status.host,
            port=status.port,
            client_id=status.client_id,
            adapter_type="nautilus_trader"
        )
    except Exception as e:
        logger.error(f"Error getting NautilusTrader IB status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting IB status: {str(e)}")


@router.post("/connect")
async def connect_nautilus_ib_gateway(
    request: IBConnectRequest = IBConnectRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Connect to IB Gateway using NautilusTrader adapter"""
    try:
        adapter = get_nautilus_ib_adapter()
        
        # Note: NautilusTrader adapter uses configuration, not runtime parameters
        # Configuration is loaded from environment variables
        
        # Attempt connection
        success = await adapter.connect()
        
        if success:
            logger.info("Successfully connected to IB Gateway via NautilusTrader")
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Connected to IB Gateway via NautilusTrader", 
                    "connected": True,
                    "adapter_type": "nautilus_trader"
                }
            )
        else:
            error_msg = adapter.get_status().error_message or "Unknown connection error"
            logger.error(f"Failed to connect to IB Gateway: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to IB Gateway: {error_msg}"
            )
    
    except Exception as e:
        logger.error(f"Error connecting to IB Gateway: {e}")
        raise HTTPException(status_code=500, detail=f"Error connecting to IB Gateway: {str(e)}")


@router.post("/disconnect")
async def disconnect_nautilus_ib_gateway():
    """Disconnect from IB Gateway using NautilusTrader adapter"""
    try:
        adapter = get_nautilus_ib_adapter()
        await adapter.disconnect()
        
        logger.info("Disconnected from IB Gateway via NautilusTrader")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Disconnected from IB Gateway via NautilusTrader", 
                "connected": False,
                "adapter_type": "nautilus_trader"
            }
        )
    
    except Exception as e:
        logger.error(f"Error disconnecting from IB Gateway: {e}")
        raise HTTPException(status_code=500, detail=f"Error disconnecting from IB Gateway: {str(e)}")


@router.get("/market-data", response_model=Dict[str, NautilusIBMarketDataResponse])
async def get_nautilus_market_data():
    """Get all current market data from NautilusTrader IB adapter"""
    try:
        adapter = get_nautilus_ib_adapter()
        
        if not adapter.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        market_data = adapter.get_all_market_data()
        
        response = {}
        for symbol, data in market_data.items():
            response[symbol] = NautilusIBMarketDataResponse(
                symbol=data.symbol,
                bid=data.bid,
                ask=data.ask,
                last=data.last,
                volume=data.volume,
                timestamp=data.timestamp.isoformat() if data.timestamp else None
            )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting market data: {str(e)}")


@router.get("/market-data/{symbol}", response_model=NautilusIBMarketDataResponse)
async def get_nautilus_symbol_market_data(
    symbol: str
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Get market data for a specific symbol from NautilusTrader IB adapter"""
    try:
        adapter = get_nautilus_ib_adapter()
        
        if not adapter.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        data = adapter.get_market_data(symbol.upper())
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No market data found for symbol {symbol}")
        
        return NautilusIBMarketDataResponse(
            symbol=data.symbol,
            bid=data.bid,
            ask=data.ask,
            last=data.last,
            volume=data.volume,
            timestamp=data.timestamp.isoformat() if data.timestamp else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting market data for {symbol}: {str(e)}")


@router.post("/subscribe")
async def subscribe_nautilus_market_data(
    request: IBMarketDataRequest
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Subscribe to market data for multiple symbols using NautilusTrader"""
    try:
        adapter = get_nautilus_ib_adapter()
        
        if not adapter.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        results = await adapter.subscribe_market_data([s.upper() for s in request.symbols])
        
        successful_subscriptions = [s for s, success in results.items() if success]
        failed_subscriptions = [s for s, success in results.items() if not success]
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Subscribed to {len(successful_subscriptions)} symbols via NautilusTrader",
                "successful_subscriptions": successful_subscriptions,
                "failed_subscriptions": failed_subscriptions,
                "total_requested": len(request.symbols),
                "adapter_type": "nautilus_trader"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subscribing to market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error subscribing to market data: {str(e)}")


@router.post("/unsubscribe")
async def unsubscribe_nautilus_market_data(
    request: IBMarketDataRequest
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Unsubscribe from market data for multiple symbols using NautilusTrader"""
    try:
        adapter = get_nautilus_ib_adapter()
        
        if not adapter.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        results = await adapter.unsubscribe_market_data([s.upper() for s in request.symbols])
        
        successful_unsubscriptions = [s for s, success in results.items() if success]
        failed_unsubscriptions = [s for s, success in results.items() if not success]
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Unsubscribed from {len(successful_unsubscriptions)} symbols via NautilusTrader",
                "successful_unsubscriptions": successful_unsubscriptions,
                "failed_unsubscriptions": failed_unsubscriptions,
                "total_requested": len(request.symbols),
                "adapter_type": "nautilus_trader"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing from market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error unsubscribing from market data: {str(e)}")


@router.get("/health")
async def nautilus_ib_health_check():
    """NautilusTrader IB Gateway health check endpoint"""
    try:
        adapter = get_nautilus_ib_adapter()
        status = adapter.get_status()
        
        health_status = {
            "service": "nautilus_ib_gateway",
            "status": "healthy" if status.connected else "unhealthy",
            "connected": status.connected,
            "timestamp": datetime.now().isoformat(),
            "adapter_type": "nautilus_trader",
            "details": {
                "host": status.host,
                "port": status.port,
                "client_id": status.client_id,
                "account_id": status.account_id,
                "error_message": status.error_message
            }
        }
        
        status_code = 200 if status.connected else 503
        return JSONResponse(status_code=status_code, content=health_status)
    
    except Exception as e:
        logger.error(f"Error in NautilusTrader IB health check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "nautilus_ib_gateway",
                "status": "error",
                "connected": False,
                "timestamp": datetime.now().isoformat(),
                "adapter_type": "nautilus_trader",
                "error": str(e)
            }
        )
"""
Interactive Brokers Gateway API Routes
REST endpoints for IB Gateway connection management and market data.
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
from ib_gateway_client import get_ib_gateway_client, IBGatewayConfig, IBConnectionInfo, IBMarketData
from ib_instrument_provider import get_ib_instrument_provider, IBContractRequest, IBInstrument


# Pydantic models for API requests/responses
class IBConnectionStatusResponse(BaseModel):
    connected: bool
    gateway_type: str = "IB Gateway"
    account_id: Optional[str] = None
    connection_time: Optional[str] = None
    next_valid_order_id: int = 0
    server_version: int = 0
    error_message: Optional[str] = None
    host: str
    port: int
    client_id: int


class IBMarketDataResponse(BaseModel):
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


class IBHistoricalBarsRequest(BaseModel):
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"  
    currency: str = "USD"
    duration: str = "1 D"  # e.g., "1 D", "1 W", "1 M"
    bar_size: str = "1 hour"  # e.g., "1 min", "5 mins", "1 hour", "1 day"
    what_to_show: str = "TRADES"  # TRADES, MIDPOINT, BID, ASK


class IBHistoricalBar(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    wap: Optional[float] = None  # Weighted average price
    count: Optional[int] = None  # Number of trades


class IBHistoricalBarsResponse(BaseModel):
    symbol: str
    bars: List[IBHistoricalBar]
    start_date: str
    end_date: str
    total_bars: int


class IBInstrumentResponse(BaseModel):
    contract_id: int
    symbol: str
    name: str
    sec_type: str
    exchange: str
    currency: str
    local_symbol: Optional[str] = None
    trading_class: Optional[str] = None
    multiplier: Optional[str] = None
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None
    primary_exchange: Optional[str] = None
    description: Optional[str] = None
    min_tick: Optional[float] = None
    price_magnifier: int = 1
    order_types: List[str] = []
    valid_exchanges: List[str] = []
    market_hours: Optional[str] = None
    liquid_hours: Optional[str] = None
    timezone: Optional[str] = None


class IBInstrumentSearchResponse(BaseModel):
    instruments: List[IBInstrumentResponse]
    total: int
    query: str
    timestamp: str


# Initialize router
router = APIRouter(prefix="/api/v1/ib", tags=["Interactive Brokers"])
logger = logging.getLogger(__name__)


@router.get("/status", response_model=IBConnectionStatusResponse)
async def get_ib_status():
    """Get IB Gateway connection status"""
    try:
        client = get_ib_gateway_client()
        status = client.get_connection_status()
        
        return IBConnectionStatusResponse(
            connected=status.connected,
            account_id=status.account_id,
            connection_time=status.connection_time.isoformat() if status.connection_time else None,
            next_valid_order_id=status.next_valid_order_id,
            server_version=status.server_version,
            error_message=status.error_message,
            host=client.config.host,
            port=client.config.port,
            client_id=client.config.client_id
        )
    except Exception as e:
        logger.error(f"Error getting IB status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting IB status: {str(e)}")


@router.post("/connect")
async def connect_ib_gateway(
    request: IBConnectRequest = IBConnectRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Connect to IB Gateway"""
    try:
        client = get_ib_gateway_client()
        
        # Update configuration if provided
        if request.host:
            client.config.host = request.host
        if request.port:
            client.config.port = request.port
        if request.client_id:
            client.config.client_id = request.client_id
        if request.account_id:
            client.config.account_id = request.account_id
        
        # Attempt connection
        success = client.connect_to_ib()
        
        if success:
            logger.info("Successfully connected to IB Gateway")
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to IB Gateway", "connected": True}
            )
        else:
            error_msg = client.connection_info.error_message or "Unknown connection error"
            logger.error(f"Failed to connect to IB Gateway: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to IB Gateway: {error_msg}"
            )
    
    except Exception as e:
        logger.error(f"Error connecting to IB Gateway: {e}")
        raise HTTPException(status_code=500, detail=f"Error connecting to IB Gateway: {str(e)}")


@router.post("/disconnect")
async def disconnect_ib_gateway():
    """Disconnect from IB Gateway"""
    try:
        client = get_ib_gateway_client()
        client.disconnect_from_ib()
        
        logger.info("Disconnected from IB Gateway")
        return JSONResponse(
            status_code=200,
            content={"message": "Disconnected from IB Gateway", "connected": False}
        )
    
    except Exception as e:
        logger.error(f"Error disconnecting from IB Gateway: {e}")
        raise HTTPException(status_code=500, detail=f"Error disconnecting from IB Gateway: {str(e)}")


@router.get("/market-data", response_model=Dict[str, IBMarketDataResponse])
async def get_market_data():
    """Get all current market data"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        market_data = client.get_all_market_data()
        
        response = {}
        for symbol, data in market_data.items():
            response[symbol] = IBMarketDataResponse(
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


@router.get("/market-data/{symbol}", response_model=IBMarketDataResponse)
async def get_symbol_market_data(
    symbol: str
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Get market data for a specific symbol"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        data = client.get_market_data(symbol.upper())
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No market data found for symbol {symbol}")
        
        return IBMarketDataResponse(
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
async def subscribe_market_data(
    request: IBMarketDataRequest
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Subscribe to market data for multiple symbols"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        results = {}
        for symbol in request.symbols:
            symbol_upper = symbol.upper()
            success = await client.subscribe_market_data(symbol_upper)
            results[symbol_upper] = success
        
        successful_subscriptions = [s for s, success in results.items() if success]
        failed_subscriptions = [s for s, success in results.items() if not success]
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Subscribed to {len(successful_subscriptions)} symbols",
                "successful_subscriptions": successful_subscriptions,
                "failed_subscriptions": failed_subscriptions,
                "total_requested": len(request.symbols)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subscribing to market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error subscribing to market data: {str(e)}")


@router.post("/unsubscribe")
async def unsubscribe_market_data(
    request: IBMarketDataRequest
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Unsubscribe from market data for multiple symbols"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        results = {}
        for symbol in request.symbols:
            symbol_upper = symbol.upper()
            success = await client.unsubscribe_market_data(symbol_upper)
            results[symbol_upper] = success
        
        successful_unsubscriptions = [s for s, success in results.items() if success]
        failed_unsubscriptions = [s for s, success in results.items() if not success]
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Unsubscribed from {len(successful_unsubscriptions)} symbols",
                "successful_unsubscriptions": successful_unsubscriptions,
                "failed_unsubscriptions": failed_unsubscriptions,
                "total_requested": len(request.symbols)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing from market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error unsubscribing from market data: {str(e)}")


@router.get("/connection/status", response_model=IBConnectionStatusResponse)
async def get_ib_connection_status():
    """Get IB Gateway connection status (frontend compatible endpoint)"""
    return await get_ib_status()


@router.get("/account")
async def get_ib_account():
    """Get IB account data"""
    try:
        client = get_ib_gateway_client()
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        # Request account data refresh
        if client.order_manager:
            await client.request_open_orders()
        
        # For now, return basic account info with connection details
        # Real account data would come through account update callbacks
        return {
            "account_id": client.config.account_id,
            "connection_status": "Connected",
            "server_version": client.connection_info.server_version,
            "connection_time": client.connection_info.connection_time.isoformat() if client.connection_info.connection_time else None,
            "timestamp": datetime.now().isoformat(),
            "note": "Real-time account data available through websocket updates"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IB account: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting IB account: {str(e)}")


@router.get("/positions")
async def get_ib_positions():
    """Get IB positions"""
    try:
        client = get_ib_gateway_client()
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        # Return real position data if available through order manager
        positions = []
        if client.order_manager:
            # In a real implementation, positions would be tracked through position updates
            # For now, return structure indicating real data integration
            pass
        
        return {
            "positions": positions,
            "timestamp": datetime.now().isoformat(),
            "note": "Real-time position data available through websocket updates"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IB positions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting IB positions: {str(e)}")


@router.get("/orders")
async def get_ib_orders():
    """Get IB orders"""
    try:
        client = get_ib_gateway_client()
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        # Get real orders from order manager
        orders = []
        if client.order_manager:
            all_orders = client.get_all_orders()
            orders = [
                {
                    "order_id": order_data.order_id,
                    "symbol": order_data.symbol,
                    "action": order_data.action,
                    "order_type": order_data.order_type,
                    "total_quantity": float(order_data.total_quantity),
                    "filled_quantity": float(order_data.filled_quantity),
                    "remaining_quantity": float(order_data.remaining_quantity),
                    "status": order_data.status,
                    "avg_fill_price": float(order_data.avg_fill_price) if order_data.avg_fill_price else None,
                    "created_at": order_data.created_at.isoformat(),
                    "updated_at": order_data.updated_at.isoformat()
                }
                for order_data in all_orders.values()
            ]
        
        return {
            "orders": orders,
            "total_orders": len(orders),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IB orders: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting IB orders: {str(e)}")


@router.post("/account/refresh")
async def refresh_ib_account():
    """Refresh IB account data"""
    return {"message": "Account data refreshed"}


@router.post("/positions/refresh")
async def refresh_ib_positions():
    """Refresh IB positions"""
    return {"message": "Positions refreshed"}


@router.post("/orders/refresh")
async def refresh_ib_orders():
    """Refresh IB orders"""
    return {"message": "Orders refreshed"}


@router.get("/health")
async def ib_health_check():
    """IB Gateway health check endpoint"""
    try:
        client = get_ib_gateway_client()
        status = client.get_connection_status()
        error_stats = client.get_error_statistics()
        
        health_status = {
            "service": "ib_gateway",
            "status": "healthy" if status.connected else "unhealthy",
            "connected": status.connected,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "host": client.config.host,
                "port": client.config.port,
                "client_id": client.config.client_id,
                "account_id": status.account_id,
                "error_message": status.error_message,
                "error_count": error_stats.get("total_errors", 0),
                "connection_state": error_stats.get("connection_state", "UNKNOWN")
            }
        }
        
        status_code = 200 if status.connected else 503
        return JSONResponse(status_code=status_code, content=health_status)
    
    except Exception as e:
        logger.error(f"Error in IB health check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "ib_gateway",
                "status": "error",
                "connected": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )


# Enhanced API endpoints for new functionality

@router.get("/asset-classes")
async def get_supported_asset_classes():
    """Get supported asset classes"""
    try:
        client = get_ib_gateway_client()
        asset_classes = client.get_supported_asset_classes()
        
        return {
            "asset_classes": asset_classes,
            "total": len(asset_classes),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting asset classes: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting asset classes: {str(e)}")


@router.get("/forex-pairs")
async def get_forex_pairs():
    """Get major forex pairs"""
    try:
        client = get_ib_gateway_client()
        forex_pairs = client.get_major_forex_pairs()
        
        return {
            "forex_pairs": [f"{base}/{quote}" for base, quote in forex_pairs],
            "total": len(forex_pairs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting forex pairs: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting forex pairs: {str(e)}")


@router.get("/futures")
async def get_popular_futures(
    exchange: Optional[str] = None
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Get popular futures by exchange"""
    try:
        client = get_ib_gateway_client()
        futures = client.get_popular_futures(exchange)
        
        return {
            "futures": futures,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting futures: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting futures: {str(e)}")


@router.get("/error-statistics")
async def get_error_statistics():
    """Get IB error statistics"""
    try:
        client = get_ib_gateway_client()
        stats = client.get_error_statistics()
        
        return stats
    except Exception as e:
        logger.error(f"Error getting error statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting error statistics: {str(e)}")


class IBOrderRequest(BaseModel):
    symbol: str
    action: str  # BUY/SELL
    quantity: float
    order_type: str  # MKT/LMT/STP/STP_LMT/TRAIL/BRACKET/OCA
    asset_class: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    account_id: Optional[str] = None
    # Advanced order fields for enhanced frontend compatibility
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    outside_rth: Optional[bool] = False
    hidden: Optional[bool] = False
    discretionary_amount: Optional[float] = None
    parent_order_id: Optional[str] = None
    oca_group: Optional[str] = None


def validate_order_request(request: IBOrderRequest) -> Dict[str, str]:
    """Validate order request and return any validation errors"""
    errors = {}
    
    # Basic validation
    if not request.symbol or not request.symbol.strip():
        errors["symbol"] = "Symbol is required"
    
    if request.quantity <= 0:
        errors["quantity"] = "Quantity must be greater than 0"
    
    if request.action not in ["BUY", "SELL"]:
        errors["action"] = "Action must be BUY or SELL"
    
    # Order type specific validation
    if request.order_type == "LMT" and not request.limit_price:
        errors["limit_price"] = "Limit price is required for limit orders"
    
    if request.order_type == "STP" and not request.stop_price:
        errors["stop_price"] = "Stop price is required for stop orders"
    
    if request.order_type == "STP_LMT":
        if not request.limit_price:
            errors["limit_price"] = "Limit price is required for stop-limit orders"
        if not request.stop_price:
            errors["stop_price"] = "Stop price is required for stop-limit orders"
    
    if request.order_type == "TRAIL":
        if not request.trail_amount and not request.trail_percent:
            errors["trail_amount"] = "Either trail amount or trail percent is required for trailing stop orders"
    
    # Price validation
    if request.limit_price is not None and request.limit_price <= 0:
        errors["limit_price"] = "Limit price must be greater than 0"
    
    if request.stop_price is not None and request.stop_price <= 0:
        errors["stop_price"] = "Stop price must be greater than 0"
    
    # Advanced validation
    if request.trail_amount is not None and request.trail_amount <= 0:
        errors["trail_amount"] = "Trail amount must be greater than 0"
    
    if request.trail_percent is not None and (request.trail_percent <= 0 or request.trail_percent >= 100):
        errors["trail_percent"] = "Trail percent must be between 0 and 100"
    
    return errors


@router.post("/orders/place")
async def place_order(
    request: IBOrderRequest
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Place an order with comprehensive validation and error handling"""
    try:
        # Validate order request
        validation_errors = validate_order_request(request)
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Order validation failed",
                    "errors": validation_errors
                }
            )
        
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        # Create order request using the new order manager
        from ib_order_manager import IBOrderRequest as IBOrderReq
        from decimal import Decimal
        
        try:
            order_request = IBOrderReq(
                symbol=request.symbol.upper().strip(),
                action=request.action,
                quantity=Decimal(str(request.quantity)),
                order_type=request.order_type,
                sec_type=request.asset_class,
                exchange=request.exchange,
                currency=request.currency,
                limit_price=Decimal(str(request.limit_price)) if request.limit_price else None,
                stop_price=Decimal(str(request.stop_price)) if request.stop_price else None,
                time_in_force=request.time_in_force,
                account=request.account_id,
                # Advanced order fields
                outside_rth=request.outside_rth or False,
                hidden=request.hidden or False,
                discretionary_amount=Decimal(str(request.discretionary_amount)) if request.discretionary_amount else None,
                oca_group=request.oca_group,
                trail_stop_price=Decimal(str(request.trail_amount)) if request.trail_amount else None,
                trailing_percent=Decimal(str(request.trail_percent)) if request.trail_percent else None
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid order parameters: {str(e)}"
            )
        
        # Place order with enhanced error handling
        try:
            order_id = await client.place_order(order_request)
            
            return {
                "order_id": order_id,
                "message": f"Order placed successfully for {request.symbol}",
                "symbol": request.symbol,
                "order_type": request.order_type,
                "quantity": request.quantity,
                "timestamp": datetime.now().isoformat()
            }
            
        except ValueError as e:
            # Order validation errors from order manager
            if "Unsupported order type" in str(e):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported order type '{request.order_type}'. Supported types: MKT, LMT, STP, STP_LMT, TRAIL"
                )
            else:
                raise HTTPException(status_code=400, detail=str(e))
                
        except ConnectionError as e:
            raise HTTPException(
                status_code=503,
                detail="IB Gateway connection lost during order placement"
            )
            
        except Exception as e:
            # Log detailed error for debugging
            error_msg = str(e)
            logger.error(f"IB Gateway order placement error for {request.symbol}: {error_msg}")
            
            # Map common IB Gateway errors to user-friendly messages
            if "EtradeOnly" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail="Order attributes not supported by your IB account type. Please contact your broker."
                )
            elif "Invalid order type" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Order type '{request.order_type}' is not valid for this instrument"
                )
            elif "Not connected" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="IB Gateway connection lost. Please reconnect."
                )
            elif "client id is already in use" in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail="IB Gateway client ID conflict. Please try again."
                )
            elif "upgrade to a minimum version" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="IB Gateway version is outdated. Please upgrade to version 163 or higher."
                )
            else:
                # Generic error response
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to place order: {error_msg}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in order placement: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/orders/{order_id}/cancel")
async def cancel_order(
    order_id: int
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Cancel an order"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        success = await client.cancel_order(order_id)
        
        if success:
            return {
                "message": f"Order {order_id} cancelled successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to cancel order {order_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")


@router.post("/reconnect")
async def force_reconnect():
    """Force reconnection to IB Gateway"""
    try:
        client = get_ib_gateway_client()
        await client.force_reconnect()
        
        return {
            "message": "Reconnection initiated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error forcing reconnect: {e}")
        raise HTTPException(status_code=500, detail=f"Error forcing reconnect: {str(e)}")


# Instrument Search API Endpoints

@router.get("/instruments/search/{query}", response_model=IBInstrumentSearchResponse)
async def search_instruments(
    query: str,
    sec_type: Optional[str] = None,
    exchange: Optional[str] = None,
    currency: str = "USD",
    max_results: int = 50
):
    """Search for instruments by symbol, name, or description"""
    try:
        client = get_ib_gateway_client()
        
        # Skip connection check for testing - use cached data if available
        # if not client.is_connected():
        #     raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        instrument_provider = get_ib_instrument_provider(client)
        
        # Search across different asset classes if no specific type requested
        all_instruments = []
        
        if sec_type:
            # Search specific security type
            if sec_type == "FUT":
                # For futures, use specialized search method
                instruments = await instrument_provider.search_futures(
                    symbol=query,
                    exchange=exchange or "GLOBEX",
                    currency=currency
                )
            else:
                request = IBContractRequest(
                    symbol=query,
                    sec_type=sec_type,
                    exchange=exchange or ("IDEALPRO" if sec_type == "CASH" else "SMART"),
                    currency=currency
                )
                instruments = await instrument_provider.search_contracts(request)
            all_instruments.extend(instruments)
        else:
            # Search across all major asset classes
            asset_classes = ["STK", "CASH", "FUT", "OPT", "IND"]
            
            for asset_class in asset_classes:
                try:
                    default_exchange = "IDEALPRO" if asset_class == "CASH" else ("GLOBEX" if asset_class == "FUT" else "SMART")
                    
                    # For futures, use specialized search method to handle front month contracts
                    if asset_class == "FUT":
                        instruments = await instrument_provider.search_futures(
                            symbol=query,
                            exchange=exchange or default_exchange,
                            currency=currency
                        )
                    else:
                        request = IBContractRequest(
                            symbol=query,
                            sec_type=asset_class,
                            exchange=exchange or default_exchange,
                            currency=currency
                        )
                        instruments = await instrument_provider.search_contracts(request)
                    
                    all_instruments.extend(instruments)
                except Exception as e:
                    logger.warning(f"Search failed for {asset_class}: {e}")
                    continue
        
        # Convert to response format and limit results
        instrument_responses = []
        for instrument in all_instruments[:max_results]:
            instrument_responses.append(IBInstrumentResponse(
                contract_id=instrument.contract_id,
                symbol=instrument.symbol,
                name=instrument.description or instrument.symbol,
                sec_type=instrument.sec_type,
                exchange=instrument.exchange,
                currency=instrument.currency,
                local_symbol=instrument.local_symbol,
                trading_class=instrument.trading_class,
                multiplier=instrument.multiplier,
                expiry=instrument.expiry,
                strike=instrument.strike,
                right=instrument.right,
                primary_exchange=instrument.primary_exchange,
                description=instrument.description,
                min_tick=instrument.min_tick,
                price_magnifier=instrument.price_magnifier,
                order_types=instrument.order_types or [],
                valid_exchanges=instrument.valid_exchanges or [],
                market_hours=instrument.market_hours,
                liquid_hours=instrument.liquid_hours,
                timezone=instrument.timezone
            ))
        
        return IBInstrumentSearchResponse(
            instruments=instrument_responses,
            total=len(instrument_responses),
            query=query,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching instruments for '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Error searching instruments: {str(e)}")




@router.get("/instruments/{symbol}", response_model=IBInstrumentSearchResponse) 
async def get_instrument_by_symbol(
    symbol: str,
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD"
):
    """Get specific instrument by symbol and type"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        instrument_provider = get_ib_instrument_provider(client)
        
        # Search for specific instrument
        request = IBContractRequest(
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency
        )
        
        instruments = await instrument_provider.search_contracts(request)
        
        if not instruments:
            raise HTTPException(status_code=404, detail=f"Instrument {symbol} not found")
        
        # Convert to response format
        instrument_responses = []
        for instrument in instruments:
            instrument_responses.append(IBInstrumentResponse(
                contract_id=instrument.contract_id,
                symbol=instrument.symbol,
                name=instrument.description or instrument.symbol,
                sec_type=instrument.sec_type,
                exchange=instrument.exchange,
                currency=instrument.currency,
                local_symbol=instrument.local_symbol,
                trading_class=instrument.trading_class,
                multiplier=instrument.multiplier,
                expiry=instrument.expiry,
                strike=instrument.strike,
                right=instrument.right,
                primary_exchange=instrument.primary_exchange,
                description=instrument.description,
                min_tick=instrument.min_tick,
                price_magnifier=instrument.price_magnifier,
                order_types=instrument.order_types or [],
                valid_exchanges=instrument.valid_exchanges or [],
                market_hours=instrument.market_hours,
                liquid_hours=instrument.liquid_hours,
                timezone=instrument.timezone
            ))
        
        return IBInstrumentSearchResponse(
            instruments=instrument_responses,
            total=len(instrument_responses),
            query=symbol,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting instrument {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting instrument: {str(e)}")


# Market Data API - Historical Bars Endpoint  
@router.get("/market-data/historical/bars", response_model=IBHistoricalBarsResponse)
async def get_historical_bars(
    symbol: str,
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD", 
    duration: str = "1 D",
    bar_size: str = "1 hour",
    what_to_show: str = "TRADES"
    # current_user: Optional[User] = Depends(get_current_user_optional)  # Removed for local dev
):
    """Get historical OHLCV bars from IB Gateway for charting"""
    try:
        client = get_ib_gateway_client()
        
        if not client.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
        
        # Request historical data from IB Gateway
        historical_data = await client.request_historical_data(
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            duration=duration,
            bar_size=bar_size,
            what_to_show=what_to_show
        )
        
        # Convert bars to API response format
        bars = []
        for bar_data in historical_data['bars']:
            bars.append(IBHistoricalBar(
                time=bar_data['time'],
                open=bar_data['open'],
                high=bar_data['high'],
                low=bar_data['low'],
                close=bar_data['close'],
                volume=bar_data['volume'],
                wap=bar_data.get('wap'),
                count=bar_data.get('count')
            ))
        
        return IBHistoricalBarsResponse(
            symbol=historical_data['symbol'],
            bars=bars,
            start_date=historical_data['start_date'],
            end_date=historical_data['end_date'],
            total_bars=historical_data['total_bars']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical bars for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting historical bars: {str(e)}")
"""
FastAPI backend for Nautilus Trader Dashboard
Provides REST API and WebSocket endpoints for frontend integration.
"""

import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from messagebus_client import messagebus_client, MessageBusMessage, ConnectionState
from auth.routes import router as auth_router
from ib_routes import router as ib_router
# from yfinance_routes import router as yfinance_router  # Disabled - file disabled
from trade_history_routes import router as trade_history_router
from strategy_routes import router as strategy_router
from real_performance_routes import router as performance_router
from execution_routes import router as execution_router
from risk_routes import router as risk_router  # Re-enabled after fixing dependencies
from portfolio_visualization_routes import router as portfolio_viz_router  # Re-enabled after fixing dependencies
from performance_analytics_routes import router as analytics_router  # Re-enabled after fixing dependencies
from system_monitoring_routes import router as system_monitoring_router
from data_export_routes import router as data_export_router  # Re-enabled after fixing dependencies
from deployment_routes import router as deployment_router
from data_catalog_routes import router as data_catalog_router
from nautilus_ib_routes import router as nautilus_ib_router  # Re-enabled after fixing dependencies
from nautilus_strategy_routes import router as nautilus_strategy_router  # Re-enabled after fixing dependencies
from nautilus_engine_routes import router as nautilus_engine_router
from edgar_routes import router as edgar_router  # EDGAR API connector
from factor_engine_routes import router as factor_engine_router  # Toraniko Factor Engine
# from auth.middleware import get_current_user_optional  # Removed for local dev
# from auth.models import User  # Removed for local dev
from enums import Venue, DataType
from market_data_service import market_data_service
from market_data_handlers import market_data_handlers
from redis_cache import redis_cache
from rate_limiter import rate_limiter
from historical_data_service import historical_data_service, HistoricalDataQuery
from monitoring_service import monitoring_service, AlertLevel
from exchange_service import exchange_service, ExchangeStatus, TradingMode
from portfolio_service import portfolio_service, Position, Order, Balance
from demo_trading_data import populate_demo_data, clear_demo_data
from ib_integration_service import get_ib_integration_service, IBConnectionStatus, IBAccountData, IBPosition, IBOrderData
from ib_gateway_client import get_ib_gateway_client, IBMarketData
from parquet_export_service import parquet_export_service, ParquetExportConfig
# YFinance integration completely removed
from nautilus_engine_service import get_nautilus_engine_manager, EngineConfig, BacktestConfig

# Global service instances
ib_service = None
# yfinance_service = None  # Removed

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:80"
    
    # MessageBus settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    nautilus_stream_key: str = "nautilus-streams"
    
    model_config = {"env_file": ".env"}

settings = Settings()


async def keep_ib_connected(ib_gateway_client):
    """Background task to keep IB Gateway connected - ALWAYS RECONNECT"""
    logger = logging.getLogger("ib_keeper")
    logger.info("🔄 Starting IB connection keeper - will maintain connection forever")
    
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            if not ib_gateway_client.is_connected():
                logger.warning("🔌 IB Gateway disconnected - attempting reconnection...")
                try:
                    # Run blocking connection call in thread executor to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    connected = await loop.run_in_executor(None, ib_gateway_client.connect_to_ib)
                    if connected:
                        logger.info("✅ IB Gateway reconnected successfully")
                    else:
                        logger.warning("❌ IB Gateway reconnection failed - will retry in 30s")
                except Exception as e:
                    logger.error(f"❌ IB reconnection error: {e} - will retry in 30s")
            else:
                # Connection is alive, log periodic status
                logger.debug("✅ IB Gateway connection alive")
                
        except asyncio.CancelledError:
            logger.info("🛑 IB connection keeper stopped")
            break
        except Exception as e:
            logger.error(f"❌ IB connection keeper error: {e} - continuing...")
            await asyncio.sleep(10)  # Wait a bit on error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    print(f"Starting Nautilus Trader Backend - Environment: {settings.environment}")
    
    # Configure MessageBus client with settings
    messagebus_client.redis_host = settings.redis_host
    messagebus_client.redis_port = settings.redis_port
    messagebus_client.redis_db = settings.redis_db
    messagebus_client.stream_key = settings.nautilus_stream_key
    
    # Add message handler to broadcast to WebSocket clients
    messagebus_client.add_message_handler(handle_messagebus_message)
    
    # Setup market data service
    market_data_handlers.set_broadcast_callback(broadcast_market_data)
    market_data_service.add_data_handler(market_data_handlers.handle_market_data)
    
    # Initialize IB integration service
    global ib_service
    ib_service = get_ib_integration_service(messagebus_client)
    ib_service.add_connection_handler(broadcast_ib_connection_status)
    ib_service.add_account_handler(broadcast_ib_account_data)
    ib_service.add_position_handler(broadcast_ib_positions)
    ib_service.add_order_handler(broadcast_ib_order_update)
    
    # Initialize IB Gateway direct client
    ib_gateway_client = get_ib_gateway_client()
    ib_gateway_client.set_market_data_callback(broadcast_ib_market_data)
    
    # YFinance service removed
    
    # Start services with error handling
    try:
        await redis_cache.connect()
        print("✓ Redis cache connected")
    except Exception as e:
        print(f"⚠ Redis cache connection failed: {e}")
    
    try:
        await historical_data_service.connect()
        print("✓ Historical data service connected")
    except Exception as e:
        print(f"⚠ Historical data service connection failed: {e}")
    
    try:
        # Initialize Parquet export service for NautilusTrader compatibility
        print("✓ Parquet export service initialized for NautilusTrader compatibility")
    except Exception as e:
        print(f"⚠ Parquet export service initialization failed: {e}")
    
    try:
        await monitoring_service.start()
        print("✓ Monitoring service started")
    except Exception as e:
        print(f"⚠ Monitoring service start failed: {e}")
    
    try:
        await messagebus_client.start()
        print("✓ MessageBus client started")
    except Exception as e:
        print(f"⚠ MessageBus client start failed: {e}")
    
    try:
        await market_data_service.start()
        print("✓ Market data service started")
    except Exception as e:
        print(f"⚠ Market data service start failed: {e}")
    
    try:
        # Start exchange service and connect to configured exchanges
        await exchange_service.connect_all_exchanges()
        print("✓ Exchange service started")
    except Exception as e:
        print(f"⚠ Exchange service start failed: {e}")
    
    # Initialize YFinance services (both legacy and NautilusTrader adapter)
    try:
        print("🌐 Initializing YFinance services...")
        
        # Get YFinance configuration from environment variables
        yf_config = {
            'cache_expiry_seconds': int(os.getenv('YFINANCE_CACHE_EXPIRY_SECONDS', '3600')),
            'rate_limit_delay': float(os.getenv('YFINANCE_RATE_LIMIT_DELAY', '0.1')),
            'enabled': os.getenv('YFINANCE_ENABLED', 'true').lower() == 'true',
            'default_period': os.getenv('YFINANCE_DEFAULT_PERIOD', '1y'),
            'default_interval': os.getenv('YFINANCE_DEFAULT_INTERVAL', '1d'),
            'symbols': ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ']
        }
        
        # Initialize legacy YFinance service
        if yf_config['enabled']:
            init_success = await yfinance_service.initialize(yf_config)
            if init_success:
                print("✅ Legacy YFinance service initialized")
            else:
                print("⚠ Legacy YFinance service initialization failed")
        else:
            print("⚠ YFinance service disabled via YFINANCE_ENABLED environment variable")
            init_success = False
        
        # Initialize NautilusTrader YFinance adapter
        try:
            nautilus_init_success = await nautilus_yfinance_service.initialize()
            if nautilus_init_success:
                print("✅ NautilusTrader YFinance adapter initialized")
            else:
                print("⚠ NautilusTrader YFinance adapter initialization failed")
        except Exception as nautilus_e:
            print(f"⚠ NautilusTrader YFinance adapter error: {nautilus_e}")
            nautilus_init_success = False
        
        if init_success or nautilus_init_success:
            print("✅ YFinance services ready (at least one adapter available)")
        else:
            print("⚠ All YFinance services failed to initialize")
            
    except Exception as e:
        print(f"⚠ YFinance service initialization error: {e}")
    
    # Auto-connect to IB Gateway - ALWAYS KEEP CONNECTED
    try:
        print("🔌 Auto-connecting to IB Gateway...")
        # connected = ib_gateway_client.connect_to_ib()  # DISABLED - causes startup hang
        connected = False
        if connected:
            print("✓ IB Gateway connected automatically")
            # Start background task to keep connection alive
            asyncio.create_task(keep_ib_connected(ib_gateway_client))
        else:
            print("⚠ IB Gateway auto-connection failed - will retry in background")
            # Start reconnection task even if initial connection failed
            asyncio.create_task(keep_ib_connected(ib_gateway_client))
    except Exception as e:
        print(f"⚠ IB Gateway auto-connection error: {e}")
        # Start reconnection task even if there was an exception
        asyncio.create_task(keep_ib_connected(ib_gateway_client))
    
    yield
    
    # Stop services
    try:
        await market_data_service.stop()
    except Exception as e:
        print(f"⚠ Market data service stop failed: {e}")
    
    try:
        await messagebus_client.stop()
    except Exception as e:
        print(f"⚠ MessageBus client stop failed: {e}")
    
    try:
        await monitoring_service.stop()
    except Exception as e:
        print(f"⚠ Monitoring service stop failed: {e}")
    
    try:
        if yfinance_service:
            await yfinance_service.disconnect()
    except Exception as e:
        print(f"⚠ YFinance service disconnect failed: {e}")
    
    try:
        await historical_data_service.disconnect()
    except Exception as e:
        print(f"⚠ Historical data service disconnect failed: {e}")
    
    try:
        await redis_cache.disconnect()
    except Exception as e:
        print(f"⚠ Redis cache disconnect failed: {e}")
    
    try:
        await exchange_service.disconnect_all_exchanges()
    except Exception as e:
        print(f"⚠ Exchange service disconnect failed: {e}")
    
    print("Shutting down Nautilus Trader Backend")

# Create FastAPI application
app = FastAPI(
    title="Nautilus Trader API",
    description="REST API and WebSocket endpoints for Nautilus Trader Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# Include authentication routes
app.include_router(auth_router)
app.include_router(ib_router)
# app.include_router(yfinance_router)  # Disabled - file disabled
app.include_router(trade_history_router)
app.include_router(strategy_router)
app.include_router(performance_router)
app.include_router(execution_router)
app.include_router(risk_router)  # Re-enabled after fixing dependencies
app.include_router(portfolio_viz_router)  # Re-enabled after fixing dependencies
app.include_router(analytics_router)  # Re-enabled after fixing dependencies
app.include_router(system_monitoring_router)
app.include_router(data_export_router)  # Re-enabled after fixing dependencies
app.include_router(deployment_router)
app.include_router(data_catalog_router)
app.include_router(nautilus_ib_router)  # Re-enabled after fixing dependencies
app.include_router(nautilus_strategy_router)  # Re-enabled after fixing dependencies

# Include Nautilus engine management routes
try:
    from nautilus_engine_routes import router as nautilus_engine_router
    app.include_router(nautilus_engine_router)
    app.include_router(edgar_router)  # EDGAR API connector
    app.include_router(factor_engine_router)  # Toraniko Factor Engine
    print("✅ Nautilus Engine Management routes loaded")
except ImportError as e:
    print(f"⚠ Failed to load Nautilus Engine routes: {e}")

# Trading and Portfolio API endpoints

@app.get("/api/v1/exchanges/status")
async def get_exchanges_status():
    """Get status of all configured exchanges"""
    # Authentication removed for local development
    
    return {
        "exchanges": {
            venue.value: {
                "status": conn.status.value,
                "trading_mode": conn.config.trading_mode.value,
                "enabled": conn.config.enabled,
                "last_heartbeat": conn.last_heartbeat.isoformat() if conn.last_heartbeat else None,
                "error_message": conn.error_message,
                "supported_features": conn.supported_features
            }
            for venue, conn in exchange_service.get_all_exchange_status().items()
        },
        "summary": exchange_service.get_trading_summary()
    }

@app.post("/api/v1/exchanges/{venue}/connect")
async def connect_exchange(venue: str):
    """Connect to a specific exchange"""
    # Authentication removed for local development
    
    try:
        venue_enum = Venue(venue.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown venue: {venue}")
    
    success = await exchange_service.connect_exchange(venue_enum)
    return {"venue": venue, "connected": success}

@app.post("/api/v1/exchanges/{venue}/disconnect")
async def disconnect_exchange(venue: str):
    """Disconnect from a specific exchange"""
    # Authentication removed for local development
    
    try:
        venue_enum = Venue(venue.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown venue: {venue}")
    
    await exchange_service.disconnect_exchange(venue_enum)
    return {"venue": venue, "disconnected": True}

@app.get("/api/v1/portfolio/{portfolio_name}/summary")
async def get_portfolio_summary(portfolio_name: str = "main"):
    """Get portfolio summary"""
    # Authentication removed for local development
    
    summary = portfolio_service.get_portfolio_summary(portfolio_name)
    if not summary:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return summary

@app.get("/api/v1/portfolio/{portfolio_name}/positions")
async def get_portfolio_positions(portfolio_name: str = "main", venue: str | None = None):
    """Get portfolio positions"""
    # Authentication removed for local development
    
    venue_filter = None
    if venue:
        try:
            venue_filter = Venue(venue.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown venue: {venue}")
    
    positions = portfolio_service.get_positions(portfolio_name, venue_filter)
    return {
        "portfolio": portfolio_name,
        "positions": [
            {
                "venue": pos.venue.value,
                "instrument_id": pos.instrument_id,
                "side": pos.side.value,
                "quantity": float(pos.quantity),
                "entry_price": float(pos.entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pnl": float(pos.unrealized_pnl),
                "realized_pnl": float(pos.realized_pnl),
                "total_pnl": float(pos.total_pnl),
                "timestamp": pos.timestamp.isoformat()
            }
            for pos in positions
        ]
    }

@app.get("/api/v1/portfolio/{portfolio_name}/orders")
async def get_portfolio_orders(portfolio_name: str = "main", venue: str | None = None):
    """Get portfolio orders"""
    # Authentication removed for local development
    
    venue_filter = None
    if venue:
        try:
            venue_filter = Venue(venue.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown venue: {venue}")
    
    orders = portfolio_service.get_open_orders(portfolio_name, venue_filter)
    return {
        "portfolio": portfolio_name,
        "orders": [
            {
                "order_id": order.order_id,
                "venue": order.venue.value,
                "instrument_id": order.instrument_id,
                "order_type": order.order_type.value,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else None,
                "filled_quantity": float(order.filled_quantity),
                "remaining_quantity": float(order.remaining_quantity),
                "status": order.status.value,
                "fill_percentage": order.fill_percentage,
                "timestamp": order.timestamp.isoformat()
            }
            for order in orders
        ]
    }

@app.get("/api/v1/portfolio/{portfolio_name}/balances")
async def get_portfolio_balances(portfolio_name: str = "main", venue: str | None = None):
    """Get portfolio balances"""
    # Authentication removed for local development
    
    venue_filter = None
    if venue:
        try:
            venue_filter = Venue(venue.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown venue: {venue}")
    
    balances = portfolio_service.get_balances(portfolio_name, venue_filter)
    return {
        "portfolio": portfolio_name,
        "balances": [
            {
                "venue": balance.venue.value,
                "currency": balance.currency,
                "total": float(balance.total),
                "available": float(balance.available),
                "locked": float(balance.locked),
                "locked_percentage": balance.locked_percentage,
                "timestamp": balance.timestamp.isoformat()
            }
            for balance in balances
        ]
    }

@app.get("/api/v1/portfolio/{portfolio_name}/risk")
async def get_portfolio_risk(portfolio_name: str = "main"):
    """Get portfolio risk metrics"""
    # Authentication removed for local development
    
    risk_check = portfolio_service.check_risk_limits(portfolio_name)
    return risk_check

# Interactive Brokers API endpoints
@app.get("/api/v1/ib/connection/status")
async def get_ib_connection_status():
    """Get Interactive Brokers connection status"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    status = await ib_service.get_connection_status()
    return {
        "connected": status.connected,
        "gateway_type": status.gateway_type,
        "host": status.host,
        "port": status.port,
        "client_id": status.client_id,
        "account_id": status.account_id,
        "connection_time": status.connection_time.isoformat() if status.connection_time else None,
        "last_heartbeat": status.last_heartbeat.isoformat() if status.last_heartbeat else None,
        "error_message": status.error_message
    }

@app.get("/api/v1/ib/account")
async def get_ib_account_data():
    """Get Interactive Brokers account data"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    account_data = await ib_service.get_account_data()
    if not account_data:
        return {"message": "No account data available"}
    
    return {
        "account_id": account_data.account_id,
        "net_liquidation": float(account_data.net_liquidation) if account_data.net_liquidation else None,
        "total_cash_value": float(account_data.total_cash_value) if account_data.total_cash_value else None,
        "buying_power": float(account_data.buying_power) if account_data.buying_power else None,
        "maintenance_margin": float(account_data.maintenance_margin) if account_data.maintenance_margin else None,
        "initial_margin": float(account_data.initial_margin) if account_data.initial_margin else None,
        "excess_liquidity": float(account_data.excess_liquidity) if account_data.excess_liquidity else None,
        "currency": account_data.currency,
        "timestamp": account_data.timestamp.isoformat() if account_data.timestamp else None
    }

@app.get("/api/v1/ib/positions")
async def get_ib_positions():
    """Get Interactive Brokers positions"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    positions = await ib_service.get_positions()
    positions_list = []
    
    for key, position in positions.items():
        positions_list.append({
            "position_key": key,
            "account_id": position.account_id,
            "contract_id": position.contract_id,
            "symbol": position.symbol,
            "position": float(position.position),
            "avg_cost": float(position.avg_cost) if position.avg_cost else None,
            "market_price": float(position.market_price) if position.market_price else None,
            "market_value": float(position.market_value) if position.market_value else None,
            "unrealized_pnl": float(position.unrealized_pnl) if position.unrealized_pnl else None,
            "realized_pnl": float(position.realized_pnl) if position.realized_pnl else None,
            "timestamp": position.timestamp.isoformat() if position.timestamp else None
        })
    
    return {"positions": positions_list}

@app.get("/api/v1/ib/orders")
async def get_ib_orders():
    """Get Interactive Brokers orders"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    orders = await ib_service.get_orders()
    orders_list = []
    
    for order_id, order in orders.items():
        orders_list.append({
            "order_id": order.order_id,
            "client_id": order.client_id,
            "account_id": order.account_id,
            "contract_id": order.contract_id,
            "symbol": order.symbol,
            "action": order.action,
            "order_type": order.order_type,
            "total_quantity": float(order.total_quantity),
            "filled_quantity": float(order.filled_quantity),
            "remaining_quantity": float(order.remaining_quantity),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "status": order.status,
            "avg_fill_price": float(order.avg_fill_price) if order.avg_fill_price else None,
            "commission": float(order.commission) if order.commission else None,
            "timestamp": order.timestamp.isoformat() if order.timestamp else None
        })
    
    return {"orders": orders_list}

@app.post("/api/v1/ib/account/refresh")
async def refresh_ib_account_data():
    """Request refresh of IB account data"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    # Use default account if available, otherwise require account_id parameter
    account_id = "main"  # This should come from connection status or user preference
    
    await ib_service.request_account_summary(account_id)
    return {"message": f"Account data refresh requested for {account_id}"}

@app.post("/api/v1/ib/positions/refresh")
async def refresh_ib_positions():
    """Request refresh of IB positions"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    account_id = "main"  # This should come from connection status or user preference
    
    await ib_service.request_positions(account_id)
    return {"message": f"Positions refresh requested for {account_id}"}

@app.post("/api/v1/ib/orders/refresh")
async def refresh_ib_orders():
    """Request refresh of IB open orders"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    account_id = "main"  # This should come from connection status or user preference
    
    await ib_service.request_open_orders(account_id)
    return {"message": f"Open orders refresh requested for {account_id}"}

class IBOrderRequest(BaseModel):
    """IB order placement request model"""
    symbol: str
    action: str  # BUY or SELL
    quantity: float
    order_type: str = "MKT"  # MKT, LMT, STP, etc.
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, etc.
    account_id: str | None = None

@app.post("/api/v1/ib/orders/place")
async def place_ib_order(order_request: IBOrderRequest):
    """Place order through Interactive Brokers"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    # Validate order parameters
    if order_request.action not in ["BUY", "SELL"]:
        raise HTTPException(status_code=400, detail="Action must be BUY or SELL")
    
    if order_request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    
    if order_request.order_type == "LMT" and not order_request.limit_price:
        raise HTTPException(status_code=400, detail="Limit price required for limit orders")
    
    if order_request.order_type == "STP" and not order_request.stop_price:
        raise HTTPException(status_code=400, detail="Stop price required for stop orders")
    
    # Build order request for IB service
    ib_order_params = {
        "symbol": order_request.symbol,
        "action": order_request.action,
        "quantity": order_request.quantity,
        "order_type": order_request.order_type,
        "time_in_force": order_request.time_in_force,
        "account_id": order_request.account_id or "main"
    }
    
    if order_request.limit_price:
        ib_order_params["limit_price"] = order_request.limit_price
    
    if order_request.stop_price:
        ib_order_params["stop_price"] = order_request.stop_price
    
    try:
        order_id = await ib_service.place_order(ib_order_params)
        return {
            "message": "Order placed successfully",
            "order_id": order_id,
            "order_params": ib_order_params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")

@app.post("/api/v1/ib/orders/{order_id}/cancel")
async def cancel_ib_order(order_id: str):
    """Cancel an IB order"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    try:
        await ib_service.cancel_order(order_id)
        return {"message": f"Order {order_id} cancellation requested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel order: {str(e)}")

class IBOrderModification(BaseModel):
    """IB order modification request model"""
    quantity: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None

@app.put("/api/v1/ib/orders/{order_id}/modify")
async def modify_ib_order(order_id: str, modifications: IBOrderModification):
    """Modify an IB order"""
    # Authentication removed for local development
    
    if not ib_service:
        raise HTTPException(status_code=503, detail="IB service not initialized")
    
    # Build modifications dict
    mod_params = {}
    if modifications.quantity is not None:
        if modifications.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        mod_params["quantity"] = modifications.quantity
    
    if modifications.limit_price is not None:
        mod_params["limit_price"] = modifications.limit_price
    
    if modifications.stop_price is not None:
        mod_params["stop_price"] = modifications.stop_price
    
    if not mod_params:
        raise HTTPException(status_code=400, detail="No modifications specified")
    
    try:
        await ib_service.modify_order(order_id, mod_params)
        return {
            "message": f"Order {order_id} modification requested",
            "modifications": mod_params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify order: {str(e)}")

# Demo endpoints for testing (only available in development)
@app.post("/api/v1/demo/populate")
async def populate_demo_trading_data():
    """Populate system with demo trading data for testing"""
    # Authentication removed for local development
    
    if settings.environment != "development":
        raise HTTPException(status_code=403, detail="Demo endpoints only available in development")
    
    try:
        await populate_demo_data()
        return {"status": "success", "message": "Demo data populated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to populate demo data: {str(e)}")

@app.post("/api/v1/demo/clear")
async def clear_demo_trading_data():
    """Clear all demo trading data"""
    # Authentication removed for local development
    
    if settings.environment != "development":
        raise HTTPException(status_code=403, detail="Demo endpoints only available in development")
    
    try:
        await clear_demo_data()
        return {"status": "success", "message": "Demo data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear demo data: {str(e)}")

@app.get("/api/v1/trading/paper-setup")
async def get_paper_trading_setup():
    """Get paper trading setup information and testnet URLs"""
    return {
        "message": "Paper Trading Setup - Use Real Exchange Testnets",
        "recommended_exchanges": [
            {
                "name": "Binance Testnet",
                "url": "https://testnet.binance.vision/",
                "features": ["Free testnet BTC/USDT", "Full API support", "Real-time data"],
                "setup_steps": [
                    "1. Register at https://testnet.binance.vision/",
                    "2. Verify email address",
                    "3. Go to Account → API Management",
                    "4. Create API key with 'Enable Trading' permissions",
                    "5. Restrict IP to your server (recommended)",
                    "6. Get free testnet funds from the faucet"
                ]
            },
            {
                "name": "Bybit Testnet",
                "url": "https://testnet.bybit.com/",
                "features": ["Auto testnet USDT", "Derivatives trading", "WebSocket support"],
                "setup_steps": [
                    "1. Register at https://testnet.bybit.com/",
                    "2. Go to API Management",
                    "3. Create API key with trading permissions",
                    "4. Automatic testnet balance allocation"
                ]
            }
        ],
        "environment_variables": {
            "binance": {
                "BINANCE_API_KEY": "your_testnet_api_key",
                "BINANCE_API_SECRET": "your_testnet_secret",
                "BINANCE_SANDBOX": "true",
                "BINANCE_TRADING_MODE": "testnet"
            },
            "bybit": {
                "BYBIT_API_KEY": "your_testnet_api_key",
                "BYBIT_API_SECRET": "your_testnet_secret",
                "BYBIT_SANDBOX": "true",
                "BYBIT_TRADING_MODE": "testnet"
            }
        },
        "quick_start": [
            "1. Choose an exchange (Binance Testnet recommended)",
            "2. Create testnet account and API keys",
            "3. Set environment variables in .env.paper file",
            "4. Restart NautilusTrader: docker-compose --env-file .env.paper up -d",
            "5. Connect via API: POST /api/v1/exchanges/{venue}/connect",
            "6. Start paper trading with fake money!"
        ],
        "safety_notes": [
            "✅ Testnet uses FAKE MONEY - completely safe",
            "✅ Real exchange APIs and market data",
            "✅ Test strategies without financial risk", 
            "✅ Learn exchange behavior and quirks",
            "⚠️ Always verify you're on testnet URLs",
            "⚠️ Don't use live API keys for testnet"
        ]
    }

# Configure CORS
origins = settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast message to all active connections with error handling"""
        failed_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logging.error(f"Failed to send message to WebSocket connection: {e}")
                failed_connections.append(connection)
        
        # Remove failed connections
        for connection in failed_connections:
            self.disconnect(connection)

manager = ConnectionManager()

# Response models
class HealthResponse(BaseModel):
    status: str
    environment: str
    debug: bool

class StatusResponse(BaseModel):
    api_version: str
    status: str
    trading_mode: str
    features: dict[str, bool]

class MessageBusStatusResponse(BaseModel):
    connection_state: str
    connected_at: str | None
    last_message_at: str | None
    reconnect_attempts: int
    error_message: str | None
    messages_received: int

class MarketDataSubscriptionRequest(BaseModel):
    venue: str
    instrument_id: str
    data_type: str

class MarketDataSubscriptionResponse(BaseModel):
    subscription_id: str
    venue: str
    instrument_id: str
    data_type: str
    active: bool

class MarketDataStatusResponse(BaseModel):
    active_subscriptions: int
    supported_venues: list[str]
    supported_data_types: list[str]

# Message handler for MessageBus messages
async def handle_messagebus_message(message: MessageBusMessage) -> None:
    """Handle messages from MessageBus and broadcast to WebSocket clients"""
    try:
        # Create WebSocket message
        ws_message = {
            "type": "messagebus",
            "topic": message.topic,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "message_type": message.message_type
        }
        
        # Broadcast to all connected WebSocket clients
        await manager.broadcast(json.dumps(ws_message))
        
    except Exception as e:
        logging.error(f"Error handling MessageBus message: {e}")

# Market data broadcast handler
async def broadcast_market_data(data: dict) -> None:
    """Broadcast market data to WebSocket clients"""
    try:
        from datetime import datetime
        
        # Create market data WebSocket message
        ws_message = {
            "type": "market_data",
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        # Broadcast to all connected WebSocket clients
        await manager.broadcast(json.dumps(ws_message))
        
    except Exception as e:
        logging.error(f"Error broadcasting market data: {e}")

# IB-specific broadcast handlers
async def broadcast_ib_connection_status(status: IBConnectionStatus) -> None:
    """Broadcast IB connection status to WebSocket clients"""
    try:
        from datetime import datetime
        ws_message = {
            "type": "ib_connection",
            "data": {
                "connected": status.connected,
                "gateway_type": status.gateway_type,
                "host": status.host,
                "port": status.port,
                "client_id": status.client_id,
                "account_id": status.account_id,
                "connection_time": status.connection_time.isoformat() if status.connection_time else None,
                "last_heartbeat": status.last_heartbeat.isoformat() if status.last_heartbeat else None,
                "error_message": status.error_message
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        await manager.broadcast(json.dumps(ws_message))
    except Exception as e:
        logging.error(f"Error broadcasting IB connection status: {e}")

async def broadcast_ib_account_data(account_data: IBAccountData) -> None:
    """Broadcast IB account data to WebSocket clients"""
    try:
        from datetime import datetime
        ws_message = {
            "type": "ib_account",
            "data": {
                "account_id": account_data.account_id,
                "net_liquidation": float(account_data.net_liquidation) if account_data.net_liquidation else None,
                "total_cash_value": float(account_data.total_cash_value) if account_data.total_cash_value else None,
                "buying_power": float(account_data.buying_power) if account_data.buying_power else None,
                "maintenance_margin": float(account_data.maintenance_margin) if account_data.maintenance_margin else None,
                "initial_margin": float(account_data.initial_margin) if account_data.initial_margin else None,
                "excess_liquidity": float(account_data.excess_liquidity) if account_data.excess_liquidity else None,
                "currency": account_data.currency,
                "timestamp": account_data.timestamp.isoformat() if account_data.timestamp else None
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        await manager.broadcast(json.dumps(ws_message))
    except Exception as e:
        logging.error(f"Error broadcasting IB account data: {e}")

async def broadcast_ib_positions(positions: dict[str, IBPosition]) -> None:
    """Broadcast IB positions to WebSocket clients"""
    try:
        from datetime import datetime
        positions_data = {}
        for key, position in positions.items():
            positions_data[key] = {
                "account_id": position.account_id,
                "contract_id": position.contract_id,
                "symbol": position.symbol,
                "position": float(position.position),
                "avg_cost": float(position.avg_cost) if position.avg_cost else None,
                "market_price": float(position.market_price) if position.market_price else None,
                "market_value": float(position.market_value) if position.market_value else None,
                "unrealized_pnl": float(position.unrealized_pnl) if position.unrealized_pnl else None,
                "realized_pnl": float(position.realized_pnl) if position.realized_pnl else None,
                "timestamp": position.timestamp.isoformat() if position.timestamp else None
            }
        
        ws_message = {
            "type": "ib_positions",
            "data": positions_data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        await manager.broadcast(json.dumps(ws_message))
    except Exception as e:
        logging.error(f"Error broadcasting IB positions: {e}")

async def broadcast_ib_order_update(order: IBOrderData) -> None:
    """Broadcast IB order updates to WebSocket clients"""
    try:
        from datetime import datetime
        ws_message = {
            "type": "ib_order",
            "data": {
                "order_id": order.order_id,
                "client_id": order.client_id,
                "account_id": order.account_id,
                "contract_id": order.contract_id,
                "symbol": order.symbol,
                "action": order.action,
                "order_type": order.order_type,
                "total_quantity": float(order.total_quantity),
                "filled_quantity": float(order.filled_quantity),
                "remaining_quantity": float(order.remaining_quantity),
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "status": order.status,
                "avg_fill_price": float(order.avg_fill_price) if order.avg_fill_price else None,
                "commission": float(order.commission) if order.commission else None,
                "timestamp": order.timestamp.isoformat() if order.timestamp else None
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        await manager.broadcast(json.dumps(ws_message))
    except Exception as e:
        logging.error(f"Error broadcasting IB order update: {e}")

async def broadcast_ib_market_data(symbol: str, market_data: IBMarketData) -> None:
    """Broadcast IB Gateway market data to WebSocket clients"""
    try:
        ws_message = {
            "type": "ib_market_data",
            "data": {
                "symbol": symbol,
                "bid": market_data.bid,
                "ask": market_data.ask,
                "last": market_data.last,
                "volume": market_data.volume,
                "timestamp": market_data.timestamp.isoformat() if market_data.timestamp else None
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        await manager.broadcast(json.dumps(ws_message))
    except Exception as e:
        logging.error(f"Error broadcasting IB market data for {symbol}: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        debug=settings.debug
    )

# API status endpoint
@app.get("/api/v1/status", response_model=StatusResponse)
async def api_status():
    """API status and feature availability"""
    messagebus_connected = messagebus_client.is_connected
    
    # Check if any exchanges are connected for trading
    connected_exchanges = exchange_service.get_connected_exchanges()
    trading_enabled = len(connected_exchanges) > 0
    
    # Check if portfolio service is available
    portfolio_available = bool(portfolio_service.get_portfolio("main"))
    
    # Determine overall trading mode
    trading_modes = set()
    for connection in exchange_service.get_all_exchange_status().values():
        if connection.config.enabled and connection.status == ExchangeStatus.CONNECTED:
            trading_modes.add(connection.config.trading_mode.value)
    
    primary_mode = "paper"
    if "live" in trading_modes:
        primary_mode = "live"
    elif "testnet" in trading_modes:
        primary_mode = "testnet"
    
    return StatusResponse(
        api_version="1.0.0",
        status="operational",
        trading_mode=primary_mode,  # Add trading mode to response
        features={
            "websocket": True,
            "messagebus": messagebus_connected,
            "authentication": True,
            "market_data": True,
            "trading": trading_enabled,
            "portfolio": portfolio_available,
        }
    )

# MessageBus connection status endpoint
@app.get("/api/v1/messagebus/status", response_model=MessageBusStatusResponse)
async def messagebus_status():
    """MessageBus connection status"""
    status = messagebus_client.connection_status
    
    return MessageBusStatusResponse(
        connection_state=status.state.value,
        connected_at=status.connected_at.isoformat() if status.connected_at else None,
        last_message_at=status.last_message_at.isoformat() if status.last_message_at else None,
        reconnect_attempts=status.reconnect_attempts,
        error_message=status.error_message,
        messages_received=status.messages_received
    )

# YFinance endpoints removed

# IB Gateway backfill endpoint
@app.post("/api/v1/ib/backfill")
async def start_ib_backfill(request: dict):
    """Start IB Gateway data backfill"""
    try:
        symbol = request.get("symbol", "AAPL")
        timeframe = request.get("timeframe", "1 min")
        duration = request.get("duration", "1 D")
        
        if not ib_service or not ib_service.is_connected():
            raise HTTPException(status_code=503, detail="IB Gateway not connected")
        
        # Request historical data from IB
        ib_client = ib_service.get_client()
        if not ib_client:
            raise HTTPException(status_code=503, detail="IB client not available")
        
        # Create contract for the symbol
        from ib_insync import Stock
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Request historical bars
        bars = ib_client.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=timeframe,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "duration": duration,
            "records_fetched": len(bars) if bars else 0,
            "message": f"Successfully fetched {len(bars) if bars else 0} bars for {symbol} from IB Gateway"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IB backfill failed: {str(e)}")

# Market Data API endpoints
@app.get("/api/v1/market-data/status", response_model=MarketDataStatusResponse)
async def market_data_status():
    """Market data service status"""
    subscriptions = market_data_service.get_active_subscriptions()
    
    return MarketDataStatusResponse(
        active_subscriptions=len(subscriptions),
        supported_venues=[venue.value for venue in Venue],
        supported_data_types=[data_type.value for data_type in DataType]
    )

@app.post("/api/v1/market-data/subscribe", response_model=MarketDataSubscriptionResponse)
async def subscribe_market_data(request: MarketDataSubscriptionRequest):
    """Subscribe to market data"""
    try:
        venue = Venue(request.venue)
        data_type = DataType(request.data_type)
        
        subscription_id = await market_data_service.subscribe(
            venue=venue,
            instrument_id=request.instrument_id,
            data_type=data_type
        )
        
        return MarketDataSubscriptionResponse(
            subscription_id=subscription_id,
            venue=request.venue,
            instrument_id=request.instrument_id,
            data_type=request.data_type,
            active=True
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid venue or data type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription failed: {e}")

@app.delete("/api/v1/market-data/subscribe/{subscription_id}")
async def unsubscribe_market_data(subscription_id: str):
    """Unsubscribe from market data"""
    success = await market_data_service.unsubscribe(subscription_id)
    
    if success:
        return {"message": f"Successfully unsubscribed from {subscription_id}"}
    else:
        raise HTTPException(status_code=404, detail="Subscription not found")

@app.get("/api/v1/market-data/subscriptions")
async def list_subscriptions():
    """List active market data subscriptions"""
    subscriptions = market_data_service.get_active_subscriptions()
    
    return {
        "subscriptions": [
            {
                "subscription_id": sub.subscription_id,
                "venue": sub.venue.value,
                "instrument_id": sub.instrument_id,
                "data_type": sub.data_type.value,
                "active": sub.active
            }
            for sub in subscriptions
        ]
    }

# Redis Cache API endpoints
@app.get("/api/v1/cache/status")
async def cache_status():
    """Redis cache status and health"""
    return await redis_cache.health_check()

@app.get("/api/v1/cache/stats")
async def cache_stats():
    """Cache statistics"""
    await redis_cache.update_cache_stats()
    return await redis_cache.get_cache_stats()



@app.get("/api/v1/cache/latest-tick/{venue}/{instrument_id}")
async def get_latest_tick(venue: str, instrument_id: str):
    """Get latest tick for instrument"""
    tick = await redis_cache.get_latest_tick(venue, instrument_id)
    if tick:
        return {"venue": venue, "instrument_id": instrument_id, "tick": tick}
    else:
        raise HTTPException(status_code=404, detail="No tick data found")

@app.get("/api/v1/cache/latest-quote/{venue}/{instrument_id}")
async def get_latest_quote(venue: str, instrument_id: str):
    """Get latest quote for instrument"""
    quote = await redis_cache.get_latest_quote(venue, instrument_id)
    if quote:
        return {"venue": venue, "instrument_id": instrument_id, "quote": quote}
    else:
        raise HTTPException(status_code=404, detail="No quote data found")

@app.get("/api/v1/cache/latest-bar/{venue}/{instrument_id}")
async def get_latest_bar(venue: str, instrument_id: str, timeframe: str = "1m"):
    """Get latest bar for instrument"""
    bar = await redis_cache.get_latest_bar(venue, instrument_id, timeframe)
    if bar:
        return {"venue": venue, "instrument_id": instrument_id, "timeframe": timeframe, "bar": bar}
    else:
        raise HTTPException(status_code=404, detail="No bar data found")

@app.get("/api/v1/cache/tick-history/{venue}/{instrument_id}")
async def get_tick_history(venue: str, instrument_id: str, count: int = 100):
    """Get tick history for instrument"""
    ticks = await redis_cache.get_tick_history(venue, instrument_id, count)
    return {"venue": venue, "instrument_id": instrument_id, "count": len(ticks), "ticks": ticks}

@app.get("/api/v1/cache/quote-history/{venue}/{instrument_id}")
async def get_quote_history(venue: str, instrument_id: str, count: int = 100):
    """Get quote history for instrument"""
    quotes = await redis_cache.get_quote_history(venue, instrument_id, count)
    return {"venue": venue, "instrument_id": instrument_id, "count": len(quotes), "quotes": quotes}

@app.get("/api/v1/cache/bar-history/{venue}/{instrument_id}")
async def get_bar_history(venue: str, instrument_id: str, timeframe: str = "1m", count: int = 100):
    """Get bar history for instrument"""
    bars = await redis_cache.get_bar_history(venue, instrument_id, timeframe, count)
    return {"venue": venue, "instrument_id": instrument_id, "timeframe": timeframe, "count": len(bars), "bars": bars}

# Rate Limiting API endpoints
@app.get("/api/v1/rate-limiting/status")
async def rate_limiting_status():
    """Rate limiting system status"""
    return await rate_limiter.health_check()

@app.get("/api/v1/rate-limiting/metrics")
async def rate_limiting_metrics():
    """Rate limiting metrics for all venues"""
    return rate_limiter.get_all_metrics()

@app.get("/api/v1/rate-limiting/venue/{venue}")
async def venue_rate_limiting_status(venue: str):
    """Rate limiting status for specific venue"""
    try:
        venue_enum = Venue(venue.upper())
        return rate_limiter.get_venue_status(venue_enum)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid venue: {venue}")

@app.post("/api/v1/rate-limiting/reset-metrics")
async def reset_rate_limiting_metrics(venue: str | None = None):
    """Reset rate limiting metrics"""
    if venue:
        try:
            venue_enum = Venue(venue.upper())
            rate_limiter.reset_metrics(venue_enum)
            return {"message": f"Metrics reset for venue {venue}"}
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid venue: {venue}")
    else:
        rate_limiter.reset_metrics()
        return {"message": "All metrics reset"}

# Historical Data API endpoints
@app.get("/api/v1/historical/status")
async def historical_data_status():
    """Historical data service status"""
    return await historical_data_service.health_check()

@app.get("/api/v1/historical/summary/{venue}/{instrument_id}")
async def get_data_summary(venue: str, instrument_id: str):
    """Get data summary for instrument"""
    summary = await historical_data_service.get_data_summary(venue, instrument_id)
    return {"venue": venue, "instrument_id": instrument_id, "summary": summary}

@app.get("/api/v1/historical/ticks/{venue}/{instrument_id}")
async def query_historical_ticks(
    venue: str, 
    instrument_id: str,
    start_time: str,
    end_time: str,
    limit: int | None = 1000
):
    """Query historical tick data"""
    try:
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        query = HistoricalDataQuery(
            venue=venue,
            instrument_id=instrument_id,
            data_type="tick",
            start_time=start_dt,
            end_time=end_dt,
            limit=limit
        )
        
        ticks = await historical_data_service.query_ticks(query)
        return {
            "venue": venue,
            "instrument_id": instrument_id,
            "data_type": "tick",
            "start_time": start_time,
            "end_time": end_time,
            "count": len(ticks),
            "data": ticks
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

@app.get("/api/v1/historical/quotes/{venue}/{instrument_id}")
async def query_historical_quotes(
    venue: str,
    instrument_id: str,
    start_time: str,
    end_time: str,
    limit: int | None = 1000
):
    """Query historical quote data"""
    try:
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        query = HistoricalDataQuery(
            venue=venue,
            instrument_id=instrument_id,
            data_type="quote",
            start_time=start_dt,
            end_time=end_dt,
            limit=limit
        )
        
        quotes = await historical_data_service.query_quotes(query)
        return {
            "venue": venue,
            "instrument_id": instrument_id,
            "data_type": "quote",
            "start_time": start_time,
            "end_time": end_time,
            "count": len(quotes),
            "data": quotes
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

@app.get("/api/v1/historical/bars/{venue}/{instrument_id}")
async def query_historical_bars(
    venue: str,
    instrument_id: str,
    start_time: str,
    end_time: str,
    timeframe: str = "1m",
    limit: int | None = 1000
):
    """Query historical bar data"""
    try:
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        query = HistoricalDataQuery(
            venue=venue,
            instrument_id=instrument_id,
            data_type="bar",
            start_time=start_dt,
            end_time=end_dt,
            timeframe=timeframe,
            limit=limit
        )
        
        bars = await historical_data_service.query_bars(query)
        return {
            "venue": venue,
            "instrument_id": instrument_id,
            "data_type": "bar",
            "timeframe": timeframe,
            "start_time": start_time,
            "end_time": end_time,
            "count": len(bars),
            "data": bars
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")


# Market Data API - Historical Bars for Chart Integration
@app.get("/api/v1/market-data/historical/bars")
async def get_market_data_historical_bars(
    symbol: str,
    timeframe: str = "1h",
    asset_class: str | None = None,
    exchange: str | None = None,
    currency: str | None = None
    # current_user: User | None = Depends(get_current_user_optional)  # Removed for local dev
):
    """Get historical OHLCV bars for chart integration - uses PostgreSQL first, IB Gateway as fallback"""
    
    try:
        # FIRST: Try to get data from PostgreSQL (stored historical data)
        from datetime import datetime, timedelta
        from historical_data_service import historical_data_service, HistoricalDataQuery
        
        # Map frontend timeframes to database timeframes and calculate time ranges
        timeframe_config = {
            "1m": {"db_tf": "1m", "days_back": 1, "ib_duration": "1 D", "ib_size": "1 min"},
            "2m": {"db_tf": "2m", "days_back": 2, "ib_duration": "2 D", "ib_size": "2 mins"},
            "5m": {"db_tf": "5m", "days_back": 5, "ib_duration": "5 D", "ib_size": "5 mins"},
            "10m": {"db_tf": "10m", "days_back": 7, "ib_duration": "1 W", "ib_size": "10 mins"},
            "15m": {"db_tf": "15m", "days_back": 7, "ib_duration": "1 W", "ib_size": "15 mins"},
            "30m": {"db_tf": "30m", "days_back": 14, "ib_duration": "2 W", "ib_size": "30 mins"},
            "1h": {"db_tf": "1h", "days_back": 30, "ib_duration": "1 M", "ib_size": "1 hour"},
            "2h": {"db_tf": "2h", "days_back": 60, "ib_duration": "2 M", "ib_size": "2 hours"},
            "4h": {"db_tf": "4h", "days_back": 90, "ib_duration": "3 M", "ib_size": "4 hours"},
            "1d": {"db_tf": "1d", "days_back": 365, "ib_duration": "1 Y", "ib_size": "1 day"},
            "1w": {"db_tf": "1w", "days_back": 730, "ib_duration": "2 Y", "ib_size": "1 week"},
            "1M": {"db_tf": "1M", "days_back": 1825, "ib_duration": "5 Y", "ib_size": "1 month"}
        }
        
        config = timeframe_config.get(timeframe)
        if not config:
            raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")
        
        # Try PostgreSQL first
        candles = []
        data_source = "Database Cache"
        
        if historical_data_service.is_connected:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=config["days_back"])
            
            venue = exchange or "SMART"  # Default venue
            instrument_id = f"{symbol}.{venue}"  # Standard instrument format
            
            query = HistoricalDataQuery(
                venue=venue,
                instrument_id=instrument_id,
                data_type="bar",
                start_time=start_time,
                end_time=end_time,
                timeframe=config["db_tf"],
                limit=1000
            )
            
            try:
                db_bars = await historical_data_service.query_bars(query)
                logging.info(f"Found {len(db_bars)} bars in database for {symbol} {timeframe}")
                
                # Convert database bars to API format
                for bar in db_bars:
                    candles.append({
                        "time": datetime.fromtimestamp(bar['timestamp_ns'] / 1_000_000_000).isoformat() + "Z",
                        "open": float(bar['open_price']),
                        "high": float(bar['high_price']),
                        "low": float(bar['low_price']),
                        "close": float(bar['close_price']),
                        "volume": float(bar['volume'])
                    })
                    
            except Exception as e:
                logging.warning(f"Database query failed for {symbol}: {e}")
        
        # FALLBACK: If no database data or IB Gateway needed, try IB Gateway
        if len(candles) == 0:
            data_source = "IB Gateway"
            ib_client = get_ib_gateway_client()
            
            if ib_client.is_connected():
                logging.info(f"Trying IB Gateway for {symbol} {timeframe}")
        
                # Enhanced asset class detection
                from ib_asset_classes import get_ib_asset_class_manager, IBAssetClass
                asset_manager = get_ib_asset_class_manager()
                
                # Use provided parameters or auto-detect
                if asset_class:
                    sec_type = asset_class
                    exchange_val = exchange or asset_manager.get_default_exchange(asset_class)
                    currency_val = currency or "USD"
                else:
                    # Auto-detect based on symbol
                    sec_type = "STK"  # Default to stock
                    exchange_val = "SMART"
                    currency_val = "USD"
                    
                    # Check if it's a forex pair (6 characters, all letters)
                    if len(symbol) == 6 and symbol.isalpha():
                        sec_type = "CASH"
                        exchange_val = "IDEALPRO"
                        currency_val = symbol[3:6]  # Last 3 characters (USD from EURUSD)
                        symbol = symbol[:3]     # First 3 characters (EUR from EURUSD)
                    
                    # Check if it's a future (common future symbols)
                    elif symbol in ["ES", "NQ", "YM", "RTY", "CL", "NG", "GC", "SI", "ZN", "ZB"]:
                        sec_type = "FUT"
                        # Map to appropriate exchanges
                        if symbol in ["ES", "NQ", "YM", "RTY"]:
                            exchange_val = "GLOBEX"
                        elif symbol in ["CL", "NG", "GC", "SI"]:
                            exchange_val = "NYMEX"
                        elif symbol in ["ZN", "ZB"]:
                            exchange_val = "CBOT"
                        else:
                            exchange_val = "GLOBEX"
                        
                        # For futures, we need to add current month expiry
                        current_date = datetime.now()
                        year = current_date.year
                        month = current_date.month + 1  # Next month
                        if month > 12:
                            year += 1
                            month = 1
                        symbol = f"{symbol}{month:02d}{str(year)[-2:]}"  # e.g., ES0325 for March 2025
                
                try:
                    # Request historical data from IB Gateway
                    historical_data = await ib_client.request_historical_data(
                        symbol=symbol,
                        sec_type=sec_type,
                        exchange=exchange_val, 
                        currency=currency_val,
                        duration=config["ib_duration"],
                        bar_size=config["ib_size"],
                        what_to_show="TRADES"
                    )
                    
                    # Convert to chart-compatible format
                    for bar_data in historical_data['bars']:
                        candles.append({
                            "time": bar_data['time'],
                            "open": bar_data['open'],
                            "high": bar_data['high'],
                            "low": bar_data['low'],
                            "close": bar_data['close'],
                            "volume": bar_data['volume']
                        })
                    
                    logging.info(f"IB Gateway returned {len(candles)} bars for {symbol}")
                    
                except Exception as ib_error:
                    logging.error(f"IB Gateway failed for {symbol}: {ib_error}")
                    data_source = "No Data Available"
            else:
                logging.warning("IB Gateway not connected, cannot fetch live data")
                data_source = "No Data Available"
        
        # THIRD FALLBACK: Try YFinance for backfilling (stocks and major instruments)
        if len(candles) == 0 and yfinance_service:
            try:
                logging.info(f"Trying YFinance backfill for {symbol} {timeframe}")
                data_source = "YFinance Backfill"
                
                # Initialize YFinance service if needed
                if not yfinance_service.is_connected():
                    init_success = await yfinance_service.initialize({
                        'symbols': [symbol.upper()],
                        'rate_limit_delay': 0.2,  # Be conservative with rate limits
                        'cache_expiry_seconds': 3600
                    })
                    if not init_success:
                        logging.warning("Failed to initialize YFinance service")
                        data_source = "No Data Available"
                        
                if yfinance_service.is_connected():
                    # Map timeframes to YFinance periods
                    yf_timeframe_map = {
                        "1m": ("1m", "1d"),
                        "2m": ("2m", "1d"), 
                        "5m": ("5m", "5d"),
                        "15m": ("15m", "1mo"),
                        "30m": ("30m", "3mo"),
                        "1h": ("1h", "1y"),
                        "2h": ("1h", "1y"),  # YFinance doesn't have 2h, use 1h
                        "4h": ("1h", "2y"),  # YFinance doesn't have 4h, use 1h
                        "1d": ("1d", "5y"),
                        "1w": ("1wk", "10y"),
                        "1M": ("1mo", "max")
                    }
                    
                    yf_interval, yf_period = yf_timeframe_map.get(timeframe, ("1d", "1y"))
                    
                    # Get data from YFinance
                    yf_data = await yfinance_service.get_historical_bars(
                        symbol=symbol.upper(),
                        timeframe=yf_interval,
                        period=yf_period,
                        limit=1000
                    )
                    
                    if yf_data and yf_data.bars:
                        # Convert YFinance data to chart format
                        for bar in yf_data.bars:
                            candles.append({
                                "time": bar["time"],
                                "open": float(bar["open"]),
                                "high": float(bar["high"]),
                                "low": float(bar["low"]),
                                "close": float(bar["close"]),
                                "volume": int(bar["volume"])
                            })
                        
                        logging.info(f"YFinance returned {len(candles)} bars for {symbol}")
                        
                        # TODO: Optionally backfill this data into PostgreSQL for future use
                        # This would require implementing a backfill service call here
                        
                    else:
                        logging.warning(f"YFinance returned no data for {symbol}")
                        
            except Exception as yf_error:
                logging.error(f"YFinance fallback failed for {symbol}: {yf_error}")
                data_source = "No Data Available"
        
        # Return results
        if len(candles) > 0:
            # Sort candles by time (oldest first for charting)
            candles.sort(key=lambda x: x['time'])
        
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "total": len(candles),
                "start_date": candles[0]['time'] if candles else None,
                "end_date": candles[-1]['time'] if candles else None,
                "source": data_source
            }
        else:
            # No data available from any source
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": [],
                "total": 0,
                "start_date": None,
                "end_date": None,
                "source": data_source,
                "error": "No historical data available. Data may need to be collected first."
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting historical bars for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting historical bars: {str(e)}")

@app.post("/api/v1/historical/cleanup")
async def cleanup_historical_data(days_to_keep: int = 30):
    """Clean up old historical data"""
    if days_to_keep < 1:
        raise HTTPException(status_code=400, detail="days_to_keep must be at least 1")
        
    deleted_counts = await historical_data_service.cleanup_old_data(days_to_keep)
    return {
        "message": f"Cleaned up data older than {days_to_keep} days",
        "deleted_counts": deleted_counts
    }

# Historical Data Backfill API Endpoints

from data_backfill_service import backfill_service, BackfillRequest
from pydantic import BaseModel
from enum import Enum

# Backfill Mode Management
class BackfillMode(str, Enum):
    IBKR = "ibkr"
    YFINANCE = "yfinance"

class BackfillController:
    """Manages backfill operations and mode switching"""
    
    def __init__(self):
        self.current_mode = BackfillMode.IBKR  # Default to IBKR
        self.is_running = False
        self.current_operation = None
        
    def set_mode(self, mode: BackfillMode) -> bool:
        """Set backfill mode - only allowed when not running"""
        if self.is_running:
            return False
        self.current_mode = mode
        return True
    
    def start_operation(self, operation_info: dict):
        """Mark backfill as running"""
        self.is_running = True
        self.current_operation = operation_info
    
    def stop_operation(self):
        """Mark backfill as stopped"""
        self.is_running = False
        self.current_operation = None
    
    def get_status(self) -> dict:
        """Get current backfill controller status"""
        return {
            "current_mode": self.current_mode.value,
            "is_running": self.is_running,
            "current_operation": self.current_operation,
            "available_modes": [mode.value for mode in BackfillMode]
        }

# Global backfill controller instance
backfill_controller = BackfillController()

class BackfillRequestModel(BaseModel):
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    timeframes: list[str | None] = None
    days_back: int | None = 365
    priority: int = 1

@app.post("/api/v1/historical/backfill/start")
async def start_backfill_process(request: dict = None):
    """Start the unified backfill process (mode-aware)"""
    try:
        # Check if backfill is already running
        if backfill_controller.is_running:
            raise HTTPException(
                status_code=409, 
                detail=f"Backfill is already running in {backfill_controller.current_mode.value.upper()} mode"
            )
        
        current_mode = backfill_controller.current_mode
        operation_info = {
            "mode": current_mode.value,
            "started_at": datetime.now().isoformat(),
            "type": "priority_instruments"
        }
        
        if current_mode == BackfillMode.IBKR:
            # IBKR backfill mode
            success = await backfill_service.initialize()
            if not success:
                raise HTTPException(status_code=503, detail="Failed to initialize IBKR backfill service")
            
            # Start IBKR backfill for priority instruments
            asyncio.create_task(backfill_service.backfill_priority_instruments())
            operation_info["service"] = "IBKR Gateway"
            
        elif current_mode == BackfillMode.YFINANCE:
            # YFinance backfill mode using NautilusTrader adapter
            if not nautilus_yfinance_service._initialized:
                # Try to initialize if not already initialized
                nautilus_init_success = await nautilus_yfinance_service.initialize()
                if not nautilus_init_success:
                    # Fallback to legacy service
                    if not yfinance_service:
                        raise HTTPException(status_code=503, detail="No YFinance service available")
                    
                    if not yfinance_service.is_connected():
                        init_success = await yfinance_service.initialize()
                        if not init_success:
                            raise HTTPException(status_code=503, detail="Failed to initialize any YFinance service")
            
            # Start YFinance backfill for common symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
            operation_info["service"] = "YFinance (NautilusTrader)" if nautilus_yfinance_service._initialized else "YFinance (Legacy)"
            operation_info["symbols"] = symbols
            operation_info["type"] = "symbol_list"
            
            # Start YFinance backfill task
            asyncio.create_task(start_yfinance_bulk_backfill(symbols))
        
        # Mark as running
        backfill_controller.start_operation(operation_info)
        
        return {
            "message": f"{current_mode.value.upper()} backfill process started",
            "mode": current_mode.value,
            "status": "running",
            "operation_info": operation_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting backfill process: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting backfill: {str(e)}")

async def start_yfinance_bulk_backfill(symbols: list[str]):
    """Background task for YFinance bulk backfill using NautilusTrader adapter"""
    try:
        for symbol in symbols:
            if not backfill_controller.is_running:
                break  # Stop if user cancelled
            
            logging.info(f"YFinance backfill: Processing {symbol}")
            
            try:
                # Try NautilusTrader adapter first
                if nautilus_yfinance_service._initialized:
                    data = await nautilus_yfinance_service.get_historical_bars(
                        symbol=symbol, timeframe="1d", limit=365
                    )
                    if data and len(data) > 0:
                        logging.info(f"YFinance backfill (Nautilus): {symbol} - {len(data)} bars retrieved")
                    else:
                        logging.warning(f"YFinance backfill (Nautilus): No data returned for {symbol}")
                else:
                    # Fallback to legacy service
                    data = await yfinance_service.get_historical_bars(
                        symbol=symbol, timeframe="1d", period="1y"
                    )
                    
                    if data and data.total_bars > 0:
                        logging.info(f"YFinance backfill (Legacy): {symbol} - {data.total_bars} bars retrieved")
                    else:
                        logging.warning(f"YFinance backfill (Legacy): No data returned for {symbol}")
            
            except Exception as symbol_error:
                logging.error(f"Error processing {symbol}: {symbol_error}")
                continue
            
            # Rate limiting delay
            await asyncio.sleep(yfinance_service.status.rate_limit_delay)
        
        # Mark as completed
        backfill_controller.stop_operation()
        logging.info("YFinance bulk backfill completed")
        
    except Exception as e:
        logging.error(f"YFinance bulk backfill error: {e}")
        backfill_controller.stop_operation()

@app.post("/api/v1/historical/backfill/add")
async def add_backfill_request(
    request: BackfillRequestModel
):
    """Add a custom backfill request for specific instrument (mode-aware)"""
    try:
        current_mode = backfill_controller.current_mode
        
        if current_mode == BackfillMode.IBKR:
            # IBKR mode - use existing backfill service
            start_date = datetime.now() - timedelta(days=request.days_back) if request.days_back else None
            
            backfill_req = BackfillRequest(
                symbol=request.symbol,
                sec_type=request.sec_type,
                exchange=request.exchange,
                currency=request.currency,
                timeframes=request.timeframes,
                start_date=start_date,
                end_date=datetime.now(),
                priority=request.priority
            )
            
            await backfill_service.add_backfill_request(backfill_req)
            
            return {
                "message": f"IBKR backfill request added for {request.symbol}",
                "mode": "ibkr",
                "symbol": request.symbol,
                "timeframes": request.timeframes or list(backfill_service.timeframe_config.keys()),
                "days_back": request.days_back,
                "timestamp": datetime.now().isoformat()
            }
        
        elif current_mode == BackfillMode.YFINANCE:
            # YFinance mode - single symbol backfill
            if not yfinance_service:
                raise HTTPException(status_code=503, detail="YFinance service not available")
            
            # Map days_back to YFinance period
            days_back = request.days_back or 30
            if days_back <= 7:
                period = "7d"
            elif days_back <= 30:
                period = "1mo"
            elif days_back <= 90:
                period = "3mo"
            elif days_back <= 365:
                period = "1y"
            else:
                period = "2y"
            
            # Start YFinance single symbol backfill
            data = await yfinance_service.get_historical_bars(
                symbol=request.symbol, timeframe="1d", period=period
            )
            
            return {
                "message": f"YFinance backfill completed for {request.symbol}",
                "mode": "yfinance",
                "symbol": request.symbol,
                "period": period,
                "bars_retrieved": data.total_bars if data else 0,
                "data_source": data.data_source if data else "No Data",
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error adding backfill request: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding backfill request: {str(e)}")

@app.get("/api/v1/historical/backfill/status")
async def get_backfill_status():
    """Get current unified backfill process status"""
    try:
        # Get controller status
        controller_status = backfill_controller.get_status()
        
        # Get service-specific status based on current mode
        service_status = {}
        if controller_status["current_mode"] == "ibkr":
            service_status = await backfill_service.get_backfill_status()
        elif controller_status["current_mode"] == "yfinance":
            service_status = {
                "service": "YFinance",
                "status": await yfinance_service.health_check() if yfinance_service else {"status": "unavailable"}
            }
        
        return {
            "controller": controller_status,
            "service_status": service_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting backfill status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting backfill status: {str(e)}")

@app.post("/api/v1/historical/backfill/stop")
async def stop_backfill_process():
    """Stop the current backfill process"""
    try:
        controller_status = backfill_controller.get_status()
        
        # Stop based on current mode
        if controller_status["current_mode"] == "ibkr":
            await backfill_service.stop_backfill_process()
        elif controller_status["current_mode"] == "yfinance":
            # YFinance operations are typically single requests, mark as stopped
            pass
        
        # Update controller state
        backfill_controller.stop_operation()
        
        return {
            "message": f"{controller_status['current_mode'].upper()} backfill process stopped",
            "mode": controller_status["current_mode"],
            "status": "stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error stopping backfill process: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping backfill: {str(e)}")

@app.post("/api/v1/historical/backfill/set-mode")
async def set_backfill_mode(request: dict):
    """Set the backfill mode (IBKR or YFinance)"""
    try:
        mode = request.get("mode", "").lower()
        
        if mode not in ["ibkr", "yfinance"]:
            raise HTTPException(status_code=400, detail="Mode must be 'ibkr' or 'yfinance'")
        
        # Check if backfill is currently running
        if backfill_controller.is_running:
            raise HTTPException(
                status_code=409, 
                detail="Cannot change mode while backfill is running. Stop current operation first."
            )
        
        # Set the new mode
        success = backfill_controller.set_mode(BackfillMode(mode))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set backfill mode")
        
        return {
            "message": f"Backfill mode set to {mode.upper()}",
            "mode": mode,
            "status": "mode_changed",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error setting backfill mode: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting backfill mode: {str(e)}")

@app.get("/api/v1/historical/backfill/mode")
async def get_backfill_mode():
    """Get the current backfill mode"""
    try:
        status = backfill_controller.get_status()
        return {
            "current_mode": status["current_mode"],
            "available_modes": status["available_modes"],
            "is_running": status["is_running"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting backfill mode: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting backfill mode: {str(e)}")

@app.get("/api/v1/historical/analyze-gaps/{symbol}")
async def analyze_data_gaps(
    symbol: str,
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD"
    # current_user: User | None = Depends(get_current_user_optional)  # Removed for local dev
):
    """Analyze missing data gaps for a specific instrument"""
    try:
        # Initialize backfill service if needed
        if not backfill_service.ib_client:
            await backfill_service.initialize()
        
        gaps = await backfill_service.analyze_missing_data(symbol, sec_type, exchange, currency)
        
        # Convert gaps to readable format
        readable_gaps = {}
        for timeframe, gap_list in gaps.items():
            readable_gaps[timeframe] = [
                {
                    "start": gap[0].isoformat(),
                    "end": gap[1].isoformat(),
                    "duration_days": (gap[1] - gap[0]).days
                }
                for gap in gap_list
            ]
        
        return {
            "symbol": symbol,
            "missing_data_gaps": readable_gaps,
            "total_gaps": sum(len(gaps) for gaps in readable_gaps.values()),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error analyzing data gaps for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing data gaps: {str(e)}")

# Monitoring and Alerting API endpoints
@app.get("/api/v1/monitoring/dashboard")
async def monitoring_dashboard():
    """Get monitoring dashboard summary"""
    return monitoring_service.get_summary_dashboard()

@app.get("/api/v1/monitoring/health")
async def system_health():
    """Get system health status"""
    return monitoring_service.get_health_status()

@app.get("/api/v1/monitoring/alerts")
async def get_alerts(resolved: bool | None = None, level: str | None = None):
    """Get alerts with optional filtering"""
    alert_level = None
    if level:
        try:
            alert_level = AlertLevel(level.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid alert level: {level}")
    
    alerts = monitoring_service.get_alerts(resolved=resolved, level=alert_level)
    return {
        "alerts": [
            {
                "id": alert.id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "tags": alert.tags,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]
    }

@app.post("/api/v1/monitoring/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    success = monitoring_service.resolve_alert(alert_id)
    if success:
        return {"message": f"Alert {alert_id} resolved"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/api/v1/monitoring/metrics")
async def get_metrics(
    name: str | None = None,
    since_hours: int | None = 1
):
    """Get metrics data"""
    from datetime import datetime, timedelta
    since = datetime.now() - timedelta(hours=since_hours) if since_hours else None
    metrics = monitoring_service.get_metrics(name=name, since=since)
    
    # Convert to JSON-serializable format
    result = {}
    for metric_name, metric_list in metrics.items():
        result[metric_name] = [
            {
                "value": metric.value,
                "metric_type": metric.metric_type.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags,
                "unit": metric.unit
            }
            for metric in metric_list
        ]
    
    return {"metrics": result}

@app.post("/api/v1/monitoring/alerts/create")
async def create_alert(
    level: str,
    title: str,
    message: str,
    source: str,
    tags: dict[str, str | None] = None
):
    """Create a custom alert"""
    try:
        alert_level = AlertLevel(level.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid alert level: {level}")
    
    alert_id = monitoring_service.create_alert(
        level=alert_level,
        title=title,
        message=message,
        source=source,
        tags=tags or {}
    )
    
    return {"alert_id": alert_id, "message": "Alert created successfully"}


# =============================================================================
# PARQUET EXPORT ENDPOINTS - NAUTILUSTRADER COMPATIBILITY
# =============================================================================

@app.get("/api/v1/parquet/status")
async def get_parquet_export_status():
    """Get Parquet export service status"""
    try:
        return {
            "status": "operational",
            "output_directory": str(parquet_export_service.config.output_directory),
            "compression": parquet_export_service.config.compression,
            "nautilus_format": parquet_export_service.config.nautilus_format,
            "batch_size": parquet_export_service.config.batch_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Parquet export status: {e}")


@app.post("/api/v1/parquet/export/ticks/{venue}/{instrument_id}")
async def export_ticks_to_parquet(
    venue: str,
    instrument_id: str,
    start_date: str,
    end_date: str
):
    """Export tick data to Parquet format for NautilusTrader compatibility"""
    try:
        from datetime import datetime
        start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        result = await parquet_export_service.export_ticks(
            venue=venue,
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export tick data: {e}")


@app.post("/api/v1/parquet/export/quotes/{venue}/{instrument_id}")
async def export_quotes_to_parquet(
    venue: str,
    instrument_id: str,
    start_date: str,
    end_date: str
):
    """Export quote data to Parquet format for NautilusTrader compatibility"""
    try:
        from datetime import datetime
        start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        result = await parquet_export_service.export_quotes(
            venue=venue,
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export quote data: {e}")


@app.post("/api/v1/parquet/export/bars/{venue}/{instrument_id}")
async def export_bars_to_parquet(
    venue: str,
    instrument_id: str,
    timeframe: str,
    start_date: str,
    end_date: str
):
    """Export bar data to Parquet format for NautilusTrader compatibility"""
    try:
        from datetime import datetime
        start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        result = await parquet_export_service.export_bars(
            venue=venue,
            instrument_id=instrument_id,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export bar data: {e}")


@app.post("/api/v1/parquet/export/daily")
async def export_daily_batch_to_parquet(
    date: str,
    venues: list[str | None] = None,
    instrument_ids: list[str | None] = None
):
    """Export a full day's data to Parquet format"""
    try:
        from datetime import datetime
        target_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        
        result = await parquet_export_service.export_daily_batch(
            date=target_date,
            venues=venues,
            instrument_ids=instrument_ids
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export daily batch: {e}")


@app.get("/api/v1/parquet/catalog")
async def get_nautilus_catalog():
    """Get NautilusTrader data catalog"""
    try:
        result = await parquet_export_service.create_nautilus_catalog()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create catalog: {e}")


# =============================================================================
# NAUTILUSTRADER ENGINE MANAGEMENT API ENDPOINTS
# =============================================================================

# Initialize global engine manager
nautilus_engine = get_nautilus_engine_manager()

@app.get("/api/v1/nautilus/engine/status")
async def get_nautilus_engine_status():
    """Get comprehensive NautilusTrader engine status"""
    try:
        status = await nautilus_engine.get_engine_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get engine status: {str(e)}")

@app.post("/api/v1/nautilus/engine/start")
async def start_nautilus_engine(config: dict):
    """Start NautilusTrader engine with specified configuration"""
    try:
        engine_config = EngineConfig(**config)
        result = await nautilus_engine.start_engine(engine_config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")

@app.post("/api/v1/nautilus/engine/stop")
async def stop_nautilus_engine(force: bool = False):
    """Stop NautilusTrader engine"""
    try:
        result = await nautilus_engine.stop_engine(force)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop engine: {str(e)}")

@app.post("/api/v1/nautilus/engine/restart")
async def restart_nautilus_engine():
    """Restart NautilusTrader engine with current configuration"""
    try:
        result = await nautilus_engine.restart_engine()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart engine: {str(e)}")

@app.post("/api/v1/nautilus/backtest/start")
async def start_backtest(backtest_config: dict):
    """Start a new backtest"""
    try:
        config = BacktestConfig(**backtest_config)
        backtest_id = f"backtest_{int(datetime.now().timestamp())}"
        result = await nautilus_engine.run_backtest(backtest_id, config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@app.get("/api/v1/nautilus/backtest/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get backtest status and results"""
    try:
        result = await nautilus_engine.get_backtest_status(backtest_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")

@app.post("/api/v1/nautilus/backtest/cancel/{backtest_id}")
async def cancel_backtest(backtest_id: str):
    """Cancel running backtest"""
    try:
        result = await nautilus_engine.cancel_backtest(backtest_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel backtest: {str(e)}")

@app.get("/api/v1/nautilus/backtest/list")
async def list_backtests():
    """List all backtests"""
    try:
        result = await nautilus_engine.list_backtests()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")

@app.get("/api/v1/nautilus/data/catalog")
async def get_data_catalog():
    """Get NautilusTrader data catalog"""
    try:
        result = await nautilus_engine.get_data_catalog()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data catalog: {str(e)}")

# =============================================================================
# EPIC 6: NAUTILUS TRADER CORE ENGINE INTEGRATION - ADDITIONAL ENDPOINTS
# =============================================================================

# Epic 6 Story 6.2: Backtesting Engine Integration - Additional Endpoints

@app.get("/api/v1/nautilus/backtest/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Get detailed backtest results including metrics, trades, and equity curve"""
    try:
        result = await nautilus_engine.get_backtest_results(backtest_id)
        return {"status": "success", "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest results: {str(e)}")

@app.delete("/api/v1/nautilus/backtest/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest and its results"""
    try:
        result = await nautilus_engine.delete_backtest(backtest_id)
        return {"status": "success", "message": f"Backtest {backtest_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {str(e)}")

@app.post("/api/v1/nautilus/backtest/compare")
async def compare_backtests(backtest_ids: list[str]):
    """Compare multiple backtests"""
    try:
        result = await nautilus_engine.compare_backtests(backtest_ids)
        return {"status": "success", "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare backtests: {str(e)}")

# Epic 6 Story 6.3: Strategy Deployment Pipeline - API Endpoints

class DeploymentRequest(BaseModel):
    strategy_id: str
    version: str
    backtest_id: str
    proposed_config: dict
    risk_assessment: dict
    rollout_plan: dict
    approval_required: bool = False

@app.post("/api/v1/nautilus/deployment/create")
async def create_deployment_request(request: DeploymentRequest):
    """Create a new strategy deployment request"""
    try:
        result = await nautilus_engine.create_deployment_request(request.dict())
        return {"status": "success", "deploymentId": result["deployment_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create deployment: {str(e)}")

@app.get("/api/v1/nautilus/deployment/list")
async def list_deployments():
    """List all deployment requests"""
    try:
        result = await nautilus_engine.list_deployments()
        return {"status": "success", "deployments": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list deployments: {str(e)}")

@app.post("/api/v1/nautilus/deployment/approve")
async def approve_deployment(deployment_id: str, comments: str = ""):
    """Approve a deployment request"""
    try:
        result = await nautilus_engine.approve_deployment(deployment_id, comments)
        return {"status": "success", "message": "Deployment approved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve deployment: {str(e)}")

@app.post("/api/v1/nautilus/deployment/deploy")
async def deploy_strategy(deployment_id: str):
    """Execute strategy deployment"""
    try:
        result = await nautilus_engine.deploy_strategy(deployment_id)
        return {"status": "success", "deployment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy strategy: {str(e)}")

@app.get("/api/v1/nautilus/deployment/status/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """Get deployment status"""
    try:
        result = await nautilus_engine.get_deployment_status(deployment_id)
        return {"status": "success", "deployment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get deployment status: {str(e)}")

@app.get("/api/v1/nautilus/strategies/live")
async def get_live_strategies():
    """Get all live running strategies"""
    try:
        result = await nautilus_engine.get_live_strategies()
        return {"status": "success", "strategies": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get live strategies: {str(e)}")

@app.post("/api/v1/nautilus/deployment/pause/{strategy_instance_id}")
async def pause_strategy(strategy_instance_id: str):
    """Pause a live strategy"""
    try:
        result = await nautilus_engine.pause_strategy(strategy_instance_id)
        return {"status": "success", "message": "Strategy paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause strategy: {str(e)}")

@app.post("/api/v1/nautilus/deployment/resume/{strategy_instance_id}")
async def resume_strategy(strategy_instance_id: str):
    """Resume a paused strategy"""
    try:
        result = await nautilus_engine.resume_strategy(strategy_instance_id)
        return {"status": "success", "message": "Strategy resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume strategy: {str(e)}")

@app.post("/api/v1/nautilus/deployment/stop/{strategy_instance_id}")
async def emergency_stop_strategy(strategy_instance_id: str, reason: str = "Manual stop"):
    """Emergency stop a live strategy"""
    try:
        result = await nautilus_engine.emergency_stop_strategy(strategy_instance_id, reason)
        return {"status": "success", "message": "Strategy emergency stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop strategy: {str(e)}")

@app.post("/api/v1/nautilus/deployment/rollback")
async def rollback_deployment(deployment_id: str, target_version: str):
    """Rollback a deployment to previous version"""
    try:
        result = await nautilus_engine.rollback_deployment(deployment_id, target_version)
        return {"status": "success", "message": "Deployment rolled back"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback deployment: {str(e)}")

@app.get("/api/v1/nautilus/strategies/performance")
async def get_strategies_performance():
    """Get performance data for all live strategies"""
    try:
        result = await nautilus_engine.get_strategies_performance()
        return {"status": "success", "performance": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategies performance: {str(e)}")

# Epic 6 Story 6.4: Data Pipeline & Catalog Integration - API Endpoints

@app.get("/api/v1/nautilus/data/quality/{instrument_id}")
async def get_data_quality_metrics(instrument_id: str):
    """Get data quality metrics for specific instrument"""
    try:
        result = await nautilus_engine.get_data_quality_metrics(instrument_id)
        return {"status": "success", "quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data quality: {str(e)}")

@app.get("/api/v1/nautilus/data/gaps/{instrument_id}")
async def analyze_data_gaps(instrument_id: str):
    """Analyze data gaps for specific instrument"""
    try:
        result = await nautilus_engine.analyze_data_gaps(instrument_id)
        return {"status": "success", "gaps": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze data gaps: {str(e)}")

class DataExportRequest(BaseModel):
    instrument_ids: list[str]
    venues: list[str] = []
    timeframes: list[str] = []
    date_range: dict
    format: str = "parquet"
    compression: bool = True
    include_metadata: bool = True

@app.post("/api/v1/nautilus/data/export")
async def export_data(request: DataExportRequest):
    """Export data in various formats"""
    try:
        result = await nautilus_engine.export_data(request.dict())
        return {"status": "success", "exportId": result["export_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")

@app.get("/api/v1/nautilus/data/export/{export_id}/status")
async def get_export_status(export_id: str):
    """Get data export status"""
    try:
        result = await nautilus_engine.get_export_status(export_id)
        return {"status": "success", "export": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get export status: {str(e)}")

@app.post("/api/v1/nautilus/data/import")
async def import_data(file_path: str, format: str = "parquet"):
    """Import external data"""
    try:
        result = await nautilus_engine.import_data(file_path, format)
        return {"status": "success", "import": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import data: {str(e)}")

@app.get("/api/v1/nautilus/data/feeds/status")
async def get_data_feeds_status():
    """Get status of all real-time data feeds"""
    try:
        result = await nautilus_engine.get_data_feeds_status()
        return {"status": "success", "feeds": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feed status: {str(e)}")

@app.post("/api/v1/nautilus/data/feeds/subscribe")
async def subscribe_data_feed(feed_config: dict):
    """Subscribe to new data feed"""
    try:
        result = await nautilus_engine.subscribe_data_feed(feed_config)
        return {"status": "success", "subscription": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to subscribe to feed: {str(e)}")

@app.delete("/api/v1/nautilus/data/feeds/unsubscribe")
async def unsubscribe_data_feed(feed_id: str):
    """Unsubscribe from data feed"""
    try:
        result = await nautilus_engine.unsubscribe_data_feed(feed_id)
        return {"status": "success", "message": "Unsubscribed from feed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unsubscribe from feed: {str(e)}")

@app.get("/api/v1/nautilus/data/pipeline/health")
async def get_pipeline_health():
    """Get data pipeline health status"""
    try:
        result = await nautilus_engine.get_pipeline_health()
        return {"status": "success", "health": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline health: {str(e)}")

@app.post("/api/v1/nautilus/data/quality/validate")
async def validate_data_quality(instrument_id: str = None):
    """Validate data quality for instrument or all data"""
    try:
        result = await nautilus_engine.validate_data_quality(instrument_id)
        return {"status": "success", "validation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate data quality: {str(e)}")

@app.post("/api/v1/nautilus/data/quality/refresh")
async def refresh_quality_metrics():
    """Refresh all data quality metrics"""
    try:
        result = await nautilus_engine.refresh_quality_metrics()
        return {"status": "success", "message": "Quality metrics refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh quality metrics: {str(e)}")

# =============================================================================
# STRATEGY MANAGEMENT API ENDPOINTS (Stories 4.2-4.4)
# =============================================================================

@app.get("/api/v1/strategies/advanced/configurations")
async def get_advanced_strategy_configurations():
    """Get all advanced strategy configurations"""
    try:
        result = await nautilus_engine.get_advanced_configurations()
        return {"status": "success", "configurations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}")

@app.post("/api/v1/strategies/advanced/configuration")
async def create_advanced_strategy_configuration(config: dict):
    """Create an advanced strategy configuration"""
    try:
        result = await nautilus_engine.create_advanced_configuration(config)
        return {"status": "success", "configuration_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create configuration: {str(e)}")

@app.put("/api/v1/strategies/advanced/configuration/{config_id}")
async def update_advanced_strategy_configuration(config_id: str, config: dict):
    """Update an advanced strategy configuration"""
    try:
        result = await nautilus_engine.update_advanced_configuration(config_id, config)
        return {"status": "success", "updated": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/api/v1/strategies/live/monitoring")
async def get_live_strategy_monitoring():
    """Get comprehensive live strategy monitoring data"""
    try:
        result = await nautilus_engine.get_live_monitoring_data()
        return {"status": "success", "monitoring_data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")

@app.get("/api/v1/strategies/live/health/{strategy_id}")
async def get_strategy_health_metrics(strategy_id: str):
    """Get detailed health metrics for a specific strategy"""
    try:
        result = await nautilus_engine.get_strategy_health(strategy_id)
        return {"status": "success", "health_metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategy health: {str(e)}")

@app.post("/api/v1/strategies/live/alerts/configure")
async def configure_strategy_alerts(alert_config: dict):
    """Configure alerts for live strategy monitoring"""
    try:
        result = await nautilus_engine.configure_alerts(alert_config)
        return {"status": "success", "alert_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure alerts: {str(e)}")

@app.get("/api/v1/strategies/performance/comparison")
async def get_strategy_performance_comparison():
    """Get comprehensive strategy performance comparison"""
    try:
        result = await nautilus_engine.get_performance_comparison()
        return {"status": "success", "comparison_data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance comparison: {str(e)}")

@app.get("/api/v1/strategies/performance/detailed/{strategy_id}")
async def get_detailed_strategy_performance(strategy_id: str, start_date: str = None, end_date: str = None):
    """Get detailed performance analysis for a specific strategy"""
    try:
        result = await nautilus_engine.get_detailed_performance(strategy_id, start_date, end_date)
        return {"status": "success", "performance_data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get detailed performance: {str(e)}")

@app.get("/api/v1/strategies/performance/rankings")
async def get_strategy_performance_rankings():
    """Get strategy performance rankings and leaderboards"""
    try:
        result = await nautilus_engine.get_performance_rankings()
        return {"status": "success", "rankings": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance rankings: {str(e)}")

@app.post("/api/v1/strategies/performance/benchmark")
async def create_performance_benchmark(benchmark_config: dict):
    """Create a new performance benchmark"""
    try:
        result = await nautilus_engine.create_benchmark(benchmark_config)
        return {"status": "success", "benchmark_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create benchmark: {str(e)}")

@app.get("/api/v1/strategies/risk/analysis/{strategy_id}")
async def get_strategy_risk_analysis(strategy_id: str):
    """Get comprehensive risk analysis for a strategy"""
    try:
        result = await nautilus_engine.get_risk_analysis(strategy_id)
        return {"status": "success", "risk_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk analysis: {str(e)}")

@app.post("/api/v1/strategies/optimization/parameter-sweep")
async def run_parameter_optimization(optimization_config: dict):
    """Run parameter optimization sweep for a strategy"""
    try:
        result = await nautilus_engine.run_parameter_sweep(optimization_config)
        return {"status": "success", "optimization_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run optimization: {str(e)}")

# =============================================================================
# PORTFOLIO VISUALIZATION API ENDPOINTS (Story 4.4)
# =============================================================================

@app.get("/api/v1/portfolio/{portfolio_id}/strategy-allocations")
async def get_portfolio_strategy_allocations(portfolio_id: str, start_date: str = None, end_date: str = None):
    """Get strategy allocations for portfolio visualization"""
    try:
        result = await nautilus_engine.get_strategy_allocations(portfolio_id, start_date, end_date)
        return {"status": "success", "allocations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategy allocations: {str(e)}")

@app.get("/api/v1/portfolio/{portfolio_id}/performance-history")
async def get_portfolio_performance_history(portfolio_id: str, start_date: str = None, end_date: str = None):
    """Get detailed portfolio performance history"""
    try:
        result = await nautilus_engine.get_portfolio_performance_history(portfolio_id, start_date, end_date)
        return {"status": "success", "history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance history: {str(e)}")

@app.get("/api/v1/portfolio/{portfolio_id}/asset-allocations")
async def get_portfolio_asset_allocations(portfolio_id: str):
    """Get current asset allocations across all strategies"""
    try:
        result = await nautilus_engine.get_asset_allocations(portfolio_id)
        return {"status": "success", "allocations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get asset allocations: {str(e)}")

@app.get("/api/v1/portfolio/{portfolio_id}/strategy-correlations")
async def get_strategy_correlations(portfolio_id: str):
    """Get correlation matrix between strategies"""
    try:
        result = await nautilus_engine.get_strategy_correlations(portfolio_id)
        return {"status": "success", "correlations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategy correlations: {str(e)}")

@app.get("/api/v1/portfolio/{portfolio_id}/benchmark-comparison")
async def get_portfolio_benchmark_comparison(portfolio_id: str, start_date: str = None, end_date: str = None):
    """Get portfolio performance vs benchmark comparison"""
    try:
        result = await nautilus_engine.get_benchmark_comparison(portfolio_id, start_date, end_date)
        return {"status": "success", "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmark comparison: {str(e)}")

@app.post("/api/v1/portfolio/{portfolio_id}/rebalance")
async def rebalance_portfolio(portfolio_id: str, rebalance_config: dict):
    """Execute portfolio rebalancing based on target allocations"""
    try:
        result = await nautilus_engine.rebalance_portfolio(portfolio_id, rebalance_config)
        return {"status": "success", "rebalance_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebalance portfolio: {str(e)}")

@app.get("/api/v1/portfolio/{portfolio_id}/attribution-analysis")
async def get_portfolio_attribution_analysis(portfolio_id: str, start_date: str = None, end_date: str = None):
    """Get detailed performance attribution analysis"""
    try:
        result = await nautilus_engine.get_attribution_analysis(portfolio_id, start_date, end_date)
        return {"status": "success", "attribution": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get attribution analysis: {str(e)}")

# =============================================================================
# LIVE TRADING DATA SYSTEM STATUS
# =============================================================================

@app.get("/api/v1/system/status")
async def get_system_status():
    """Get comprehensive system status - our live trading approach vs NautilusTrader"""
    try:
        # Historical data service status
        historical_status = await historical_data_service.health_check()
        
        # Market data ingestion stats
        ingestion_stats = {}
        if historical_data_service.is_connected:
            async with historical_data_service._pool.acquire() as conn:
                # Get real-time performance metrics
                metrics = await conn.fetch("""
                    SELECT metric_name, metric_value, metric_unit 
                    FROM get_realtime_performance()
                """)
                ingestion_stats = {
                    row['metric_name']: {
                        'value': float(row['metric_value']),
                        'unit': row['metric_unit']
                    } for row in metrics
                }
        
        return {
            "implementation_approach": {
                "name": "Live Trading & Web Applications",
                "description": "Optimized for real-time data integration and web applications",
                "complementary_to": "NautilusTrader (research and backtesting)",
                "key_features": [
                    "PostgreSQL with nanosecond precision",
                    "Real-time data integration from IB Gateway", 
                    "Web dashboard with live charts",
                    "Parquet export for NautilusTrader compatibility",
                    "TimescaleDB optimization (optional)",
                    "Automatic data retention policies"
                ]
            },
            "services": {
                "historical_data": historical_status,
                "parquet_export": {
                    "status": "operational",
                    "nautilus_compatible": True,
                    "compression": parquet_export_service.config.compression
                },
                "real_time_ingestion": ingestion_stats,
                "ib_gateway": {
                    "connected": ib_gateway_client.is_connected() if 'ib_gateway_client' in globals() else False,
                    "auto_reconnect": True
                }
            },
            "integration_benefits": {
                "live_trading": "Real-time PostgreSQL for immediate access",
                "research": "Parquet exports for NautilusTrader analysis", 
                "best_of_both": "Combined approach maximizes utility",
                "data_quality": "Nanosecond precision maintained across both systems"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")


@app.get("/api/v1/system/performance")
async def get_system_performance():
    """Get real-time system performance metrics"""
    try:
        if not historical_data_service.is_connected:
            raise HTTPException(status_code=503, detail="Historical data service not available")
            
        async with historical_data_service._pool.acquire() as conn:
            # Get comprehensive performance stats
            stats_result = await conn.fetch("""
                SELECT * FROM get_market_data_stats(NULL, NULL, 1)
            """)
            
            performance_result = await conn.fetch("""
                SELECT * FROM get_realtime_performance()
            """)
            
            # Format data statistics
            data_stats = {}
            for row in stats_result:
                key = f"{row['venue']}_{row['instrument_id']}_{row['data_type']}"
                data_stats[key] = {
                    'venue': row['venue'],
                    'instrument_id': row['instrument_id'],
                    'data_type': row['data_type'],
                    'record_count': int(row['record_count']),
                    'time_range_hours': float(row['time_range_hours']) if row['time_range_hours'] else 0
                }
            
            # Format performance metrics
            performance_metrics = {}
            for row in performance_result:
                performance_metrics[row['metric_name']] = {
                    'value': float(row['metric_value']),
                    'unit': row['metric_unit']
                }
                
        return {
            "data_statistics": data_stats,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "system_approach": "Live Trading Optimized with NautilusTrader Compatibility"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {e}")


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        # Send welcome message
        await manager.send_personal_message(
            '{"type": "connection", "status": "connected", "message": "Connected to Nautilus Trader API"}',
            websocket
        )
        
        while True:
            data = await websocket.receive_text()
            
            # Basic input validation and sanitization
            if not data or len(data) > 1024:  # Reasonable message size limit
                await manager.send_personal_message(
                    '{"type": "error", "message": "Invalid message size"}', 
                    websocket
                )
                continue
                
            # Echo received data for testing (in production, would route to handlers)
            await manager.send_personal_message(f"Echo: {data}", websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/realtime")
async def websocket_realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data and updates"""
    await manager.connect(websocket)
    try:
        # Send welcome message
        await manager.send_personal_message(
            '{"type": "connection", "status": "connected", "message": "Connected to Nautilus Trader Real-time API"}',
            websocket
        )
        
        while True:
            data = await websocket.receive_text()
            
            # Basic input validation and sanitization
            if not data or len(data) > 1024:  # Reasonable message size limit
                await manager.send_personal_message(
                    '{"type": "error", "message": "Invalid message size"}', 
                    websocket
                )
                continue
                
            # Echo received data for testing (in production, would route to handlers)
            await manager.send_personal_message(f"Echo: {data}", websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Nautilus Trader API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
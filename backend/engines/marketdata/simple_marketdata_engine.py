#!/usr/bin/env python3
"""
Simple Market Data Engine - Containerized Market Data Processing Service
High-performance market data ingestion, processing, and distribution
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    TICK = "tick"
    QUOTE = "quote"
    BAR = "bar"
    TRADE = "trade"
    LEVEL2 = "level2"
    NEWS = "news"

class DataSource(Enum):
    IBKR = "ibkr"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    YAHOO = "yahoo"
    INTERNAL = "internal"
    MOCK = "mock"

@dataclass
class MarketDataPoint:
    symbol: str
    data_type: DataType
    source: DataSource
    timestamp: datetime
    data: Dict[str, Any]
    sequence: int
    latency_ms: float

@dataclass
class DataFeed:
    feed_id: str
    symbol: str
    data_source: DataSource
    data_types: List[DataType]
    is_active: bool
    last_update: datetime
    message_count: int
    error_count: int

@dataclass
class MarketDataSubscription:
    subscription_id: str
    symbols: List[str]
    data_types: List[DataType]
    callback_url: Optional[str]
    is_active: bool
    created_at: datetime

class SimpleMarketDataEngine:
    """
    Simple Market Data Engine demonstrating containerization approach
    High-performance data ingestion and real-time distribution
    """
    
    def __init__(self):
        self.app = FastAPI(title="Nautilus Simple Market Data Engine", version="1.0.0")
        self.is_running = False
        self.messages_processed = 0
        self.data_points_stored = 0
        self.subscriptions_served = 0
        self.start_time = time.time()
        
        # Market data state
        self.active_feeds: Dict[str, DataFeed] = {}
        self.subscriptions: Dict[str, MarketDataSubscription] = {}
        self.data_cache: Dict[str, List[MarketDataPoint]] = {}  # symbol -> recent data points
        self.symbols_tracked: set = set()
        
        # Performance metrics
        self.latency_stats = {"min": 0, "max": 0, "avg": 0, "p95": 0}
        self.throughput_stats = {"messages_per_second": 0, "data_points_per_second": 0}
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.data_generation_task = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "messages_processed": self.messages_processed,
                "data_points_stored": self.data_points_stored,
                "subscriptions_served": self.subscriptions_served,
                "active_feeds": len(self.active_feeds),
                "symbols_tracked": len(self.symbols_tracked),
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            uptime = time.time() - self.start_time
            return {
                "messages_per_second": self.messages_processed / max(1, uptime),
                "data_points_per_second": self.data_points_stored / max(1, uptime),
                "total_messages": self.messages_processed,
                "total_data_points": self.data_points_stored,
                "active_subscriptions": len(self.subscriptions),
                "active_feeds": len(self.active_feeds),
                "symbols_tracked": len(self.symbols_tracked),
                "latency_stats": self.latency_stats,
                "throughput_stats": self.throughput_stats,
                "cache_size": sum(len(points) for points in self.data_cache.values()),
                "uptime": uptime,
                "engine_type": "market_data",
                "containerized": True
            }
        
        @self.app.get("/feeds")
        async def get_active_feeds():
            """Get all active data feeds"""
            feeds = []
            for feed in self.active_feeds.values():
                feeds.append({
                    "feed_id": feed.feed_id,
                    "symbol": feed.symbol,
                    "data_source": feed.data_source.value,
                    "data_types": [dt.value for dt in feed.data_types],
                    "is_active": feed.is_active,
                    "last_update": feed.last_update.isoformat(),
                    "message_count": feed.message_count,
                    "error_count": feed.error_count
                })
            
            return {
                "feeds": feeds,
                "count": len(feeds),
                "total_messages": sum(feed.message_count for feed in self.active_feeds.values())
            }
        
        @self.app.post("/feeds")
        async def create_data_feed(feed_config: Dict[str, Any]):
            """Create new data feed"""
            try:
                feed = DataFeed(
                    feed_id=f"feed_{len(self.active_feeds)}_{int(time.time())}",
                    symbol=feed_config.get("symbol", "AAPL"),
                    data_source=DataSource(feed_config.get("data_source", "mock")),
                    data_types=[DataType(dt) for dt in feed_config.get("data_types", ["tick", "quote"])],
                    is_active=True,
                    last_update=datetime.now(),
                    message_count=0,
                    error_count=0
                )
                
                self.active_feeds[feed.feed_id] = feed
                self.symbols_tracked.add(feed.symbol)
                
                return {
                    "status": "created",
                    "feed_id": feed.feed_id,
                    "symbol": feed.symbol,
                    "data_source": feed.data_source.value
                }
                
            except Exception as e:
                logger.error(f"Feed creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/{symbol}")
        async def get_market_data(symbol: str, data_type: str = "all", limit: int = 100):
            """Get recent market data for symbol"""
            try:
                if symbol not in self.data_cache:
                    return {"symbol": symbol, "data": [], "count": 0}
                
                data_points = self.data_cache[symbol]
                
                # Filter by data type if specified
                if data_type != "all":
                    data_points = [dp for dp in data_points if dp.data_type.value == data_type]
                
                # Apply limit
                data_points = data_points[-limit:]
                
                return {
                    "symbol": symbol,
                    "data_type": data_type,
                    "data": [
                        {
                            "timestamp": dp.timestamp.isoformat(),
                            "data_type": dp.data_type.value,
                            "source": dp.source.value,
                            "data": dp.data,
                            "latency_ms": dp.latency_ms
                        }
                        for dp in data_points
                    ],
                    "count": len(data_points)
                }
                
            except Exception as e:
                logger.error(f"Market data retrieval error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/subscriptions")
        async def create_subscription(subscription_config: Dict[str, Any]):
            """Create market data subscription"""
            try:
                subscription = MarketDataSubscription(
                    subscription_id=f"sub_{len(self.subscriptions)}_{int(time.time())}",
                    symbols=subscription_config.get("symbols", ["AAPL"]),
                    data_types=[DataType(dt) for dt in subscription_config.get("data_types", ["quote"])],
                    callback_url=subscription_config.get("callback_url"),
                    is_active=True,
                    created_at=datetime.now()
                )
                
                self.subscriptions[subscription.subscription_id] = subscription
                self.subscriptions_served += 1
                
                # Add symbols to tracking
                for symbol in subscription.symbols:
                    self.symbols_tracked.add(symbol)
                
                return {
                    "status": "created",
                    "subscription_id": subscription.subscription_id,
                    "symbols": subscription.symbols,
                    "data_types": [dt.value for dt in subscription.data_types]
                }
                
            except Exception as e:
                logger.error(f"Subscription creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/subscriptions")
        async def get_subscriptions():
            """Get all active subscriptions"""
            subscriptions = []
            for sub in self.subscriptions.values():
                subscriptions.append({
                    "subscription_id": sub.subscription_id,
                    "symbols": sub.symbols,
                    "data_types": [dt.value for dt in sub.data_types],
                    "callback_url": sub.callback_url,
                    "is_active": sub.is_active,
                    "created_at": sub.created_at.isoformat()
                })
            
            return {
                "subscriptions": subscriptions,
                "count": len(subscriptions)
            }
        
        @self.app.post("/data/ingest")
        async def ingest_market_data(data_batch: Dict[str, Any]):
            """Ingest batch of market data"""
            try:
                ingested_count = 0
                
                for data_item in data_batch.get("data", []):
                    data_point = MarketDataPoint(
                        symbol=data_item.get("symbol", "UNKNOWN"),
                        data_type=DataType(data_item.get("data_type", "tick")),
                        source=DataSource(data_item.get("source", "internal")),
                        timestamp=datetime.fromisoformat(data_item.get("timestamp", datetime.now().isoformat())),
                        data=data_item.get("data", {}),
                        sequence=self.data_points_stored + ingested_count,
                        latency_ms=data_item.get("latency_ms", 1.0)
                    )
                    
                    await self._store_data_point(data_point)
                    ingested_count += 1
                
                return {
                    "status": "ingested",
                    "count": ingested_count,
                    "total_stored": self.data_points_stored
                }
                
            except Exception as e:
                logger.error(f"Data ingestion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/symbols")
        async def get_tracked_symbols():
            """Get all tracked symbols"""
            return {
                "symbols": list(self.symbols_tracked),
                "count": len(self.symbols_tracked)
            }
        
        @self.app.post("/symbols/{symbol}/snapshot")
        async def get_symbol_snapshot(symbol: str):
            """Get current market data snapshot for symbol"""
            try:
                # Generate mock snapshot
                snapshot = await self._generate_market_snapshot(symbol)
                
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "snapshot": snapshot
                }
                
            except Exception as e:
                logger.error(f"Snapshot generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def start_engine(self):
        """Start the market data engine"""
        try:
            logger.info("Starting Simple Market Data Engine...")
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Initialize default feeds and subscriptions
            await self._initialize_default_feeds()
            
            # Start data generation task
            self.data_generation_task = asyncio.create_task(self._generate_mock_data())
            
            self.is_running = True
            logger.info(f"Simple Market Data Engine started successfully with {len(self.active_feeds)} feeds")
            
        except Exception as e:
            logger.error(f"Failed to start Market Data Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the market data engine"""
        logger.info("Stopping Simple Market Data Engine...")
        self.is_running = False
        
        # Cancel data generation
        if self.data_generation_task:
            self.data_generation_task.cancel()
        
        if self.messagebus:
            await self.messagebus.stop()
        
        logger.info("Simple Market Data Engine stopped")
    
    async def _initialize_default_feeds(self):
        """Initialize default data feeds"""
        default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        for symbol in default_symbols:
            feed = DataFeed(
                feed_id=f"default_{symbol.lower()}",
                symbol=symbol,
                data_source=DataSource.MOCK,
                data_types=[DataType.TICK, DataType.QUOTE, DataType.BAR],
                is_active=True,
                last_update=datetime.now(),
                message_count=0,
                error_count=0
            )
            
            self.active_feeds[feed.feed_id] = feed
            self.symbols_tracked.add(symbol)
            self.data_cache[symbol] = []
        
        logger.info(f"Initialized {len(default_symbols)} default feeds")
    
    async def _generate_mock_data(self):
        """Generate mock market data"""
        try:
            while self.is_running:
                for symbol in self.symbols_tracked:
                    # Generate various data types
                    data_types = [DataType.TICK, DataType.QUOTE, DataType.BAR]
                    
                    for data_type in data_types:
                        data_point = await self._generate_mock_data_point(symbol, data_type)
                        await self._store_data_point(data_point)
                        
                        # Update feed statistics
                        for feed in self.active_feeds.values():
                            if feed.symbol == symbol:
                                feed.message_count += 1
                                feed.last_update = datetime.now()
                                break
                
                await asyncio.sleep(0.1)  # 10 updates per second per symbol
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Mock data generation error: {e}")
    
    async def _generate_mock_data_point(self, symbol: str, data_type: DataType) -> MarketDataPoint:
        """Generate mock data point"""
        base_price = 100 + hash(symbol) % 400  # Deterministic base price per symbol
        current_time = datetime.now()
        
        if data_type == DataType.TICK:
            data = {
                "price": round(base_price + np.random.normal(0, 2), 2),
                "size": np.random.randint(100, 1000),
                "exchange": "NASDAQ"
            }
        elif data_type == DataType.QUOTE:
            spread = np.random.uniform(0.01, 0.1)
            mid = base_price + np.random.normal(0, 1)
            data = {
                "bid": round(mid - spread/2, 2),
                "ask": round(mid + spread/2, 2),
                "bid_size": np.random.randint(100, 500),
                "ask_size": np.random.randint(100, 500)
            }
        elif data_type == DataType.BAR:
            open_price = base_price + np.random.normal(0, 1)
            data = {
                "open": round(open_price, 2),
                "high": round(open_price + np.random.uniform(0, 2), 2),
                "low": round(open_price - np.random.uniform(0, 2), 2),
                "close": round(open_price + np.random.normal(0, 0.5), 2),
                "volume": np.random.randint(10000, 100000)
            }
        else:
            data = {"value": np.random.random()}
        
        return MarketDataPoint(
            symbol=symbol,
            data_type=data_type,
            source=DataSource.MOCK,
            timestamp=current_time,
            data=data,
            sequence=self.data_points_stored,
            latency_ms=np.random.uniform(0.5, 5.0)
        )
    
    async def _store_data_point(self, data_point: MarketDataPoint):
        """Store data point and update metrics"""
        symbol = data_point.symbol
        
        # Initialize cache if needed
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        # Add to cache (keep last 1000 points per symbol)
        self.data_cache[symbol].append(data_point)
        if len(self.data_cache[symbol]) > 1000:
            self.data_cache[symbol] = self.data_cache[symbol][-1000:]
        
        # Update counters
        self.data_points_stored += 1
        self.messages_processed += 1
        
        # Update latency statistics
        self._update_latency_stats(data_point.latency_ms)
        
        # Update throughput statistics
        self._update_throughput_stats()
    
    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics"""
        # Simple running statistics (could be improved with proper sliding window)
        if self.latency_stats["min"] == 0 or latency_ms < self.latency_stats["min"]:
            self.latency_stats["min"] = latency_ms
        if latency_ms > self.latency_stats["max"]:
            self.latency_stats["max"] = latency_ms
        
        # Simple running average
        current_avg = self.latency_stats["avg"]
        self.latency_stats["avg"] = (current_avg * 0.99) + (latency_ms * 0.01)
        self.latency_stats["p95"] = self.latency_stats["max"] * 0.95  # Approximation
    
    def _update_throughput_stats(self):
        """Update throughput statistics"""
        uptime = time.time() - self.start_time
        if uptime > 0:
            self.throughput_stats["messages_per_second"] = self.messages_processed / uptime
            self.throughput_stats["data_points_per_second"] = self.data_points_stored / uptime
    
    async def _generate_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Generate market snapshot for symbol"""
        base_price = 100 + hash(symbol) % 400
        
        return {
            "last_trade": {
                "price": round(base_price + np.random.normal(0, 1), 2),
                "size": np.random.randint(100, 1000),
                "timestamp": datetime.now().isoformat()
            },
            "quote": {
                "bid": round(base_price - 0.05, 2),
                "ask": round(base_price + 0.05, 2),
                "bid_size": np.random.randint(500, 2000),
                "ask_size": np.random.randint(500, 2000)
            },
            "daily_stats": {
                "open": round(base_price * 0.98, 2),
                "high": round(base_price * 1.03, 2),
                "low": round(base_price * 0.95, 2),
                "volume": np.random.randint(1000000, 5000000)
            }
        }

# Create and start the engine
simple_marketdata_engine = SimpleMarketDataEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        # For now, use simple engine with hybrid mode flag
        logger.info("Hybrid MarketData Engine integration enabled (using enhanced simple engine)")
        app = simple_marketdata_engine.app
        engine_instance = simple_marketdata_engine
        # Add hybrid flag to engine
        engine_instance.hybrid_enabled = True
    except Exception as e:
        logger.warning(f"Hybrid MarketData Engine setup failed: {e}. Using simple engine.")
        app = simple_marketdata_engine.app
        engine_instance = simple_marketdata_engine
else:
    logger.info("Using Simple MarketData Engine (hybrid disabled)")
    app = simple_marketdata_engine.app
    engine_instance = simple_marketdata_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8800"))
    
    logger.info(f"Starting MarketData Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
    # Start the engine on startup
    async def lifespan():
        await engine_instance.start_engine()
    
    # Run startup
    asyncio.run(lifespan())
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
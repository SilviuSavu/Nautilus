#!/usr/bin/env python3
"""
Simple Analytics Engine - Containerized Performance Analytics Service
Demonstrates containerization approach with basic MessageBus integration
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass
import json
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
# Import MarketData Client - MANDATORY for all market data access
from marketdata_client import create_marketdata_client, DataType, DataSource
from universal_enhanced_messagebus_client import EngineType, MessageType

# Dual MessageBus integration - MIGRATED
from dual_messagebus_client import DualMessageBusClient, DualBusConfig, create_dual_bus_client
# Real data integration
from real_data_integration import RealDataIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    portfolio_id: str
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

class SimpleAnalyticsEngine:
    """
    Simple Analytics Engine demonstrating containerization approach
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Nautilus Real Data Analytics Engine", 
            version="2.0.0",
            description="Analytics Engine with real market data integration",
            lifespan=self.lifespan
        )
        self.is_running = False
        self.processed_count = 0
        self.start_time = time.time()
        
        # Dual MessageBus configuration - MIGRATED
        self.dual_bus_config = DualBusConfig(
            engine_type=EngineType.ANALYTICS,
            engine_instance_id=f"analytics-{int(time.time()*1000)%10000}",
            marketdata_redis_host=os.getenv("MARKETDATA_REDIS_HOST", "localhost"),
            marketdata_redis_port=int(os.getenv("MARKETDATA_REDIS_PORT", "6380")),
            engine_logic_redis_host=os.getenv("ENGINE_LOGIC_REDIS_HOST", "localhost"),
            engine_logic_redis_port=int(os.getenv("ENGINE_LOGIC_REDIS_PORT", "6381"))
        )
        
        self.dual_messagebus = None
        
        # MarketData Client (MANDATORY - replaces all direct API calls)
        self.marketdata_client = None
        self.marketdata_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0
        }
        
        # Real data integration
        database_url = os.getenv("DATABASE_URL", "postgresql://nautilus:nautilus123@postgres:5432/nautilus")
        marketdata_url = os.getenv("MARKETDATA_ENGINE_URL", "http://marketdata-engine:8800")
        self.real_data = RealDataIntegration(database_url, marketdata_url)
        
        self.setup_routes()
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan management with connection pooling"""
        # Startup
        await self.start_engine()
        yield
        # Shutdown
        await self.stop_engine()
    
    async def start_engine(self):
        """Start the analytics engine with connection pooling and real data integration"""
        try:
            logger.info("Starting Real Data Analytics Engine...")
            
            # Initialize MarketData Client (MANDATORY for sub-5ms performance)
            self.marketdata_client = create_marketdata_client(
                EngineType.ANALYTICS, 
                8100
            )
            logger.info("✅ MarketData Client initialized - all market data via Centralized Hub")
            
            # Initialize Real Data Integration
            try:
                await self.real_data.initialize()
                logger.info("Real Data Integration initialized successfully")
            except Exception as e:
                logger.error(f"Real Data Integration initialization failed: {e}")
                # Continue without real data integration
            
            # Initialize Dual MessageBus - MIGRATED
            try:
                self.dual_messagebus = create_dual_bus_client(EngineType.ANALYTICS)
                await self.dual_messagebus.initialize()
                logger.info("✅ Dual MessageBus connected - MarketData Bus (6380) + Engine Logic Bus (6381)")
            except Exception as e:
                logger.warning(f"Dual MessageBus connection failed: {e}. Running without MessageBus.")
                self.dual_messagebus = None
            
            self.is_running = True
            logger.info("Real Data Analytics Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Analytics Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the analytics engine"""
        logger.info("Stopping Real Data Analytics Engine...")
        self.is_running = False
        
        # Close real data integration
        if self.real_data:
            await self.real_data.cleanup()
        
        if self.dual_messagebus:
            await self.dual_messagebus.close()
        
        logger.info("Real Data Analytics Engine stopped")
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "processed_count": self.processed_count,
                "uptime_seconds": time.time() - self.start_time,
                "dual_messagebus_connected": self.dual_messagebus is not None and self.dual_messagebus._initialized,
                "real_data_integration": self.real_data is not None,
                "marketdata_client_connected": self.marketdata_client is not None,
                "marketdata_metrics": self.marketdata_client.get_metrics() if self.marketdata_client else {},
                "engine_version": "2.0.0-MarketDataHub",
                "data_sources": ["centralized_hub", "database"]
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "processed_analytics": self.processed_count,
                "processing_rate": self.processed_count / max(1, time.time() - self.start_time),
                "uptime": time.time() - self.start_time,
                "engine_type": "real_data_analytics",
                "containerized": True,
                "real_data_enabled": True,
                "data_sources": ["centralized_hub", "database"],
                "marketdata_hub_performance": self.marketdata_client.get_metrics() if self.marketdata_client else {}
            }
        
        @self.app.post("/analytics/performance/{portfolio_id}")
        async def calculate_performance(portfolio_id: str, data: Dict[str, Any]):
            """Calculate portfolio performance metrics using real data"""
            try:
                # Extract symbol from data if provided, otherwise use default
                symbol = data.get("symbol", "SPY.SMART")
                period_days = data.get("period_days", 252)  # Default to 1 year
                
                # Calculate real performance metrics
                if self.real_data:
                    performance_result = await self.real_data.calculate_performance_metrics(
                        symbol, period_days=period_days
                    )
                    
                    if performance_result:
                        result = {
                            "portfolio_id": portfolio_id,
                            "symbol": symbol,
                            "timestamp": performance_result.period_end.isoformat(),
                            "period_start": performance_result.period_start.isoformat(),
                            "period_end": performance_result.period_end.isoformat(),
                            "total_return_pct": performance_result.total_return,
                            "sharpe_ratio": performance_result.sharpe_ratio,
                            "max_drawdown_pct": performance_result.max_drawdown,
                            "volatility_pct": performance_result.volatility,
                            "beta": performance_result.beta,
                            "alpha": performance_result.alpha,
                            "calmar_ratio": performance_result.calmar_ratio,
                            "sortino_ratio": performance_result.sortino_ratio,
                            "win_rate_pct": performance_result.win_rate,
                            "profit_factor": performance_result.profit_factor,
                            "calculation_type": "real_data_performance",
                            "data_source": "market_database"
                        }
                    else:
                        result = await self._calculate_performance_metrics(portfolio_id, data)  # Fallback
                        result["data_source"] = "fallback_mock"
                else:
                    result = await self._calculate_performance_metrics(portfolio_id, data)
                    result["data_source"] = "mock_data"
                
                self.processed_count += 1
                
                # If Dual MessageBus is available, publish results
                if self.dual_messagebus and self.dual_messagebus._initialized:
                    await self._publish_results(result)
                
                return {"status": "success", "result": result}
                
            except Exception as e:
                logger.error(f"Performance calculation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/symbol/{symbol}")
        async def get_symbol_analysis(symbol: str):
            """Get comprehensive analysis for a specific symbol using real data"""
            try:
                if not self.real_data:
                    raise HTTPException(status_code=503, detail="Real data integration not available")
                
                result = await self.real_data.get_comprehensive_analysis(symbol)
                self.processed_count += 1
                return result
                
            except Exception as e:
                logger.error(f"Symbol analysis error for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/technical-indicators/{symbol}")
        async def get_technical_indicators(symbol: str, limit: int = 100):
            """Get technical indicators for a symbol using real price data"""
            try:
                if not self.real_data:
                    raise HTTPException(status_code=503, detail="Real data integration not available")
                
                # Get recent market data
                market_data = await self.real_data.get_market_data_from_db(symbol, limit=limit)
                
                if not market_data:
                    raise HTTPException(status_code=404, detail=f"No market data found for symbol {symbol}")
                
                # Calculate technical indicators
                indicators = self.real_data.calculate_technical_indicators(market_data)
                
                if not indicators:
                    raise HTTPException(status_code=400, detail="Insufficient data for technical indicators")
                
                result = {
                    "symbol": symbol,
                    "timestamp": indicators.timestamp.isoformat(),
                    "indicators": {
                        "sma_20": indicators.sma_20,
                        "sma_50": indicators.sma_50,
                        "ema_12": indicators.ema_12,
                        "ema_26": indicators.ema_26,
                        "rsi_14": indicators.rsi_14,
                        "macd": indicators.macd,
                        "macd_signal": indicators.macd_signal,
                        "macd_histogram": indicators.macd_histogram,
                        "bollinger_bands": {
                            "upper": indicators.bb_upper,
                            "middle": indicators.bb_middle,
                            "lower": indicators.bb_lower
                        },
                        "atr_14": indicators.atr_14,
                        "stochastic": {
                            "k": indicators.stoch_k,
                            "d": indicators.stoch_d
                        }
                    },
                    "data_points_used": len(market_data),
                    "calculation_type": "real_market_data"
                }
                
                self.processed_count += 1
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Technical indicators error for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/performance-metrics/{symbol}")
        async def get_performance_metrics(symbol: str, period_days: int = 252, benchmark: str = "SPY.SMART"):
            """Get detailed performance metrics for a symbol using real data"""
            try:
                if not self.real_data:
                    raise HTTPException(status_code=503, detail="Real data integration not available")
                
                performance = await self.real_data.calculate_performance_metrics(
                    symbol, period_days=period_days, benchmark_symbol=benchmark
                )
                
                if not performance:
                    raise HTTPException(status_code=404, detail=f"Unable to calculate performance metrics for {symbol}")
                
                result = {
                    "symbol": symbol,
                    "benchmark": benchmark,
                    "period": {
                        "days": period_days,
                        "start_date": performance.period_start.isoformat(),
                        "end_date": performance.period_end.isoformat()
                    },
                    "returns": {
                        "total_return_pct": performance.total_return,
                        "annualized_return_pct": (performance.total_return / 100) * (365 / period_days) * 100
                    },
                    "risk_metrics": {
                        "volatility_pct": performance.volatility,
                        "max_drawdown_pct": performance.max_drawdown,
                        "beta": performance.beta,
                        "alpha": performance.alpha
                    },
                    "risk_adjusted_returns": {
                        "sharpe_ratio": performance.sharpe_ratio,
                        "calmar_ratio": performance.calmar_ratio,
                        "sortino_ratio": performance.sortino_ratio
                    },
                    "trading_metrics": {
                        "win_rate_pct": performance.win_rate,
                        "avg_win_pct": performance.avg_win,
                        "avg_loss_pct": performance.avg_loss,
                        "profit_factor": performance.profit_factor
                    },
                    "calculation_type": "real_market_data",
                    "data_source": "database"
                }
                
                self.processed_count += 1
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Performance metrics error for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/market-data/{symbol}")
        async def get_market_data(symbol: str, limit: int = 50):
            """Get recent market data points for a symbol"""
            try:
                if not self.real_data:
                    raise HTTPException(status_code=503, detail="Real data integration not available")
                
                market_data = await self.real_data.get_market_data_from_db(symbol, limit=limit)
                
                if not market_data:
                    raise HTTPException(status_code=404, detail=f"No market data found for symbol {symbol}")
                
                result = {
                    "symbol": symbol,
                    "data_points": len(market_data),
                    "latest_timestamp": market_data[0].timestamp.isoformat() if market_data else None,
                    "oldest_timestamp": market_data[-1].timestamp.isoformat() if market_data else None,
                    "market_data": [
                        {
                            "timestamp": dp.timestamp.isoformat(),
                            "open": dp.open_price,
                            "high": dp.high_price,
                            "low": dp.low_price,
                            "close": dp.close_price,
                            "volume": dp.volume,
                            "venue": dp.venue
                        } for dp in market_data[:10]  # Limit response size
                    ],
                    "summary": {
                        "latest_price": market_data[0].close_price if market_data else 0,
                        "price_change": ((market_data[0].close_price - market_data[-1].close_price) / market_data[-1].close_price * 100) if len(market_data) > 1 else 0,
                        "avg_volume": sum(dp.volume for dp in market_data) / len(market_data) if market_data else 0,
                        "high_52w": max(dp.high_price for dp in market_data) if market_data else 0,
                        "low_52w": min(dp.low_price for dp in market_data) if market_data else 0
                    }
                }
                
                self.processed_count += 1
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Market data error for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/available-symbols")
        async def get_available_symbols():
            """Get list of symbols available in the database"""
            try:
                if not self.real_data or not self.real_data.db_pool:
                    raise HTTPException(status_code=503, detail="Database connection not available")
                
                query = """
                    SELECT DISTINCT instrument_id, venue, COUNT(*) as data_points,
                           MIN(timestamp_ns) as first_data, MAX(timestamp_ns) as last_data
                    FROM market_bars 
                    GROUP BY instrument_id, venue 
                    ORDER BY data_points DESC
                """
                
                async with self.real_data.db_pool.acquire() as conn:
                    rows = await conn.fetch(query)
                
                symbols = []
                for row in rows:
                    symbols.append({
                        "symbol": row["instrument_id"],
                        "venue": row["venue"],
                        "data_points": row["data_points"],
                        "first_data": datetime.fromtimestamp(row["first_data"] / 1_000_000_000).isoformat(),
                        "last_data": datetime.fromtimestamp(row["last_data"] / 1_000_000_000).isoformat(),
                    })
                
                return {
                    "total_symbols": len(symbols),
                    "total_data_points": sum(s["data_points"] for s in symbols),
                    "symbols": symbols
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error fetching available symbols: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/calculate/{portfolio_id}")
        async def get_analytics(portfolio_id: str):
            """Get analytics for a portfolio using real data"""
            try:
                # For portfolio analysis, we'll use a representative symbol
                # In production, this would query actual portfolio holdings
                symbol = "SPY.SMART"  # Default to SPY for portfolio-level analysis
                
                # Get comprehensive real data analysis
                if self.real_data:
                    result = await self.real_data.get_comprehensive_analysis(symbol)
                    result["portfolio_id"] = portfolio_id
                    result["analysis_type"] = "portfolio_performance"
                else:
                    # Fallback to mock data if real data not available
                    result = {
                        "portfolio_id": portfolio_id,
                        "timestamp": datetime.now().isoformat(),
                        "error": "Real data integration not available",
                        "metrics": {
                            "total_pnl": float(np.random.normal(10000, 5000)),
                            "daily_pnl": float(np.random.normal(500, 200)),
                            "sharpe_ratio": float(np.random.normal(1.5, 0.3)),
                            "max_drawdown": abs(float(np.random.normal(-0.15, 0.05))),
                            "win_rate": float(np.random.uniform(0.45, 0.65)),
                            "total_trades": int(np.random.randint(100, 1000))
                        },
                        "processing_time_ms": 1.0
                    }
                
                self.processed_count += 1
                return result
                
            except Exception as e:
                logger.error(f"Analytics calculation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    
    async def _calculate_performance_metrics(self, portfolio_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        # Simulate calculation time
        await asyncio.sleep(0.001)
        
        # Mock performance calculation
        performance = PerformanceMetrics(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            total_pnl=np.random.normal(10000, 5000),
            daily_pnl=np.random.normal(500, 200),
            sharpe_ratio=np.random.normal(1.5, 0.3),
            max_drawdown=abs(np.random.normal(-0.15, 0.05)),
            win_rate=np.random.uniform(0.45, 0.65),
            total_trades=np.random.randint(100, 1000)
        )
        
        return {
            "portfolio_id": portfolio_id,
            "timestamp": performance.timestamp.isoformat(),
            "total_pnl": performance.total_pnl,
            "daily_pnl": performance.daily_pnl,
            "sharpe_ratio": performance.sharpe_ratio,
            "max_drawdown": performance.max_drawdown,
            "win_rate": performance.win_rate,
            "total_trades": performance.total_trades,
            "calculation_type": "portfolio_performance"
        }
    
    async def _publish_results(self, result: Dict[str, Any]):
        """Publish results to Dual MessageBus if available - MIGRATED"""
        if self.dual_messagebus and self.dual_messagebus._initialized:
            try:
                # Publish analytics results to Engine Logic Bus
                await self.dual_messagebus.publish_message(
                    MessageType.ANALYTICS_RESULT,
                    {
                        "topic": "analytics.results",
                        "data": result,
                        "timestamp": time.time(),
                        "source": "analytics-engine"
                    }
                )
                logger.debug(f"✅ Published analytics result to Engine Logic Bus: {result.get('portfolio_id', 'N/A')}")
            except Exception as e:
                logger.error(f"Failed to publish results to Dual MessageBus: {e}")

# Create and start the engine
simple_analytics_engine = SimpleAnalyticsEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        from hybrid_analytics_engine import hybrid_analytics_engine
        logger.info("Hybrid Analytics Engine integration enabled")
        # Use hybrid engine as the primary engine
        app = hybrid_analytics_engine.app
        engine_instance = hybrid_analytics_engine
    except ImportError as e:
        logger.warning(f"Hybrid Analytics Engine not available: {e}. Using simple engine.")
        app = simple_analytics_engine.app
        engine_instance = simple_analytics_engine
else:
    logger.info("Using Simple Analytics Engine (hybrid disabled)")
    app = simple_analytics_engine.app
    engine_instance = simple_analytics_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8100"))
    
    logger.info(f"Starting Analytics Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
    # Start FastAPI server with lifespan management
    if hasattr(engine_instance, 'app'):
        uvicorn.run(
            engine_instance.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    else:
        # Fallback for engines without app attribute
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
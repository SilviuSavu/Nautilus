"""
Nautilus Volatility Forecasting Engine

This is the main service that coordinates all volatility forecasting capabilities,
providing a unified interface for real-time volatility estimation and forecasting.
"""

import asyncio
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import json

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Data processing
import pandas as pd
import numpy as np

# Internal imports
from ..config import VolatilityConfig, get_default_volatility_config
from ..ensemble.orchestrator import EnsembleOrchestrator, EnsembleForecast
from ..models.base import ModelStatus
from ..streaming.messagebus_client import create_volatility_messagebus_client, VolatilityMessageBusClient, MarketDataEvent

# Redis integration
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Database integration
try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)


class VolatilityEngine:
    """
    Main Volatility Forecasting Engine
    
    This class provides the core functionality for the volatility forecasting service,
    coordinating multiple models through the ensemble orchestrator and providing
    real-time volatility estimates and forecasts.
    """
    
    def __init__(self, config: Optional[VolatilityConfig] = None):
        """
        Initialize the volatility engine.
        
        Args:
            config: Volatility configuration (uses defaults if None)
        """
        self.config = config or VolatilityConfig.from_environment()
        
        # Validate configuration
        if not self.config.validate_config():
            raise ValueError("Invalid volatility engine configuration")
        
        # Initialize core components
        self.orchestrator = EnsembleOrchestrator(self.config)
        
        # Service state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.processed_forecasts = 0
        self.active_symbols: set = set()
        
        # External connections
        self.redis_client: Optional[redis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.messagebus_client: Optional[VolatilityMessageBusClient] = None
        
        # Real-time data buffers
        self.data_streams: Dict[str, asyncio.Queue] = {}
        self.forecast_cache: Dict[str, EnsembleForecast] = {}
        
        # Performance monitoring
        self.performance_stats = {
            'total_forecasts': 0,
            'avg_forecast_time_ms': 0.0,
            'models_active': 0,
            'hardware_acceleration_used': self.config.hardware.auto_hardware_routing
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the volatility engine and all its components"""
        try:
            self.logger.info("Initializing Volatility Forecasting Engine...")
            
            # Initialize external connections
            await self._initialize_redis()
            await self._initialize_postgres()
            await self._initialize_messagebus()
            
            # Initialize the orchestrator (models will be initialized on-demand)
            self.start_time = datetime.utcnow()
            self.is_running = True
            
            self.logger.info("Volatility Forecasting Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize volatility engine: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for caching and streaming"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, caching disabled")
            return
        
        try:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
            self.redis_client = redis.from_url(redis_url)
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL connection for data storage"""
        if not POSTGRES_AVAILABLE:
            self.logger.warning("PostgreSQL not available, persistence disabled")
            return
        
        try:
            db_url = os.getenv("DATABASE_URL", "postgresql://nautilus:nautilus123@postgres:5432/nautilus")
            self.postgres_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
            
            self.logger.info("PostgreSQL connection pool created")
            
        except Exception as e:
            self.logger.warning(f"Failed to connect to PostgreSQL: {e}")
            self.postgres_pool = None
    
    async def _initialize_messagebus(self) -> None:
        """Initialize MessageBus client for real-time data streaming"""
        try:
            # Create MessageBus configuration
            messagebus_config = {
                'redis_host': os.getenv('REDIS_HOST', 'redis'),
                'redis_port': int(os.getenv('REDIS_PORT', '6379')),
                'stream_key': self.config.data.redis_stream_key,
                'consumer_group': 'volatility-engine-group',
                'enable_streaming': True
            }
            
            # Create and initialize MessageBus client
            self.messagebus_client = create_volatility_messagebus_client(messagebus_config)
            
            if self.messagebus_client:
                await self.messagebus_client.initialize()
                
                # Register event handlers
                self.messagebus_client.register_data_handler('volatility_trigger', self._handle_volatility_trigger)
                
                # Start streaming
                await self.messagebus_client.start_streaming()
                
                self.logger.info("✅ MessageBus client initialized for real-time volatility updates")
            else:
                self.logger.warning("⚠️ MessageBus not available - real-time updates disabled")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize MessageBus client: {e}")
            self.messagebus_client = None
    
    async def add_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Add a symbol to the volatility engine and initialize its models.
        
        Args:
            symbol: Trading symbol to add
            
        Returns:
            Initialization results
        """
        try:
            if symbol in self.active_symbols:
                return {'status': 'already_active', 'symbol': symbol}
            
            self.logger.info(f"Adding symbol {symbol} to volatility engine")
            
            # Initialize models for the symbol
            models = await self.orchestrator.initialize_models(symbol)
            
            # Create data stream queue
            self.data_streams[symbol] = asyncio.Queue(maxsize=1000)
            
            # Add to active symbols
            self.active_symbols.add(symbol)
            
            # Add symbol to MessageBus streaming
            if self.messagebus_client:
                self.messagebus_client.add_symbol(symbol)
                self.logger.info(f"🔄 Added {symbol} to real-time MessageBus streaming")
            
            # Update performance stats
            self.performance_stats['models_active'] = len(self.active_symbols) * len(models)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'models_initialized': len(models),
                'model_types': list(models.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add symbol {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add symbol {symbol}")
    
    async def train_symbol_models(self, 
                                symbol: str,
                                training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models for a specific symbol.
        
        Args:
            symbol: Trading symbol
            training_data: Historical price data
            
        Returns:
            Training results
        """
        try:
            if symbol not in self.active_symbols:
                await self.add_symbol(symbol)
            
            self.logger.info(f"Training models for {symbol}")
            
            # Train models using orchestrator
            results = await self.orchestrator.train_models(symbol, training_data)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'training_results': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train models for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed for {symbol}")
    
    async def generate_forecast(self, 
                              symbol: str,
                              recent_data: Optional[pd.DataFrame] = None,
                              horizon: int = 5,
                              confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate ensemble volatility forecast for a symbol.
        
        Args:
            symbol: Trading symbol
            recent_data: Recent market data
            horizon: Forecast horizon in days
            confidence_level: Confidence level for intervals
            
        Returns:
            Ensemble forecast results
        """
        try:
            start_time = time.time()
            
            if symbol not in self.active_symbols:
                raise HTTPException(status_code=404, detail=f"Symbol {symbol} not initialized")
            
            # Generate ensemble forecast
            ensemble_forecast = await self.orchestrator.generate_ensemble_forecast(
                symbol, recent_data, horizon, confidence_level
            )
            
            # Cache forecast
            self.forecast_cache[symbol] = ensemble_forecast
            
            # Store in Redis if available
            if self.redis_client:
                await self._cache_forecast(symbol, ensemble_forecast)
            
            # Store in database if available
            if self.postgres_pool:
                await self._store_forecast(ensemble_forecast)
            
            # Update performance stats
            forecast_time_ms = (time.time() - start_time) * 1000
            self.processed_forecasts += 1
            
            # Update average forecast time
            current_avg = self.performance_stats['avg_forecast_time_ms']
            new_avg = ((current_avg * (self.processed_forecasts - 1)) + forecast_time_ms) / self.processed_forecasts
            self.performance_stats['avg_forecast_time_ms'] = new_avg
            self.performance_stats['total_forecasts'] = self.processed_forecasts
            
            return {
                'status': 'success',
                'forecast': ensemble_forecast.to_dict(),
                'generation_time_ms': forecast_time_ms
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to generate forecast for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Forecast generation failed for {symbol}")
    
    async def _handle_volatility_trigger(self, trigger_event: Dict[str, Any]) -> None:
        """Handle volatility update trigger from MessageBus"""
        try:
            symbol = trigger_event.get('symbol', '').upper()
            trigger_type = trigger_event.get('trigger_type', 'unknown')
            return_magnitude = trigger_event.get('return_magnitude')
            
            if symbol not in self.active_symbols:
                return
            
            self.logger.debug(f"🔄 Processing volatility trigger for {symbol} ({trigger_type})")
            
            # Get recent data from MessageBus client
            recent_data = []
            if self.messagebus_client:
                recent_data = await self.messagebus_client.get_recent_data(symbol, limit=50)
            
            # Convert to DataFrame format for models
            if recent_data:
                data_dicts = []
                for event in recent_data:
                    if event.close is not None or event.price is not None:
                        data_dicts.append({
                            'timestamp': event.timestamp,
                            'open': event.open or event.price,
                            'high': event.high or event.price,
                            'low': event.low or event.price,
                            'close': event.close or event.price,
                            'volume': event.volume or 0
                        })
                
                if data_dicts:
                    market_df = pd.DataFrame(data_dicts)
                    market_df.set_index('timestamp', inplace=True)
                    
                    # Generate new forecast based on updated data
                    forecast_result = await self.generate_forecast(symbol, recent_data=market_df)
                    
                    if forecast_result['status'] == 'success':
                        self.logger.info(f"✅ Updated volatility forecast for {symbol} via MessageBus trigger")
                        
                        # Publish updated forecast back to MessageBus
                        if self.messagebus_client and hasattr(self.messagebus_client, 'client'):
                            await self.messagebus_client.client.publish(
                                f"volatility.forecasts.{symbol.lower()}",
                                {
                                    'symbol': symbol,
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'trigger_type': trigger_type,
                                    'forecast': forecast_result['forecast'],
                                    'source': 'volatility_engine'
                                }
                            )
        
        except Exception as e:
            self.logger.error(f"Error handling volatility trigger for {symbol}: {e}")
    
    async def update_real_time_data(self, 
                                  symbol: str,
                                  market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update real-time market data and trigger forecast update if needed.
        
        Args:
            symbol: Trading symbol
            market_data: Real-time market data (OHLCV)
            
        Returns:
            Updated forecast if triggered, None otherwise
        """
        try:
            if symbol not in self.active_symbols:
                return None
            
            # Add to data stream
            if symbol in self.data_streams:
                try:
                    self.data_streams[symbol].put_nowait(market_data)
                except asyncio.QueueFull:
                    # Remove oldest item and add new one
                    try:
                        self.data_streams[symbol].get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self.data_streams[symbol].put_nowait(market_data)
            
            # Update orchestrator with real-time data
            updated_forecast = await self.orchestrator.update_real_time(symbol, market_data)
            
            if updated_forecast:
                # Cache updated forecast
                self.forecast_cache[symbol] = updated_forecast
                
                # Store updates
                if self.redis_client:
                    await self._cache_forecast(symbol, updated_forecast)
                
                return {
                    'status': 'updated',
                    'forecast': updated_forecast.to_dict(),
                    'update_triggered': True
                }
            
            return {
                'status': 'received',
                'update_triggered': False
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update real-time data for {symbol}: {e}")
            return None
    
    async def get_latest_forecast(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest forecast for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest forecast or None if not available
        """
        try:
            # Try cache first
            if symbol in self.forecast_cache:
                forecast = self.forecast_cache[symbol]
                return {
                    'status': 'success',
                    'source': 'cache',
                    'forecast': forecast.to_dict()
                }
            
            # Try Redis cache
            if self.redis_client:
                cached_forecast = await self._get_cached_forecast(symbol)
                if cached_forecast:
                    return {
                        'status': 'success',
                        'source': 'redis',
                        'forecast': cached_forecast
                    }
            
            # Try orchestrator history
            forecast = self.orchestrator.get_latest_ensemble_forecast(symbol)
            if forecast:
                return {
                    'status': 'success',
                    'source': 'orchestrator',
                    'forecast': forecast.to_dict()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest forecast for {symbol}: {e}")
            return None
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        # Get model status for all symbols
        model_status = {}
        for symbol in self.active_symbols:
            model_status[symbol] = self.orchestrator.get_model_status(symbol)
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': uptime,
            'active_symbols': list(self.active_symbols),
            'models_per_symbol': {
                symbol: status.get('total_models', 0) 
                for symbol, status in model_status.items()
            },
            'performance_stats': self.performance_stats,
            'hardware_acceleration': {
                'enabled': self.config.hardware.auto_hardware_routing,
                'metal_gpu': self.config.hardware.use_metal_gpu,
                'neural_engine': self.config.hardware.use_neural_engine,
                'cpu_optimization': self.config.hardware.use_cpu_optimization
            },
            'external_services': {
                'redis_connected': self.redis_client is not None,
                'postgres_connected': self.postgres_pool is not None,
                'messagebus_connected': self.messagebus_client is not None and self.messagebus_client.is_running,
                'messagebus_streaming_stats': await self._get_messagebus_stats() if self.messagebus_client else {}
            },
            'configuration': {
                'ensemble_method': self.config.ensemble.method.value,
                'min_models': self.config.ensemble.min_models,
                'max_models': self.config.ensemble.max_models,
                'rebalance_frequency': str(self.config.ensemble.rebalance_frequency)
            },
            'model_status': model_status
        }
    
    async def _cache_forecast(self, symbol: str, forecast: EnsembleForecast) -> None:
        """Cache forecast in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"volatility:forecast:{symbol}"
            value = json.dumps(forecast.to_dict(), default=str)
            await self.redis_client.setex(key, self.config.data.cache_ttl_seconds, value)
        except Exception as e:
            self.logger.warning(f"Failed to cache forecast for {symbol}: {e}")
    
    async def _get_cached_forecast(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached forecast from Redis"""
        if not self.redis_client:
            return None
        
        try:
            key = f"volatility:forecast:{symbol}"
            cached = await self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            self.logger.warning(f"Failed to get cached forecast for {symbol}: {e}")
        
        return None
    
    async def _store_forecast(self, forecast: EnsembleForecast) -> None:
        """Store forecast in PostgreSQL"""
        if not self.postgres_pool:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO volatility_forecasts (
                        symbol, forecast_timestamp, ensemble_volatility, 
                        ensemble_variance, ensemble_confidence, model_weights,
                        forecast_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, forecast_timestamp) 
                    DO UPDATE SET
                        ensemble_volatility = EXCLUDED.ensemble_volatility,
                        ensemble_variance = EXCLUDED.ensemble_variance,
                        ensemble_confidence = EXCLUDED.ensemble_confidence,
                        model_weights = EXCLUDED.model_weights,
                        forecast_data = EXCLUDED.forecast_data
                """, 
                forecast.symbol,
                forecast.forecast_timestamp,
                forecast.ensemble_volatility,
                forecast.ensemble_variance,
                forecast.ensemble_confidence,
                json.dumps(forecast.model_weights),
                json.dumps(forecast.to_dict(), default=str)
            )
        except Exception as e:
            self.logger.warning(f"Failed to store forecast: {e}")
    
    async def _get_messagebus_stats(self) -> Dict[str, Any]:
        """Get MessageBus streaming statistics"""
        try:
            if self.messagebus_client:
                return await self.messagebus_client.get_streaming_stats()
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to get MessageBus stats: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the volatility engine"""
        try:
            self.logger.info("Shutting down Volatility Forecasting Engine...")
            
            self.is_running = False
            
            # Cleanup orchestrator
            await self.orchestrator.cleanup()
            
            # Close external connections
            if self.messagebus_client:
                await self.messagebus_client.stop_streaming()
                
            if self.redis_client:
                await self.redis_client.close()
            
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            # Clear caches
            self.forecast_cache.clear()
            self.data_streams.clear()
            self.active_symbols.clear()
            
            self.logger.info("Volatility Forecasting Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global engine instance
volatility_engine: Optional[VolatilityEngine] = None


async def get_engine() -> VolatilityEngine:
    """Get the global volatility engine instance"""
    global volatility_engine
    
    if volatility_engine is None:
        config = VolatilityConfig.from_environment()
        volatility_engine = VolatilityEngine(config)
        await volatility_engine.initialize()
    
    return volatility_engine


async def shutdown_engine() -> None:
    """Shutdown the global engine instance"""
    global volatility_engine
    
    if volatility_engine:
        await volatility_engine.shutdown()
        volatility_engine = None
#!/usr/bin/env python3
"""
Simple Feature Engineering Engine - Containerized Feature Processing Service
High-performance feature calculation and engineering for trading signals
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO_ECONOMIC = "macro_economic"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

@dataclass
class Feature:
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    value: float
    confidence: float
    timestamp: datetime
    lookback_period: int = 20

@dataclass
class FeatureSet:
    symbol: str
    timestamp: datetime
    features: Dict[str, Feature]
    total_features: int
    processing_time_ms: float

class SimpleFeaturesEngine:
    """
    Simple Feature Engineering Engine demonstrating containerization approach
    """
    
    def __init__(self):
        self.app = FastAPI(title="Nautilus Simple Features Engine", version="1.0.0")
        self.is_running = False
        self.features_calculated = 0
        self.feature_sets_processed = 0
        self.start_time = time.time()
        
        # Feature computation state
        self.available_features: Dict[str, str] = {}
        self.feature_cache: Dict[str, Dict] = {}
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "features_calculated": self.features_calculated,
                "feature_sets_processed": self.feature_sets_processed,
                "available_features": len(self.available_features),
                "cache_size": len(self.feature_cache),
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            uptime = time.time() - self.start_time
            return {
                "features_per_second": self.features_calculated / max(1, uptime),
                "feature_sets_per_second": self.feature_sets_processed / max(1, uptime),
                "total_features": self.features_calculated,
                "total_feature_sets": self.feature_sets_processed,
                "available_features": len(self.available_features),
                "cache_hit_ratio": self._calculate_cache_hit_ratio(),
                "uptime": uptime,
                "engine_type": "feature_engineering",
                "containerized": True
            }
        
        @self.app.get("/features/available")
        async def get_available_features():
            """Get all available feature types"""
            return {
                "features": self.available_features,
                "count": len(self.available_features),
                "categories": [ft.value for ft in FeatureType]
            }
        
        @self.app.post("/features/calculate/{symbol}")
        async def calculate_features(symbol: str, market_data: Dict[str, Any]):
            """Calculate comprehensive feature set for symbol"""
            try:
                start_time = time.time()
                
                # Calculate feature set
                feature_set = await self._calculate_feature_set(symbol, market_data)
                self.feature_sets_processed += 1
                
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "completed",
                    "symbol": symbol,
                    "feature_set": {
                        "timestamp": feature_set.timestamp.isoformat(),
                        "total_features": feature_set.total_features,
                        "features": {
                            name: {
                                "value": feat.value,
                                "type": feat.feature_type.value,
                                "confidence": feat.confidence
                            }
                            for name, feat in feature_set.features.items()
                        }
                    },
                    "processing_time_ms": processing_time
                }
                
            except Exception as e:
                logger.error(f"Feature calculation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/technical/{symbol}")
        async def calculate_technical_features(symbol: str, price_data: Dict[str, Any]):
            """Calculate technical analysis features"""
            try:
                features = await self._calculate_technical_features(symbol, price_data)
                self.features_calculated += len(features)
                
                return {
                    "status": "completed",
                    "symbol": symbol,
                    "feature_type": "technical",
                    "features": features,
                    "count": len(features)
                }
                
            except Exception as e:
                logger.error(f"Technical features error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/fundamental/{symbol}")
        async def calculate_fundamental_features(symbol: str, fundamental_data: Dict[str, Any]):
            """Calculate fundamental analysis features"""
            try:
                features = await self._calculate_fundamental_features(symbol, fundamental_data)
                self.features_calculated += len(features)
                
                return {
                    "status": "completed",
                    "symbol": symbol,
                    "feature_type": "fundamental",
                    "features": features,
                    "count": len(features)
                }
                
            except Exception as e:
                logger.error(f"Fundamental features error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/features/batch")
        async def calculate_batch_features(batch_request: Dict[str, Any]):
            """Calculate features for multiple symbols"""
            try:
                symbols = batch_request.get("symbols", [])
                market_data = batch_request.get("market_data", {})
                
                results = {}
                for symbol in symbols:
                    if symbol in market_data:
                        feature_set = await self._calculate_feature_set(symbol, market_data[symbol])
                        results[symbol] = {
                            "total_features": feature_set.total_features,
                            "timestamp": feature_set.timestamp.isoformat(),
                            "processing_time_ms": feature_set.processing_time_ms
                        }
                        self.feature_sets_processed += 1
                
                return {
                    "status": "completed",
                    "batch_size": len(symbols),
                    "processed": len(results),
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Batch features error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def start_engine(self):
        """Start the features engine"""
        try:
            logger.info("Starting Simple Features Engine...")
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Initialize available features registry
            await self._initialize_feature_registry()
            
            self.is_running = True
            logger.info(f"Simple Features Engine started successfully with {len(self.available_features)} feature types")
            
        except Exception as e:
            logger.error(f"Failed to start Features Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the features engine"""
        logger.info("Stopping Simple Features Engine...")
        self.is_running = False
        
        if self.messagebus:
            await self.messagebus.stop()
        
        logger.info("Simple Features Engine stopped")
    
    async def _initialize_feature_registry(self):
        """Initialize available features registry"""
        self.available_features = {
            # Technical Features
            "sma_20": "20-period Simple Moving Average",
            "ema_20": "20-period Exponential Moving Average", 
            "rsi_14": "14-period Relative Strength Index",
            "macd": "Moving Average Convergence Divergence",
            "bollinger_bands": "Bollinger Bands (2 std dev)",
            "atr_14": "14-period Average True Range",
            "volume_sma": "Volume Simple Moving Average",
            
            # Momentum Features
            "momentum_10": "10-period Price Momentum",
            "roc_12": "12-period Rate of Change",
            "williams_r": "Williams %R Oscillator",
            "stochastic_k": "Stochastic %K",
            "stochastic_d": "Stochastic %D",
            
            # Volatility Features
            "realized_vol": "Realized Volatility (20-day)",
            "garch_vol": "GARCH Volatility Estimate",
            "vol_regime": "Volatility Regime Indicator",
            "vol_clustering": "Volatility Clustering Score",
            
            # Volume Features
            "volume_profile": "Volume Profile Analysis",
            "vwap": "Volume Weighted Average Price",
            "volume_rsi": "Volume-based RSI",
            "accumulation_distribution": "Accumulation/Distribution Line",
            
            # Fundamental Features (when data available)
            "pe_ratio": "Price-to-Earnings Ratio",
            "pb_ratio": "Price-to-Book Ratio",
            "roe": "Return on Equity",
            "debt_to_equity": "Debt-to-Equity Ratio",
            "revenue_growth": "Revenue Growth Rate"
        }
        
        logger.info(f"Initialized {len(self.available_features)} feature definitions")
    
    async def _calculate_feature_set(self, symbol: str, market_data: Dict[str, Any]) -> FeatureSet:
        """Calculate comprehensive feature set for symbol"""
        start_time = time.time()
        
        # Simulate feature calculation processing time
        await asyncio.sleep(0.003)  # 3ms processing time
        
        features = {}
        
        # Extract market data
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])
        current_price = market_data.get("current_price", 100.0)
        
        if not prices:
            # Generate mock price series
            prices = [current_price * (1 + np.random.normal(0, 0.02)) for _ in range(20)]
        if not volumes:
            volumes = [np.random.randint(1000, 10000) for _ in range(len(prices))]
        
        # Calculate technical features
        tech_features = await self._calculate_technical_features(symbol, {
            "prices": prices,
            "volumes": volumes,
            "current_price": current_price
        })
        
        # Convert to Feature objects
        for feature_name, value in tech_features.items():
            features[feature_name] = Feature(
                feature_id=f"{symbol}_{feature_name}_{int(time.time())}",
                feature_name=feature_name,
                feature_type=self._determine_feature_type(feature_name),
                value=value,
                confidence=np.random.uniform(0.6, 0.95),
                timestamp=datetime.now()
            )
        
        processing_time = (time.time() - start_time) * 1000
        self.features_calculated += len(features)
        
        return FeatureSet(
            symbol=symbol,
            timestamp=datetime.now(),
            features=features,
            total_features=len(features),
            processing_time_ms=processing_time
        )
    
    async def _calculate_technical_features(self, symbol: str, price_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate technical analysis features"""
        prices = np.array(price_data.get("prices", []))
        volumes = np.array(price_data.get("volumes", []))
        current_price = price_data.get("current_price", 100.0)
        
        if len(prices) == 0:
            prices = np.array([current_price])
        
        features = {}
        
        try:
            # Moving averages
            if len(prices) >= 20:
                features["sma_20"] = np.mean(prices[-20:])
                features["ema_20"] = self._calculate_ema(prices, 20)
            else:
                features["sma_20"] = np.mean(prices)
                features["ema_20"] = current_price
            
            # RSI
            features["rsi_14"] = self._calculate_rsi(prices, 14)
            
            # MACD
            macd_line, signal_line = self._calculate_macd(prices)
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = macd_line - signal_line
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            features["bb_upper"] = bb_upper
            features["bb_middle"] = bb_middle
            features["bb_lower"] = bb_lower
            features["bb_width"] = (bb_upper - bb_lower) / bb_middle
            
            # Volatility
            features["realized_vol"] = np.std(np.diff(np.log(prices))) * np.sqrt(252)
            
            # Momentum
            if len(prices) >= 10:
                features["momentum_10"] = (prices[-1] - prices[-10]) / prices[-10]
            else:
                features["momentum_10"] = 0.0
            
            # Volume features (if available)
            if len(volumes) > 0:
                features["volume_sma"] = np.mean(volumes)
                features["vwap"] = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else current_price
            
        except Exception as e:
            logger.warning(f"Error calculating technical features: {e}")
            # Return basic features on error
            features = {
                "sma_20": current_price,
                "rsi_14": 50.0,
                "realized_vol": 0.2,
                "momentum_10": 0.0
            }
        
        return features
    
    async def _calculate_fundamental_features(self, symbol: str, fundamental_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fundamental analysis features"""
        features = {}
        
        # Extract fundamental data
        market_cap = fundamental_data.get("market_cap", 0)
        pe_ratio = fundamental_data.get("pe_ratio", 15.0)
        pb_ratio = fundamental_data.get("pb_ratio", 2.0)
        
        # Calculate or use provided fundamental features
        features["pe_ratio"] = pe_ratio
        features["pb_ratio"] = pb_ratio
        features["market_cap_log"] = np.log10(market_cap) if market_cap > 0 else 0
        
        # Mock additional fundamental features
        features["roe"] = np.random.uniform(0.05, 0.25)
        features["debt_to_equity"] = np.random.uniform(0.1, 2.0)
        features["revenue_growth"] = np.random.uniform(-0.1, 0.3)
        
        return features
    
    def _determine_feature_type(self, feature_name: str) -> FeatureType:
        """Determine feature type based on name"""
        if any(tech in feature_name.lower() for tech in ["sma", "ema", "rsi", "macd", "bb"]):
            return FeatureType.TECHNICAL
        elif any(vol in feature_name.lower() for vol in ["vol", "volatility"]):
            return FeatureType.VOLATILITY
        elif any(mom in feature_name.lower() for mom in ["momentum", "roc"]):
            return FeatureType.MOMENTUM
        elif any(fund in feature_name.lower() for fund in ["pe", "pb", "roe", "debt"]):
            return FeatureType.FUNDAMENTAL
        elif "volume" in feature_name.lower():
            return FeatureType.VOLUME
        else:
            return FeatureType.TECHNICAL
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Simplified signal line calculation
        signal_line = macd_line * 0.9  # Mock signal line
        
        return macd_line, signal_line
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices)
        else:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        if self.feature_sets_processed == 0:
            return 0.0
        return min(1.0, len(self.feature_cache) / self.feature_sets_processed)

# Create and start the engine
simple_features_engine = SimpleFeaturesEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        # For now, use simple engine with hybrid mode flag
        logger.info("Hybrid Features Engine integration enabled (using enhanced simple engine)")
        app = simple_features_engine.app
        engine_instance = simple_features_engine
        # Add hybrid flag to engine
        engine_instance.hybrid_enabled = True
    except Exception as e:
        logger.warning(f"Hybrid Features Engine setup failed: {e}. Using simple engine.")
        app = simple_features_engine.app
        engine_instance = simple_features_engine
else:
    logger.info("Using Simple Features Engine (hybrid disabled)")
    app = simple_features_engine.app
    engine_instance = simple_features_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8500"))
    
    logger.info(f"Starting Features Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
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
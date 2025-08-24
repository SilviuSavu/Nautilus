#!/usr/bin/env python3
"""
Simple ML Inference Engine - Containerized Machine Learning Service
High-performance ML inference with model prediction capabilities
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Hardware router for intelligent workload routing
try:
    from backend.hardware_router import (
        HardwareRouter,
        WorkloadType,
        WorkloadCharacteristics,
        hardware_accelerated,
        route_ml_workload
    )
    from backend.acceleration import (
        risk_predict,
        initialize_coreml_acceleration,
        get_acceleration_status,
        is_m4_max_detected
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False
    # logger will be defined below

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log hardware acceleration status
if not HARDWARE_ACCELERATION_AVAILABLE:
    logger.warning("Hardware acceleration not available - running in CPU-only mode")

class ModelType(Enum):
    PRICE_PREDICTION = "price_prediction"
    REGIME_DETECTION = "regime_detection"
    VOLATILITY_FORECASTING = "volatility_forecasting"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_PREDICTION = "risk_prediction"

@dataclass
class MLModel:
    model_id: str
    model_type: ModelType
    version: str
    accuracy: float
    last_trained: datetime
    enabled: bool = True

@dataclass
class MLPrediction:
    prediction_id: str
    model_id: str
    input_data: Dict[str, Any]
    prediction: Any
    confidence: float
    processing_time_ms: float
    timestamp: datetime

class SimpleMLEngine:
    """
    Simple ML Inference Engine demonstrating containerization approach
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Nautilus Simple ML Engine", 
            version="1.0.0",
            lifespan=self.lifespan
        )
        self.is_running = False
        self.predictions_made = 0
        self.models_loaded = 0
        self.start_time = time.time()
        
        # ML models registry
        self.loaded_models: Dict[str, MLModel] = {}
        
        # Hardware acceleration
        self.hardware_router = None
        self.m4_max_detected = False
        self.neural_acceleration_available = False
        self.hardware_acceleration_metrics = {
            "neural_engine_predictions": 0,
            "cpu_fallback_predictions": 0,
            "avg_neural_inference_time_ms": 0.0,
            "avg_cpu_inference_time_ms": 0.0,
            "hardware_acceleration_ratio": 0.0
        }
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.setup_routes()
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan management"""
        # Startup
        await self.start_engine()
        yield
        # Shutdown
        await self.stop_engine()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "predictions_made": self.predictions_made,
                "models_loaded": self.models_loaded,
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "predictions_per_second": self.predictions_made / max(1, time.time() - self.start_time),
                "total_predictions": self.predictions_made,
                "models_loaded": self.models_loaded,
                "active_models": len([m for m in self.loaded_models.values() if m.enabled]),
                "uptime": time.time() - self.start_time,
                "engine_type": "ml_inference",
                "containerized": True,
                "m4_max_detected": self.m4_max_detected,
                "neural_acceleration_available": self.neural_acceleration_available,
                "hardware_acceleration_metrics": self.hardware_acceleration_metrics
            }
        
        @self.app.get("/models")
        async def get_loaded_models():
            """Get all loaded models"""
            return {
                "models": [
                    {
                        "model_id": model.model_id,
                        "model_type": model.model_type.value,
                        "version": model.version,
                        "accuracy": model.accuracy,
                        "last_trained": model.last_trained.isoformat(),
                        "enabled": model.enabled
                    }
                    for model in self.loaded_models.values()
                ],
                "count": len(self.loaded_models)
            }
        
        @self.app.post("/ml/predict/{model_id}")
        async def make_prediction(model_id: str, input_data: Dict[str, Any]):
            """Make ML prediction using specified model"""
            try:
                if model_id not in self.loaded_models:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                
                model = self.loaded_models[model_id]
                if not model.enabled:
                    raise HTTPException(status_code=400, detail=f"Model {model_id} is disabled")
                
                # Make prediction
                prediction_result = await self._make_prediction(model, input_data)
                self.predictions_made += 1
                
                return {
                    "status": "success",
                    "prediction_id": prediction_result.prediction_id,
                    "model_id": model_id,
                    "prediction": prediction_result.prediction,
                    "confidence": prediction_result.confidence,
                    "processing_time_ms": prediction_result.processing_time_ms
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"ML prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/price")
        async def predict_price(data: Dict[str, Any]):
            """Predict price movement"""
            try:
                # Use price prediction model
                model = self.loaded_models.get("price_model_v1")
                if not model:
                    # Create mock prediction
                    prediction = await self._mock_price_prediction(data)
                else:
                    prediction_result = await self._make_prediction(model, data)
                    prediction = prediction_result.prediction
                
                self.predictions_made += 1
                
                return {
                    "prediction_type": "price_movement",
                    "symbol": data.get("symbol", "UNKNOWN"),
                    "current_price": data.get("current_price", 0),
                    "predicted_direction": prediction.get("direction", "NEUTRAL"),
                    "predicted_change_percent": prediction.get("change_percent", 0.0),
                    "confidence": prediction.get("confidence", 0.5),
                    "time_horizon": data.get("time_horizon", "1d")
                }
                
            except Exception as e:
                logger.error(f"Price prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/regime")
        async def detect_market_regime(data: Dict[str, Any]):
            """Detect current market regime"""
            try:
                # Market regime detection
                regime_result = await self._detect_market_regime(data)
                self.predictions_made += 1
                
                return {
                    "prediction_type": "market_regime",
                    "current_regime": regime_result.get("regime", "NORMAL"),
                    "confidence": regime_result.get("confidence", 0.5),
                    "regime_probabilities": regime_result.get("probabilities", {}),
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Regime detection error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/volatility")
        async def forecast_volatility(data: Dict[str, Any]):
            """Forecast volatility"""
            try:
                # Volatility forecasting
                vol_result = await self._forecast_volatility(data)
                self.predictions_made += 1
                
                return {
                    "prediction_type": "volatility_forecast",
                    "symbol": data.get("symbol", "UNKNOWN"),
                    "current_volatility": data.get("current_vol", 0),
                    "forecasted_volatility": vol_result.get("forecast", 0),
                    "forecast_horizon": data.get("horizon", "1w"),
                    "confidence": vol_result.get("confidence", 0.5)
                }
                
            except Exception as e:
                logger.error(f"Volatility forecasting error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def start_engine(self):
        """Start the ML engine"""
        try:
            logger.info("Starting Simple ML Engine...")
            
            # Initialize hardware acceleration if available
            if HARDWARE_ACCELERATION_AVAILABLE:
                await self._initialize_hardware_acceleration()
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Load default models
            await self._load_default_models()
            
            self.is_running = True
            logger.info(f"Simple ML Engine started successfully with {self.models_loaded} models")
            if self.neural_acceleration_available:
                logger.info("🚀 Neural Engine acceleration ACTIVE for ML inference")
            
        except Exception as e:
            logger.error(f"Failed to start ML Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the ML engine"""
        logger.info("Stopping Simple ML Engine...")
        self.is_running = False
        
        if self.messagebus:
            await self.messagebus.stop()
        
        logger.info("Simple ML Engine stopped")
    
    async def _initialize_hardware_acceleration(self):
        """Initialize M4 Max hardware acceleration"""
        try:
            logger.info("Initializing ML Engine hardware acceleration...")
            
            # Initialize hardware router
            self.hardware_router = HardwareRouter()
            
            # Check M4 Max detection
            self.m4_max_detected = is_m4_max_detected()
            
            # Initialize Neural Engine acceleration
            acceleration_status = await initialize_coreml_acceleration(enable_logging=True)
            self.neural_acceleration_available = acceleration_status.get("neural_engine_available", False)
            
            if self.neural_acceleration_available:
                logger.info("✅ Neural Engine acceleration initialized for ML inference")
            else:
                logger.info("ℹ️ Neural Engine not available - using CPU-only inference")
            
            logger.info(f"M4 Max detected: {self.m4_max_detected}")
            
        except Exception as e:
            logger.warning(f"Hardware acceleration initialization failed: {e}")
            self.hardware_router = None
            self.neural_acceleration_available = False
    
    async def _load_default_models(self):
        """Load default ML models"""
        default_models = [
            MLModel(
                model_id="price_model_v1",
                model_type=ModelType.PRICE_PREDICTION,
                version="1.0.0",
                accuracy=0.72,
                last_trained=datetime.now(),
                enabled=True
            ),
            MLModel(
                model_id="regime_detector_v2",
                model_type=ModelType.REGIME_DETECTION,
                version="2.1.0",
                accuracy=0.68,
                last_trained=datetime.now(),
                enabled=True
            ),
            MLModel(
                model_id="volatility_forecaster",
                model_type=ModelType.VOLATILITY_FORECASTING,
                version="1.5.0",
                accuracy=0.65,
                last_trained=datetime.now(),
                enabled=True
            ),
            MLModel(
                model_id="risk_predictor",
                model_type=ModelType.RISK_PREDICTION,
                version="1.2.0",
                accuracy=0.74,
                last_trained=datetime.now(),
                enabled=True
            )
        ]
        
        for model in default_models:
            self.loaded_models[model.model_id] = model
            self.models_loaded += 1
        
        logger.info(f"Loaded {len(default_models)} default ML models")
    
    async def _make_prediction(self, model: MLModel, input_data: Dict[str, Any]) -> MLPrediction:
        """Make prediction using intelligent hardware routing"""
        start_time = time.time()
        prediction = None
        used_hardware = "CPU"
        
        # Try hardware-accelerated prediction first
        if self.hardware_router and self.neural_acceleration_available:
            try:
                # Get routing decision for ML workload
                data_size = len(str(input_data))  # Approximate data size
                routing_decision = await route_ml_workload(data_size=data_size)
                
                # If Neural Engine is recommended, try it first
                if routing_decision.primary_hardware.name == 'NEURAL_ENGINE':
                    neural_start = time.time()
                    
                    # Use Neural Engine for inference
                    neural_prediction = await risk_predict(
                        {
                            "model_type": model.model_type.value,
                            "input_data": input_data,
                            "model_id": model.model_id
                        },
                        model_id=f"ml_{model.model_type.value}_v1"
                    )
                    
                    if neural_prediction and not neural_prediction.get("error"):
                        neural_time = (time.time() - neural_start) * 1000
                        
                        # Update metrics
                        self.hardware_acceleration_metrics["neural_engine_predictions"] += 1
                        self.hardware_acceleration_metrics["avg_neural_inference_time_ms"] = (
                            0.9 * self.hardware_acceleration_metrics["avg_neural_inference_time_ms"] + 
                            0.1 * neural_time
                        )
                        
                        prediction = neural_prediction
                        used_hardware = "Neural Engine"
                        
                        logger.debug(f"Neural Engine prediction completed in {neural_time:.2f}ms "
                                   f"(estimated gain: {routing_decision.estimated_performance_gain:.1f}x)")
            
            except Exception as e:
                logger.debug(f"Neural Engine prediction failed: {e} - falling back to CPU")
        
        # Fallback to CPU prediction if Neural Engine failed or unavailable
        if prediction is None:
            cpu_start = time.time()
            
            # Simulate model inference time (longer for CPU)
            await asyncio.sleep(0.005)  # 5ms inference time for CPU
            
            # Generate mock prediction based on model type
            if model.model_type == ModelType.PRICE_PREDICTION:
                prediction = await self._mock_price_prediction(input_data)
            elif model.model_type == ModelType.REGIME_DETECTION:
                prediction = await self._detect_market_regime(input_data)
            elif model.model_type == ModelType.VOLATILITY_FORECASTING:
                prediction = await self._forecast_volatility(input_data)
            elif model.model_type == ModelType.RISK_PREDICTION:
                prediction = await self._predict_risk(input_data)
            else:
                prediction = {"value": np.random.normal(0, 1), "confidence": 0.5}
            
            cpu_time = (time.time() - cpu_start) * 1000
            
            # Update CPU metrics
            self.hardware_acceleration_metrics["cpu_fallback_predictions"] += 1
            self.hardware_acceleration_metrics["avg_cpu_inference_time_ms"] = (
                0.9 * self.hardware_acceleration_metrics["avg_cpu_inference_time_ms"] + 
                0.1 * cpu_time
            )
        
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        
        # Update acceleration ratio
        if self.hardware_acceleration_metrics["cpu_fallback_predictions"] > 0:
            cpu_avg = self.hardware_acceleration_metrics["avg_cpu_inference_time_ms"]
            neural_avg = self.hardware_acceleration_metrics["avg_neural_inference_time_ms"]
            if neural_avg > 0:
                self.hardware_acceleration_metrics["hardware_acceleration_ratio"] = cpu_avg / neural_avg
        
        return MLPrediction(
            prediction_id=f"pred_{int(time.time())}{np.random.randint(1000, 9999)}",
            model_id=model.model_id,
            input_data=input_data,
            prediction={**prediction, "used_hardware": used_hardware},
            confidence=prediction.get("confidence", 0.5),
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
    
    async def _mock_price_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock price prediction"""
        current_price = data.get("current_price", 100)
        
        # Simulate ML model prediction
        direction_prob = np.random.random()
        direction = "UP" if direction_prob > 0.5 else "DOWN"
        change_percent = np.random.normal(0, 0.02)  # 2% volatility
        confidence = np.random.uniform(0.3, 0.9)
        
        return {
            "direction": direction,
            "change_percent": change_percent,
            "target_price": current_price * (1 + change_percent),
            "confidence": confidence
        }
    
    async def _detect_market_regime(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market regime"""
        regimes = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE", "CALM"]
        
        # Mock regime probabilities
        probabilities = {}
        remaining_prob = 1.0
        
        for i, regime in enumerate(regimes[:-1]):
            prob = np.random.uniform(0, remaining_prob)
            probabilities[regime] = prob
            remaining_prob -= prob
        probabilities[regimes[-1]] = remaining_prob
        
        # Determine dominant regime
        dominant_regime = max(probabilities.keys(), key=lambda k: probabilities[k])
        
        return {
            "regime": dominant_regime,
            "confidence": probabilities[dominant_regime],
            "probabilities": probabilities
        }
    
    async def _forecast_volatility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast volatility"""
        current_vol = data.get("current_vol", 0.2)
        
        # Mock volatility forecast
        vol_change = np.random.normal(0, 0.05)  # 5% volatility change
        forecasted_vol = max(0.01, current_vol + vol_change)  # Minimum 1% vol
        
        confidence = np.random.uniform(0.4, 0.8)
        
        return {
            "forecast": forecasted_vol,
            "confidence": confidence,
            "vol_change": vol_change
        }
    
    async def _predict_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risk metrics"""
        portfolio_value = data.get("portfolio_value", 100000)
        
        # Mock risk prediction
        var_prediction = -abs(np.random.normal(portfolio_value * 0.05, portfolio_value * 0.02))
        risk_score = np.random.uniform(0, 1)
        confidence = np.random.uniform(0.5, 0.85)
        
        return {
            "predicted_var": var_prediction,
            "risk_score": risk_score,
            "confidence": confidence
        }

# Create and start the engine
simple_ml_engine = SimpleMLEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        from hybrid_ml_engine import hybrid_ml_engine
        logger.info("Hybrid ML Engine integration enabled")
        # Use hybrid engine as the primary engine
        app = hybrid_ml_engine.app
        engine_instance = hybrid_ml_engine
    except ImportError as e:
        logger.warning(f"Hybrid ML Engine not available: {e}. Using simple engine.")
        app = simple_ml_engine.app
        engine_instance = simple_ml_engine
else:
    logger.info("Using Simple ML Engine (hybrid disabled)")
    app = simple_ml_engine.app
    engine_instance = simple_ml_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8400"))
    
    logger.info(f"Starting ML Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
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